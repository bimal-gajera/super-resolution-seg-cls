"""Segmentation decoder in the style of SEED-SR's U-Net decoder.

IMPORTANT SCOPE NOTE. SEED-SR's Sec. 4.3.2 / Table 11 describes the decoder of
its *diffusion* U-Net ``G_theta``, which predicts a clean HR embedding. Its
actual *segmentation* decoder is ``D_HSM``, the frozen half of the HR-FM, whose
reference [49] is a Google blog post with no published architecture. So this
module reproduces the published decoder's *design* -- transposed-convolution
blocks, SFT conditioning, a late cross-attention block, a final convolution --
applied to our segmentation problem. It is a faithful adaptation, not a port.

Their layout (Table 11), and ours beside it:

    SEED-SR                         here
    ---------------------------     ------------------------------------------
    Decoder Block 1 (15,15,640)     Block 1   99 -> 198,  384 ch, skip DINOv3 tap 2
    SFT Block 1     <- e_l          SFT 1     conditioned on the LR S2 input
    (attention at the end)          Attention cross-attn, Q=features, K/V=context
    Decoder Block 2 (30,30,560)     Block 2   198 -> 396, 256 ch, skip tap 1
    Decoder Block 3 (60,60,480)     Block 3   396 -> 792, 128 ch, skip tap 0
    SFT Block 2     <- e_r          SFT 2     conditioned on the LR input again
    Decoder Block 4 (120,120,320)   Block 4   792,  64 ch, no upsample
    Attention Block                 (moved earlier, see below)
    Convolution Block               head      1x1 conv -> 1 logit, resize to 960

Three deviations, all forced by our setup rather than chosen:

1. **No reference embedding.** They condition SFT 2 on ``e_r``, an HR image from
   another date. We have none, so both SFT blocks take the LR input. The block
   is kept so a reference can be dropped in later.
2. **No timestep embedding.** Nothing here is diffusing.
3. **Attention is applied earlier.** Theirs runs at the final 120x120 = 14,400
   tokens. Our final stage is 792x792 = 627k tokens, where that is not
   affordable, so it runs at 198x198 = 39k tokens instead. Keys/values stay a
   small pooled 7x7 context, mirroring the 7x7 downsampling they use to make
   attention tractable.

Only 3 of the 4 blocks upsample: their bottleneck is 7x7 feeding a 120x120
output (~17x), ours is a 99x99 token grid feeding 960x960 (~9.7x, i.e. ~3.3
doublings).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import DINOV3_EMBED_DIM

DECODER_WIDTHS = (384, 256, 128, 64)
STEM_WIDTHS = (24, 48, 96)          # features at 792, 396, 198


class HighResStem(nn.Module):
    """Strided convolutions over the input image, giving the decoder real
    high-frequency skips.

    Without this the decoder has none: all four DINOv3 taps sit at 99x99
    (4.85 m/token), so they differ in *depth* but not *scale*, and a 9.7x
    upsample to 960x960 has to invent every edge. That is what makes the
    predictions rounded blobs rather than straight-edged parcels.

    SEED-SR's decoder does not have this problem -- its conv encoder supplies
    genuinely multi-scale skips (120/60/30/15/7 in Table 11), so its last
    decoder block receives 120x120 detail directly. Adding this stem restores
    that property; it is a fidelity fix, not an embellishment.
    """

    def __init__(self, in_channels: int = 3, widths: tuple[int, ...] = STEM_WIDTHS) -> None:
        super().__init__()
        w1, w2, w3 = widths
        self.down1 = self._block(in_channels, w1)   # 1584 -> 792
        self.down2 = self._block(w1, w2)            # 792  -> 396
        self.down3 = self._block(w2, w3)            # 396  -> 198

    @staticmethod
    def _block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f792 = self.down1(image)
        f396 = self.down2(f792)
        f198 = self.down3(f396)
        return f198, f396, f792


class SFTBlock(nn.Module):
    """Spatial Feature Transform (Wang et al. 2018): ``F_out = gamma * F_in + beta``.

    Internals follow SEED-SR C.3.3, which specifies its own variant: "each SFT
    block first processes its conditioning input through a pair of 3x3
    convolutional layers with ReLU activations (transforming channels to 128,
    then 64)", applying ``F_out = gamma * F_in + beta``.

    That differs from both cited sources, so the divergences are deliberate:

    * **SFTGAN [58]** (``xinntao/SFTGAN``) uses 1x1 convs, LeakyReLU 0.1, and
      ``x * (scale + 1) + shift``. We keep only the ``+1``, which makes the
      block identity at init -- SEED-SR writes a plain ``gamma`` but an
      unbiased ``gamma`` near zero would zero the features on step one.
    * **RefDiff [10]** (``dongrunmin/RefDiff``, ``training/networks_cond_v14.py``)
      concatenates *features* with the condition, runs separate mul/add trunks,
      and puts a ``sigmoid`` gate on the multiplier. SEED-SR explicitly says
      "its conditioning input" and gives no sigmoid, so we follow SEED-SR.
      RefDiff also applies two SFTs inside *every* U-Net block; SEED-SR's
      Table 11 lists exactly two SFT blocks, so we place two.

    Where SEED-SR is silent we follow RefDiff: the condition is resized with
    ``nearest`` (not bilinear), and the block modulates *normalised* features
    (RefDiff's order is ``norm -> sft -> conv``).
    """

    def __init__(self, cond_channels: int, feat_channels: int, groups: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups, feat_channels)   # RefDiff: norm -> sft -> conv
        self.trunk = nn.Sequential(
            nn.Conv2d(cond_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.to_scale = nn.Conv2d(64, feat_channels, 3, padding=1)
        self.to_shift = nn.Conv2d(64, feat_channels, 3, padding=1)
        # zero-init both heads: gamma = 0 -> (gamma + 1) = 1 and beta = 0, so the
        # block is exactly the identity at step 0 and learns modulation from there.
        for head in (self.to_scale, self.to_shift):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.shape[-2:] != features.shape[-2:]:
            # nearest, as in RefDiff -- the condition is a low-resolution image and
            # bilinear would invent intermediate radiometry it never had
            condition = F.interpolate(condition, size=features.shape[-2:], mode="nearest")
        hidden = self.trunk(condition)
        normed = self.norm(features)
        return normed * (self.to_scale(hidden) + 1.0) + self.to_shift(hidden)


class ContextSummary(nn.Module):
    """Pool the deepest tap to a small 7x7 grid: the cross-attention keys/values.

    SEED-SR downsamples its fused input to 7x7 explicitly to make attention
    tractable ("memory from 14 GB to 2.75 MB"). Same idea here.
    """

    def __init__(self, in_channels: int, out_channels: int, grid: int = 7) -> None:
        super().__init__()
        self.grid = grid
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LayerNorm([out_channels, grid, grid]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(F.adaptive_avg_pool2d(x, self.grid))


class CrossAttentionBlock(nn.Module):
    """Queries from decoder features, keys/values from the pooled context."""

    def __init__(self, dim: int, context_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, dim)
        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_kv = nn.Conv2d(context_dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.to_q(self.norm(x)).reshape(b, self.heads, c // self.heads, h * w).transpose(-1, -2)
        k, v = self.to_kv(context).chunk(2, dim=1)
        n = context.shape[-1] * context.shape[-2]
        k = k.reshape(b, self.heads, c // self.heads, n).transpose(-1, -2)
        v = v.reshape(b, self.heads, c // self.heads, n).transpose(-1, -2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)          # zero-init projection -> identity at start


class DecoderBlock(nn.Module):
    """Transposed convolution (optional), skip fusion, then a convolution block.

    Mirrors SEED-SR: "Each of the 4 decoder blocks comprises transposed
    convolution and convolutional layers. The decoder path reconstructs the
    spatial resolution using skip connections from the encoder."
    """

    def __init__(
        self, in_channels: int, out_channels: int, skip_channels: int | None = None,
        upsample: bool = True,
    ) -> None:
        super().__init__()
        if upsample:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        else:
            self.up = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(skip_channels, out_channels, 1) if skip_channels is not None else None
        )
        self.block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if self.skip is not None and skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.skip(skip)
        return self.block(x)


class SeedSRDecoder(nn.Module):
    """Frozen-ViT features (+ the LR image as conditioning) -> segmentation logits.

    forward(features, condition, out_size) where ``features`` is the tuple of
    DINOv3 taps, shallow-to-deep, and ``condition`` is the LR S2 image in [0, 1]
    standing in for SEED-SR's ``e_l``.
    """

    def __init__(
        self,
        in_channels: int = DINOV3_EMBED_DIM,
        cond_channels: int = 3,
        widths: tuple[int, ...] = DECODER_WIDTHS,
        num_classes: int = 1,
        out_size: int | tuple[int, int] | None = None,
        attention_heads: int = 4,
        hires_stem: bool = True,
    ) -> None:
        super().__init__()
        self.out_size = (out_size, out_size) if isinstance(out_size, int) else out_size
        w1, w2, w3, w4 = widths

        self.project_in = nn.Conv2d(in_channels, w1 * 2, 3, padding=1)
        self.context = ContextSummary(in_channels, w1)

        self.block1 = DecoderBlock(w1 * 2, w1, skip_channels=in_channels, upsample=True)
        self.sft1 = SFTBlock(cond_channels, w1)
        self.attention = CrossAttentionBlock(w1, w1, heads=attention_heads)

        self.block2 = DecoderBlock(w1, w2, skip_channels=in_channels, upsample=True)
        self.block3 = DecoderBlock(w2, w3, skip_channels=in_channels, upsample=True)
        self.sft2 = SFTBlock(cond_channels, w3)
        self.block4 = DecoderBlock(w3, w4, skip_channels=None, upsample=False)

        # high-resolution path: real detail for the last three stages
        self.stem = HighResStem(cond_channels) if hires_stem else None
        if hires_stem:
            s198, s396, s792 = STEM_WIDTHS[2], STEM_WIDTHS[1], STEM_WIDTHS[0]
            self.fuse1 = nn.Conv2d(s198, w1, 1)
            self.fuse2 = nn.Conv2d(s396, w2, 1)
            self.fuse3 = nn.Conv2d(s792, w3, 1)
            for layer in (self.fuse1, self.fuse2, self.fuse3):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.head = nn.Conv2d(w4, num_classes, 1)

    def forward(
        self,
        features: tuple[torch.Tensor, ...],
        condition: torch.Tensor,
        image: torch.Tensor | None = None,
        out_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(f"expected 4 feature maps, got {len(features)}")
        shallow, mid, deep, deepest = features

        hires = (None, None, None)
        if self.stem is not None:
            if image is None:
                raise ValueError("hires_stem is on but no `image` was passed")
            hires = self.stem(image)
        h198, h396, h792 = hires

        x = self.project_in(deepest)
        context = self.context(deepest)

        x = self.block1(x, deep)
        if h198 is not None:
            x = x + self.fuse1(F.interpolate(h198, size=x.shape[-2:], mode="bilinear",
                                             align_corners=False))
        x = self.sft1(x, condition)
        x = self.attention(x, context)

        x = self.block2(x, mid)
        if h396 is not None:
            x = x + self.fuse2(F.interpolate(h396, size=x.shape[-2:], mode="bilinear",
                                             align_corners=False))
        x = self.block3(x, shallow)
        if h792 is not None:
            x = x + self.fuse3(F.interpolate(h792, size=x.shape[-2:], mode="bilinear",
                                             align_corners=False))
        x = self.sft2(x, condition)
        x = self.block4(x)

        logits = self.head(x)
        target = out_size or self.out_size
        if target is not None and tuple(logits.shape[-2:]) != tuple(target):
            logits = F.interpolate(logits, size=target, mode="bilinear", align_corners=False)
        return logits
