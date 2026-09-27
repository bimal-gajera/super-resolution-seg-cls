"""DPT decoder head over frozen ViT features (Ranftl et al., 2021).

This is the only trainable module in either flow, and both flows use an
identical instance of it -- so any difference in the final numbers comes from
the features DINOv3 produced, not from decoder capacity.

Four token maps at the same spatial size (H/16) come in. `Reassemble` puts
them at four different scales, imitating a convolutional pyramid:

    tap 0  ->  H/4    (upsample 4x)   256 ch
    tap 1  ->  H/8    (upsample 2x)   512 ch
    tap 2  ->  H/16   (unchanged)    1024 ch
    tap 3  ->  H/32   (downsample 2) 1024 ch

They are then fused coarse-to-fine by RefineNet-style blocks, each doubling
resolution, ending at H/4. A final head resamples to the label grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import DINOV3_EMBED_DIM

REASSEMBLE_CHANNELS = (256, 512, 1024, 1024)


class Reassemble(nn.Module):
    """Project one token map to `out_channels` and resample it by `scale`."""

    def __init__(self, in_channels: int, out_channels: int, scale: float) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if scale > 1:
            self.resample = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=int(scale), stride=int(scale)
            )
        elif scale < 1:
            self.resample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3,
                stride=int(round(1 / scale)), padding=1,
            )
        else:
            self.resample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resample(self.project(x))


class ResidualConvUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(self.conv1(F.relu(x)))
        out = self.bn2(self.conv2(F.relu(out)))
        return out + x


class FeatureFusion(nn.Module):
    """Add the coarser branch (if any), refine, then double the resolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pre = ResidualConvUnit(channels)
        self.post = ResidualConvUnit(channels)
        self.project = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, coarser: torch.Tensor | None = None) -> torch.Tensor:
        if coarser is not None:
            if coarser.shape[-2:] != x.shape[-2:]:
                coarser = F.interpolate(
                    coarser, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            x = x + self.pre(coarser)
        x = self.post(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.project(x)


class DPTHead(nn.Module):
    """Frozen-ViT features -> segmentation logits at `out_size`.

    Args:
        in_channels: token dimension of each tap (1024 for ViT-L).
        fusion_channels: width of the fusion pyramid.
        num_classes: 1 for binary (use with BCE-style losses).
        out_size: spatial size of the returned logits, e.g. 960 for the 0.5 m grid.
    """

    def __init__(
        self,
        in_channels: int = DINOV3_EMBED_DIM,
        fusion_channels: int = 256,
        num_classes: int = 1,
        out_size: int | tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.out_size = (out_size, out_size) if isinstance(out_size, int) else out_size

        scales = (4, 2, 1, 0.5)
        self.reassemble = nn.ModuleList(
            Reassemble(in_channels, ch, scale)
            for ch, scale in zip(REASSEMBLE_CHANNELS, scales)
        )
        self.align = nn.ModuleList(
            nn.Conv2d(ch, fusion_channels, 3, padding=1, bias=False)
            for ch in REASSEMBLE_CHANNELS
        )
        self.fusion = nn.ModuleList(FeatureFusion(fusion_channels) for _ in REASSEMBLE_CHANNELS)

        self.head = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels // 2, num_classes, kernel_size=1),
        )

    def forward(
        self, features: tuple[torch.Tensor, ...], out_size: tuple[int, int] | None = None
    ) -> torch.Tensor:
        if len(features) != len(self.reassemble):
            raise ValueError(f"expected {len(self.reassemble)} feature maps, got {len(features)}")

        pyramid = [
            align(reassemble(f))
            for f, reassemble, align in zip(features, self.reassemble, self.align)
        ]

        # coarse -> fine, each fusion doubling the resolution
        out = self.fusion[-1](pyramid[-1])
        for level in range(len(pyramid) - 2, -1, -1):
            out = self.fusion[level](pyramid[level], out)

        logits = self.head(out)

        target = out_size or self.out_size
        if target is not None and tuple(logits.shape[-2:]) != tuple(target):
            logits = F.interpolate(logits, size=target, mode="bilinear", align_corners=False)
        return logits
