"""Load and run the PixelDiT super-resolution model.

Adapted from the prior repo's ``core/inference.py``. Two checkpoints exist and
both are used here, so the conditioning width is inferred from the weights
rather than assumed:

* ``pixeldit_rgb.ckpt`` -- 3 conditioning channels. ``pixel_embedder.proj`` is
  (16, 6) = 3 target + 3 RGB. Used by the segmentation flows, which only ever
  had RGB to give it.
* ``pixeldit_weights.ckpt`` -- 12 conditioning channels, proj (16, 15). The
  original S2 model. Used by the classification arms, where the datasets ship
  all 13 Sentinel-2 bands and there is no reason to throw nine of them away.

Both carry ``diffusion_trainer.proj.*``, the REPA alignment head, so
``load_proj`` works against either.

GEOMETRY. PixelDiT is tied to one footprint: 48 LR px at 10 m = 480 m across,
generated at 1584x1584 (0.303 m/px). ``prepare_lr_cond`` bicubic-upsamples
whatever it is handed to 1584 without checking the input's real-world extent,
so the caller is responsible for passing a tensor that genuinely covers 480 m.
A differently-scaled input is silently out of distribution, not rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import COARSE, HR_SIZE, PIXELDIT_PATCH, RGB_STATS_INDICES
from ..core.diffusion import (
    FlowDPMSolverSampler,
    LinearScheduler,
    ode_step_fn,
    simple_guidance_fn,
)
from ..core.pixeldit_c2i import PixDiTSR

__all__ = [
    "load_state", "load_denoiser", "load_proj", "make_sampler",
    "rgb_stats", "band_stats", "to_unit_range", "prepare_lr_cond", "generate_hr",
    "BlockTap", "fold_pixel_tokens",
]


def load_state(ckpt: str | Path) -> dict:
    return torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]


def load_denoiser(state: dict, prefix: str, device: str) -> nn.Module:
    """Build PixDiTSR and load one weight group ('denoiser.' or 'ema_denoiser.').

    Generation should use ``ema_denoiser.`` -- the EMA copy produces
    noticeably better images than the raw weights.
    """
    weights = {
        k[len(prefix):].replace("_orig_mod.", ""): v
        for k, v in state.items()
        if k.startswith(prefix)
    }
    if not weights:
        raise KeyError(f"no tensors with prefix {prefix!r} in this checkpoint")

    # pixel_embedder.proj is (16, in_channels + cond_channels), so the
    # conditioning width is readable straight off the weights. Two checkpoints
    # exist: the RGB one (6 = 3 + 3) used by the segmentation flows, and the
    # 12-band S2 one (15 = 3 + 12) used by the classification arms. Inferring
    # it means a checkpoint can never be loaded under the wrong width.
    cond_in = weights["pixel_embedder.proj.weight"].shape[1]
    cond_channels = cond_in - 3
    if cond_channels not in (3, 12):
        raise ValueError(
            f"pixel_embedder.proj has {cond_in} input channels, so conditioning "
            f"width is {cond_channels}; expected 3 (RGB checkpoint) or 12 "
            "(S2 checkpoint)."
        )

    net = PixDiTSR(
        in_channels=3, cond_channels=cond_channels, num_groups=16, hidden_size=1152,
        pixel_hidden_size=16, patch_depth=26, pixel_depth=4, patch_size=PIXELDIT_PATCH,
    ).to(device)
    missing, unexpected = net.load_state_dict(weights, strict=False)
    if unexpected:
        raise ValueError(f"unexpected tensors in checkpoint: {sorted(unexpected)[:5]}")
    return net.eval()


def make_sampler(
    device: str, num_steps: int = 50, guidance: float = 2.5,
    timeshift: float = 1.0, interval: str = "0.3:1",
) -> FlowDPMSolverSampler:
    """The tuned settings the model was evaluated with. Changing these changes
    image quality; they are not arbitrary defaults."""
    lo, hi = (float(v) for v in interval.split(":"))
    return FlowDPMSolverSampler(
        num_steps=num_steps, guidance=guidance, timeshift=timeshift,
        guidance_interval_min=lo, guidance_interval_max=hi,
        scheduler=LinearScheduler(), w_scheduler=LinearScheduler(),
        guidance_fn=simple_guidance_fn, step_fn=ode_step_fn,
    ).to(device)


def rgb_stats(stats_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel (low, high-low) for R, G, B from the 12-band stats file.

    The file lists bands in the order B01 B02 B03 B04 B05 B06 B07 B08 B8A B09
    B11 B12, so red/green/blue are indices 3/2/1 (B4/B3/B2). The stats belong
    to the 12-band model's training data; for this RGB checkpoint they are the
    best available estimate, which is why `scripts/seam_check.py` compares
    them against per-image normalization before anything expensive runs.
    """
    bands = json.loads(Path(stats_path).read_text())["bands"]
    lo = torch.tensor([float(bands[i]["low"]) for i in RGB_STATS_INDICES])
    hi = torch.tensor([float(bands[i]["high"]) for i in RGB_STATS_INDICES])
    return lo.view(1, 3, 1, 1), (hi - lo).view(1, 3, 1, 1)


def to_unit_range(
    rgb: torch.Tensor,
    lo: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    *,
    mode: str = "stats",
    percentiles: tuple[float, float] = (1.0, 99.0),
) -> torch.Tensor:
    """(B, 3, H, W) raw band values -> [0, 1], per channel.

    ``mode="stats"`` applies the training percentiles from
    `s2_band_stats.json`; ``mode="per-image"`` uses each patch's own.

    AI4SmallFarms is radiometrically much narrower than PixelDiT's training
    data -- measured over 8 tiles, p99 lands at only 0.18-0.24 of the training
    range -- which suggests the training stretch would leave the model a
    near-black image. Empirically that reasoning is wrong: `seam_check.py`
    shows "stats" producing markedly cleaner fields with crisper boundaries,
    and "per-image" producing blotchy, speckled output. PixelDiT evidently
    learned an *absolute* reflectance mapping, and rescaling each patch
    independently destroys exactly that signal. So conditioning defaults to
    "stats".
    """
    x = rgb.float()
    if mode == "per-image":
        flat = x.flatten(2)
        q = torch.tensor(percentiles, device=x.device, dtype=x.dtype) / 100.0
        lo_i = flat.quantile(q[0], dim=2)[..., None, None]
        hi_i = flat.quantile(q[1], dim=2)[..., None, None]
        x = (x - lo_i) / (hi_i - lo_i).clamp(min=1e-6)
    elif mode == "stats":
        if lo is None or scale is None:
            raise ValueError("stats normalization needs lo and scale (see rgb_stats)")
        x = (x - lo.to(x)) / scale.to(x)
    else:
        raise ValueError(f"unknown normalization mode {mode!r}")
    return x.clamp(0, 1)


def prepare_lr_cond(
    rgb: torch.Tensor,
    lo: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    *,
    mode: str = "stats",
) -> torch.Tensor:
    """(B, 3, H, W) raw band values -> (B, 3, 1584, 1584) in [-1, 1].

    Defaults to the training statistics -- see `to_unit_range` for why.
    """
    x = to_unit_range(rgb, lo, scale, mode=mode) * 2 - 1
    x = F.interpolate(x, size=(HR_SIZE, HR_SIZE), mode="bicubic", align_corners=False)
    return x.clamp(-1, 1)


@torch.no_grad()
def generate_hr(
    net: nn.Module, sampler: FlowDPMSolverSampler, lr_cond: torch.Tensor,
    device: str, seed: int = 0,
) -> torch.Tensor:
    """Run the sampler. This is the expensive step -- ~3.4 s/tile on an H100.

    lr_cond: (B, 3, 1584, 1584). Returns (B, 3, 1584, 1584) in [-1, 1].
    """
    batch = lr_cond.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((batch, 3, HR_SIZE, HR_SIZE), generator=generator).to(device)
    uncond = torch.zeros_like(lr_cond)
    return sampler(net, noise, lr_cond, uncond).float().clamp(-1, 1)


def band_stats(stats_path: str | Path, n_bands: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel (low, high-low) for all twelve conditioning bands.

    The companion of :func:`rgb_stats` for the 12-band checkpoint. The file
    lists bands in the order documented in ``constants.S2_12BAND_CODES``, which
    is also the order the checkpoint's conditioning channels are in, so no
    reindexing is needed -- just take them straight through.
    """
    bands = json.loads(Path(stats_path).read_text())["bands"]
    if len(bands) != n_bands:
        raise ValueError(f"{stats_path} lists {len(bands)} bands, expected {n_bands}")
    lo = torch.tensor([float(b["low"]) for b in bands])
    hi = torch.tensor([float(b["high"]) for b in bands])
    return lo.view(1, n_bands, 1, 1), (hi - lo).clamp_min(1e-6).view(1, n_bands, 1, 1)


def load_proj(state: dict, device: str) -> nn.Module:
    """The REPA alignment head stored in the checkpoint: 1152 -> 1152 -> 1024.

    Only REPA training runs carry this. It is what turns a raw DiT block-8
    activation into the 1024-d embedding that was aligned against DINOv3, and
    it is the whole reason ``pixeldit_repa`` features are usable without
    generating an image.
    """
    prefix = "diffusion_trainer.proj."
    weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not weights:
        raise KeyError(
            "no diffusion_trainer.proj.* in this checkpoint -- not a REPA run, so "
            "there is no alignment head and pixeldit_repa features are unavailable"
        )
    d_in = weights["0.0.weight"].shape[1]
    d_hidden = weights["0.2.weight"].shape[0]
    d_out = weights["0.4.weight"].shape[0]
    # Double nesting is not a mistake: the checkpoint stores the head as a
    # Sequential holding one Sequential, so the key prefixes are "0.0", "0.2",
    # "0.4". Flattening it here would fail to load.
    proj = nn.Sequential(nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.SiLU(),
        nn.Linear(d_hidden, d_hidden), nn.SiLU(),
        nn.Linear(d_hidden, d_out),
    ))
    proj.load_state_dict(weights)
    return proj.to(device).eval()


class BlockTap:
    """Capture one module's output during a forward pass.

    Used to read PixelDiT's internal activations -- DiT patch block 8 for the
    REPA embedding, and the last PiT pixel block for the full-resolution
    branch -- without modifying the model.
    """

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.value: torch.Tensor | None = None
        self._handle = None

    def __enter__(self) -> BlockTap:
        def hook(_module, _inputs, output):
            self.value = output[0] if isinstance(output, tuple) else output

        self._handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def fold_pixel_tokens(tokens: torch.Tensor, batch: int = 1) -> torch.Tensor:
    """(B*1089, 2304, C) pixel tokens -> (B, C, 1584, 1584).

    Mirrors ``PixDiTSR.forward``'s own view/permute/fold. The tokens are stored
    patch-by-patch, so a reshape of merely the right *size* would scramble the
    geometry into a grid of shuffled 48x48 blocks -- silently, since the shape
    would still be correct.
    """
    channels = tokens.shape[-1]
    x = tokens.view(batch, COARSE * COARSE, PIXELDIT_PATCH * PIXELDIT_PATCH, channels)
    x = x.permute(0, 3, 2, 1).contiguous()
    x = x.view(batch, channels * PIXELDIT_PATCH * PIXELDIT_PATCH, COARSE * COARSE)
    return F.fold(x, (HR_SIZE, HR_SIZE), kernel_size=PIXELDIT_PATCH, stride=PIXELDIT_PATCH)
