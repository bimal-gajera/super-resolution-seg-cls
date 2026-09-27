"""Frozen DINOv3-SAT encoder exposing four intermediate feature maps.

The prior repo's ``core.diffusion.DINOv3`` returns only the final layer, which
was all REPA alignment needed. A DPT decoder needs several depths, so this
wrapper calls ``get_intermediate_layers`` instead.

The SAT-493M checkpoint is NOT an ImageNet model: it was distilled on ~0.6 m
satellite imagery and carries its own normalization statistics. Using ImageNet
mean/std here would quietly degrade every feature.

Weights travel as a local file and the source tree is vendored because the
HuggingFace repo is gated and compute nodes may have no outbound network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from ..torch_utils import amp_dtype
from ..constants import (
    DINOV3_EMBED_DIM,
    DINOV3_PATCH,
    DINOV3_TAPS,
    SAT493M_MEAN,
    SAT493M_STD,
)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO / "third_party" / "dinov3"
# The filename matters: dinov3's hub loader parses the 8-char hash out of it
# with `-(.{8}).pth`, and for ViT-L the hash `eadcf0ff` additionally sets
# `untie_global_and_local_cls_norm=True`. Renaming this file does not just
# fail to load -- it would quietly build a different architecture.
DEFAULT_WEIGHTS = REPO / "weights" / "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"


class FrozenDINOv3(nn.Module):
    """DINOv3 ViT-L/16 (SAT-493M), frozen, returning ``len(taps)`` feature maps.

    forward((B, 3, H, W) in [0, 1]) -> tuple of (B, 1024, H/16, W/16), one per tap.
    """

    def __init__(
        self,
        source_dir: str | Path = DEFAULT_SOURCE,
        weights: str | Path = DEFAULT_WEIGHTS,
        arch: str = "dinov3_vitl16",
        taps: tuple[int, ...] = DINOV3_TAPS,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # V100 has no bfloat16; pick what this GPU actually supports.
        dtype = dtype or amp_dtype("cuda")
        source_dir, weights = Path(source_dir), Path(weights)
        if not source_dir.is_dir():
            raise FileNotFoundError(
                f"DINOv3 source tree not found at {source_dir} -- run env/create_env.sh"
            )
        if not weights.is_file():
            raise FileNotFoundError(f"DINOv3 weights not found at {weights}")

        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        import dinov3.hub.backbones as backbones  # noqa: PLC0415

        self.encoder = getattr(backbones, arch)(pretrained=True, weights=str(weights))
        self.encoder = self.encoder.to(dtype).eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        self.taps = tuple(taps)
        self.embed_dim = getattr(self.encoder, "embed_dim", DINOV3_EMBED_DIM)
        self.patch_size = getattr(self.encoder, "patch_size", DINOV3_PATCH)
        self._dtype = dtype

        self.register_buffer("_mean", torch.tensor(SAT493M_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(SAT493M_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):  # noqa: D102 -- stays frozen regardless
        return super().train(False)

    @torch.no_grad()
    def forward(self, images01: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if images01.shape[1] != 3:
            raise ValueError(f"expected 3 channels, got {images01.shape[1]}")
        height, width = images01.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"input {height}x{width} is not divisible by patch size {self.patch_size}"
            )

        x = (images01 - self._mean.to(images01)) / self._std.to(images01)
        feats = self.encoder.get_intermediate_layers(
            x.to(self._dtype), n=self.taps, reshape=True, norm=True
        )
        return tuple(f.float() for f in feats)
