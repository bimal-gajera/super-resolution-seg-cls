"""Segmentation loss for the DPT decoder.

SEED-SR's only training objective is a latent-diffusion MSE, which has nothing
to attach to here -- in both flows PixelDiT and DINOv3 are frozen and nothing
is diffusion-trained. The decoder is the only trainable module, so it gets a
conventional binary segmentation loss.

Dice + BCE is the standard pairing: BCE gives well-calibrated per-pixel
gradients, Dice supplies a region-overlap signal that is insensitive to class
imbalance. The optional boundary term upweights pixels near a field edge,
which is where the resolution difference between the two flows should show up
-- it is off by default so the headline comparison uses the plain objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice over each image in the batch, then averaged."""
    probs = torch.sigmoid(logits).flatten(1)
    target = target.flatten(1)
    intersection = (probs * target).sum(1)
    cardinality = probs.sum(1) + target.sum(1)
    return (1.0 - (2.0 * intersection + eps) / (cardinality + eps)).mean()


def boundary_weight_map(target: torch.Tensor, width: int = 5) -> torch.Tensor:
    """1 everywhere, higher within `width` pixels of a label transition.

    Found by max-pooling the target and its complement: a pixel is near an edge
    when both the target and the background are present in its neighbourhood.
    """
    pad = width // 2
    dilated = F.max_pool2d(target, width, stride=1, padding=pad)
    eroded = -F.max_pool2d(-target, width, stride=1, padding=pad)
    return (dilated - eroded).clamp(0, 1)


class DiceBCELoss(nn.Module):
    """``w_dice * Dice + w_bce * BCE`` (+ optional boundary-weighted BCE).

    Args:
        w_dice, w_bce: term weights.
        boundary_weight: if > 0, add this much extra BCE on near-edge pixels.
        boundary_width: neighbourhood size, in output pixels, defining "near".
            At 0.5 m/px the default 5 means 2.5 m either side of an edge.
    """

    def __init__(
        self,
        w_dice: float = 0.5,
        w_bce: float = 0.5,
        boundary_weight: float = 0.0,
        boundary_width: int = 5,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.w_dice = w_dice
        self.w_bce = w_bce
        self.boundary_weight = boundary_weight
        self.boundary_width = boundary_width
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if logits.shape != target.shape:
            raise ValueError(f"logits {tuple(logits.shape)} != target {tuple(target.shape)}")

        pos_weight = (
            torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
            if self.pos_weight is not None
            else None
        )
        bce_map = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight, reduction="none"
        )
        bce = bce_map.mean()
        dice = dice_loss(logits, target)
        total = self.w_dice * dice + self.w_bce * bce

        parts = {"dice": dice.detach(), "bce": bce.detach()}
        if self.boundary_weight > 0:
            weights = boundary_weight_map(target, self.boundary_width)
            denom = weights.sum().clamp(min=1.0)
            boundary = (bce_map * weights).sum() / denom
            total = total + self.boundary_weight * boundary
            parts["boundary"] = boundary.detach()

        parts["total"] = total
        return parts
