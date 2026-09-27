"""Small plotting helpers for the visual checks.

These exist so the gate in `scripts/seam_check.py` produces something a
person can actually judge -- the question "does this super-resolved farmland
look plausible?" has no numeric answer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def to_display(image: np.ndarray, percentile: float = 2.0) -> np.ndarray:
    """(C, H, W) or (H, W) -> (H, W, 3) uint8, contrast-stretched for viewing.

    Stretching is per-panel and for display only; it never touches what the
    model sees.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)

    lo = np.percentile(array, percentile)
    hi = np.percentile(array, 100 - percentile)
    if hi <= lo:
        lo, hi = float(array.min()), float(array.max()) or 1.0
    return (np.clip((array - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def save_panels(
    panels: list[tuple[str, np.ndarray]],
    path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 130,
) -> Path:
    """Write a labelled row of images."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
    if len(panels) == 1:
        axes = [axes]
    for ax, (label, image) in zip(axes, panels):
        ax.imshow(to_display(image))
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def seam_strip(mosaic: np.ndarray, seam: int, halfwidth: int = 120) -> np.ndarray:
    """Crop a vertical band centred on a seam, for close inspection."""
    lo = max(0, seam - halfwidth)
    hi = min(mosaic.shape[-1], seam + halfwidth)
    return mosaic[..., lo:hi]
