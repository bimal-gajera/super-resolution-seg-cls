"""Reassemble per-window predictions into whole-tile maps.

Windows are cut at stride 48 over the 10 m grid, so they tile the image
exactly except at the right and bottom edges, where the final window is
shifted flush with the border and overlaps its neighbour.

Accumulate-then-divide handles that automatically: sum each window's
contribution into a canvas, count how many windows touched each pixel, and
divide at the end. Pixels covered once are unchanged; pixels covered twice
become the mean of both views, which removes the seam without any special
casing.

If a future configuration uses a *wide* overlap, prefer `weight="cosine"`:
averaging two independently-generated diffusion outputs over a broad band
blurs it, whereas a feathered blend does not.
"""

from __future__ import annotations

import numpy as np

from .constants import LABEL_SIZE


def cosine_window(size: int, floor: float = 1e-3) -> np.ndarray:
    """Separable raised-cosine taper, ~0 at the border and 1 at the centre.

    Clamped away from zero: at a tile's outer edge a pixel may be covered by
    exactly one window, and dividing by a corner weight of ~1e-11 would
    amplify float error into visible garbage. The floor costs nothing in the
    interior, where weights are O(1).
    """
    ramp = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(size) + 0.5) / size)
    return np.maximum(np.outer(ramp, ramp), floor)


class TileAccumulator:
    """Sum windows into a full-tile canvas, then average by coverage."""

    def __init__(self, height: int, width: int, weight: str = "uniform") -> None:
        self.total = np.zeros((height, width), dtype=np.float64)
        self.coverage = np.zeros((height, width), dtype=np.float64)
        if weight not in ("uniform", "cosine"):
            raise ValueError(f"unknown weight {weight!r}")
        self.weight = weight
        self._window_cache: dict[int, np.ndarray] = {}

    def _weights(self, size: int) -> np.ndarray:
        if self.weight == "uniform":
            return np.ones((size, size))
        if size not in self._window_cache:
            self._window_cache[size] = cosine_window(size)
        return self._window_cache[size]

    def add(self, patch: np.ndarray, row: int, col: int) -> None:
        """Place one prediction at (row, col) in label-grid pixels."""
        height, width = patch.shape
        weights = self._weights(height) if height == width else np.ones_like(patch)
        canvas_h, canvas_w = self.total.shape
        if row + height > canvas_h or col + width > canvas_w:
            raise ValueError(
                f"window at ({row}, {col}) size {height}x{width} "
                f"exceeds canvas {canvas_h}x{canvas_w}"
            )
        self.total[row : row + height, col : col + width] += patch * weights
        self.coverage[row : row + height, col : col + width] += weights

    def result(self, fill: float = 0.0) -> np.ndarray:
        """Coverage-averaged map. Uncovered pixels take `fill`."""
        out = np.full(self.total.shape, fill, dtype=np.float32)
        seen = self.coverage > 0
        out[seen] = (self.total[seen] / self.coverage[seen]).astype(np.float32)
        return out

    @property
    def uncovered(self) -> int:
        return int((self.coverage == 0).sum())


def accumulate_tile(
    predictions: list[tuple[np.ndarray, int, int]],
    height: int,
    width: int,
    weight: str = "uniform",
) -> np.ndarray:
    """Convenience wrapper: list of (patch, label_row, label_col) -> full tile."""
    acc = TileAccumulator(height, width, weight=weight)
    for patch, row, col in predictions:
        acc.add(patch, row, col)
    return acc.result()
