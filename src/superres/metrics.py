"""Evaluation metrics, following SEED-SR's definitions (arXiv 2511.14481, App. B.2).

Every metric is computed *per image* and then averaged over the test set --
not accumulated globally. That distinction matters: a global IoU is dominated
by large tiles, a per-image mean is not, and the paper reports the latter.

Semantic (v1):

    IoU_S = (1/N) Σ_i  TP_i / (TP_i + FP_i + FN_i)

plus Accuracy, Precision, Recall and F1 from the same per-image counts.

Instance (v2):

    IoU_I = (1/M) Σ_i  (P_m^i ∩ P^i) / (P_m^i ∪ P^i)

where M is the number of ground-truth instances, P^i is ground-truth instance
i, and P_m^i is the union of predicted instances overlapping it by more than
`overlap_threshold`. The paper does not state its threshold; 0.5 is used here
and is reported alongside the metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Counts:
    tp: int
    tn: int
    fp: int
    fn: int


def confusion(pred: np.ndarray, target: np.ndarray) -> Counts:
    """Binary confusion counts for one image. Both arrays boolean-castable."""
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
    return Counts(
        tp=int(np.count_nonzero(pred & target)),
        tn=int(np.count_nonzero(~pred & ~target)),
        fp=int(np.count_nonzero(pred & ~target)),
        fn=int(np.count_nonzero(~pred & target)),
    )


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def semantic_metrics(counts: Counts) -> dict[str, float]:
    """Per-image semantic metrics. NaN where a metric is undefined for the image.

    IoU is NaN when an image has neither predicted nor true positives -- the
    metric is genuinely undefined there, and scoring it 0 or 1 would both be
    wrong. `aggregate` skips NaNs, matching the convention of averaging over
    images where the quantity exists.
    """
    precision = _safe_div(counts.tp, counts.tp + counts.fp)
    recall = _safe_div(counts.tp, counts.tp + counts.fn)
    # 2TP / (2TP + FP + FN) -- algebraically the harmonic mean of the two above,
    # but defined directly from counts so it survives either being undefined.
    f1 = _safe_div(2 * counts.tp, 2 * counts.tp + counts.fp + counts.fn)
    return {
        "iou": _safe_div(counts.tp, counts.tp + counts.fp + counts.fn),
        "accuracy": _safe_div(counts.tp + counts.tn, counts.tp + counts.tn + counts.fp + counts.fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate(per_image: list[dict[str, float]]) -> dict[str, float]:
    """Mean over images, ignoring NaNs. Keys become `mIoU_S`, `mAccuracy`, ..."""
    if not per_image:
        return {}
    out = {}
    for key in per_image[0]:
        values = np.array([m[key] for m in per_image], dtype=float)
        finite = values[np.isfinite(values)]
        name = "mIoU_S" if key == "iou" else f"m{key.capitalize()}"
        out[name] = float(finite.mean()) if finite.size else float("nan")
        out[f"{name}_n"] = int(finite.size)
    return out


# --------------------------------------------------------------------------- #
# instance-level (v2)
# --------------------------------------------------------------------------- #
def watershed_instances(
    probability: np.ndarray,
    threshold: float = 0.5,
    mode: str = "gradient",
    marker_threshold: float = 0.8,
    min_distance: int = 10,
) -> np.ndarray:
    """Split a field-probability map into labelled instances. 0 = background.

    ``mode="gradient"`` follows SEED-SR's cited watershed (Ng et al. 2006, ref
    [40]): a *marker-controlled watershed on the gradient magnitude*. Field
    boundaries show up as ridges where the probability falls, and confident
    interiors seed the basins. This is the right tool when parcels share
    straight edges.

    ``mode="distance"`` is the classic distance-transform variant, which seeds
    on maxima of the distance to background. It is built to split touching
    *round* blobs and systematically merges neighbouring fields that share an
    edge -- measured on four test windows it under-counted parcels every time
    (e.g. 10 found against 17 true). Kept for comparison only.
    """
    from scipy import ndimage
    from skimage.feature import peak_local_max
    from skimage.filters import sobel
    from skimage.segmentation import watershed

    mask = probability >= threshold
    if not mask.any():
        return np.zeros(probability.shape, dtype=np.int32)

    if mode == "gradient":
        elevation = sobel(probability.astype(np.float32))
        seeds = mask & (probability >= marker_threshold)
        markers, count = ndimage.label(seeds)
        if count == 0:
            markers, _ = ndimage.label(mask)
        return watershed(elevation, markers, mask=mask).astype(np.int32)

    if mode == "distance":
        distance = ndimage.distance_transform_edt(mask)
        coords = peak_local_max(distance, min_distance=min_distance, labels=mask)
        markers = np.zeros(distance.shape, dtype=np.int32)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i
        if markers.max() == 0:
            markers, _ = ndimage.label(mask)
        return watershed(-distance, markers, mask=mask).astype(np.int32)

    raise ValueError(f"unknown watershed mode {mode!r}")


def instance_iou(
    predicted: np.ndarray,
    truth: np.ndarray,
    overlap_threshold: float = 0.5,
) -> tuple[float, int]:
    """SEED-SR's IoU_I. Returns (mean IoU over GT instances, instance count).

    For each ground-truth instance, gather every predicted instance that covers
    more than `overlap_threshold` of it, union them, and take IoU against the
    truth. Ground-truth instances with no qualifying prediction score 0.
    """
    truth_ids = np.unique(truth)
    truth_ids = truth_ids[truth_ids != 0]
    if truth_ids.size == 0:
        return float("nan"), 0

    scores = []
    for tid in truth_ids:
        gt_mask = truth == tid
        gt_size = int(gt_mask.sum())
        overlapping = np.unique(predicted[gt_mask])
        overlapping = overlapping[overlapping != 0]

        matched = np.zeros_like(gt_mask)
        for pid in overlapping:
            pred_mask = predicted == pid
            if int((pred_mask & gt_mask).sum()) / gt_size > overlap_threshold:
                matched |= pred_mask

        union = int((matched | gt_mask).sum())
        scores.append(int((matched & gt_mask).sum()) / union if union else 0.0)

    return float(np.mean(scores)), int(truth_ids.size)
