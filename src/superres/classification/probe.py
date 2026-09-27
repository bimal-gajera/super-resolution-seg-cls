"""kNN and linear probes over cached features, using torchgeo-bench's own code.

Deliberately thin. ``evaluate_knn`` and ``evaluate_logistic`` upstream take
plain numpy arrays and know nothing about dataloaders, so they can be driven
from a feature cache without reimplementing anything. That matters: the probe
protocol -- kNN-5 on raw L2 distance, a 40-point C sweep selected on val, train
and val merged for the final fit, 200-sample bootstrap CIs, ECE alongside -- is
what makes our numbers comparable to published ones, and reimplementing it would
quietly diverge.

What this module adds is the one thing the upstream default gets wrong for us.

**Feature scale.** Neither probe normalizes its input: ``KNNClassifier`` indexes
raw vectors under ``faiss.IndexFlatL2``, and ``LogisticRegression`` has no
scaler. Our arms are not on comparable scales -- ``pixeldit_repa`` runs to
std ~900 because REPA trained with a cosine loss and never learned magnitude,
against ~0.2 for the DINOv3 arms. kNN survives that, since a uniform rescale
does not reorder neighbours, but a fixed C sweep is not scale-invariant, so the
same C means something different to each arm.

So every arm is probed **twice**: once on raw features, which is what the
leaderboard protocol does and therefore comparable outward, and once
standardized per dimension with statistics fit on train alone, which is fair
between arms. Probing costs seconds against hours of extraction, so there is no
reason to choose. The ``feature_norm`` column says which is which.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from .arms import ARMS
from .datasets import cache_group
from .features import DEFAULT_CACHE_ROOT, FeatureCache

__all__ = ["FEATURE_NORMS", "standardize", "probe_arm", "DEFAULT_RESULTS_DIR"]

logger = logging.getLogger(__name__)

FEATURE_NORMS = ("raw", "zscore")
DEFAULT_RESULTS_DIR = Path("results/models")


def standardize(
    train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Per-dimension z-score with statistics from ``train`` only.

    Fitting on train alone and applying the same shift and scale to val and test
    keeps the evaluation splits out of the fit. Zero-variance dimensions are
    left alone rather than divided by ~0.
    """
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return tuple(((x - mean) / std).astype(np.float32) for x in (train, *others))


def _config_hash(payload: dict) -> str:
    """Short, stable fingerprint of what produced a row."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _load_splits(
    dataset: str,
    arm: str,
    geometry: str,
    cache_root: Path,
    split_ids: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """Gather cached features for each split, in that split's own order.

    ``eurosat-spatial`` resolves against the ``eurosat`` cache here: same images,
    different split membership, so the lookup is by id rather than by position.
    """
    cache = FeatureCache(cache_root, cache_group(dataset), arm, geometry)
    if not cache.is_complete():
        raise FileNotFoundError(
            f"feature cache {cache.dir} is incomplete; run scripts/cls_extract.py first"
        )
    return {split: cache.load_for(ids) for split, ids in split_ids.items()}


def probe_arm(
    dataset: str,
    arm: str,
    geometry: str,
    *,
    split_ids: dict[str, list[str]],
    split_labels: dict[str, np.ndarray],
    num_classes: int,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    seed: int = 0,
    knn_k: int = 5,
    bootstrap: int = 200,
    c_range: tuple[float, float, int] = (-6.0, 4.0, 40),
    merge_val: bool = True,
    device: str = "cpu",
    feature_norms: tuple[str, ...] = FEATURE_NORMS,
    extra_meta: dict | None = None,
    write: bool = True,
) -> list[dict]:
    """Probe one cached arm on one dataset and return (optionally write) rows.

    Args:
        dataset: torchgeo-bench dataset name.
        arm: Arm name from :data:`~superres.classification.arms.ARMS`.
        geometry: ``"crop"`` or ``"resize"`` -- selects the cache, and recorded.
        split_ids: ``{"train"/"val"/"test": [sample id, ...]}``.
        split_labels: Integer labels aligned with ``split_ids``.
        num_classes: For the results row.
        cache_root: Feature cache root.
        results_dir: One CSV per arm is written here.
        seed: Passed to the probes and the bootstrap.
        knn_k: Neighbours for kNN.
        bootstrap: Bootstrap resamples for the CI.
        c_range: ``log10`` (start, stop, num) for the logistic C sweep.
        merge_val: Refit on train+val at the selected C.
        device: Probe device; ``"cpu"`` avoids needing a GPU FAISS build.
        feature_norms: Which scalings to report. Defaults to both.
        extra_meta: Merged into the config fingerprint.
        write: Append to the CSV. ``False`` returns rows without touching disk.

    Returns:
        One row per (feature_norm, method).
    """
    from torchgeo_bench.main import evaluate_knn, evaluate_logistic
    from torchgeo_bench.results import append_rows_atomic, metric_row, model_results_path

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    features = _load_splits(dataset, arm, geometry, Path(cache_root), split_ids)
    labels = {s: np.asarray(v) for s, v in split_labels.items()}
    for split, x in features.items():
        if len(x) != len(labels[split]):
            raise ValueError(
                f"{dataset}/{split}: {len(x)} feature rows but {len(labels[split])} labels"
            )

    c_values = np.logspace(c_range[0], c_range[1], int(c_range[2]))
    n_counts = {s: len(v) for s, v in labels.items()}
    rows: list[dict] = []

    for feature_norm in feature_norms:
        if feature_norm == "raw":
            x_train, x_val, x_test = (features[s] for s in ("train", "val", "test"))
        elif feature_norm == "zscore":
            x_train, x_val, x_test = standardize(
                features["train"], features["val"], features["test"]
            )
        else:
            raise ValueError(f"unknown feature_norm {feature_norm!r}")

        common = {
            "dataset": dataset,
            "seed": seed,
            "model": arm,
            "name": f"{arm}__{geometry}__{feature_norm}",
            # The dataset layer hands raw band values straight through, so the
            # only normalization that happened is the arm's own -- not one of
            # torchgeo-bench's strategies.
            "normalization": "arm_native",
            "image_size": 48,
            "interpolation": geometry,
            "partition": "default",
            "bands": "pixeldit12",
            "num_classes": num_classes,
            "config_hash": _config_hash(
                {
                    "arm": arm, "geometry": geometry, "feature_norm": feature_norm,
                    "seed": seed, "knn_k": knn_k, "bootstrap": bootstrap,
                    "c_range": list(c_range), "merge_val": merge_val,
                    **(extra_meta or {}),
                }
            ),
            "c_range_start": float(c_range[0]),
            "c_range_stop": float(c_range[1]),
            "c_range_num": int(c_range[2]),
            "merge_val": merge_val,
            "bootstrap": bootstrap,
        }
        dim = x_train.shape[1]

        metric, lo, hi, calibration, n_bins = evaluate_knn(
            x_train, labels["train"], x_test, labels["test"],
            seed=seed, n_bootstrap=bootstrap, device=device, n_neighbors=knn_k,
        )
        rows.append(metric_row(
            dict(common), method=f"knn{knn_k}", metric_name="accuracy",
            metric_value=metric, feature_dim=dim, n_counts=n_counts,
            ci_lower=lo, ci_upper=hi, pool="mean",
            calibration_n_bins=n_bins, **calibration,
        ))
        logger.info(
            "%s/%s/%s/%s knn%d accuracy=%.4f (%.4f-%.4f)",
            dataset, arm, geometry, feature_norm, knn_k, metric, lo, hi,
        )

        metric, lo, hi, best_c, calibration, calibration_ts = evaluate_logistic(
            x_train, labels["train"], x_val, labels["val"], x_test, labels["test"],
            c_values=c_values, seed=seed, n_bootstrap=bootstrap,
            merge_val=merge_val, device=device, temp_scale=False,
        )
        rows.append(metric_row(
            dict(common), method="linear", metric_name="accuracy",
            metric_value=metric, feature_dim=dim, n_counts=n_counts,
            ci_lower=lo, ci_upper=hi, best_c=best_c, pool="mean",
            calibration_n_bins=15, **calibration, **calibration_ts,
        ))
        logger.info(
            "%s/%s/%s/%s linear accuracy=%.4f (%.4f-%.4f) C=%.4g",
            dataset, arm, geometry, feature_norm, metric, lo, hi, best_c,
        )

    if write:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        append_rows_atomic(str(model_results_path(results_dir, arm)), rows)
    return rows
