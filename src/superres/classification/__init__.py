"""Classification track: PixelDiT features probed on four Sentinel-2 benchmarks.

The segmentation track asks whether super-resolution helps place field
boundaries. This one asks the cheaper, broader question: do PixelDiT's features
carry information a frozen encoder on the plain low-resolution image does not?
Same checkpoint, same 480 m frame, but scene-level labels and a kNN / linear
probe instead of a trained decoder -- so a result costs GPU-hours, not GPU-days.

Datasets come from torchgeo-bench, and so do the probes, but the pipeline does
not: extraction is decoupled from probing by an on-disk feature cache, because
two of the six arms cost ~4 s per image and must never be recomputed for a
probing decision. See :mod:`.features` for why, and :mod:`.bands` for the
measured band order that has to be right before any of it runs.

Stages, each independently resumable::

    scripts/cls_band_gate.py    GATE: confirm band order and radiometry
    scripts/cls_extract.py      images -> feature cache
    scripts/cls_probe.py        feature cache -> results/models/<arm>.csv
"""

from __future__ import annotations

from .arms import (
    ARMS,
    CHEAP_ARMS,
    GENERATIVE_ARMS,
    CheapExtractor,
    GenerativeExtractor,
    aggregate_views,
)
from .bands import CONDITIONING_NORM, TRUE_BAND_ORDER, pixeldit_indices, verify_band_order
from .datasets import (
    CLASSIFICATION_DATASETS,
    GEOMETRIES,
    SPLITS,
    TILE_POOLS,
    ClsSplit,
    cache_group,
    canonical_dataset,
    footprint_views,
    group_ids,
    load_split,
    tile_offsets,
    to_footprint,
)
from .features import DEFAULT_CACHE_ROOT, FeatureCache
from .probe import FEATURE_NORMS, probe_arm

__all__ = [
    "ARMS", "CHEAP_ARMS", "GENERATIVE_ARMS", "CheapExtractor", "GenerativeExtractor",
    "aggregate_views",
    "CONDITIONING_NORM", "TRUE_BAND_ORDER", "pixeldit_indices", "verify_band_order",
    "CLASSIFICATION_DATASETS", "GEOMETRIES", "TILE_POOLS", "SPLITS", "ClsSplit",
    "cache_group", "canonical_dataset", "footprint_views", "group_ids", "load_split",
    "tile_offsets", "to_footprint",
    "DEFAULT_CACHE_ROOT", "FeatureCache",
    "FEATURE_NORMS", "probe_arm",
]
