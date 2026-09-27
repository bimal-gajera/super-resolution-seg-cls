"""Classification datasets, stable sample ids, and the 480 m footprint.

Wraps the four torchgeo-bench classification sets so that every arm sees the
same twelve bands, in PixelDiT's order, over the same ground.

TWO THINGS THIS LAYER OWNS.

**Stable ids.** Features are extracted once and probed many times, so every
sample needs a name that survives across processes and runs. Both backends
supply one: GeoBench V1 exposes ``sample_ids`` (its partition JSON), and
torchgeo's EuroSAT exposes ``samples`` (ImageFolder paths). Index order is
deterministic in both, which is what lets an interrupted extraction resume.

**The footprint.** PixelDiT is rigid: 48 px at 10 m/px = 480 m across, and it
has no way to be told otherwise. Every one of these datasets is 64 px at
10 m/px = 640 m. Three ways to bridge that, and they are not equivalent:

======  =====  =========  =======  ==========================================
mode    views  footprint  GSD      what it costs
======  =====  =========  =======  ==========================================
resize      1  640 m      13.3 m   a 1.33x scale error, applied uniformly
tile4       4  640 m      10.0 m   4x the compute; needs a pooling choice
crop        1  480 m      10.0 m   DISCARDS 44% of the labelled footprint
======  =====  =========  =======  ==========================================

The ordering that matters is **label integrity first**. A centre crop keeps only
56% of the area, so on a presence/absence task the label can say the object is
present while the pixels no longer contain it. That is not noise to be averaged away --
it is a corrupted target, and no arm can recover from it. ``resize`` and
``tile4`` both see the entire labelled footprint; ``crop`` does not, which is why
it is **not** the default and should only be used deliberately.

``tile4`` is the default, on measurement rather than principle. Probed on 384
m-eurosat images across 5 splits it beat ``resize`` on 9 of 10 arm x seed
combinations (+0.079 mean for dinov3_lr_528, +0.044 for pixeldit_repa), so the
1.33x scale error is not free. Four 48 px windows at offsets {0, 16} in each axis
tile the 64 px source with
**complete coverage** -- every source pixel appears in at least one window -- and
each window is exactly 480 m at exactly 10 m/px, so PixelDiT is fully in
distribution and nothing labelled is thrown away. The price is 4x compute and one
extra decision, how to pool the four view features:

* ``mean`` -- the default. Right for land cover, where the label describes the
  whole patch and averaging views is averaging evidence.
* ``max`` -- right for presence/absence, where one view containing the object is
  enough. No current dataset needs it; it exists for a detection task.

``resize`` remains useful. It costs 1x compute against tile4's 4x, which matters
for the generative pass at ~4 s/image, and its scale error is uniform across arms
so a comparison made under it is still controlled -- just uniformly handicapped.
Measured on an H100, the cheap pass over all 48,061 unique images is ~1.5 h under
``resize`` and ~5.9 h under ``tile4`` for two arms, so the cheap arms can afford
both and the geometry effect never has to be assumed.

Whatever the mode, geometry must be the **same for every arm** in a comparison,
since it decides what ground each arm sees.

``interpolation`` defaults to ``area``. torchgeo-bench's own ``_ResizeTransform``
calls ``F.interpolate`` without ``antialias``, so a bilinear or bicubic 64->48
downsample aliases; ``area`` is the box filter that does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..constants import LR_PATCH, TILE_METRES
from .bands import CONDITIONING_NORM, pixeldit_indices

__all__ = [
    "CLASSIFICATION_DATASETS",
    "SPLITS",
    "GEOMETRIES",
    "TILE_POOLS",
    "footprint_views",
    "tile_offsets",
    "ClsSplit",
    "cache_group",
    "canonical_dataset",
    "group_ids",
    "load_split",
    "to_footprint",
    "native_gsd",
]

# The three sets this track uses, all EuroSAT-derived. eurosat and
# eurosat-spatial are the SAME 27,000 images under two different split
# assignments (random vs longitude-based), so their features are extracted once
# and shared -- see cache_group. m-eurosat is a 4,000-image subset of them, so
# the track currently covers one image source and one task type. Adding a
# dataset means registering it here, in _CACHE_GROUPS/_GROUP_CANONICAL, and in
# bands.TRUE_BAND_ORDER + bands.CONDITIONING_NORM, then re-running the gate.
CLASSIFICATION_DATASETS = ("m-eurosat", "eurosat", "eurosat-spatial")

SPLITS = ("train", "val", "test")
# Ordered by preference. "tile4" is the default: it is the only mode that both
# preserves the whole labelled footprint and holds PixelDiT's exact GSD, and it
# measured better than "resize" on every arm tried. "crop" preserves neither the
# footprint nor the label, and is kept only as an ablation.
GEOMETRIES = ("tile4", "resize", "crop")
TILE_POOLS = ("mean", "max")

# Sets whose extracted features are interchangeable, keyed by the group name the
# cache is stored under. Membership means "identical images, identical order".
_CACHE_GROUPS = {
    "eurosat": "eurosat",
    "eurosat-spatial": "eurosat",
    "m-eurosat": "m-eurosat",
}

# The dataset whose split enumeration defines a group's canonical id list. For
# the eurosat group either member would do -- the image set is the same -- so one
# is fixed here to keep the cache's id order reproducible.
_GROUP_CANONICAL = {"eurosat": "eurosat", "m-eurosat": "m-eurosat"}


@dataclass
class ClsSplit:
    """One split of one dataset, with everything an extractor or probe needs."""

    dataset: str
    split: str
    loader: DataLoader
    ids: list[str]
    labels: torch.Tensor
    num_classes: int
    norm_mode: str
    band_indices: list[int]

    def __len__(self) -> int:
        return len(self.ids)


def cache_group(dataset: str) -> str:
    """The name under which this dataset's features are cached.

    ``eurosat`` and ``eurosat-spatial`` share one group: same 27,000 images,
    only the train/val/test assignment differs, so extracting twice would
    double the most expensive stage of the pipeline for nothing.
    """
    return _CACHE_GROUPS[dataset]


def canonical_dataset(group: str) -> str:
    """The dataset used to enumerate a group's images for extraction.

    Extraction always runs over the canonical member, so ``eurosat-spatial``
    never triggers a second pass: it shares ``eurosat``'s cache and simply asks
    for its own split membership by id at probe time.
    """
    return _GROUP_CANONICAL[group]


def group_ids(group: str, **loader_kwargs) -> tuple[list[str], dict[str, list[str]]]:
    """Every image in a group, in canonical order, plus the per-split id lists.

    The concatenated train/val/test order of the canonical dataset *is* the cache
    order. It is deterministic because each split loader is unshuffled and its
    ids come from the backend's own index order.
    """
    canonical = canonical_dataset(group)
    per_split = {}
    for split in SPLITS:
        per_split[split] = load_split(canonical, split, **loader_kwargs).ids
    flat = [sid for split in SPLITS for sid in per_split[split]]
    if len(set(flat)) != len(flat):
        raise ValueError(
            f"group {group!r}: sample ids are not unique across splits, so they "
            "cannot key a shared cache"
        )
    return flat, per_split


def native_gsd(dataset: str, image_size: int) -> float:
    """Metres per pixel of the delivered image. All four sets are 10 m Sentinel-2."""
    del dataset
    return TILE_METRES / LR_PATCH if image_size == LR_PATCH else 10.0


def _sample_ids(inner: Dataset, dataset: str, split: str) -> list[str]:
    """Stable per-sample names, in the dataset's own index order."""
    if hasattr(inner, "sample_ids"):                       # GeoBench V1 backends
        return [str(s) for s in inner.sample_ids]
    if hasattr(inner, "samples"):                          # torchgeo ImageFolder
        return [Path(p).stem for p, _ in inner.samples]
    raise TypeError(
        f"{dataset}/{split}: {type(inner).__name__} exposes neither .sample_ids "
        "nor .samples, so samples cannot be given stable cache keys"
    )


class _GatherBands:
    """Sample transform: keep PixelDiT's twelve bands, in PixelDiT's order.

    Applied inside the dataset rather than after collation so the dataloader
    workers do the gather and only 12 of 13 channels cross the process
    boundary. Indices come from the *measured* band order (see bands.py), never
    from the dataset's own metadata, which is wrong for m-eurosat.
    """

    def __init__(self, indices: list[int]) -> None:
        self.indices = torch.tensor(indices, dtype=torch.long)

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        if image.shape[0] <= int(self.indices.max()):
            raise ValueError(
                f"image has {image.shape[0]} channels but band index "
                f"{int(self.indices.max())} was requested; band order is stale"
            )
        sample["image"] = image.index_select(0, self.indices)
        return sample


def load_split(
    dataset: str,
    split: str,
    *,
    batch_size: int = 32,
    num_workers: int = 8,
    norm_mode: str | None = None,
) -> ClsSplit:
    """Build a deterministic, non-shuffled loader over one split.

    Yields ``image`` of shape ``(B, 12, 64, 64)`` in raw band values -- no
    normalization and no resizing. Both are the arm's business: normalization
    because PixelDiT and DINOv3 want different stretches, and geometry because
    ``to_footprint`` is a cheap batched GPU op.

    Args:
        dataset: One of :data:`CLASSIFICATION_DATASETS`.
        split: ``"train"``, ``"val"`` or ``"test"``.
        batch_size: Samples per batch.
        num_workers: Dataloader workers; 0 disables multiprocessing.
        norm_mode: Override the per-dataset conditioning normalization.

    Returns:
        A :class:`ClsSplit`. ``loader`` is unshuffled, so batch order matches
        ``ids`` and ``labels`` exactly -- which is what makes shard-level
        resume safe.
    """
    from torchgeo_bench.datasets import get_bench_dataset_class

    if dataset not in CLASSIFICATION_DATASETS:
        raise ValueError(f"{dataset!r} is not one of {CLASSIFICATION_DATASETS}")
    if split not in SPLITS:
        raise ValueError(f"{split!r} is not one of {SPLITS}")

    spec = get_bench_dataset_class(dataset)()
    indices = pixeldit_indices(dataset)
    # bands=None asks for every channel in the file's own order, which is what
    # the measured indices above are relative to. Passing band *names* here
    # would route through the dataset's broken metadata instead.
    inner = spec.get_dataset(split, bands=None, transform=_GatherBands(indices))
    ids = _sample_ids(inner, dataset, split)

    loader = DataLoader(
        inner,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )
    labels = _collect_labels(inner, spec)
    return ClsSplit(
        dataset=dataset,
        split=split,
        loader=loader,
        ids=ids,
        labels=labels,
        num_classes=spec.num_classes,
        norm_mode=norm_mode or CONDITIONING_NORM[dataset],
        band_indices=indices,
    )


def _collect_labels(inner: Dataset, spec: object) -> torch.Tensor:
    """Labels in index order, read without decoding imagery where possible."""
    targets = getattr(inner, "targets", None)
    if targets is None and hasattr(inner, "samples"):
        targets = [c for _, c in inner.samples]
    if targets is not None:
        return torch.as_tensor(list(targets), dtype=torch.long)
    # GeoBench V1 stores the label in per-sample metadata, so it has to be read
    # sample by sample. Cheap next to feature extraction, and done once.
    out = [int(inner[i]["label"]) for i in range(len(inner))]  # type: ignore[index]
    return torch.tensor(out, dtype=torch.long)


def tile_offsets(source: int, size: int) -> list[int]:
    """Top-left offsets of the minimum window set that fully covers ``source``.

    Windows of width ``size`` at these offsets leave no source pixel uncovered,
    which is the whole point: a tiling that missed pixels would reintroduce the
    centre-crop problem of a label describing ground the model never saw.

    For the 64 -> 48 case this returns ``[0, 16]``, so a 2x2 grid of four
    windows. Offsets are spread evenly rather than packed at the edges, so the
    overlap sits in the middle where it is least wasteful.

    Raises:
        ValueError: if ``size`` is larger than ``source`` (nothing to tile).
    """
    if size > source:
        raise ValueError(f"window {size} is larger than the source {source}")
    if size == source:
        return [0]
    # ceil((source - size) / size) + 1 windows is the fewest that can span the
    # gap, since each step may advance by at most `size` without leaving a hole.
    count = -(-(source - size) // size) + 1
    if count == 1:
        return [0]
    span = source - size
    return [round(i * span / (count - 1)) for i in range(count)]


def footprint_views(
    images: torch.Tensor,
    *,
    geometry: str = "tile4",
    size: int = LR_PATCH,
    interpolation: str = "area",
) -> torch.Tensor:
    """Bring ``(B, C, H, W)`` onto PixelDiT's 48 px frame as ``(V, B, C, 48, 48)``.

    The leading view axis is what lets ``tile4`` exist without special-casing the
    single-view modes: every geometry returns a stack, and the caller pools over
    it. See the module docstring for which geometry to use.

    Args:
        images: ``(B, C, H, W)`` raw band values.
        geometry: ``"resize"`` (1 view, full footprint, 1.33x scale error),
            ``"tile4"`` (4 views, full footprint, exact GSD), or ``"crop"``
            (1 view, exact GSD, discards 44% of the footprint).
        size: Target side, normally 48.
        interpolation: ``area`` / ``bilinear`` / ``bicubic`` / ``nearest``. Used
            only by ``resize``; ``area`` is the default because the others alias
            when downsampling (``F.interpolate`` applies no antialias filter).

    Returns:
        ``(V, B, C, size, size)`` -- V is 1 for ``resize`` and ``crop``, and
        ``len(tile_offsets(H, size)) ** 2`` for ``tile4`` (4 for 64 -> 48).
    """
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
    height, width = images.shape[-2:]
    if height != width:
        raise ValueError(f"expected a square patch, got {height}x{width}")

    if geometry == "resize":
        if height == size:
            return images.unsqueeze(0)
        align = False if interpolation in ("bilinear", "bicubic") else None
        resized = F.interpolate(
            images.float(), size=(size, size), mode=interpolation, align_corners=align
        )
        return resized.unsqueeze(0)

    if geometry == "crop":
        if height < size:
            raise ValueError(
                f"cannot centre-crop {height}x{width} to {size}x{size}; "
                "the source is smaller than the target"
            )
        top = left = (height - size) // 2
        return images[..., top : top + size, left : left + size].unsqueeze(0)

    offsets = tile_offsets(height, size)
    views = [
        images[..., top : top + size, left : left + size]
        for top in offsets
        for left in offsets
    ]
    return torch.stack(views, dim=0)


def to_footprint(
    images: torch.Tensor,
    *,
    geometry: str = "resize",
    size: int = LR_PATCH,
    interpolation: str = "area",
) -> torch.Tensor:
    """Single-view convenience wrapper over :func:`footprint_views`.

    Raises on a multi-view geometry rather than silently returning only the first
    window, which would quietly be a crop.
    """
    views = footprint_views(
        images, geometry=geometry, size=size, interpolation=interpolation
    )
    if views.shape[0] != 1:
        raise ValueError(
            f"geometry {geometry!r} produces {views.shape[0]} views; use "
            "footprint_views and pool over them"
        )
    return views[0]
