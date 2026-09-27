"""On-disk feature cache: extract once, probe many times.

WHY THIS EXISTS. torchgeo-bench's classification pipeline runs the backbone
inside the dataloader loop, once per run, holding features only in RAM. That is
the right design for a backbone that costs 50 ms per image. Two of our arms run
a 50-step diffusion sampler and cost ~4 s per image, which changes the
arithmetic completely:

* Re-probing with a different ``knn_k``, a different C range, or standardized
  instead of raw features would mean re-generating every image.
* ``eurosat`` and ``eurosat-spatial`` are the *same* 27,000 images under two
  split assignments. Extracting per dataset would generate all of them twice.
* A wall-clock timeout partway through a 40-hour split would lose the split.

So features land on disk, keyed by stable sample id, in contiguous shards. A
shard is the unit of resume: an interrupted run rejoins at the first shard that
is missing, and nothing already computed is recomputed.

LAYOUT::

    <root>/<group>/<arm>__<geometry>/
        manifest.json      arm, group, geometry, dim, ids, shard size, provenance
        shard_00000.npz    ids + (n, dim) float32
        shard_00001.npz
        ...

``group`` is not always the dataset name: ``eurosat`` and ``eurosat-spatial``
share the group ``eurosat`` (see :func:`~superres.classification.datasets.cache_group`),
which is what makes the second one free.

Features are stored **raw and unnormalized**, deliberately. The arms sit on wildly
different scales -- ``pixeldit_repa`` has std ~919 because REPA trained with a
cosine loss and never learned magnitude, against DINOv3's ~0.2 -- and which
normalization is fair is a question for the probe, not the extractor. Storing raw
keeps both answers available from one extraction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["FeatureCache", "CacheManifest", "DEFAULT_CACHE_ROOT"]

DEFAULT_CACHE_ROOT = Path("data/cls_features")
_SHARD_FMT = "shard_{:05d}.npz"


@dataclass
class CacheManifest:
    """What a cache directory holds, and what produced it."""

    group: str
    arm: str
    geometry: str
    dim: int
    shard_size: int
    ids: list[str]
    provenance: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "group": self.group,
                "arm": self.arm,
                "geometry": self.geometry,
                "dim": self.dim,
                "shard_size": self.shard_size,
                "n": len(self.ids),
                "ids": self.ids,
                "provenance": self.provenance,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> CacheManifest:
        d = json.loads(text)
        return cls(
            group=d["group"], arm=d["arm"], geometry=d["geometry"], dim=d["dim"],
            shard_size=d["shard_size"], ids=list(d["ids"]),
            provenance=d.get("provenance", {}),
        )


class FeatureCache:
    """Sharded, resumable store of one arm's features over one image group."""

    def __init__(
        self,
        root: Path | str,
        group: str,
        arm: str,
        geometry: str,
    ) -> None:
        self.dir = Path(root) / group / f"{arm}__{geometry}"
        self.group, self.arm, self.geometry = group, arm, geometry

    # ---- manifest ---------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def read_manifest(self) -> CacheManifest | None:
        if not self.manifest_path.is_file():
            return None
        return CacheManifest.from_json(self.manifest_path.read_text())

    def open_for_write(
        self,
        ids: list[str],
        dim: int,
        *,
        shard_size: int = 512,
        provenance: dict | None = None,
    ) -> CacheManifest:
        """Create or reuse the cache for ``ids``, returning the manifest.

        Reusing requires the id list to match exactly. A mismatch means the
        enumeration changed -- a different split assignment, a re-downloaded
        dataset -- and silently appending to it would interleave features from
        two different image sets, so it is an error.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        existing = self.read_manifest()
        wanted = CacheManifest(
            group=self.group, arm=self.arm, geometry=self.geometry, dim=dim,
            shard_size=shard_size, ids=list(ids), provenance=provenance or {},
        )
        if existing is not None:
            if existing.ids != wanted.ids:
                raise ValueError(
                    f"{self.dir}: cached id list ({len(existing.ids)} ids) differs from "
                    f"the requested one ({len(wanted.ids)} ids). Delete the directory to "
                    "re-extract, rather than mixing two image sets in one cache."
                )
            if existing.dim != dim:
                raise ValueError(
                    f"{self.dir}: cached feature dim is {existing.dim}, requested {dim}"
                )
            if existing.shard_size != shard_size:
                raise ValueError(
                    f"{self.dir}: cached shard_size is {existing.shard_size}, requested "
                    f"{shard_size}; resume needs the same shard boundaries"
                )
            return existing
        self.manifest_path.write_text(wanted.to_json())
        return wanted

    # ---- shards -----------------------------------------------------------

    def shard_path(self, index: int) -> Path:
        return self.dir / _SHARD_FMT.format(index)

    def n_shards(self, n_items: int, shard_size: int) -> int:
        return (n_items + shard_size - 1) // shard_size

    def missing_shards(self, n_items: int, shard_size: int) -> list[int]:
        """Shard indices not yet on disk, in order -- the work still to do."""
        return [
            i for i in range(self.n_shards(n_items, shard_size))
            if not self.shard_path(i).is_file()
        ]

    def write_shard(self, index: int, ids: list[str], features: np.ndarray) -> None:
        """Write one shard atomically, so a killed job leaves no partial file."""
        if len(ids) != len(features):
            raise ValueError(f"{len(ids)} ids but {len(features)} feature rows")
        tmp = self.shard_path(index).with_suffix(".npz.tmp")
        # Write through a file object, not a path: np.savez appends ".npz" to any
        # path that does not already end in it, which would silently produce
        # "shard_00000.npz.tmp.npz" and leave the rename with nothing to move.
        with tmp.open("wb") as handle:
            np.savez(handle, ids=np.array(ids, dtype=object),
                     features=features.astype(np.float32))
        tmp.replace(self.shard_path(index))

    # ---- reading ----------------------------------------------------------

    def load_all(self) -> tuple[list[str], np.ndarray]:
        """Every cached feature, in manifest id order.

        Raises if any shard is missing, because a probe on a partial cache would
        silently score a subset of the test set.
        """
        manifest = self.read_manifest()
        if manifest is None:
            raise FileNotFoundError(f"no manifest at {self.manifest_path}")
        missing = self.missing_shards(len(manifest.ids), manifest.shard_size)
        if missing:
            raise FileNotFoundError(
                f"{self.dir}: {len(missing)} of "
                f"{self.n_shards(len(manifest.ids), manifest.shard_size)} shards missing "
                f"(first: {missing[0]}). Finish extraction before probing."
            )
        ids: list[str] = []
        blocks: list[np.ndarray] = []
        for i in range(self.n_shards(len(manifest.ids), manifest.shard_size)):
            with np.load(self.shard_path(i), allow_pickle=True) as z:
                ids.extend(str(s) for s in z["ids"])
                blocks.append(z["features"])
        features = np.concatenate(blocks, axis=0)
        if ids != manifest.ids:
            raise ValueError(
                f"{self.dir}: shard ids do not reproduce the manifest order; "
                "the cache is inconsistent and should be rebuilt"
            )
        return ids, features

    def load_for(self, ids: list[str]) -> np.ndarray:
        """Gather features for ``ids``, in the order given.

        This is what lets ``eurosat-spatial`` reuse ``eurosat``'s cache: it asks
        for its own split membership by id and gets rows back in its own order.
        """
        cached_ids, features = self.load_all()
        index = {sid: i for i, sid in enumerate(cached_ids)}
        missing = [sid for sid in ids if sid not in index]
        if missing:
            raise KeyError(
                f"{self.dir}: {len(missing)} requested ids are not in the cache "
                f"(first: {missing[:3]}). The cache was built over a different image set."
            )
        return features[np.array([index[sid] for sid in ids])]

    def is_complete(self) -> bool:
        manifest = self.read_manifest()
        if manifest is None:
            return False
        return not self.missing_shards(len(manifest.ids), manifest.shard_size)
