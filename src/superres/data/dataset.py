"""Torch dataset over the prepared windows.

One item is one 480 m patch:

    lr      (3, 48, 48)    float32, raw B4/B3/B2 values -- normalization is the
                           model's job, not the loader's
    label   (1, 960, 960)  float32 {0, 1}, field areas at 0.5 m
    lines   (1, 960, 960)  float32 {0, 1}, boundaries -- only if want_lines
    sr      (3, 1584, 1584) float32 [0, 1], the cached PixelDiT image -- flow 1 only

Rasters are opened lazily and cached per worker: 30 tiles, two files each, and
a DataLoader worker would otherwise reopen them on every single item.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window as RioWindow
from torch.utils.data import Dataset

from ..constants import LABEL_SIZE, LR_PATCH, RGB_BAND_NAMES


class AI4SmallFarmsPatches(Dataset):
    """Windows from `manifest.parquet`, one split at a time."""

    def __init__(
        self,
        prepared_dir: str | Path,
        split: str,
        *,
        sr_cache: str | Path | None = None,
        sr_seed: int = 0,
        want_lines: bool = False,
        target: str = "interior",
        tiles: list[str] | None = None,
        min_field_frac: float = 0.0,
        manifest_name: str = "manifest.parquet",
    ) -> None:
        self.prepared = Path(prepared_dir)
        self.split = split
        self.sr_cache = Path(sr_cache) if sr_cache else None
        self.sr_seed = sr_seed
        self.want_lines = want_lines
        if target not in ("interior", "areas", "lines"):
            raise ValueError(f"target must be 'interior', 'areas' or 'lines', got {target!r}")
        self.target = target

        manifest = pd.read_parquet(self.prepared / manifest_name)
        rows = manifest[manifest["split"] == split]
        if tiles:
            rows = rows[rows["tile"].isin(tiles)]
        if min_field_frac > 0:
            rows = rows[rows["area_frac"] >= min_field_frac]
        if rows.empty:
            raise ValueError(f"no windows for split {split!r} in {self.prepared}")
        self.rows = rows.reset_index(drop=True)

        self._readers: dict[str, rasterio.DatasetReader] = {}
        self._band_index: dict[str, tuple[int, ...]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _reader(self, key: str, path: Path) -> rasterio.DatasetReader:
        reader = self._readers.get(key)
        if reader is None or reader.closed:
            reader = rasterio.open(path)
            self._readers[key] = reader
        return reader

    def _rgb_bands(self, reader: rasterio.DatasetReader, tile: str) -> tuple[int, ...]:
        """1-based band indices for R, G, B, resolved from band descriptions.

        The rasters declare ('B2','B3','B4','B8'); we want B4/B3/B2 in that
        order. Reading positionally would silently give blue-green-red.
        """
        cached = self._band_index.get(tile)
        if cached is not None:
            return cached
        names = {(d or "").upper(): i + 1 for i, d in enumerate(reader.descriptions)}
        missing = [b for b in RGB_BAND_NAMES if b not in names]
        if missing:
            raise ValueError(
                f"{tile}: bands {missing} not found; raster declares {list(names)}"
            )
        index = tuple(names[b] for b in RGB_BAND_NAMES)
        self._band_index[tile] = index
        return index

    def sr_path(self, row) -> Path:
        """Mirrors `scripts/cache_sr.py:cache_path`.

        Seed 0 lives at the cache root; other seeds (the paper's 4-sample
        test-time averaging) live under `seed<N>/`.
        """
        if self.sr_cache is None:
            raise ValueError("no sr_cache configured")
        base = self.sr_cache if self.sr_seed == 0 else self.sr_cache / f"seed{self.sr_seed}"
        return base / row.split / f"{row.tile}_r{row.row:04d}_c{row.col:04d}.npy"

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.rows.iloc[idx]

        s2 = self._reader(f"s2:{row.tile}", self.prepared / row.split / f"{row.tile}_s2_10m.tif")
        bands = self._rgb_bands(s2, row.tile)
        lr = s2.read(
            bands,
            window=RioWindow(row.col, row.row, LR_PATCH, LR_PATCH),
        ).astype(np.float32)

        labels = self._reader(
            f"lb:{row.tile}", self.prepared / row.split / f"{row.tile}_label_0p5m.tif"
        )
        window = RioWindow(row.label_col, row.label_row, LABEL_SIZE, LABEL_SIZE)
        # band 1 = field areas (polygon union), band 2 = boundary lines
        #
        # "interior" = areas minus the boundary, and is the default. Adjacent
        # parcels share edges, so the bare polygon union merges them: measured
        # on a test crop, a PERFECT prediction of `areas` recovers only 49 of
        # 127 parcels, while `interior` recovers all 127. Predicting the gaps
        # is what makes instances recoverable at all.
        # Always read both bands: they live in the same windowed read, so the
        # saving from reading one is negligible and getting the indexing wrong
        # is not (an earlier version silently returned an empty array for
        # target="lines").
        label = labels.read((1, 2), window=window).astype(np.float32)

        if self.target == "interior":
            target_array = np.clip(label[0] - label[1], 0.0, 1.0)[None]
        elif self.target == "areas":
            target_array = label[0:1]
        else:
            target_array = label[1:2]

        item: dict[str, torch.Tensor | str] = {
            "lr": torch.from_numpy(lr),
            "label": torch.from_numpy(target_array),
            "tile": row.tile,
            "row": int(row.row),
            "col": int(row.col),
        }
        if self.want_lines:
            item["lines"] = torch.from_numpy(label[1:2])
        if self.sr_cache is not None:
            sr = np.load(self.sr_path(row))          # (3, 1584, 1584) uint8
            item["sr"] = torch.from_numpy(sr).float().div_(255.0)
        return item

    def __del__(self) -> None:
        for reader in getattr(self, "_readers", {}).values():
            try:
                reader.close()
            except Exception:
                pass
