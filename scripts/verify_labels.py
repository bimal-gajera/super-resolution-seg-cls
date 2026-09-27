#!/usr/bin/env python3
"""Check the 0.5 m rasterized labels against the imagery and the shipped masks.

Two things worth confirming before training on these labels:

1. **Alignment.** Overlay the `areas` and `lines` bands on the (upsampled) S2
   window. Field edges should follow visible structure.
2. **Provenance.** The shipped `masks/*.tif` are field *boundary lines*, not
   areas -- so our `lines` band, max-pooled to 10 m, should agree with them,
   and our `areas` band should not.

    python scripts/verify_labels.py [--tile 47_vietnam]

Needs no GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window as RioWindow

from superres.constants import LABEL_SIZE, LR_PATCH, RGB_BAND_NAMES
from superres.data.prepare import LABEL_SCALE
from superres.models.pipeline import bicubic_to_hr
from superres.models.pixeldit import to_unit_range
from superres.viz import save_panels

REPO = Path(__file__).resolve().parent.parent


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, default=REPO / "data" / "prepared")
    ap.add_argument("--raw", type=Path, default=REPO / "data" / "raw" / "sentinel-2-asia")
    ap.add_argument("--tile", default=None, help="default: the busiest tile")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "label_check")
    args = ap.parse_args()

    manifest = pd.read_parquet(args.prepared / "manifest.parquet")
    tile = args.tile or manifest.loc[manifest.area_frac.idxmax(), "tile"]
    rows = manifest[manifest.tile == tile]
    split = rows.iloc[0].split
    row = rows.loc[rows.area_frac.sub(0.6).abs().idxmin()]

    # ---- 1: alignment ----------------------------------------------------
    s2_path = args.prepared / split / f"{tile}_s2_10m.tif"
    with rasterio.open(s2_path) as src:
        names = {(d or "").upper(): i + 1 for i, d in enumerate(src.descriptions)}
        lr = src.read(
            tuple(names[b] for b in RGB_BAND_NAMES),
            window=RioWindow(row.col, row.row, LR_PATCH, LR_PATCH),
        ).astype(np.float32)
        transform_10m, crs, h10, w10 = src.transform, src.crs, src.height, src.width

    label_path = args.prepared / split / f"{tile}_label_0p5m.tif"
    with rasterio.open(label_path) as src:
        patch = src.read(
            (1, 2), window=RioWindow(row.label_col, row.label_row, LABEL_SIZE, LABEL_SIZE)
        )
        areas_full, lines_full = src.read(1) > 0, src.read(2) > 0

    image = bicubic_to_hr(
        to_unit_range(torch.from_numpy(lr).unsqueeze(0), mode="per-image"), LABEL_SIZE
    )[0].numpy()
    tinted = image.copy()
    tinted[1] = np.clip(tinted[1] + patch[0] * 0.25, 0, 1)
    outlined = image.copy()
    outlined[0] = np.maximum(outlined[0], patch[1] * 0.9)

    out = save_panels(
        [("S2 upsampled (10 m data)", image), ("+ areas band", tinted), ("+ lines band", outlined)],
        args.out / f"labels_{tile}_r{row.row:04d}_c{row.col:04d}.png",
        title=f"{tile} r{row.row} c{row.col} — 0.5 m labels over 10 m imagery "
              f"(field frac {row.area_frac:.2f})",
    )
    print(f"alignment panel: {out}")

    # ---- 2: provenance ---------------------------------------------------
    shipped = np.zeros((h10, w10), dtype=np.uint8)
    with rasterio.open(args.raw / split / "masks" / f"{tile}.tif") as src:
        reproject(
            rasterio.band(src, 1), shipped,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform_10m, dst_crs=crs, resampling=Resampling.nearest,
        )
    shipped = shipped > 0

    pooled = {
        name: band.reshape(h10, LABEL_SCALE, w10, LABEL_SCALE).max(axis=(1, 3))
        for name, band in (("lines", lines_full), ("areas", areas_full))
    }
    print(f"\nshipped mask positive fraction: {shipped.mean():.3f}")
    for name, band in pooled.items():
        print(f"  our {name:6s} @10m: {band.mean():.3f}   IoU vs shipped = {iou(band, shipped):.3f}")
    print("\nExpect `lines` to agree and `areas` not to: the shipped masks are boundaries.")


if __name__ == "__main__":
    main()
