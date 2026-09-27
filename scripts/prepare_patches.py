#!/usr/bin/env python3
"""Regrid tiles to 10 m, rasterize 0.5 m labels, enumerate 48x48 windows.

    python scripts/prepare_patches.py [--raw data/raw] [--out data/prepared]

Writes one regridded S2 raster and one 2-band label raster per tile, plus
`manifest.parquet` listing every window. See `superres.data.prepare` for why
each step exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from superres.constants import SPLITS
from superres.data.prepare import prepare_tile, write_manifest

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "prepared")
    ap.add_argument("--region", default="sentinel-2-asia")
    ap.add_argument("--country", default="vietnam")
    ap.add_argument(
        "--boundary-width-m",
        type=float,
        default=2.0,
        help="physical width to buffer boundary lines to before rasterizing",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tiles, for a trial")
    args = ap.parse_args()

    root = args.raw / args.region
    reference = root / "reference"
    if not reference.is_dir():
        raise SystemExit(f"no reference/ under {root} -- run scripts/download.py first")

    jobs = []
    for split in SPLITS:
        images = sorted((root / split / "images").glob(f"*{args.country}*.tif"))
        for image_path in images:
            tile = image_path.stem
            areas = reference / f"{tile}_areas.gpkg"
            lines = reference / f"{tile}_lines.gpkg"
            missing = [p.name for p in (areas, lines) if not p.exists()]
            if missing:
                print(f"  !! {tile}: missing {missing}, skipping")
                continue
            jobs.append((tile, split, image_path, areas, lines))

    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        raise SystemExit(f"no {args.country} tiles found under {root}")

    print(f"{len(jobs)} tiles to prepare -> {args.out}")
    all_windows = []
    for i, (tile, split, image_path, areas, lines) in enumerate(jobs, 1):
        windows = prepare_tile(
            tile, split, image_path, areas, lines, args.out,
            boundary_width_m=args.boundary_width_m, overwrite=args.overwrite,
        )
        all_windows.extend(windows)
        print(f"  [{i}/{len(jobs)}] {split:9s} {tile:16s} {len(windows):4d} windows")

    manifest_path = args.out / "manifest.parquet"
    df = write_manifest(all_windows, manifest_path)

    print(f"\nwrote {manifest_path}  ({len(df)} windows)")
    summary = df.groupby("split").agg(
        tiles=("tile", "nunique"),
        windows=("tile", "size"),
        mean_field_frac=("area_frac", "mean"),
        empty_windows=("area_frac", lambda s: int((s == 0).sum())),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
