#!/usr/bin/env python3
"""Download the AI4SmallFarms Vietnam subset from the DANS data station.

    python scripts/download.py [--dest data/raw] [--country vietnam]

Public record, CC-BY-4.0, no account needed. Resumable: files already present
and intact are skipped, so re-running after an interruption is cheap.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from superres.constants import AI4SMALLFARMS_DOI
from superres.data.download import download, list_files, select

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=REPO / "data" / "raw")
    ap.add_argument(
        "--country",
        default="vietnam",
        help="'vietnam' (default), 'cambodia', or 'all' for both",
    )
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched, then stop")
    args = ap.parse_args()

    print(f"listing {AI4SMALLFARMS_DOI} ...")
    everything = list_files()
    picked = select(everything, country=None if args.country == "all" else args.country)

    by_kind = Counter(f.kind for f in picked)
    tiles = {f.tile for f in picked if f.kind == "images"}
    total_mb = sum(f.size for f in picked) / 1e6
    print(f"{len(everything)} files in record; {len(picked)} selected ({total_mb:.0f} MB)")
    print(f"  by kind: {dict(by_kind)}")
    print(f"  {len(tiles)} image tiles")
    for split in ("train", "validate", "test"):
        n = len({f.tile for f in picked if f.kind == "images" and f.split == split})
        print(f"    {split:9s} {n:3d} tiles")

    if args.dry_run:
        return

    print(f"\ndownloading to {args.dest}")
    written = download(picked, args.dest)
    print(f"\n{len(written)} files present under {args.dest}")


if __name__ == "__main__":
    main()
