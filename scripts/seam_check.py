#!/usr/bin/env python3
"""GATE: verify PixelDiT produces sensible imagery before spending GPU hours.

Three questions, all visual, all cheap relative to a full cache run:

1. **Normalization.** The 12-band `s2_band_stats.json` belongs to the older
   12-band checkpoint. For this RGB model its indices 3/2/1 (B4/B3/B2) are the
   best available estimate, but an estimate. This renders the same tiles under
   the training statistics and under per-image percentiles, side by side.

2. **Geometry.** Confirms a 48 px window really is 480 m and that the model
   returns 1584x1584.

3. **Seams.** Generates a 2x2 block of adjacent stride-48 windows and mosaics
   them. Adjacent windows get independent noise and hallucinate independently,
   so the same field can be rendered differently on either side of a join --
   which the decoder could read as a field boundary. If the joins are clean,
   the cheap non-overlapping tiling stands; if not, add overlap and blend.

    python scripts/seam_check.py --n 4 --out results/seam_check

Nothing downstream should run until these panels look right.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window as RioWindow

from superres.constants import HR_SIZE, LR_GSD, LR_PATCH, RGB_BAND_NAMES, TILE_METRES
from superres.models.pipeline import bicubic_to_hr, normalize_rgb
from superres.models.pixeldit import (
    generate_hr, load_denoiser, load_state, make_sampler, prepare_lr_cond, rgb_stats,
)
from superres.viz import save_panels, seam_strip

REPO = Path(__file__).resolve().parent.parent


def read_window(tile_path: Path, row: int, col: int, size: int = LR_PATCH) -> np.ndarray:
    with rasterio.open(tile_path) as src:
        names = {(d or "").upper(): i + 1 for i, d in enumerate(src.descriptions)}
        bands = tuple(names[b] for b in RGB_BAND_NAMES)
        return src.read(bands, window=RioWindow(col, row, size, size)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, default=REPO / "data" / "prepared")
    ap.add_argument("--ckpt", type=Path, default=REPO / "weights" / "pixeldit_rgb.ckpt")
    ap.add_argument("--stats", type=Path, default=REPO / "configs" / "s2_band_stats.json")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "seam_check")
    ap.add_argument("--n", type=int, default=4, help="patches for the normalization check")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--norm", default="stats", choices=("stats", "per-image"),
                help="normalization used for the seam mosaic")
    ap.add_argument("--skip-seams", action="store_true")
    args = ap.parse_args()

    manifest = pd.read_parquet(args.prepared / "manifest.parquet")
    lo, scale = rgb_stats(args.stats)

    print(f"loading {args.ckpt} ...")
    state = load_state(args.ckpt)
    net = load_denoiser(state, "ema_denoiser.", args.device)   # EMA -> better images
    del state
    sampler = make_sampler(args.device, num_steps=args.steps)

    # ---- 1 + 2: normalization and geometry -------------------------------
    busy = manifest[manifest.area_frac.between(0.3, 0.8)]
    picks = busy.sample(n=min(args.n, len(busy)), random_state=args.seed)
    print(f"\n[1/2] normalization + geometry on {len(picks)} patches")

    for _, row in picks.iterrows():
        tile_path = args.prepared / row.split / f"{row.tile}_s2_10m.tif"
        lr = torch.from_numpy(read_window(tile_path, row.row, row.col)).unsqueeze(0)
        ground_m = LR_PATCH * LR_GSD
        assert abs(ground_m - TILE_METRES) < 1e-6, f"window covers {ground_m} m, expected 480"

        panels = [("LR bicubic (input)", bicubic_to_hr(normalize_rgb(lr, lo, scale))[0].numpy())]
        for label, kwargs in (
            ("SR — training stats", {"mode": "stats"}),
            ("SR — per-image pct", {"mode": "per-image"}),
        ):
            cond = prepare_lr_cond(lr, lo, scale, **kwargs).to(args.device)
            image = generate_hr(net, sampler, cond, args.device, seed=args.seed)
            assert image.shape[-1] == HR_SIZE, f"got {tuple(image.shape)}, expected {HR_SIZE}"
            panels.append((label, ((image[0] + 1) / 2).cpu().numpy()))

        name = f"{row.tile}_r{row.row:04d}_c{row.col:04d}"
        out = save_panels(
            panels, args.out / f"norm_{name}.png",
            title=(
                f"{name}  |  {LR_PATCH}px @ {LR_GSD:g} m = {ground_m:g} m  "
                f"→ {HR_SIZE}px @ {TILE_METRES / HR_SIZE:.3f} m  |  field {row.area_frac:.2f}"
            ),
        )
        print(f"  wrote {out}")

    if args.skip_seams:
        return

    # ---- 3: seams --------------------------------------------------------
    print("\n[2/2] seam check: 2x2 block of adjacent windows")
    # Pick a window that actually HAS all three neighbours -- a randomly chosen
    # busy patch is often on a tile edge, where the 2x2 block cannot be formed.
    tile, base_r, base_c = None, None, None
    for candidate_tile, group in manifest.groupby("tile"):
        present = set(zip(group.row, group.col))
        options = [
            (r, c)
            for r, c in present
            if {(r + LR_PATCH, c), (r, c + LR_PATCH), (r + LR_PATCH, c + LR_PATCH)} <= present
        ]
        if not options:
            continue
        # prefer a block whose four windows are all field-rich, so seams are visible
        frac = {(r, c): f for r, c, f in zip(group.row, group.col, group.area_frac)}
        best = max(
            options,
            key=lambda rc: min(
                frac[(rc[0] + dr, rc[1] + dc)]
                for dr in (0, LR_PATCH) for dc in (0, LR_PATCH)
            ),
        )
        score = min(
            frac[(best[0] + dr, best[1] + dc)]
            for dr in (0, LR_PATCH) for dc in (0, LR_PATCH)
        )
        if tile is None or score > tile[1]:
            tile = (group.iloc[0], score, candidate_tile)
            base_r, base_c = best

    if tile is None:
        print("  no tile has a complete 2x2 block of windows; skipping")
        return
    tile = tile[0]
    block = [
        (r, c)
        for r in (base_r, base_r + LR_PATCH)
        for c in (base_c, base_c + LR_PATCH)
    ]
    print(f"  using {tile.tile} block at ({base_r}, {base_c})")

    tile_path = args.prepared / tile.split / f"{tile.tile}_s2_10m.tif"
    mosaic = np.zeros((3, HR_SIZE * 2, HR_SIZE * 2), dtype=np.float32)
    for r, c in block:
        lr = torch.from_numpy(read_window(tile_path, r, c)).unsqueeze(0)
        cond = prepare_lr_cond(lr, lo, scale, mode=args.norm).to(args.device)
        image = ((generate_hr(net, sampler, cond, args.device, seed=args.seed)[0] + 1) / 2)
        dr = (r - base_r) // LR_PATCH * HR_SIZE
        dc = (c - base_c) // LR_PATCH * HR_SIZE
        mosaic[:, dr : dr + HR_SIZE, dc : dc + HR_SIZE] = image.cpu().numpy()
        print(f"  generated ({r}, {c})")

    out = save_panels(
        [
            ("2x2 mosaic", mosaic),
            (f"vertical seam @ x={HR_SIZE}", seam_strip(mosaic, HR_SIZE)),
            (f"horizontal seam @ y={HR_SIZE}", seam_strip(mosaic.transpose(0, 2, 1), HR_SIZE)),
        ],
        args.out / f"seams_{tile.tile}.png",
        title=f"{tile.tile}: adjacent windows generated independently — look for a visible join",
    )
    print(f"  wrote {out}")
    print("\nInspect the panels before running scripts/cache_sr.py.")


if __name__ == "__main__":
    main()
