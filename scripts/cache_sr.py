#!/usr/bin/env python3
"""Generate and cache PixelDiT super-resolved images for every window.

The 50-step sampler costs ~3.4 s per patch on an H100 -- thousands of times
everything else in the pipeline -- so flow 1 cannot sample inside its training
loop. Each window is generated once here and written to disk as uint8.

    python scripts/cache_sr.py                    # all splits, seed 0
    python scripts/cache_sr.py --split test --seed 1

Resumable: windows already cached are skipped, so an interrupted or
time-limited job can simply be resubmitted.

Seeds: training uses a single seed. SEED-SR averages four seeds at *test*
time, so generating extra seeds for the test split is the v2 instance-metric
path; those land in `sr_cache/seed<N>/`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window as RioWindow

from superres.constants import LR_PATCH, RGB_BAND_NAMES
from superres.models.pixeldit import (
    generate_hr, load_denoiser, load_state, make_sampler, prepare_lr_cond, rgb_stats,
)

REPO = Path(__file__).resolve().parent.parent


def cache_path(root: Path, seed: int, row) -> Path:
    stem = f"{row.tile}_r{row.row:04d}_c{row.col:04d}.npy"
    base = root if seed == 0 else root / f"seed{seed}"
    return base / row.split / stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, default=REPO / "data" / "prepared")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "sr_cache")
    ap.add_argument("--ckpt", type=Path, default=REPO / "weights" / "pixeldit_rgb.ckpt")
    ap.add_argument("--stats", type=Path, default=REPO / "configs" / "s2_band_stats.json")
    ap.add_argument("--split", default=None, help="train|validate|test (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--norm", default="stats", choices=("stats", "per-image"),
                    help="normalization for PixelDiT conditioning; 'stats' wins "
                         "empirically -- PixelDiT learned an absolute reflectance "
                         "mapping, so rescaling each patch independently destroys it")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    manifest = pd.read_parquet(args.prepared / "manifest.parquet")
    if args.split:
        manifest = manifest[manifest.split == args.split]
    manifest = manifest.reset_index(drop=True)

    todo = [row for _, row in manifest.iterrows()
            if not cache_path(args.out, args.seed, row).exists()]
    cached = len(manifest) - len(todo)
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(manifest)} windows; {cached} already cached; {len(todo)} to generate"
          + (f" (--limit {args.limit})" if args.limit else ""))
    if not todo:
        return

    lo, scale = rgb_stats(args.stats)
    print(f"loading {args.ckpt} ...")
    state = load_state(args.ckpt)
    net = load_denoiser(state, "ema_denoiser.", args.device)
    del state
    sampler = make_sampler(args.device, num_steps=args.steps)

    readers: dict[str, rasterio.DatasetReader] = {}
    bands_for: dict[str, tuple[int, ...]] = {}

    def read_lr(row) -> np.ndarray:
        key = f"{row.split}/{row.tile}"
        if key not in readers:
            path = args.prepared / row.split / f"{row.tile}_s2_10m.tif"
            readers[key] = rasterio.open(path)
            names = {(d or "").upper(): i + 1 for i, d in enumerate(readers[key].descriptions)}
            bands_for[key] = tuple(names[b] for b in RGB_BAND_NAMES)
        return readers[key].read(
            bands_for[key], window=RioWindow(row.col, row.row, LR_PATCH, LR_PATCH)
        ).astype(np.float32)

    started = time.time()
    generated = 0
    for start in range(0, len(todo), args.batch_size):
        chunk = todo[start : start + args.batch_size]
        lr = torch.from_numpy(np.stack([read_lr(r) for r in chunk]))
        cond = prepare_lr_cond(lr, lo, scale, mode=args.norm).to(args.device)
        images = generate_hr(net, sampler, cond, args.device, seed=args.seed)

        for row, image in zip(chunk, images):
            array = ((image + 1) / 2).clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
            path = cache_path(args.out, args.seed, row)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part.npy")
            np.save(tmp, array)
            tmp.replace(path)

        generated += len(chunk)
        elapsed = time.time() - started
        rate = elapsed / generated
        remaining = (len(todo) - generated) * rate
        print(
            f"  {generated}/{len(todo)}  {rate:.2f}s/patch  "
            f"eta {remaining / 3600:.1f}h",
            flush=True,
        )

    for reader in readers.values():
        reader.close()
    print(f"\ndone: {generated} patches in {(time.time() - started) / 3600:.2f}h -> {args.out}")


if __name__ == "__main__":
    main()
