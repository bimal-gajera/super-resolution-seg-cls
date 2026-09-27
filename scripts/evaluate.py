#!/usr/bin/env python3
"""Evaluate a trained flow on the test split, scoring whole tiles.

Per-window logits are stitched back into full tiles before scoring, because
SEED-SR's metrics are defined per *image* and then averaged -- scoring loose
patches would answer a different question and would not be comparable.

    python scripts/evaluate.py --flow 2 --checkpoint runs/flow2/best.pt

`--seeds` averages several cached SR seeds before thresholding, which is the
paper's protocol (they use four). It requires those seeds to exist in the
cache; see scripts/cache_sr.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader

from superres.constants import LABEL_SIZE
from superres.torch_utils import autocast
from superres.data.dataset import AI4SmallFarmsPatches
from superres.data.prepare import LABEL_SCALE
from superres.metrics import aggregate, confusion, instance_iou, semantic_metrics, watershed_instances
from superres.models.dinov3 import FrozenDINOv3
from superres.models.pipeline import (
    SegmentationFlow, condition_from_batch, hr_from_batch,
)
from superres.models.pixeldit import rgb_stats
from superres.stitch import TileAccumulator

REPO = Path(__file__).resolve().parent.parent


def tile_shapes(prepared: Path, split: str, tiles: list[str]) -> dict[str, tuple[int, int]]:
    shapes = {}
    for tile in tiles:
        with rasterio.open(prepared / split / f"{tile}_label_0p5m.tif") as src:
            shapes[tile] = (src.height, src.width)
    return shapes


@torch.no_grad()
def predict_tiles(model, loader, args, lo, scale) -> dict[str, TileAccumulator]:
    prepared = Path(args.prepared)
    manifest = pd.read_parquet(prepared / "manifest.parquet")
    rows = manifest[manifest.split == args.split]
    if args.tiles:
        rows = rows[rows.tile.isin(args.tiles)]
    shapes = tile_shapes(prepared, args.split, sorted(rows.tile.unique()))
    accumulators = {t: TileAccumulator(*shapes[t], weight=args.stitch) for t in shapes}

    model.eval()
    for batch in loader:
        hr = hr_from_batch(batch, args.flow, lo, scale, args.baseline_norm, args.device)
        cond = condition_from_batch(batch, lo, scale, args.baseline_norm, args.device)
        with autocast(args.device):
            logits = model(hr, cond)
        probs = torch.sigmoid(logits.float()).cpu().numpy()

        for i, tile in enumerate(batch["tile"]):
            # manifest offsets are in 10 m pixels; the label grid is 20x finer
            row = int(batch["row"][i]) * LABEL_SCALE
            col = int(batch["col"][i]) * LABEL_SCALE
            accumulators[tile].add(probs[i, 0], row, col)
    return accumulators


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow", type=int, required=True, choices=(1, 2))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--prepared", type=Path, default=REPO / "data" / "prepared")
    ap.add_argument("--sr-cache", type=Path, default=REPO / "data" / "sr_cache")
    ap.add_argument("--stats", type=Path, default=REPO / "configs" / "s2_band_stats.json")
    ap.add_argument("--decoder", default="seedsr", choices=("seedsr", "dpt"),
                    help="'seedsr' mirrors the paper's U-Net decoder (transposed-conv "
                         "blocks, SFT conditioning, late cross-attention). 'dpt' is the "
                         "earlier head, kept for comparison")
    ap.add_argument("--target", default=None,
                    choices=("interior", "areas", "lines"),
                    help="ground truth to score against; defaults to whatever "
                         "the checkpoint was trained on")
    ap.add_argument("--watershed", default="gradient",
                    choices=("gradient", "distance"),
                    help="gradient follows SEED-SR's cited watershed (ref [40])")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0],
                    help="cached SR seeds to average over before thresholding. "
                         "SEED-SR uses four. Flow 1 only -- bicubic is deterministic, "
                         "so extra seeds would be identical images")
    ap.add_argument("--tiles", nargs="+", default=None,
                    help="score only these tiles (still whole tiles, so coverage stays complete)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--stitch", default="uniform", choices=("uniform", "cosine"))
    ap.add_argument("--instances", action="store_true", help="also compute mIoU_I (v2, slow)")
    ap.add_argument("--overlap-thresholds", nargs="+", type=float,
                    default=[0.25, 0.5, 0.75],
                    help="mIoU_I is reported at each. SEED-SR never states its t, and "
                         "the metric is insensitive to t when a model MERGES parcels but "
                         "swings ~2.2x when it SPLITS them -- so a single t can decide a "
                         "comparison between arms that fail in opposite directions")
    ap.add_argument("--baseline-norm", default="per-image", choices=("per-image", "stats"),
                    help="normalization for FLOW 2's DINOv3 input only. Per-image keeps the "
                         "baseline properly exposed; flow 1 is unaffected (it reads cached SR)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = args.out or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    lo, scale = rgb_stats(args.stats)

    seeds = sorted(set(args.seeds))
    if args.flow == 2 and len(seeds) > 1:
        print("flow 2 is deterministic; ignoring extra seeds")
        seeds = seeds[:1]

    def make_loader(seed: int) -> DataLoader:
        dataset = AI4SmallFarmsPatches(
            args.prepared, args.split,
            sr_cache=args.sr_cache if args.flow == 1 else None,
            sr_seed=seed,
            tiles=args.tiles,
        )
        return DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

    loader = make_loader(seeds[0])
    print(f"flow {args.flow}: {len(loader.dataset)} windows over "
          f"{loader.dataset.rows.tile.nunique()} tiles, seeds {seeds}")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # prefer the decoder the checkpoint was trained with, so eval cannot silently
    # rebuild the wrong architecture
    trained_with = state.get("args", {}).get("decoder", args.decoder)
    target = args.target or state.get("args", {}).get("target", "areas")
    print(f"scoring against target={target}, watershed={args.watershed}")
    if trained_with != args.decoder:
        print(f"checkpoint was trained with decoder={trained_with}; using that")
    model = SegmentationFlow(
        FrozenDINOv3(), out_size=LABEL_SIZE, decoder=trained_with
    ).to(args.device)
    model.head.load_state_dict(state["head"])
    print(f"loaded decoder from {args.checkpoint} (epoch {state.get('epoch')})")

    # SEED-SR averages the semantic logits across seeds before thresholding and
    # watershed. Averaging probabilities (post-sigmoid) is the same ordering of
    # tiles here and keeps the accumulator arithmetic in one place.
    per_seed = []
    for seed in seeds:
        if len(seeds) > 1:
            print(f"  seed {seed} ...", flush=True)
        per_seed.append(predict_tiles(model, make_loader(seed), args, lo, scale))
    accumulators = per_seed[0]

    per_image, instance_rows = [], []
    for tile, acc in sorted(accumulators.items()):
        if acc.uncovered:
            print(f"  !! {tile}: {acc.uncovered} uncovered pixels")
        probability = np.mean([p[tile].result() for p in per_seed], axis=0)
        prediction = probability >= args.threshold

        with rasterio.open(args.prepared / args.split / f"{tile}_label_0p5m.tif") as src:
            areas = src.read(1) > 0
            boundaries = src.read(2) > 0
        if target == "interior":
            truth = areas & ~boundaries
        elif target == "areas":
            truth = areas
        else:
            truth = boundaries

        metrics = semantic_metrics(confusion(prediction, truth))
        per_image.append(metrics)
        print(f"  {tile:16s} IoU {metrics['iou']:.4f}  F1 {metrics['f1']:.4f}")

        if args.instances:
            from scipy import ndimage

            # Adjacent fields SHARE an edge, so connected components of the
            # areas band alone would merge neighbouring parcels into one blob.
            # Cutting along the stored boundary band separates them.
            # `interior` already excludes the boundary; `areas` still needs cutting
            separated = truth if target == "interior" else (truth & ~boundaries)
            truth_instances, _ = ndimage.label(separated)
            predicted_instances = watershed_instances(
                probability, args.threshold, mode=args.watershed
            )
            scores = {}
            for t in args.overlap_thresholds:
                score, count = instance_iou(predicted_instances, truth_instances, t)
                scores[t] = score
            instance_rows.append((scores, count))
            shown = "  ".join(f"t={t:g}: {v:.4f}" for t, v in scores.items())
            print(f"  {'':16s} IoU_I {shown}  over {count} instances "
                  f"({int(predicted_instances.max())} predicted)")

    summary = aggregate(per_image)
    if instance_rows:
        weights = np.array([n for _, n in instance_rows], dtype=float)
        for t in args.overlap_thresholds:
            col = np.array([s[t] for s, _ in instance_rows], dtype=float)
            summary[f"mIoU_I@{t:g}"] = float(np.average(col, weights=weights))
        # keep a headline key at the conventional 0.5 when it was requested
        if 0.5 in args.overlap_thresholds:
            summary["mIoU_I"] = summary["mIoU_I@0.5"]
        summary["instances"] = int(weights.sum())

    print("\n" + json.dumps(summary, indent=2))
    path = out_dir / f"eval_{args.split}_flow{args.flow}.json"
    path.write_text(json.dumps({"summary": summary, "args": {k: str(v) for k, v in vars(args).items()}}, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
