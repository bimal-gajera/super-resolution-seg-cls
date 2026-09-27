#!/usr/bin/env python3
"""Render side-by-side predictions from both flows on the same windows.

The evaluation scripts give numbers; this gives pictures. Per 480 m window:

    S2 input | PixelDiT SR | ground truth | flow 1 pred | flow 2 pred

and with --instances a second row showing the watershed instance maps, which
is where the two arms actually differ (flow 2 leads on mIoU_I by ~24%).

Windows rather than whole tiles on purpose: a stitched tile is ~10000x15000 px
and shows nothing at screen size, whereas one window is 960x960 at 0.5 m.

    python scripts/compare_predictions.py --tiles 27_vietnam --n 4 --instances
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window as RioWindow

from superres.constants import LABEL_SIZE, LR_PATCH, RGB_BAND_NAMES
from superres.data.prepare import LABEL_SCALE
from superres.metrics import confusion, semantic_metrics, watershed_instances
from superres.models.dinov3 import FrozenDINOv3
from superres.models.pipeline import SegmentationFlow, bicubic_to_hr, normalize_rgb
from superres.models.pixeldit import rgb_stats
from superres.torch_utils import autocast
from superres.viz import save_panels

REPO = Path(__file__).resolve().parent.parent


def colour_instances(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Random colour per instance so neighbouring parcels are distinguishable."""
    rng = np.random.default_rng(seed)
    n = int(labels.max())
    palette = rng.uniform(0.25, 1.0, size=(n + 1, 3))
    palette[0] = 0.0                      # background stays black
    return palette[labels].transpose(2, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, default=REPO / "data" / "prepared")
    ap.add_argument("--sr-cache", type=Path, default=REPO / "data" / "sr_cache")
    ap.add_argument("--stats", type=Path, default=REPO / "configs" / "s2_band_stats.json")
    ap.add_argument("--flow1-ckpt", type=Path, default=REPO / "runs" / "flow1" / "best.pt")
    ap.add_argument("--flow2-ckpt", type=Path, default=REPO / "runs" / "flow2" / "best.pt")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "predictions")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tiles", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=4, help="windows to render")
    ap.add_argument("--min-field", type=float, default=0.25)
    ap.add_argument("--max-field", type=float, default=0.85)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--instances", action="store_true", help="add the watershed row")
    ap.add_argument("--target", default=None,
                    choices=("interior", "areas", "lines"),
                    help="ground truth to compare against; defaults to what the "
                         "checkpoints were trained on")
    ap.add_argument("--watershed", default="gradient",
                    choices=("gradient", "distance"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    manifest = pd.read_parquet(args.prepared / "manifest.parquet")
    rows = manifest[manifest.split == args.split]
    if args.tiles:
        rows = rows[rows.tile.isin(args.tiles)]
    rows = rows[rows.area_frac.between(args.min_field, args.max_field)]
    if rows.empty:
        raise SystemExit("no windows matched -- loosen --min-field/--max-field")
    picks = rows.sample(n=min(args.n, len(rows)), random_state=args.seed)

    lo, scale = rgb_stats(args.stats)
    encoder = FrozenDINOv3()
    models, trained_targets = {}, set()
    for flow, ckpt in ((1, args.flow1_ckpt), (2, args.flow2_ckpt)):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        kind = state.get("args", {}).get("decoder", "dpt")
        trained_targets.add(state.get("args", {}).get("target", "areas"))
        model = SegmentationFlow(encoder, out_size=LABEL_SIZE, decoder=kind).to(args.device)
        model.head.load_state_dict(state["head"])
        model.eval()
        models[flow] = model
        print(f"flow {flow}: loaded {ckpt} (epoch {state.get('epoch')}, "
              f"decoder={kind}, target={state.get('args', {}).get('target')})")

    if args.target:
        target = args.target
    elif len(trained_targets) == 1:
        target = trained_targets.pop()
    else:
        raise SystemExit(f"checkpoints disagree on target: {trained_targets}; pass --target")
    print(f"comparing against target={target}, watershed={args.watershed}")

    for _, row in picks.iterrows():
        name = f"{row.tile}_r{row.row:04d}_c{row.col:04d}"

        with rasterio.open(args.prepared / row.split / f"{row.tile}_s2_10m.tif") as src:
            bands = {(d or "").upper(): i + 1 for i, d in enumerate(src.descriptions)}
            lr = src.read(
                tuple(bands[b] for b in RGB_BAND_NAMES),
                window=RioWindow(row.col, row.row, LR_PATCH, LR_PATCH),
            ).astype(np.float32)
        with rasterio.open(args.prepared / row.split / f"{row.tile}_label_0p5m.tif") as src:
            window = RioWindow(row.label_col, row.label_row, LABEL_SIZE, LABEL_SIZE)
            areas = src.read(1, window=window) > 0
            lines = src.read(2, window=window) > 0
        # must match what the models were trained to predict, or the IoU shown
        # on each panel is measured against the wrong thing
        truth = (areas & ~lines) if target == "interior" else (lines if target == "lines" else areas)

        sr = np.load(args.sr_cache / row.split / f"{name}.npy")
        sr_t = torch.from_numpy(sr).float().div(255).unsqueeze(0)
        lr_t = torch.from_numpy(lr).unsqueeze(0)
        bicubic = bicubic_to_hr(normalize_rgb(lr_t, lo, scale, mode="per-image"))

        probs = {}
        for flow, model in models.items():
            hr = sr_t if flow == 1 else bicubic
            cond = normalize_rgb(lr_t, lo, scale, mode="per-image").to(args.device)
            with torch.no_grad(), autocast(args.device):
                logits = model(hr.to(args.device), cond)
            probs[flow] = torch.sigmoid(logits.float())[0, 0].cpu().numpy()

        preds = {f: p >= args.threshold for f, p in probs.items()}
        scores = {f: semantic_metrics(confusion(preds[f], truth))["iou"] for f in preds}

        view_lr = bicubic_to_hr(normalize_rgb(lr_t, lo, scale, mode="per-image"), LABEL_SIZE)[0].numpy()
        view_sr = torch.nn.functional.interpolate(
            sr_t, size=(LABEL_SIZE, LABEL_SIZE), mode="bilinear", align_corners=False
        )[0].numpy()

        panels = [
            ("S2 input (10 m)", view_lr),
            ("PixelDiT SR (0.3 m)", view_sr),
            ("ground truth", truth.astype(np.float32)),
            (f"flow 1 — SR   IoU {scores[1]:.3f}", preds[1].astype(np.float32)),
            (f"flow 2 — bicubic   IoU {scores[2]:.3f}", preds[2].astype(np.float32)),
        ]
        out = save_panels(
            panels, args.out / f"semantic_{name}.png",
            title=f"{name}  |  480 m window at 0.5 m  |  field fraction {row.area_frac:.2f}",
        )
        print(f"  wrote {out}   flow1 IoU {scores[1]:.3f} | flow2 IoU {scores[2]:.3f}")

        if args.instances:
            from scipy import ndimage

            # `interior` already excludes the boundary; `areas` still needs cutting
            separated = truth if target == "interior" else (truth & ~lines)
            truth_inst, n_true = ndimage.label(separated)
            inst = {
                f: watershed_instances(probs[f], args.threshold, mode=args.watershed)
                for f in probs
            }
            out = save_panels(
                [
                    ("ground truth instances", colour_instances(truth_inst)),
                    (f"flow 1 — SR   ({int(inst[1].max())} found)", colour_instances(inst[1])),
                    (f"flow 2 — bicubic   ({int(inst[2].max())} found)", colour_instances(inst[2])),
                ],
                args.out / f"instances_{name}.png",
                title=f"{name}  |  watershed instances  |  {n_true} true parcels",
            )
            print(f"  wrote {out}   true {n_true} | flow1 {int(inst[1].max())} "
                  f"| flow2 {int(inst[2].max())}")


if __name__ == "__main__":
    main()
