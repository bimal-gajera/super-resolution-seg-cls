#!/usr/bin/env python3
"""Train the segmentation decoder for one flow.

    python scripts/train.py --flow 2            # baseline, needs no SR cache
    python scripts/train.py --flow 1            # needs scripts/cache_sr.py first

Both flows must be trained with identical hyperparameters -- that is the whole
point of the comparison, so the defaults here are deliberately flow-agnostic
and `--flow` changes nothing except which pixels reach DINOv3.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from superres.constants import LABEL_SIZE
from superres.torch_utils import autocast
from superres.data.dataset import AI4SmallFarmsPatches
from superres.losses import DiceBCELoss
from superres.metrics import aggregate, confusion, semantic_metrics
from superres.models.dinov3 import FrozenDINOv3
from superres.models.pipeline import (
    SegmentationFlow, condition_from_batch, hr_from_batch,
)
from superres.models.pixeldit import rgb_stats

REPO = Path(__file__).resolve().parent.parent


def build_loader(args, split: str, shuffle: bool) -> DataLoader:
    dataset = AI4SmallFarmsPatches(
        args.prepared, split,
        sr_cache=args.sr_cache if args.flow == 1 else None,
        target=args.target,
    )
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=True, drop_last=shuffle,
        persistent_workers=args.workers > 0,
    )


@torch.no_grad()
def evaluate(model, loader, args, lo, scale, threshold: float = 0.5) -> dict[str, float]:
    """Per-*window* validation, for model selection during training.

    Not the headline number: SEED-SR's metrics are per image, so the reported
    results come from `scripts/evaluate.py`, which stitches windows back
    into whole tiles first. This is a cheap monitoring signal, and the two are
    not directly comparable.
    """
    model.eval()
    per_image = []
    for index, batch in enumerate(loader):
        if args.limit_val_batches and index >= args.limit_val_batches:
            break
        hr = hr_from_batch(batch, args.flow, lo, scale, args.baseline_norm, args.device)
        cond = condition_from_batch(batch, lo, scale, args.baseline_norm, args.device)
        target = batch["label"].to(args.device, non_blocking=True)
        with autocast(args.device):
            logits = model(hr, cond)
        predictions = (torch.sigmoid(logits.float()) >= threshold).cpu().numpy()
        truths = target.cpu().numpy() > 0.5
        for pred, truth in zip(predictions, truths):
            per_image.append(semantic_metrics(confusion(pred[0], truth[0])))
    return aggregate(per_image)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow", type=int, required=True, choices=(1, 2))
    ap.add_argument("--prepared", type=Path, default=REPO / "data" / "prepared")
    ap.add_argument("--sr-cache", type=Path, default=REPO / "data" / "sr_cache")
    ap.add_argument("--stats", type=Path, default=REPO / "configs" / "s2_band_stats.json")
    ap.add_argument("--out", type=Path, default=None, help="default: runs/flow<N>")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--decoder", default="seedsr", choices=("seedsr", "dpt"),
                    help="'seedsr' mirrors the paper's U-Net decoder (transposed-conv "
                         "blocks, SFT conditioning, late cross-attention). 'dpt' is the "
                         "earlier head, kept for comparison")
    ap.add_argument("--target", default="interior",
                    choices=("interior", "areas", "lines"),
                    help="'interior' = field areas minus the boundary (default): the "
                         "bare polygon union merges touching parcels, capping instance "
                         "recovery at ~39%%. 'areas' is the raw union. 'lines' predicts "
                         "the boundary itself (~3%% positive -- pair with --pos-weight)")
    ap.add_argument("--pos-weight", type=float, default=None,
                    help="positive-class weight in BCE; useful for the thin 'lines' target")
    ap.add_argument("--boundary-weight", type=float, default=0.0)
    ap.add_argument("--limit-steps", type=int, default=None, help="cap steps/epoch, for smoke runs")
    ap.add_argument("--limit-val-batches", type=int, default=None,
                    help="cap validation batches, for smoke runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline-norm", default="per-image", choices=("per-image", "stats"),
                    help="normalization for FLOW 2's DINOv3 input only. Per-image keeps the "
                         "baseline properly exposed; flow 1 is unaffected (it reads cached SR)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = args.out or REPO / "runs" / f"flow{args.flow}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    lo, scale = rgb_stats(args.stats)
    train_loader = build_loader(args, "train", shuffle=True)
    val_loader = build_loader(args, "validate", shuffle=False)
    print(f"flow {args.flow}: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val")

    encoder = FrozenDINOv3()
    model = SegmentationFlow(encoder, out_size=LABEL_SIZE, decoder=args.decoder).to(args.device)
    trainable = sum(p.numel() for p in model.head.parameters())
    print(f"decoder: {args.decoder}  |  trainable: {trainable / 1e6:.1f}M")

    criterion = DiceBCELoss(boundary_weight=args.boundary_weight,
                            pos_weight=args.pos_weight)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps = args.limit_steps or len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps)

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.encoder.eval()  # stays frozen
        running, started = 0.0, time.time()

        for step, batch in enumerate(train_loader):
            if args.limit_steps and step >= args.limit_steps:
                break
            hr = hr_from_batch(batch, args.flow, lo, scale, args.baseline_norm, args.device)
            cond = condition_from_batch(batch, lo, scale, args.baseline_norm, args.device)
            target = batch["label"].to(args.device, non_blocking=True)

            with autocast(args.device):
                logits = model(hr, cond)
            parts = criterion(logits.float(), target)

            optimizer.zero_grad(set_to_none=True)
            parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += float(parts["total"].detach())
            if step % 20 == 0:
                print(
                    f"  e{epoch} s{step}/{steps} "
                    f"loss {float(parts['total'].detach()):.4f} "
                    f"(dice {float(parts['dice']):.4f} bce {float(parts['bce']):.4f})",
                    flush=True,
                )

        metrics = evaluate(model, val_loader, args, lo, scale)
        record = {
            "epoch": epoch,
            "train_loss": running / max(1, min(steps, len(train_loader))),
            "seconds": round(time.time() - started, 1),
            **metrics,
        }
        history.append(record)
        print(f"epoch {epoch}: val mIoU_S {metrics['mIoU_S']:.4f}  "
              f"mF1 {metrics['mF1']:.4f}  ({record['seconds']}s)")

        if metrics["mIoU_S"] > best:
            best = metrics["mIoU_S"]
            torch.save(
                {
                    "head": model.head.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "args": {k: str(v) for k, v in vars(args).items()},
                },
                out_dir / "best.pt",
            )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2, default=str))

    print(f"\nbest val mIoU_S: {best:.4f}  ->  {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
