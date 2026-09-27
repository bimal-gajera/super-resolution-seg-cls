#!/usr/bin/env python3
"""Strip optimizer state out of the PixelDiT training checkpoint.

The Lightning checkpoint carries `optimizer_states` and `loops` alongside the
weights, which roughly doubles it (32 GB -> ~16 GB). Inference needs only:

    denoiser.*               the raw model (used for internal feature taps)
    ema_denoiser.*           the EMA copy (used for generation -- better images)
    diffusion_trainer.proj.* the REPA head, 1152 -> 1024

    python scripts/strip_checkpoint.py [--src ...] [--dst weights/pixeldit_rgb.ckpt]

This checkpoint is the 3-channel RGB-conditioned model: `pixel_embedder.proj`
is (16, 6) = 3 target + 3 conditioning channels. The older 12-band model has
(16, 15). The script refuses to write if it finds the wrong one.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
# Site-specific; set SR_PIXELDIT_CKPT or pass --src.
DEFAULT_SRC = os.environ.get("SR_PIXELDIT_CKPT")
KEEP_PREFIXES = ("denoiser.", "ema_denoiser.", "diffusion_trainer.proj.")
EXPECTED_COND_IN = 6  # 3 target + 3 conditioning


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src", type=Path, default=Path(DEFAULT_SRC) if DEFAULT_SRC else None,
        required=DEFAULT_SRC is None,
        help="raw training checkpoint (or set SR_PIXELDIT_CKPT)",
    )
    ap.add_argument("--dst", type=Path, default=REPO / "weights" / "pixeldit_rgb.ckpt")
    ap.add_argument("--force", action="store_true", help="overwrite an existing destination")
    args = ap.parse_args()

    if args.dst.exists() and not args.force:
        raise SystemExit(f"{args.dst} already exists (use --force to overwrite)")

    print(f"loading {args.src} ...")
    ckpt = torch.load(args.src, map_location="cpu", mmap=True, weights_only=False)
    state = ckpt["state_dict"]

    probe = state.get("denoiser.pixel_embedder.proj.weight")
    if probe is None:
        raise SystemExit("no denoiser.pixel_embedder.proj.weight -- not a PixDiTSR checkpoint")
    if probe.shape[1] != EXPECTED_COND_IN:
        raise SystemExit(
            f"pixel_embedder.proj is {tuple(probe.shape)}, expected (*, {EXPECTED_COND_IN}). "
            "This looks like the 12-band model (16, 15), not the RGB one."
        )
    print(f"  verified RGB conditioning: pixel_embedder.proj = {tuple(probe.shape)}")

    kept = {
        k: v.clone() for k, v in state.items() if k.startswith(KEEP_PREFIXES)
    }
    dropped = len(state) - len(kept)
    print(f"  keeping {len(kept)} tensors, dropping {dropped} + optimizer state")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.dst.with_suffix(".part")
    torch.save({"state_dict": kept}, tmp)
    tmp.replace(args.dst)

    src_gb = args.src.stat().st_size / 1e9
    dst_gb = args.dst.stat().st_size / 1e9
    print(f"\nwrote {args.dst}  ({src_gb:.1f} GB -> {dst_gb:.1f} GB)")


if __name__ == "__main__":
    main()
