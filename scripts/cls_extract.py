#!/usr/bin/env python3
"""Extract one pass of feature arms over one dataset group into the cache.

Resumable at shard granularity: rerun the same command after a wall-clock kill
and it rejoins at the first missing shard. Nothing already computed is redone,
which is the point -- the generative pass costs ~4 s per image and a 27,000-image
group is over a day of GPU time.

Run the gate first. ``scripts/cls_band_gate.py`` confirms the band order, and a
scrambled band order produces no error here, only wrong numbers later.

CHEAP PASS -- all three arms, ~0.5 s/image, no image generation::

    python scripts/cls_extract.py --dataset m-eurosat \\
        --arms dinov3_lr_528 dinov3_lr_1584 pixeldit_repa

GENERATIVE PASS -- ~4 s/image, three arms off one generation. Wants an 80 GB
card: two 2 B-parameter denoisers in fp32 is ~16 GB of weights before
activations::

    python scripts/cls_extract.py --dataset m-eurosat \\
        --arms dinov3_sr_528 dinov3_sr_1584 hybrid_1m --batch-size 4

The two passes cannot be mixed in one invocation, because they load different
weight sets and want different batch sizes. Ask for both and this script says so.

``--dataset eurosat-spatial`` is redirected to ``eurosat``: the two are the same
27,000 images under different split assignments, so they share one cache and the
second one needs no extraction at all.

Budgeting over the 31,000 unique images in all three sets (27,000 eurosat, which
also serves eurosat-spatial, plus 4,000 m-eurosat). Measured on an H100, all three
cheap arms:

    cheap pass, resize   0.232 s/image   ~2 h
    cheap pass, tile4    0.895 s/image   ~7.7 h
    generative pass      ~4 s/image      ~34 h

which is why ``--limit`` exists: measure on a few hundred images and multiply
before committing to the second one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from superres.classification.arms import (  # noqa: E402
    ARMS,
    CheapExtractor,
    GenerativeExtractor,
    aggregate_views,
    arms_for_pass,
)
from superres.classification.bands import verify_pixeldit_stack  # noqa: E402
from superres.classification.datasets import (  # noqa: E402
    CLASSIFICATION_DATASETS,
    GEOMETRIES,
    SPLITS,
    TILE_POOLS,
    cache_group,
    canonical_dataset,
    footprint_views,
    load_split,
)
from superres.classification.features import DEFAULT_CACHE_ROOT, FeatureCache  # noqa: E402
from superres.constants import LR_PATCH  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO / "weights" / "pixeldit_weights.ckpt"
DEFAULT_STATS = REPO / "configs" / "s2_band_stats.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, choices=list(CLASSIFICATION_DATASETS))
    parser.add_argument("--arms", nargs="+", required=True, choices=sorted(ARMS),
                        help="all must belong to the same pass (cheap or generative)")
    parser.add_argument("--geometry", default="tile4", choices=list(GEOMETRIES),
                        help="how to reach PixelDiT's 48 px / 480 m frame from 64 px / "
                             "640 m. 'tile4' (default) sees the whole labelled footprint "
                             "at exact GSD in 4 windows covering every pixel, 4x the "
                             "compute -- and measured better than resize on 9 of 10 "
                             "arm x seed trials. 'resize' sees the whole footprint at a "
                             "1.33x scale error for 1x compute, which is the affordable "
                             "choice for the generative pass. 'crop' throws away 44%% of "
                             "the footprint and can drop the labelled object entirely -- "
                             "an ablation, not a reporting mode.")
    parser.add_argument("--tile-pool", default="mean", choices=list(TILE_POOLS),
                        help="how to combine the 4 tile4 views: 'mean' averages evidence "
                             "(land cover), 'max' takes the per-dimension maximum "
                             "(presence/absence). Ignored unless "
                             "--geometry tile4.")
    parser.add_argument("--interpolation", default="area",
                        choices=("area", "bilinear", "bicubic", "nearest"),
                        help="resize filter; 'area' is the only one that does not alias "
                             "on a 64->48 downsample")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="32 suits the cheap pass; use 2-4 for the generative one")
    parser.add_argument("--shard-size", type=int, default=512,
                        help="samples per cache shard, and so the resume granularity")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N images -- for timing a pass before "
                             "committing to it. Does NOT write a usable cache.")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT,
                        help="point this somewhere scratch when using --limit, so a "
                             "deliberately partial cache cannot be mistaken for real")
    parser.add_argument("--norm", default=None, choices=("stats", "per-image"),
                        help="override the per-dataset conditioning normalization "
                             "(see bands.CONDITIONING_NORM); normally leave alone")
    parser.add_argument("--skip-gate", action="store_true",
                        help="skip the band-order recheck. Don't.")

    generative = parser.add_argument_group("generative pass")
    generative.add_argument("--num-steps", type=int, default=50)
    generative.add_argument("--guidance", type=float, default=2.5)
    generative.add_argument("--timeshift", type=float, default=1.0)
    generative.add_argument("--guidance-interval", default="0.3:1")
    generative.add_argument("--gen-seed", type=int, default=0,
                            help="fixed, so the same image always yields the same sample")
    generative.add_argument("--readback-t", type=float, default=0.9)
    generative.add_argument("--hybrid-grid", type=int, default=480)
    generative.add_argument("--dinov3-micro-batch", type=int, default=2,
                            help="sub-batch for the 1584 px DINOv3 pass (99x99 tokens)")
    generative.add_argument("--weights-bf16", action="store_true",
                            help="hold denoiser weights in bf16, halving ~16 GB of "
                                 "weights. Changes numerics slightly; only for cards "
                                 "that cannot fit fp32.")
    generative.add_argument("--repa-seeds", type=int, default=4,
                            help="noise draws averaged per image for pixeldit_repa")
    return parser


def cache_variant(geometry: str, tile_pool: str) -> str:
    """Cache sub-directory name for a geometry.

    ``tile4`` with different pooling yields different features, so the pool is
    part of the cache identity -- otherwise a mean run and a max run would
    overwrite each other's shards under a single manifest that describes neither.
    """
    return f"{geometry}-{tile_pool}" if geometry == "tile4" else geometry


def _flat_plan(group: str, loader_kwargs: dict) -> tuple[list[str], list[tuple[str, int]]]:
    """Canonical id list for the group, plus (split, index) for each position.

    Extraction walks splits in order and concatenates, so position p in the cache
    is a known (split, within-split index) -- which is what lets a shard be
    reconstructed without re-reading earlier splits.
    """
    canonical = canonical_dataset(group)
    ids: list[str] = []
    origin: list[tuple[str, int]] = []
    for split in SPLITS:
        split_ids = load_split(canonical, split, **loader_kwargs).ids
        ids.extend(split_ids)
        origin.extend((split, i) for i in range(len(split_ids)))
    return ids, origin


def main() -> int:
    args = build_parser().parse_args()
    cheap, generative = arms_for_pass(list(args.arms))
    if cheap and generative:
        raise SystemExit(
            f"cheap arms {cheap} and generative arms {generative} cannot share one run: "
            "they load different weights and want different batch sizes. Run twice."
        )
    arms = cheap or generative
    group = cache_group(args.dataset)
    canonical = canonical_dataset(group)
    if canonical != args.dataset:
        print(f"note: {args.dataset} shares the {group!r} cache with {canonical}; "
              f"extracting over {canonical} and reusing it.")

    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.workers,
                     "norm_mode": args.norm}
    ids, _ = _flat_plan(group, loader_kwargs)
    if args.limit:
        ids = ids[: args.limit]
        print(f"note: --limit {args.limit} -- this run is for timing only and the "
              "cache it leaves will be incomplete.")
    views = 4 if args.geometry == "tile4" else 1
    print(f"group {group}: {len(ids)} images, arms {arms}, geometry {args.geometry} "
          f"({views} view{'s' if views > 1 else ''}"
          f"{', pooled by ' + args.tile_pool if views > 1 else ''})")

    provenance = {
        "canonical_dataset": canonical,
        "geometry": args.geometry,
        "tile_pool": args.tile_pool if args.geometry == "tile4" else None,
        "interpolation": args.interpolation,
        "checkpoint": str(args.ckpt),
        "stats": str(args.stats),
        "norm_override": args.norm,
        "pass": "generative" if generative else "cheap",
    }
    if generative:
        provenance.update(
            num_steps=args.num_steps, guidance=args.guidance, timeshift=args.timeshift,
            guidance_interval=args.guidance_interval, gen_seed=args.gen_seed,
            readback_t=args.readback_t, hybrid_grid=args.hybrid_grid,
            weights_bf16=args.weights_bf16,
        )
    else:
        provenance.update(repa_seeds=args.repa_seeds)

    variant = cache_variant(args.geometry, args.tile_pool)
    caches = {}
    todo: set[int] = set()
    for arm in arms:
        cache = FeatureCache(args.cache_root, group, arm, variant)
        manifest = cache.open_for_write(
            ids, ARMS[arm].dim, shard_size=args.shard_size, provenance=provenance
        )
        caches[arm] = cache
        missing = cache.missing_shards(len(ids), manifest.shard_size)
        todo.update(missing)
        print(f"  {arm:16s} dim={ARMS[arm].dim:5d}  "
              f"{len(missing)}/{cache.n_shards(len(ids), manifest.shard_size)} shards to do")
    if not todo:
        print("nothing to do: every arm's cache is already complete.")
        return 0
    shards = sorted(todo)

    print(f"  cache variant: {variant}")
    extractor = _build_extractor(args, cheap, generative)
    print(f"\nextracting {len(shards)} shard(s) of {args.shard_size} ...")
    return _run(args, group, ids, shards, caches, extractor, loader_kwargs)


def _build_extractor(args, cheap: list[str], generative: list[str]):
    common = {"ckpt": str(args.ckpt), "stats_path": str(args.stats), "device": args.device}
    if cheap:
        return CheapExtractor(
            cheap, repa_seeds=args.repa_seeds,
            dinov3_micro_batch=args.dinov3_micro_batch, **common,
        )
    return GenerativeExtractor(
        generative, num_steps=args.num_steps, guidance=args.guidance,
        timeshift=args.timeshift, guidance_interval=args.guidance_interval,
        gen_seed=args.gen_seed, readback_t=args.readback_t,
        hybrid_grid=args.hybrid_grid, dinov3_micro_batch=args.dinov3_micro_batch,
        weights_dtype=torch.bfloat16 if args.weights_bf16 else None, **common,
    )


def _run(args, group, ids, shards, caches, extractor, loader_kwargs) -> int:
    """Walk the requested shards, extract, and write.

    Iterates split by split with an unshuffled loader, which reproduces the
    canonical order exactly, and buffers into shard-sized blocks. Batches whose
    whole shard is already on disk are skipped without running the model.
    """
    canonical = canonical_dataset(group)
    wanted = set(shards)
    shard_size = args.shard_size
    buffers: dict[str, list[np.ndarray]] = {arm: [] for arm in caches}
    buffer_ids: list[str] = []
    current_shard = None
    position = 0
    gated = args.skip_gate
    done = 0
    started = time.time()

    def flush() -> None:
        nonlocal buffers, buffer_ids, current_shard
        if current_shard is None or not buffer_ids:
            return
        if current_shard in wanted:
            for arm, cache in caches.items():
                cache.write_shard(
                    current_shard, buffer_ids, np.concatenate(buffers[arm], axis=0)
                )
        buffers = {arm: [] for arm in caches}
        buffer_ids = []
        current_shard = None

    for split in SPLITS:
        if position >= len(ids):
            break
        spec = load_split(canonical, split, **loader_kwargs)
        for batch in spec.loader:
            images = batch["image"]
            batch_positions = list(range(position, position + len(images)))
            position += len(images)
            if batch_positions[0] >= len(ids):
                break
            # Trim a batch that runs past --limit.
            keep = [p for p in batch_positions if p < len(ids)]
            if len(keep) < len(images):
                images = images[: len(keep)]

            if not gated:
                # The dataset layer has already gathered to twelve bands in
                # PixelDiT's order, so this checks the gather too -- a stronger
                # test than the raw-file check the gate script runs.
                verify_pixeldit_stack(images)
                gated = True
                print("  12-band stack re-checked against this batch: OK")

            # Skip whole batches that belong only to finished shards.
            if all(p // shard_size not in wanted for p in keep):
                continue

            views = footprint_views(
                images, geometry=args.geometry, size=LR_PATCH,
                interpolation=args.interpolation,
            )
            features = aggregate_views(
                extractor, views, spec.norm_mode, pool=args.tile_pool
            )
            for arm in caches:
                if arm not in features:
                    raise RuntimeError(f"extractor produced no output for arm {arm!r}")

            for offset, p in enumerate(keep):
                shard = p // shard_size
                if shard != current_shard:
                    flush()
                    current_shard = shard
                for arm in caches:
                    buffers[arm].append(features[arm][offset : offset + 1].numpy())
                buffer_ids.append(ids[p])
            done += len(keep)
            if done % (shard_size) < len(keep):
                rate = done / max(time.time() - started, 1e-9)
                remaining = (len(ids) - done) / max(rate, 1e-9)
                print(f"    {done}/{len(ids)}  {rate:.2f} img/s  "
                      f"eta {remaining / 3600:.2f} h", flush=True)
    flush()

    elapsed = time.time() - started
    print(f"\ndone: {done} images in {elapsed / 60:.1f} min "
          f"({done / max(elapsed, 1e-9):.2f} img/s, "
          f"{elapsed / max(done, 1):.3f} s/img)")
    for arm, cache in caches.items():
        state = "complete" if cache.is_complete() else "INCOMPLETE"
        print(f"  {arm:16s} {state}  {cache.dir}")
    timing = {
        "group": group, "arms": list(caches),
        "geometry": cache_variant(args.geometry, args.tile_pool),
        "images": done, "seconds": elapsed, "images_per_second": done / max(elapsed, 1e-9),
    }
    print("\ntiming: " + json.dumps(timing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
