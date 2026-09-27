#!/usr/bin/env python3
"""Does the `resize` scale error actually cost accuracy against exact-GSD tiling?

Getting a 64 px / 640 m patch onto PixelDiT's fixed 48 px / 480 m frame forces a
choice, and two of the three options preserve the whole labelled footprint:

* ``tile4``  -- exact 10 m/px GSD, 4 covering windows, 4x compute
* ``resize`` -- one view, but a 1.33x scale error, 1x compute

(The third, ``crop``, keeps only 56% of the area and can drop the labelled object
outright, so it is an ablation rather than a reporting mode.)

``tile4`` costs 4x, so whether that buys anything should not be an assumption.
This script probes the *same images* under both geometries and prints the gap.

Not a benchmark. It splits one extracted subset internally, so the numbers are
small-N and not comparable to anything published -- the only thing to read off is
the difference between two rows. For real numbers use ``cls_probe.py`` on a
complete cache.

Needs no GPU: kNN over a few hundred 1024-d vectors is instant.

    # after extracting the same --limit N under both geometries
    python scripts/cls_geomcheck.py --cache-root data/cls_geomcheck --n 384
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from superres.classification.datasets import load_split  # noqa: E402
from superres.classification.features import FeatureCache  # noqa: E402


def knn_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    k: int = 5,
) -> float:
    """kNN-5 accuracy, z-scored on the train part only (as ``probe.py`` does)."""
    from torchgeo_bench.knn import KNNClassifier

    mean = features[train_idx].mean(axis=0, keepdims=True)
    std = features[train_idx].std(axis=0, keepdims=True)
    scaled = (features - mean) / np.where(std < 1e-6, 1.0, std)
    clf = KNNClassifier(n_neighbors=k, device="cpu").fit(
        scaled[train_idx].astype(np.float32), labels[train_idx]
    )
    predicted = clf.predict(scaled[test_idx].astype(np.float32))
    return float((predicted == labels[test_idx]).mean())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache-root", type=Path, default=Path("data/cls_geomcheck"))
    parser.add_argument("--dataset", default="m-eurosat")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=384,
                        help="must match the --limit used when extracting")
    parser.add_argument("--arms", nargs="+",
                        default=["dinov3_lr_528", "pixeldit_repa"])
    parser.add_argument("--geometries", nargs="+",
                        default=["resize", "tile4-mean"],
                        help="cache variant names, as cls_extract.py wrote them")
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    spec = load_split(args.dataset, args.split, batch_size=64, num_workers=4)
    labels = spec.labels.numpy()[: args.n]
    counts = np.bincount(labels)
    chance = counts.max() / len(labels)
    print(f"{args.dataset}/{args.split}, first {len(labels)} images: "
          f"{int((counts > 0).sum())} classes present, majority-class rate "
          f"{chance:.4f}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(labels))
    cut = int(len(labels) * args.train_frac)
    train_idx, test_idx = order[:cut], order[cut:]
    print(f"internal split: {len(train_idx)} train / {len(test_idx)} test, seed {args.seed}\n")

    print(f"{'arm':16s} " + "".join(f"{g:>14s}" for g in args.geometries) + f"{'gap':>10s}")
    missing = False
    for arm in args.arms:
        scores: list[float | None] = []
        for geometry in args.geometries:
            cache = FeatureCache(args.cache_root, spec.dataset, arm, geometry)
            if not cache.is_complete():
                scores.append(None)
                missing = True
                continue
            features = cache.load_all()[1]
            if len(features) != len(labels):
                raise SystemExit(
                    f"{cache.dir} holds {len(features)} rows but --n is {len(labels)}; "
                    "re-run with --n matching the extraction's --limit"
                )
            scores.append(knn_accuracy(features, labels, train_idx, test_idx))
        cells = "".join("      (no cache)" if s is None else f"{s:14.4f}" for s in scores)
        gap = ""
        if len(scores) == 2 and None not in scores:
            gap = f"{scores[1] - scores[0]:+10.4f}"
        print(f"{arm:16s}{cells}{gap}")

    print(f"\nmajority-class rate is {chance:.4f}; anything near it means no signal.")
    print("'gap' is the second geometry minus the first: positive favours "
          f"{args.geometries[-1]}.")
    if missing:
        print("\nSome caches are missing -- extract both geometries at the same --limit.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
