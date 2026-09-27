#!/usr/bin/env python3
"""Probe cached features with kNN and a linear classifier, and write results.

Reads the feature cache, runs torchgeo-bench's own ``evaluate_knn`` and
``evaluate_logistic`` on it, and appends rows to ``results/models/<arm>.csv``.
No GPU needed by default -- this is seconds to minutes of CPU per arm, against
hours of extraction, which is exactly why the two stages are separate.

Every arm is scored twice, once on raw features and once standardized per
dimension. Neither probe normalizes its input, and our arms are not on
comparable scales (``pixeldit_repa`` runs to std ~900 against DINOv3's ~0.2), so
``raw`` keeps the published protocol and ``zscore`` makes the arms fair against
each other. The ``feature_norm`` part of the ``name`` column says which.

``eurosat-spatial`` needs no extraction of its own: it resolves its split
membership against the shared ``eurosat`` cache by sample id.

    # everything that has a complete cache
    python scripts/cls_probe.py --all

    # one dataset, one arm
    python scripts/cls_probe.py --datasets m-eurosat --arms pixeldit_repa

    # compare arms side by side once rows exist
    python scripts/cls_probe.py --summarize
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from superres.classification.arms import ARMS  # noqa: E402
from superres.classification.datasets import (  # noqa: E402
    CLASSIFICATION_DATASETS,
    SPLITS,
    cache_group,
    load_split,
)
from superres.classification.features import DEFAULT_CACHE_ROOT, FeatureCache  # noqa: E402
from superres.classification.probe import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    FEATURE_NORMS,
    probe_arm,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=list(CLASSIFICATION_DATASETS),
                        choices=list(CLASSIFICATION_DATASETS))
    parser.add_argument("--arms", nargs="+", default=sorted(ARMS), choices=sorted(ARMS))
    parser.add_argument("--geometry", default="tile4-mean",
                        help="cache variant to probe: 'resize', 'crop', or "
                             "'tile4-mean' / 'tile4-max'. Must match what "
                             "cls_extract.py wrote -- see its --geometry/--tile-pool.")
    parser.add_argument("--all", action="store_true",
                        help="probe every dataset/arm whose cache is complete, "
                             "skipping the rest quietly")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--feature-norms", nargs="+", default=list(FEATURE_NORMS),
                        choices=list(FEATURE_NORMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--c-range", nargs=3, type=float, default=(-6.0, 4.0, 40),
                        metavar=("LOG10_START", "LOG10_STOP", "NUM"))
    parser.add_argument("--no-merge-val", action="store_true",
                        help="fit the final logistic model on train only, leaving val "
                             "genuinely held out")
    parser.add_argument("--device", default="cpu",
                        help="'cpu' avoids needing a GPU FAISS build; the probes are "
                             "small enough that it rarely matters")
    parser.add_argument("--workers", type=int, default=4,
                        help="dataloader workers used only to enumerate ids and labels")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute and print rows without writing the CSV")
    parser.add_argument("--summarize", action="store_true",
                        help="print a comparison table from existing CSVs and exit")
    return parser


def _splits_for(dataset: str, workers: int) -> tuple[dict, dict, int]:
    """Per-split ids and labels for ``dataset``, plus its class count."""
    ids: dict[str, list[str]] = {}
    labels: dict[str, np.ndarray] = {}
    num_classes = 0
    for split in SPLITS:
        # batch_size is irrelevant here: only .ids and .labels are read.
        spec = load_split(dataset, split, batch_size=256, num_workers=workers)
        ids[split] = spec.ids
        labels[split] = spec.labels.numpy()
        num_classes = spec.num_classes
    return ids, labels, num_classes


def summarize(results_dir: Path, geometry: str) -> int:
    """Print one row per (dataset, arm, feature_norm, method) from the CSVs."""
    import pandas as pd

    files = sorted(Path(results_dir).glob("*.csv"))
    if not files:
        print(f"no result CSVs under {results_dir}")
        return 1
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame = frame[frame["interpolation"] == geometry]
    if frame.empty:
        print(f"no rows with geometry={geometry!r}")
        return 1
    frame["feature_norm"] = frame["name"].str.rsplit("__", n=1).str[-1]
    table = frame.pivot_table(
        index=["dataset", "model"], columns=["feature_norm", "method"],
        values="metric_value", aggfunc="max",
    )
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(f"\naccuracy, geometry={geometry}\n")
        print(table.round(4).to_string())
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.summarize:
        return summarize(args.results_dir, args.geometry)

    merge_val = not args.no_merge_val
    c_range = (args.c_range[0], args.c_range[1], int(args.c_range[2]))
    planned: list[tuple[str, str]] = []
    for dataset in args.datasets:
        for arm in args.arms:
            cache = FeatureCache(args.cache_root, cache_group(dataset), arm, args.geometry)
            if cache.is_complete():
                planned.append((dataset, arm))
            elif args.all:
                continue
            else:
                print(f"skip {dataset}/{arm}: cache incomplete at {cache.dir}")
    if not planned:
        print("nothing to probe -- no complete caches matched. Run scripts/cls_extract.py.")
        return 1

    print(f"probing {len(planned)} (dataset, arm) pair(s), geometry={args.geometry}, "
          f"feature_norms={args.feature_norms}")
    # Enumerating ids/labels touches the dataset, so do it once per dataset.
    cached_splits: dict[str, tuple[dict, dict, int]] = {}
    failures = 0
    for dataset, arm in planned:
        if dataset not in cached_splits:
            cached_splits[dataset] = _splits_for(dataset, args.workers)
        ids, labels, num_classes = cached_splits[dataset]
        try:
            rows = probe_arm(
                dataset, arm, args.geometry,
                split_ids=ids, split_labels=labels, num_classes=num_classes,
                cache_root=args.cache_root, results_dir=args.results_dir,
                seed=args.seed, knn_k=args.knn_k, bootstrap=args.bootstrap,
                c_range=c_range, merge_val=merge_val, device=args.device,
                feature_norms=tuple(args.feature_norms), write=not args.dry_run,
            )
        except Exception as exc:                        # noqa: BLE001 -- keep going
            failures += 1
            print(f"  FAIL {dataset}/{arm}: {type(exc).__name__}: {exc}")
            continue
        for row in rows:
            print(f"  {row['dataset']:16s} {row['name']:44s} {row['method']:8s} "
                  f"acc={row['metric_value']:.4f} "
                  f"[{row['ci_lower']:.4f}-{row['ci_upper']:.4f}]")

    if not args.dry_run:
        print(f"\nrows appended under {args.results_dir}")
        summarize(args.results_dir, args.geometry)
    if failures:
        print(f"\n{failures} pair(s) failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
