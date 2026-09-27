#!/usr/bin/env python3
"""GATE: confirm each dataset's band order and radiometry before spending GPU hours.

Run this first, and run it again after any dataset re-download. It is cheap
(seconds) and it guards against the one failure mode that produces no error at
all: handing PixelDiT twelve channels of the right shape holding the wrong
bands. The model generates a plausible image, the probes return plausible
numbers, and nothing anywhere says the input was scrambled.

That is not hypothetical. ``m-eurosat`` ships band metadata that labels channel
8 ``'08A - Vegetation Red Edge'`` when channel 8 actually holds B09; B8A is at
channel 12. Selecting bands by name -- the obvious thing to do -- returns the
wrong pixels for five of thirteen channels. This script is what caught it, by
comparing pixels against physics instead of against metadata.

WHAT IT CHECKS

1. **Band order**, via two relationships that hold for any scene: B8A (865 nm)
   is spectrally adjacent to B08 (842 nm) so the two must track each other, and
   B09/B10 are absorption bands so they must be much darker than B08. A channel
   permutation moves these numbers a long way.
2. **Radiometry against PixelDiT's training range**, per band. The conditioning
   stretch is absolute, so a dataset on a different scale silently clamps. This
   catches a dataset whose bands are on a different scale entirely -- 8-bit
   SWIR channels, for instance, collapse to exactly 0 under the training
   percentiles.
3. **Footprint**, printed so the crop-versus-resize decision stays explicit.

Exit status is non-zero if any dataset fails, so it can gate a job chain.

    python scripts/cls_band_gate.py
    python scripts/cls_band_gate.py --datasets m-eurosat --samples 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from superres.classification.bands import (  # noqa: E402
    CONDITIONING_NORM,
    TRUE_BAND_ORDER,
    BandOrderError,
    pixeldit_indices,
    verify_band_order,
)
from superres.classification.datasets import (  # noqa: E402
    CLASSIFICATION_DATASETS,
    native_gsd,
)
from superres.constants import LR_PATCH, S2_12BAND_CODES, TILE_METRES  # noqa: E402
from superres.models.pixeldit import band_stats  # noqa: E402

DEFAULT_STATS = Path(__file__).resolve().parents[1] / "configs" / "s2_band_stats.json"


def _raw_batch(dataset: str, n: int) -> tuple[torch.Tensor, list[str]]:
    """``(n, 13, H, W)`` raw values in file order, plus the file's band labels."""
    from torch.utils.data import DataLoader
    from torchgeo_bench.datasets import get_bench_dataset_class

    inner = get_bench_dataset_class(dataset)().get_dataset("train", bands=None)
    loader = DataLoader(inner, batch_size=n, shuffle=False, num_workers=4)
    images = next(iter(loader))["image"].float()
    labels = list(getattr(inner, "band_names", [f"ch{i}" for i in range(images.shape[1])]))
    return images, labels


def _check_dataset(dataset: str, samples: int, stats_path: Path) -> bool:
    print("=" * 78)
    print(f"{dataset}")
    print("=" * 78)
    images, file_labels = _raw_batch(dataset, samples)
    n, channels, height, width = images.shape
    gsd = native_gsd(dataset, width)
    print(f"  {n} samples, {channels} channels, {height}x{width} px at {gsd:g} m/px "
          f"= {width * gsd:.0f} m across")
    print(f"  PixelDiT frame is {LR_PATCH} px at {TILE_METRES / LR_PATCH:g} m/px "
          f"= {TILE_METRES:.0f} m across")
    if width != LR_PATCH:
        from superres.classification.datasets import tile_offsets

        offsets = tile_offsets(width, LR_PATCH)
        views = len(offsets) ** 2
        print(f"    tile4  -> {LR_PATCH * gsd:.0f} m at {gsd:g} m/px x {views} views "
              f"at offsets {offsets}; covers 100% of the footprint  [DEFAULT]")
        print(f"    resize -> {width * gsd:.0f} m at {width * gsd / LR_PATCH:.2f} m/px, "
              f"1 view; covers 100% ({width / LR_PATCH:.2f}x scale error)")
        print(f"    crop   -> {LR_PATCH * gsd:.0f} m at {gsd:g} m/px, 1 view; covers only "
              f"{(LR_PATCH / width) ** 2:.0%} -- can drop the labelled object")

    ok = True

    # 1. band order
    print("\n  -- band order (measured against physics, not metadata) --")
    order = TRUE_BAND_ORDER[dataset]
    try:
        measured = verify_band_order(dataset, images, strict=False)
        problems = measured.pop("_problems", [])
        print(f"     corr(B8A, B08)     = {measured['corr_b8a_b08']:+.4f}   "
              f"(adjacent wavelengths; expect > 0.60)")
        print(f"     mean(B8A)/mean(B08)= {measured['ratio_b8a_b08']:.3f}")
        for code, bound in (("B09", 0.80), ("B10", 0.20)):
            key = f"ratio_{code.lower()}_b08"
            if key in measured:
                print(f"     mean({code})/mean(B08) = {measured[key]:.3f}   "
                      f"(absorbing band; expect < {bound})")
        if problems:
            ok = False
            print("     FAIL:")
            for p in problems:
                print(f"       - {p}")
        else:
            print("     OK -- resolved order is consistent with the pixels")
    except BandOrderError as exc:
        ok = False
        print(f"     FAIL: {exc}")

    # Show the mapping being used, and flag where metadata disagrees with it.
    print("\n     channel -> band (resolved)          file metadata says")
    for i, code in enumerate(order):
        label = file_labels[i] if i < len(file_labels) else "?"
        flag = ""
        # The file's own label is only advisory; note where it contradicts us.
        if code.lstrip("B").lstrip("0") not in label.replace(" ", "").replace("-", ""):
            flag = "   <-- metadata disagrees"
        print(f"       {i:2d} -> {code:4s}  mean={images[:, i].mean():9.2f}"
              f"   {label!r:34s}{flag}")

    idx = pixeldit_indices(dataset)
    print(f"\n     PixelDiT gathers channels {idx}")
    print(f"     giving {list(S2_12BAND_CODES)}")

    # 2. radiometry against the conditioning stretch
    print("\n  -- radiometry vs PixelDiT's training percentiles --")
    lo, scale = band_stats(str(stats_path))
    gathered = images[:, idx]
    unit = ((gathered - lo) / scale).clamp(0, 1)
    print("     band   dataset mean   train range      -> stretched mean   note")
    dead = []
    for j, code in enumerate(S2_12BAND_CODES):
        raw_mean = float(gathered[:, j].mean())
        low = float(lo[0, j, 0, 0])
        high = low + float(scale[0, j, 0, 0])
        stretched = float(unit[:, j].mean())
        note = ""
        if stretched < 0.01:
            note = "CLAMPS TO ZERO"
            dead.append(code)
        elif stretched > 0.99:
            note = "saturates"
        print(f"     {code:4s}  {raw_mean:11.1f}   {low:7.0f}-{high:<7.0f}"
              f"  -> {stretched:13.4f}   {note}")
    chosen = CONDITIONING_NORM[dataset]
    if dead:
        print(f"\n     {len(dead)} band(s) {dead} are outside the training range entirely.")
        if chosen == "stats":
            ok = False
            print(f"     FAIL: conditioning norm is 'stats', which would feed PixelDiT "
                  f"{len(dead)} dead channels. Set CONDITIONING_NORM[{dataset!r}] "
                  "= 'per-image'.")
        else:
            print(f"     Handled: CONDITIONING_NORM[{dataset!r}] = 'per-image', which "
                  "rescales each image by its own percentiles instead.")
    else:
        print(f"\n     All twelve bands land inside the training range. "
              f"CONDITIONING_NORM = {chosen!r}.")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=list(CLASSIFICATION_DATASETS),
                        choices=list(CLASSIFICATION_DATASETS))
    parser.add_argument("--samples", type=int, default=256,
                        help="samples to measure per dataset; 256 is ample for an "
                             "order check and a few hundred is not a tight estimate "
                             "of the mean, which is fine -- a permutation is huge")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    args = parser.parse_args()

    np.set_printoptions(suppress=True)
    results = {d: _check_dataset(d, args.samples, args.stats) for d in args.datasets}

    print("\n" + "=" * 78)
    print("GATE SUMMARY")
    print("=" * 78)
    for dataset, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {dataset}")
    failed = [d for d, ok in results.items() if not ok]
    if failed:
        print(f"\n{len(failed)} dataset(s) failed. Fix before extracting features.")
        return 1
    print("\nAll datasets pass. Safe to run scripts/cls_extract.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
