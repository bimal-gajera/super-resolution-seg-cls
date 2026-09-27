"""Which channel of a classification dataset is which Sentinel-2 band.

This module exists because the declared metadata is wrong for one of the four
datasets, and getting it wrong is silent: PixelDiT is handed twelve channels of
the right shape holding the wrong bands, produces a plausible image, and every
number downstream is quietly meaningless.

MEASURED, not assumed. All four sets ship 13 channels at 64x64. Their true
channel order was resolved from the pixels themselves (see
``scripts/cls_band_gate.py``) using two relationships that hold for any scene:

* **B8A (865 nm) sits next to B08 (842 nm)**, so the two must correlate above
  ~0.95 and have similar magnitude.
* **B09 (945 nm, water vapour) and B10 (1375 nm, cirrus) are absorption
  bands**, so over clear land they are far darker than B08.

What that turned up:

``m-eurosat`` **is mislabelled.** Its ``bands_order`` metadata claims channel 8
is ``'08A - Vegetation Red Edge'``, but channel 8 correlates only 0.727 with
B08 at 0.34x its magnitude, while channel 12 -- labelled ``'12 - SWIR'`` --
correlates 0.979 at 1.13x. Channel 12 is B8A. The file is stored in torchgeo's
order (B8A last) and labelled in GeoBench's order (B8A ninth). Selecting bands
by canonical name therefore returns the wrong pixels for five of thirteen
channels. torchgeo-bench's own ``BandSpec`` means are computed from the real
pixels, so they disagree with the names they are attached to -- which is the
first thing that gave it away.

``eurosat`` / ``eurosat-spatial`` are labelled correctly, B8A last, and
``m-eurosat`` turns out to store its bands in exactly their order despite saying
otherwise -- an m-eurosat sample matches an EuroSAT one bit-exactly under the
identity permutation.

So the mapping below is frozen data, not inference. ``verify_band_order``
re-checks it against real pixels before any GPU hours are spent, because a
dataset re-release that changed channel order would otherwise be undetectable.
"""

from __future__ import annotations

import numpy as np
import torch

from ..constants import S2_12BAND_CODES

__all__ = [
    "TRUE_BAND_ORDER",
    "CONDITIONING_NORM",
    "pixeldit_indices",
    "verify_band_order",
    "verify_pixeldit_stack",
    "BandOrderError",
]

# The actual channel order of each dataset's image tensor, as Sentinel-2 codes.
# Index i of the tuple is channel i of the (13, 64, 64) array the loader
# returns when asked for every band.
TRUE_BAND_ORDER: dict[str, tuple[str, ...]] = {
    # Stored in torchgeo order -- B8A last -- despite metadata claiming ninth.
    "m-eurosat": (
        "B01", "B02", "B03", "B04", "B05", "B06", "B07",
        "B08", "B09", "B10", "B11", "B12", "B8A",
    ),
    "eurosat": (
        "B01", "B02", "B03", "B04", "B05", "B06", "B07",
        "B08", "B09", "B10", "B11", "B12", "B8A",
    ),
    "eurosat-spatial": (
        "B01", "B02", "B03", "B04", "B05", "B06", "B07",
        "B08", "B09", "B10", "B11", "B12", "B8A",
    ),
}

# How to map raw band values into PixelDiT's conditioning range, per dataset.
#
# "stats" applies the training percentiles from s2_band_stats.json. That is the
# right default -- PixelDiT learned an absolute reflectance mapping, and the
# segmentation project measured per-image rescaling to be visibly worse.
#
# All three current datasets are EuroSAT-derived and sit inside the training
# range, so all three use "stats". A dataset on a different radiometric scale
# needs "per-image" instead: 8-bit SWIR channels, for instance, fall entirely
# below the training percentiles and clamp to exactly zero, which silently
# deadens a quarter of the conditioning. scripts/cls_band_gate.py prints the
# stretched per-band mean for exactly this reason and fails on a dead channel.
CONDITIONING_NORM: dict[str, str] = {
    "m-eurosat": "stats",
    "eurosat": "stats",
    "eurosat-spatial": "stats",
}


class BandOrderError(RuntimeError):
    """Raised when measured pixels contradict :data:`TRUE_BAND_ORDER`."""


def pixeldit_indices(dataset: str) -> list[int]:
    """Channel indices that gather PixelDiT's twelve bands, in its own order.

    Args:
        dataset: torchgeo-bench dataset name, e.g. ``"m-eurosat"``.

    Returns:
        Twelve indices into the dataset's 13-channel image tensor, ordered as
        :data:`~superres.constants.S2_12BAND_CODES`. B10 is dropped, since
        PixelDiT was trained on L2A and never saw a cirrus band.
    """
    if dataset not in TRUE_BAND_ORDER:
        raise KeyError(
            f"no resolved band order for {dataset!r}; add it to TRUE_BAND_ORDER "
            "after confirming the order with scripts/cls_band_gate.py"
        )
    order = TRUE_BAND_ORDER[dataset]
    position = {code: i for i, code in enumerate(order)}
    missing = [c for c in S2_12BAND_CODES if c not in position]
    if missing:
        raise KeyError(f"{dataset} is missing band(s) {missing}, which PixelDiT requires")
    return [position[code] for code in S2_12BAND_CODES]


def verify_band_order(
    dataset: str,
    images: torch.Tensor,
    *,
    min_b8a_corr: float = 0.60,
    max_b09_ratio: float = 0.80,
    max_b10_ratio: float = 0.20,
    strict: bool = True,
) -> dict[str, float]:
    """Check a batch of raw images against the frozen band order.

    Two physical invariants, both independent of geography and radiometry:

    1. Of the thirteen channels, the one we call B8A must be the *best*
       non-adjacent match to B08 among the candidates B8A/B09/B10 -- adjacent
       wavelengths, so they track each other.
    2. B09 and B10 are absorption bands and must be substantially darker than
       B08.

    Thresholds are deliberately loose. This is not a quality metric; it is a
    tripwire for a channel *permutation*, which moves these numbers by a lot
    (on m-eurosat the mislabelled channel scores 0.727/0.34 against the correct
    channel's 0.979/1.13).

    Args:
        dataset: torchgeo-bench dataset name.
        images: ``(B, 13, H, W)`` raw band values, every band, in file order.
        min_b8a_corr: Floor on corr(B8A, B08).
        max_b09_ratio: Ceiling on mean(B09)/mean(B08). Separate from B10's, and
            much looser, because the two bands are not alike: B10 (cirrus,
            1375 nm) is near-zero over clear land, while B09 (water vapour,
            945 nm) scales with atmospheric water and is genuinely bright in
            some scenes -- a ratio of 0.54 has been measured on real data. One
            shared threshold would either reject such a scene or stop
            constraining B10.
        max_b10_ratio: Ceiling on mean(B10)/mean(B08). Tight, as it can be.
        strict: Raise :class:`BandOrderError` on failure rather than returning.

    Returns:
        The measured quantities, for reporting.
    """
    order = TRUE_BAND_ORDER[dataset]
    if images.shape[1] != len(order):
        raise BandOrderError(
            f"{dataset}: got {images.shape[1]} channels, but TRUE_BAND_ORDER "
            f"describes {len(order)}. The dataset layout has changed."
        )
    position = {code: i for i, code in enumerate(order)}
    flat = images.float().permute(1, 0, 2, 3).reshape(images.shape[1], -1).numpy()
    corr = np.corrcoef(flat)
    means = flat.mean(axis=1)

    b08, b8a = position["B08"], position["B8A"]
    measured = {
        "corr_b8a_b08": float(corr[b08, b8a]),
        "ratio_b8a_b08": float(means[b8a] / max(means[b08], 1e-9)),
    }
    problems: list[str] = []
    if measured["corr_b8a_b08"] < min_b8a_corr:
        problems.append(
            f"corr(B8A, B08) = {measured['corr_b8a_b08']:.3f} < {min_b8a_corr}; "
            f"channel {b8a} does not behave like B8A"
        )
    for code, bound in (("B09", max_b09_ratio), ("B10", max_b10_ratio)):
        if code not in position:
            continue
        ratio = float(means[position[code]] / max(means[b08], 1e-9))
        measured[f"ratio_{code.lower()}_b08"] = ratio
        if ratio > bound:
            problems.append(
                f"mean({code})/mean(B08) = {ratio:.3f} > {bound}; "
                f"channel {position[code]} is too bright to be {code}"
            )
    # A permutation that swapped B8A with one of the absorption bands would trip
    # the checks above; one that swapped it with a *brighter* band would not, so
    # also confirm nothing else beats B8A as B08's spectral neighbour.
    rivals = [position[c] for c in ("B09", "B10", "B11", "B12") if c in position]
    better = [r for r in rivals if corr[b08, r] > measured["corr_b8a_b08"]]
    if better:
        problems.append(
            f"channel(s) {better} correlate with B08 more strongly than the channel "
            f"claimed to be B8A ({b8a}); the order is probably permuted"
        )

    measured["ok"] = float(not problems)
    if problems and strict:
        raise BandOrderError(
            f"{dataset}: measured pixels contradict TRUE_BAND_ORDER:\n  - "
            + "\n  - ".join(problems)
        )
    measured["_problems"] = problems  # type: ignore[assignment]
    return measured


def verify_pixeldit_stack(
    images: torch.Tensor,
    *,
    min_b8a_corr: float = 0.60,
    max_absorption_ratio: float = 0.95,
    strict: bool = True,
) -> dict[str, float]:
    """Check the twelve-band stack that is actually handed to PixelDiT.

    Stronger than :func:`verify_band_order`, which inspects the raw file: this
    runs *after* the gather, so it validates the gather itself. The stack is in
    :data:`~superres.constants.S2_12BAND_CODES` order, which puts B08 at index 7,
    B8A at 8 and B09 at 9 -- and B10 is gone, so only one absorption band is left
    to test against.

    THE CORRELATION IS THE REAL DETECTOR. This runs on whatever batch the
    extractor happens to be holding, which may be only a handful of images, so
    the two checks are not equally trustworthy at that size:

    * ``corr(B8A, B08)`` is computed over every pixel in the batch -- 18k pixels
      even at batch 8 -- so it is stable regardless of how many scenes are in it.
    * A **ratio of means** over a handful of scenes is not. On real data a
      B09/B08 of 0.54 over 256 samples came out as 0.69 on one batch of 8, which
      tripped an earlier 0.60 bound and killed three jobs on sound data.

    So the ratio bounds here are swap detectors, not quality checks, and are set
    loose enough to survive small-batch noise. A genuine B08/B09 swap inverts the
    ratio to ~1.8, and any swap with a reflective band pushes it higher still, so
    a loose bound still catches what this is for. The tight, physically meaningful
    version of the check belongs in ``scripts/cls_band_gate.py``, which runs on a
    large sample.

    Args:
        images: ``(B, 12, H, W)`` raw band values in PixelDiT's band order.
        min_b8a_corr: Floor on corr(B8A, B08).
        max_absorption_ratio: Ceiling on mean(B09)/mean(B08). Deliberately loose;
            see above.
        strict: Raise :class:`BandOrderError` on failure.

    Returns:
        The measured quantities.
    """
    if images.shape[1] != len(S2_12BAND_CODES):
        raise BandOrderError(
            f"expected {len(S2_12BAND_CODES)} channels in PixelDiT order, got "
            f"{images.shape[1]}; the gather is wrong"
        )
    position = {code: i for i, code in enumerate(S2_12BAND_CODES)}
    flat = images.float().permute(1, 0, 2, 3).reshape(images.shape[1], -1).numpy()
    corr = np.corrcoef(flat)
    means = flat.mean(axis=1)
    b08, b8a, b09 = position["B08"], position["B8A"], position["B09"]

    measured = {
        "corr_b8a_b08": float(corr[b08, b8a]),
        "ratio_b8a_b08": float(means[b8a] / max(means[b08], 1e-9)),
        "ratio_b09_b08": float(means[b09] / max(means[b08], 1e-9)),
    }
    problems = []
    if measured["corr_b8a_b08"] < min_b8a_corr:
        problems.append(
            f"corr(B8A, B08) = {measured['corr_b8a_b08']:.3f} < {min_b8a_corr}"
        )
    if measured["ratio_b09_b08"] > max_absorption_ratio:
        problems.append(
            f"mean(B09)/mean(B08) = {measured['ratio_b09_b08']:.3f} > "
            f"{max_absorption_ratio} (B09 should be the darker, absorbing band)"
        )
    if measured["ratio_b8a_b08"] < 0.4 or measured["ratio_b8a_b08"] > 2.0:
        problems.append(
            f"mean(B8A)/mean(B08) = {measured['ratio_b8a_b08']:.3f} is outside [0.4, 2.0]; "
            "these are adjacent wavelengths and should be close in magnitude"
        )
    measured["ok"] = float(not problems)
    if problems and strict:
        raise BandOrderError(
            "the 12-band stack handed to PixelDiT does not look like Sentinel-2 in "
            f"{list(S2_12BAND_CODES)} order:\n  - " + "\n  - ".join(problems)
        )
    measured["_problems"] = problems  # type: ignore[assignment]
    return measured
