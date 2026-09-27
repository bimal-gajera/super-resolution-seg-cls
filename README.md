# PixelDiT-SR as a frozen front-end for Sentinel-2

Does attaching a diffusion super-resolution model to low-resolution Sentinel-2
imagery produce better features than plain upsampling? Two tracks ask this on one
frozen PixelDiT checkpoint.

**Segmentation** — smallholder field boundary delineation on
[AI4SmallFarms](https://doi.org/10.17026/dans-xy6-ngg6) (Vietnam subset).
Two arms, shape-identical end to end, so the only variable is the pixels DINOv3
sees:

```
Flow 1:  S2 RGB 48² @10m → PixelDiT → 1584² @0.303m → DINOv3 → 99×99×1024 → DPT → 960² @0.5m
Flow 2:  S2 RGB 48² @10m → bicubic  → 1584² @0.303m → DINOv3 → 99×99×1024 → DPT → 960² @0.5m
```

Flow 2 is the baseline. PixelDiT and DINOv3 are frozen; the DPT decoder is the
only trainable module and is identical in both arms.

**Classification** — the same question, cheaper, on three Sentinel-2 land-cover
benchmarks from [torchgeo-bench](https://github.com/torchgeo/torchgeo-bench)
(`m-eurosat`, `eurosat`, `eurosat-spatial`). Scene labels and a kNN / linear probe
instead of a trained decoder, so a result costs GPU-hours rather than GPU-days.
Six feature arms in two passes:

```
cheap       S2 12-band 48² → bicubic → DINOv3 → 1024              (baseline, ×2 token sizes)
            S2 12-band 48² → PixelDiT block 8 → REPA head → 1024  (no generation)
generative  S2 12-band 48² → PixelDiT → SR 1584² → DINOv3 → 1024  (×2 token sizes)
                                     ↘ read-back → 1040           (hybrid_1m)
```

Motivated by [SEED-SR](https://arxiv.org/abs/2511.14481), which argues that
super-resolving into a segmentation-aware *latent* space beats the conventional
"pixel-space SR, then segment" family. This is an entrant in that family, not a
reimplementation of SEED-SR.

## Setup

```bash
bash env/create_env.sh                # segmentation env
python scripts/download.py            # ~150 MB, DANS data station
python scripts/prepare_patches.py
python scripts/strip_checkpoint.py
```

The classification track needs the `classification` extra (`torchgeo-bench`) in
its own environment, and pulls its datasets automatically on first use (~10.5 GB).

## Running

Both tracks open with a **gate**. Run it; each one guards a failure that is
otherwise silent and costs GPU-days.

```bash
# --- segmentation ---
python scripts/seam_check.py                    # GATE: SR quality, geometry, seams
python scripts/verify_labels.py

python scripts/cache_sr.py                      # ~4.5 h H100, ~28.5 GB, resumable
python scripts/train.py --flow 2 --batch-size 8 --workers 5
python scripts/train.py --flow 1 --batch-size 8 --workers 5
python scripts/evaluate.py --flow 1 --checkpoint runs/flow1/best.pt --instances
```

```bash
# --- classification ---
python scripts/cls_band_gate.py                 # GATE: band order, radiometry

python scripts/cls_extract.py --dataset m-eurosat \
    --arms dinov3_lr_528 dinov3_lr_1584 pixeldit_repa   # ~6 h H100 for all 4 sets
python scripts/cls_probe.py --all
python scripts/cls_probe.py --summarize
```

`cache_sr.py` and `cls_extract.py` are resumable, so a job that hits its wall
clock can simply be resubmitted. Cluster job scripts are not included: they are
site-specific and carry allocation paths.

Useful flags: `--limit N` (cache/extract), `--limit-steps` / `--limit-val-batches`
(train), `--tiles <name>` and `--instances` (evaluate), `--geometry` and
`--tile-pool` (extract — how a 64 px patch reaches PixelDiT's fixed 48 px frame),
`--summarize` (probe).

## Layout

| path | what |
|---|---|
| `src/superres/data/` | download, patch preparation, torch `Dataset` |
| `src/superres/models/` | PixelDiT loading/sampling, frozen DINOv3, DPT decoder, the two flows |
| `src/superres/metrics.py` | mIoU_S, mIoU_I, pixel Acc/P/R/F1 (SEED-SR's definitions) |
| `src/superres/classification/` | band resolution, feature cache, the six arms, probes |
| `scripts/` | entry points; run order is the order shown above |
| `weights/` | gitignored; populated by setup |

Design notes, measurements and the reasoning behind each choice are kept in
`docs/` and are deliberately not tracked in git — this repo ships source, not
write-ups.
