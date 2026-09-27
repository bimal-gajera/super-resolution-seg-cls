"""The feature arms: six ways to turn a 48x48 Sentinel-2 patch into a vector.

Each arm produces one pooled descriptor per image, which is what kNN and a
linear probe consume. They fall into two passes, and the split is entirely about
cost:

CHEAP PASS -- no image is generated.

============================  ====  =========  =========================
arm                           dim   ~s/image   what it is
============================  ====  =========  =========================
``dinov3_lr_528``             1024   0.02      baseline: bicubic LR, 33x33 tokens
``dinov3_lr_1584``            1024   0.20      baseline at 99x99 tokens
``pixeldit_repa``             1024   0.40      PixelDiT block 8 -> REPA head
============================  ====  =========  =========================

GENERATIVE PASS -- one 50-step sampler run per image, ~4 s, then three readouts
off the same generated image.

============================  ====  ==========================================
arm                           dim   what it is
============================  ====  ==========================================
``dinov3_sr_528``             1024   DINOv3 on the SR image, 33x33 tokens
``dinov3_sr_1584``            1024   DINOv3 on the SR image, 99x99 tokens
``hybrid_1m``                 1040   block 8 (1024) + PiT pixel branch (16)
============================  ====  ==========================================

**The three generative arms share one generation.** That is the whole reason they
are grouped: the sampler dominates at ~4 s while a second encoder pass costs
~0.2 s, so emitting all three costs barely more than emitting one. Running them
as separate jobs would triple the only expensive part of the pipeline.

WHY BOTH 528 AND 1584. 528 px gives DINOv3 a 33x33 token grid, matching
PixelDiT's own patch grid and the geometry the REPA head was aligned in; 1584 is
the SR image at native resolution, 99x99 tokens. For a *pooled* descriptor these
are not interchangeable, and which one wins is an empirical question, so each SR
arm is paired with an LR arm at the same token geometry. That pairing is the
controlled comparison: within a pair the only difference is whether DINOv3 sees
a PixelDiT generation or a bicubic stretch of the same 48x48 input.

NORMALIZATION, and one deliberate asymmetry. PixelDiT conditioning uses the
training percentiles (it learned an absolute reflectance mapping); a dataset on a
different radiometric scale needs per-image instead -- see
``bands.CONDITIONING_NORM``. The
DINOv3-on-LR arms use per-image percentiles instead, because DINOv3 wants a
well-exposed picture rather than an absolute one, and applying the training
stretch would confine these datasets to a fraction of the range. The SR arms
feed DINOv3 the generator's own output range. So the two halves of a pair are
normalized differently, on purpose: each side gets the input its encoder
expects, which is the same choice the segmentation project made and measured.

FEATURE SCALE. Nothing here is normalized on the way out. ``pixeldit_repa`` runs
to std ~900 because REPA trained with a cosine loss and never learned magnitude,
while the DINOv3 arms sit near 0.2. Standardizing is the probe's job, and
``probe.py`` reports both raw and standardized so the choice is visible rather
than baked in.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..constants import ALIGN_LAYER, HR_SIZE, RGB_IN_PIXELDIT_BANDS
from ..models.pixeldit import (
    BlockTap,
    band_stats,
    fold_pixel_tokens,
    generate_hr,
    load_denoiser,
    load_proj,
    load_state,
    make_sampler,
    prepare_lr_cond,
    to_unit_range,
)
from ..torch_utils import amp_dtype

__all__ = [
    "aggregate_views",
    "ARMS",
    "CHEAP_ARMS",
    "GENERATIVE_ARMS",
    "ArmSpec",
    "CheapExtractor",
    "GenerativeExtractor",
    "arms_for_pass",
]


@dataclass(frozen=True)
class ArmSpec:
    """Static description of one arm."""

    name: str
    dim: int
    generative: bool
    description: str


ARMS: dict[str, ArmSpec] = {
    a.name: a
    for a in (
        ArmSpec("dinov3_lr_528", 1024, False, "DINOv3-SAT on bicubic LR, 33x33 tokens"),
        ArmSpec("dinov3_lr_1584", 1024, False, "DINOv3-SAT on bicubic LR, 99x99 tokens"),
        ArmSpec("pixeldit_repa", 1024, False, "PixelDiT DiT block 8 -> REPA head, LR only"),
        ArmSpec("dinov3_sr_528", 1024, True, "DINOv3-SAT on the SR image, 33x33 tokens"),
        ArmSpec("dinov3_sr_1584", 1024, True, "DINOv3-SAT on the SR image, 99x99 tokens"),
        ArmSpec("hybrid_1m", 1040, True, "block 8 (1024) + PiT pixel branch (16)"),
    )
}
CHEAP_ARMS = tuple(n for n, a in ARMS.items() if not a.generative)
GENERATIVE_ARMS = tuple(n for n, a in ARMS.items() if a.generative)


def arms_for_pass(names: list[str]) -> tuple[list[str], list[str]]:
    """Split requested arms into (cheap, generative). Unknown names raise."""
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; choose from {sorted(ARMS)}")
    return (
        [n for n in names if not ARMS[n].generative],
        [n for n in names if ARMS[n].generative],
    )


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _lr_to_rgb01(images: torch.Tensor) -> torch.Tensor:
    """(B, 12, H, W) raw bands -> (B, 3, H, W) in [0, 1], per-image stretched.

    Per-image 1st/99th percentiles, not the training statistics: this feeds
    DINOv3, which wants a well-exposed picture. See the module docstring.
    """
    rgb = images[:, list(RGB_IN_PIXELDIT_BANDS)].float()
    return to_unit_range(rgb, mode="per-image")


def _pool_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    """(B, C, h, w) -> (B, C) by spatial mean."""
    return feature_map.flatten(2).mean(dim=2)


class _Dinov3Runner:
    """Frozen DINOv3-SAT, final block only, pooled to one vector per image."""

    def __init__(self, device: str) -> None:
        from ..models.dinov3 import FrozenDINOv3

        # Only the last block is needed: a pooled descriptor has no use for the
        # intermediate taps the DPT decoder reads.
        self.encoder = FrozenDINOv3(taps=(23,)).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, images01: torch.Tensor, size: int, micro_batch: int = 0) -> torch.Tensor:
        x = images01
        if x.shape[-1] != size:
            x = F.interpolate(x, (size, size), mode="bicubic", align_corners=False)
        x = x.clamp(0, 1)
        # 1584 px is 99x99 = 9801 tokens through a ViT-L; splitting keeps peak
        # activation memory bounded independently of the dataloader batch size.
        step = micro_batch or len(x)
        out = [
            _pool_tokens(self.encoder(x[i : i + step].to(self.device))[0])
            for i in range(0, len(x), step)
        ]
        return torch.cat(out, dim=0).float()


def aggregate_views(
    extractor,
    views: torch.Tensor,
    norm_mode: str,
    *,
    pool: str = "mean",
) -> dict[str, torch.Tensor]:
    """Run an extractor over every view of a batch and pool the results.

    ``views`` is ``(V, B, C, 48, 48)`` from
    :func:`~superres.classification.datasets.footprint_views`. With V=1 this is a
    pass-through, so the single-view geometries cost nothing extra; with V=4
    (``tile4``) each view is extracted separately and the per-view descriptors are
    combined.

    Views are extracted one at a time rather than folded into the batch
    dimension. Folding would be faster, but it multiplies peak activation memory
    by V, and the generative pass is already close to the limit of an 80 GB card
    with two 2 B-parameter denoisers resident.

    Args:
        extractor: A :class:`CheapExtractor` or :class:`GenerativeExtractor`.
        views: ``(V, B, C, H, W)``.
        norm_mode: Conditioning normalization for this dataset.
        pool: ``"mean"`` averages evidence across views -- right when the label
            describes the whole patch, as in land cover. ``"max"`` takes the
            per-dimension maximum -- right for presence/absence, where one view
            containing the object is enough.

    Returns:
        ``{arm: (B, dim)}`` on CPU. Pooling never changes ``dim``, so the cache
        layout is independent of the geometry.
    """
    if views.ndim != 5:
        raise ValueError(f"expected (V, B, C, H, W) views, got shape {tuple(views.shape)}")
    if pool not in ("mean", "max"):
        raise ValueError(f"pool must be 'mean' or 'max', got {pool!r}")
    if views.shape[0] == 1:
        return extractor(views[0], norm_mode)

    per_view: list[dict[str, torch.Tensor]] = [
        extractor(view, norm_mode) for view in views
    ]
    out: dict[str, torch.Tensor] = {}
    for arm in per_view[0]:
        stacked = torch.stack([f[arm] for f in per_view], dim=0)   # (V, B, dim)
        out[arm] = stacked.mean(0) if pool == "mean" else stacked.amax(0)
    return out


# --------------------------------------------------------------------------- #
# cheap pass
# --------------------------------------------------------------------------- #
class CheapExtractor:
    """The three arms that need no image generation.

    Loads only what the requested arms actually use, so a DINOv3-only run never
    pays to put a 2 B-parameter denoiser on the GPU.
    """

    def __init__(
        self,
        arms: list[str],
        *,
        ckpt: str,
        stats_path: str,
        device: str = "cuda",
        repa_seeds: int = 4,
        dinov3_micro_batch: int = 2,
    ) -> None:
        bad = [a for a in arms if ARMS[a].generative]
        if bad:
            raise ValueError(f"{bad} are generative arms; use GenerativeExtractor")
        self.arms = list(arms)
        self.device = device
        self.repa_seeds = repa_seeds
        self.dinov3_micro_batch = dinov3_micro_batch
        self.amp = amp_dtype(device)

        self.dinov3 = (
            _Dinov3Runner(device)
            if any(a.startswith("dinov3_lr") for a in arms)
            else None
        )
        self.net_raw = None
        self.proj = None
        if "pixeldit_repa" in arms:
            state = load_state(ckpt)
            # RAW denoiser, not the EMA copy: the REPA head was trained against
            # the raw weights, so EMA activations fall outside its calibration.
            self.net_raw = load_denoiser(state, "denoiser.", device)
            self.proj = load_proj(state, device)
            del state
        self.lo, self.scale = band_stats(stats_path)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, norm_mode: str) -> dict[str, torch.Tensor]:
        """``(B, 12, 48, 48)`` raw bands -> {arm: (B, dim)} on CPU."""
        images = images.to(self.device, non_blocking=True).float()
        out: dict[str, torch.Tensor] = {}

        if self.dinov3 is not None:
            rgb01 = _lr_to_rgb01(images)
            for arm in ("dinov3_lr_528", "dinov3_lr_1584"):
                if arm in self.arms:
                    size = int(arm.rsplit("_", 1)[1])
                    out[arm] = self.dinov3(
                        rgb01, size, micro_batch=self.dinov3_micro_batch
                    ).cpu()

        if self.net_raw is not None:
            cond = prepare_lr_cond(images, self.lo, self.scale, mode=norm_mode)
            out["pixeldit_repa"] = self._repa(cond).cpu()
        return out

    @torch.no_grad()
    def _repa(self, cond: torch.Tensor) -> torch.Tensor:
        """Block-8 REPA embedding from pure noise at t=0.

        At t=0 the input *is* the noise, so the only thing the model has to look
        at is the conditioning -- which is what makes this a feature of the input
        rather than of a sample. The noise is random, so several draws are
        averaged; one draw is visibly jumpy.
        """
        batch = cond.shape[0]
        acc = None
        with BlockTap(self.net_raw.patch_blocks[ALIGN_LAYER - 1]) as tap:
            for seed in range(self.repa_seeds):
                generator = torch.Generator(device="cpu").manual_seed(1234 + seed)
                noise = torch.randn(
                    (batch, 3, HR_SIZE, HR_SIZE), generator=generator
                ).to(self.device)
                t = torch.zeros((batch,), device=self.device)
                with torch.autocast(device_type=self.device.split(":")[0], dtype=self.amp):
                    self.net_raw(noise, t, cond)
                    embedding = self.proj(tap.value).float()      # (B, 1089, 1024)
                acc = embedding if acc is None else acc + embedding
        return (acc / self.repa_seeds).mean(dim=1)                # pool 1089 tokens


# --------------------------------------------------------------------------- #
# generative pass
# --------------------------------------------------------------------------- #
class GenerativeExtractor:
    """One 50-step generation per image, read out three ways.

    Holds two weight sets when ``hybrid_1m`` is requested: the EMA copy
    generates (measurably better images -- 0.890 vs 0.869 test accuracy
    upstream) while the raw copy does the read-back, because that is what the
    REPA head was calibrated against. Two 2 B-parameter models in fp32 is
    ~16 GB of weights, so this pass wants an 80 GB card; ``weights_dtype`` can
    halve it at some cost in fidelity.
    """

    def __init__(
        self,
        arms: list[str],
        *,
        ckpt: str,
        stats_path: str,
        device: str = "cuda",
        num_steps: int = 50,
        guidance: float = 2.5,
        timeshift: float = 1.0,
        guidance_interval: str = "0.3:1",
        gen_seed: int = 0,
        readback_t: float = 0.9,
        hybrid_grid: int = 480,
        dinov3_micro_batch: int = 2,
        weights_dtype: torch.dtype | None = None,
    ) -> None:
        bad = [a for a in arms if not ARMS[a].generative]
        if bad:
            raise ValueError(f"{bad} are cheap arms; use CheapExtractor")
        self.arms = list(arms)
        self.device = device
        self.gen_seed = gen_seed
        self.readback_t = readback_t
        self.hybrid_grid = hybrid_grid
        self.dinov3_micro_batch = dinov3_micro_batch
        self.amp = amp_dtype(device)

        state = load_state(ckpt)
        self.net_gen = load_denoiser(state, "ema_denoiser.", device)
        needs_readback = "hybrid_1m" in arms
        self.net_raw = load_denoiser(state, "denoiser.", device) if needs_readback else None
        self.proj = load_proj(state, device) if needs_readback else None
        del state
        if weights_dtype is not None:
            self.net_gen = self.net_gen.to(weights_dtype)
            if self.net_raw is not None:
                self.net_raw = self.net_raw.to(weights_dtype)

        self.sampler = make_sampler(
            device, num_steps=num_steps, guidance=guidance,
            timeshift=timeshift, interval=guidance_interval,
        )
        self.dinov3 = (
            _Dinov3Runner(device)
            if any(a.startswith("dinov3_sr") for a in arms)
            else None
        )
        self.lo, self.scale = band_stats(stats_path)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, norm_mode: str) -> dict[str, torch.Tensor]:
        """``(B, 12, 48, 48)`` raw bands -> {arm: (B, dim)} on CPU."""
        images = images.to(self.device, non_blocking=True).float()
        cond = prepare_lr_cond(images, self.lo, self.scale, mode=norm_mode)
        generated = generate_hr(self.net_gen, self.sampler, cond, self.device, self.gen_seed)

        out: dict[str, torch.Tensor] = {}
        if self.dinov3 is not None:
            sr01 = (generated + 1) / 2                     # [-1,1] -> [0,1]
            for arm in ("dinov3_sr_528", "dinov3_sr_1584"):
                if arm in self.arms:
                    size = int(arm.rsplit("_", 1)[1])
                    out[arm] = self.dinov3(
                        sr01, size, micro_batch=self.dinov3_micro_batch
                    ).cpu()
        if "hybrid_1m" in self.arms:
            out["hybrid_1m"] = self._hybrid(generated, cond).cpu()
        del generated, cond
        return out

    @torch.no_grad()
    def _hybrid(self, generated: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Coarse REPA (1024) + fine pixel branch (16), both pooled -> 1040.

        One forward pass over the generated image taps two places at once, so
        the two halves are mutually consistent.

        The fine half is pooled over a 480x480 grid, and on its own that would
        be useless -- 16 dims averaged that wide gives an inter-tile correlation
        of 0.998 upstream. It is kept only because a classifier can weight the
        16 against the 1024 itself, which PCA cannot.
        """
        from ..core.diffusion import LinearScheduler

        scheduler = LinearScheduler()
        batch = generated.shape[0]
        t = torch.full((batch,), float(self.readback_t), device=self.device)
        noise = torch.randn_like(generated)
        # t=0.9 is 90% image, 10% noise. Not 1.0: the model never saw a
        # perfectly clean image in training, so 1.0 sits off the edge of its range.
        x_t = scheduler.alpha(t) * generated + scheduler.sigma(t) * noise

        with BlockTap(self.net_raw.patch_blocks[ALIGN_LAYER - 1]) as tap_patch:
            with BlockTap(self.net_raw.pixel_blocks[-1]) as tap_pixel:
                with torch.autocast(device_type=self.device.split(":")[0], dtype=self.amp):
                    self.net_raw(x_t, t, cond)
                    coarse = self.proj(tap_patch.value).float()        # (B, 1089, 1024)
                    fine = fold_pixel_tokens(tap_pixel.value.float(), batch)  # (B,16,1584,1584)

        grid = self.hybrid_grid
        fine_pooled = F.adaptive_avg_pool2d(fine, (grid, grid)).flatten(2).mean(dim=2)
        coarse_pooled = coarse.mean(dim=1)                            # (B, 1024)
        del fine, coarse
        return torch.cat([coarse_pooled, fine_pooled], dim=1)         # (B, 1040)
