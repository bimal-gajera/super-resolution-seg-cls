# Modified from https://github.com/MCG-NJU/PixNerd

import copy
import logging
import os
import sys
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .utils import no_grad

logger = logging.getLogger(__name__)


class BaseScheduler:
    def alpha(self, t) -> Tensor:
        raise NotImplementedError

    def sigma(self, t) -> Tensor:
        raise NotImplementedError

    def dalpha(self, t) -> Tensor:
        raise NotImplementedError

    def dsigma(self, t) -> Tensor:
        raise NotImplementedError

    def dalpha_over_alpha(self, t) -> Tensor:
        return self.dalpha(t) / self.alpha(t)

    def dsigma_mul_sigma(self, t) -> Tensor:
        return self.dsigma(t) * self.sigma(t)

    def drift_coefficient(self, t):
        alpha, sigma = self.alpha(t), self.sigma(t)
        dalpha, dsigma = self.dalpha(t), self.dsigma(t)
        return dalpha / (alpha + 1e-6)

    def diffuse_coefficient(self, t):
        alpha, sigma = self.alpha(t), self.sigma(t)
        dalpha, dsigma = self.dalpha(t), self.dsigma(t)
        return dsigma * sigma - dalpha / (alpha + 1e-6) * sigma ** 2

    def w(self, t):
        return self.sigma(t)


class LinearScheduler(BaseScheduler):
    def alpha(self, t) -> Tensor:
        return t.view(-1, 1, 1, 1)

    def sigma(self, t) -> Tensor:
        return (1 - t).view(-1, 1, 1, 1)

    def dalpha(self, t) -> Tensor:
        return torch.full_like(t, 1.0).view(-1, 1, 1, 1)

    def dsigma(self, t) -> Tensor:
        return torch.full_like(t, -1.0).view(-1, 1, 1, 1)


class BaseTrainer(nn.Module):
    def __init__(self, null_condition_p=0.1):
        super().__init__()
        self.null_condition_p = null_condition_p

    def preproprocess(self, x, condition, uncondition, metadata):
        bsz = x.shape[0]
        if self.null_condition_p > 0:
            mask = torch.rand((bsz), device=condition.device) < self.null_condition_p
            mask = mask.view(-1, *([1] * (len(condition.shape) - 1))).to(condition.dtype)
            condition = condition * (1 - mask) + uncondition * mask
        return x, condition, metadata

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        raise NotImplementedError

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def __call__(self, net, ema_net, solver, x, condition, uncondition, metadata=None):
        x, condition, metadata = self.preproprocess(x, condition, uncondition, metadata)
        return self._impl_trainstep(net, ema_net, solver, x, condition, metadata)


class BaseSampler(nn.Module):
    def __init__(
        self,
        scheduler: BaseScheduler = None,
        guidance_fn: Callable = None,
        num_steps: int = 250,
        guidance: Union[float, List[float]] = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.guidance = guidance
        self.guidance_fn = guidance_fn
        self.scheduler = scheduler

    def _impl_sampling(self, net, noise, condition, uncondition):
        raise NotImplementedError

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(self, net, noise, condition, uncondition, return_x_trajs=False, return_v_trajs=False):
        x_trajs, v_trajs = self._impl_sampling(net, noise, condition, uncondition)
        if return_x_trajs and return_v_trajs:
            return x_trajs[-1], x_trajs, v_trajs
        elif return_x_trajs:
            return x_trajs[-1], x_trajs
        elif return_v_trajs:
            return x_trajs[-1], v_trajs
        else:
            return x_trajs[-1]


def simple_guidance_fn(out, cfg):
    uncondition, condition = out.chunk(2, dim=0)
    out = uncondition + cfg * (condition - uncondition)
    return out


CFG_RESCALE_PHI = 0.7


def cfg_rescale_guidance_fn(out, cfg, phi: float = CFG_RESCALE_PHI):
    """Classifier-free guidance with the variance blow-up rescaled away.

    Plain CFG extrapolates: `uncond + cfg * (cond - uncond)`. For cfg > 1 that
    pushes the prediction outside the model's own output distribution and
    inflates its variance, which shows up as over-contrast / over-saturation.
    Measured on the S2/Maxar model at step 200k: raising cfg 1.0 -> 1.5 lifted
    the finest spatial band from 0.44 -> 0.69 of ground truth (good) but also
    blew the coarse band from 1.28 -> 2.49 and contrast to 25% above ground
    truth (bad). Sharpness and contrast were locked together.

    This is the fix from Lin et al., "Common Diffusion Noise Schedules and
    Sample Steps are Flawed": renormalise the guided prediction back to the
    *conditional* prediction's standard deviation, then blend that correction
    in with weight `phi`. It decouples the two -- you can raise cfg for detail
    without dragging contrast up with it.

    phi=0.0 reproduces `simple_guidance_fn` exactly; phi=1.0 is full rescaling.
    0.7 is the value the paper recommends.
    """
    uncondition, condition = out.chunk(2, dim=0)
    guided = uncondition + cfg * (condition - uncondition)
    if phi <= 0.0:
        return guided
    # per-sample statistics, over channels and space
    dims = tuple(range(1, guided.ndim))
    std_cond = condition.float().std(dim=dims, keepdim=True)
    std_guided = guided.float().std(dim=dims, keepdim=True)
    rescaled = guided * (std_cond / std_guided.clamp_min(1e-6)).to(guided.dtype)
    return phi * rescaled + (1.0 - phi) * guided


def make_cfg_rescale_guidance_fn(phi: float = CFG_RESCALE_PHI):
    """Return a `guidance_fn(out, cfg)` closure with `phi` baked in.

    The samplers call `guidance_fn(out, cfg)` with exactly two arguments, so
    this is how to use a non-default phi (e.g. from a sweep or a config).
    """
    def _fn(out, cfg):
        return cfg_rescale_guidance_fn(out, cfg, phi=phi)

    _fn.__name__ = f"cfg_rescale_guidance_fn(phi={phi})"
    return _fn


def shift_respace_fn(t, shift=3.0):
    return t / (t + (1 - t) * shift)


def ode_step_fn(x, v, dt, s, w):
    return x + v * dt


def sde_mean_step_fn(x, v, dt, s, w):
    return x + v * dt + s * w * dt


def sde_step_fn(x, v, dt, s, w):
    return x + v * dt + s * w * dt + torch.sqrt(2 * w * dt) * torch.randn_like(x)


def sde_preserve_step_fn(x, v, dt, s, w):
    return x + v * dt + 0.5 * s * w * dt + torch.sqrt(w * dt) * torch.randn_like(x)


class EulerSampler(BaseSampler):
    def __init__(
        self,
        w_scheduler: BaseScheduler = None,
        timeshift=1.0,
        guidance_interval_min: float = 0.0,
        guidance_interval_max: float = 1.0,
        step_fn: Callable = ode_step_fn,
        last_step=None,
        last_step_fn: Callable = ode_step_fn,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.step_fn = step_fn
        self.last_step = last_step
        self.last_step_fn = last_step_fn
        self.w_scheduler = w_scheduler
        self.timeshift = timeshift
        self.guidance_interval_min = guidance_interval_min
        self.guidance_interval_max = guidance_interval_max

        if self.last_step is None or self.num_steps == 1:
            self.last_step = 1.0 / self.num_steps

        timesteps = torch.linspace(0.0, 1 - self.last_step, self.num_steps)
        timesteps = torch.cat([timesteps, torch.tensor([1.0])], dim=0)
        self.timesteps = shift_respace_fn(timesteps, self.timeshift)

        assert self.last_step > 0.0
        assert self.scheduler is not None
        assert self.w_scheduler is not None or self.step_fn in [ode_step_fn]
        if self.w_scheduler is not None and self.step_fn == ode_step_fn:
            logger.warning("current sampler is ODE sampler, but w_scheduler is enabled")

    def _impl_sampling(self, net, noise, condition, uncondition):
        batch_size = noise.shape[0]
        steps = self.timesteps.to(noise.device, noise.dtype)
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        x = noise
        x_trajs = [noise]
        v_trajs = []
        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            dt = t_next - t_cur
            t_cur = t_cur.repeat(batch_size)
            sigma = self.scheduler.sigma(t_cur)
            dalpha_over_alpha = self.scheduler.dalpha_over_alpha(t_cur)
            dsigma_mul_sigma = self.scheduler.dsigma_mul_sigma(t_cur)
            if self.w_scheduler:
                w = self.w_scheduler.w(t_cur)
            else:
                w = 0.0

            cfg_x = torch.cat([x, x], dim=0)
            cfg_t = t_cur.repeat(2)
            out = net(cfg_x, cfg_t, cfg_condition)
            if self.guidance_interval_min < t_cur[0] < self.guidance_interval_max:
                guidance = self.guidance
                out = self.guidance_fn(out, guidance)
            else:
                out = self.guidance_fn(out, 1.0)
            v = out
            s = ((1 / dalpha_over_alpha) * v - x) / (sigma ** 2 - (1 / dalpha_over_alpha) * dsigma_mul_sigma)
            if i < self.num_steps - 1:
                x = self.step_fn(x, v, dt, s=s, w=w)
            else:
                x = self.last_step_fn(x, v, dt, s=s, w=w)
            x_trajs.append(x)
            v_trajs.append(v)
        v_trajs.append(torch.zeros_like(x))
        return x_trajs, v_trajs


class HeunSampler(BaseSampler):
    def __init__(
        self,
        scheduler: BaseScheduler = None,
        w_scheduler: BaseScheduler = None,
        exact_henu=False,
        timeshift=1.0,
        guidance_interval_min: float = 0.0,
        guidance_interval_max: float = 1.0,
        step_fn: Callable = ode_step_fn,
        last_step=None,
        last_step_fn: Callable = ode_step_fn,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.exact_henu = exact_henu
        self.step_fn = step_fn
        self.last_step = last_step
        self.last_step_fn = last_step_fn
        self.w_scheduler = w_scheduler
        self.timeshift = timeshift
        self.guidance_interval_min = guidance_interval_min
        self.guidance_interval_max = guidance_interval_max
        if self.last_step is None or self.num_steps == 1:
            self.last_step = 1.0 / self.num_steps

        timesteps = torch.linspace(0.0, 1 - self.last_step, self.num_steps)
        timesteps = torch.cat([timesteps, torch.tensor([1.0])], dim=0)
        self.timesteps = shift_respace_fn(timesteps, self.timeshift)

        assert self.last_step > 0.0
        assert self.scheduler is not None
        assert self.w_scheduler is not None or self.step_fn in [ode_step_fn]
        if self.w_scheduler is not None and self.step_fn == ode_step_fn:
            logger.warning("current sampler is ODE sampler, but w_scheduler is enabled")

    def _impl_sampling(self, net, noise, condition, uncondition):
        batch_size = noise.shape[0]
        steps = self.timesteps.to(noise.device)
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        x = noise
        v_hat, s_hat = 0.0, 0.0
        x_trajs = [noise]
        v_trajs = []
        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            dt = t_next - t_cur
            t_cur = t_cur.repeat(batch_size)
            sigma = self.scheduler.sigma(t_cur)
            alpha_over_dalpha = 1 / self.scheduler.dalpha_over_alpha(t_cur)
            dsigma_mul_sigma = self.scheduler.dsigma_mul_sigma(t_cur)
            t_hat = t_next.repeat(batch_size)
            sigma_hat = self.scheduler.sigma(t_hat)
            alpha_over_dalpha_hat = 1 / self.scheduler.dalpha_over_alpha(t_hat)
            dsigma_mul_sigma_hat = self.scheduler.dsigma_mul_sigma(t_hat)

            if self.w_scheduler:
                w = self.w_scheduler.w(t_cur)
            else:
                w = 0.0
            if i == 0 or self.exact_henu:
                cfg_x = torch.cat([x, x], dim=0)
                cfg_t_cur = t_cur.repeat(2)
                out = net(cfg_x, cfg_t_cur, cfg_condition)
                if self.guidance_interval_min < t_cur[0] < self.guidance_interval_max:
                    guidance = self.guidance
                    out = self.guidance_fn(out, guidance)
                else:
                    out = self.guidance_fn(out, 1.0)
                v = out
                s = (alpha_over_dalpha * v - x) / (sigma ** 2 - (alpha_over_dalpha) * dsigma_mul_sigma)
            else:
                v = v_hat
                s = s_hat
            x_hat = self.step_fn(x, v, dt, s=s, w=w)
            if i < self.num_steps - 1:
                cfg_x_hat = torch.cat([x_hat, x_hat], dim=0)
                cfg_t_hat = t_hat.repeat(2)
                out = net(cfg_x_hat, cfg_t_hat, cfg_condition)
                if self.guidance_interval_min < t_hat[0] < self.guidance_interval_max:
                    guidance = self.guidance
                    out = self.guidance_fn(out, guidance)
                else:
                    out = self.guidance_fn(out, 1.0)
                v_hat = out
                s_hat = (
                    (alpha_over_dalpha_hat) * v_hat - x_hat
                ) / (sigma_hat ** 2 - (alpha_over_dalpha_hat) * dsigma_mul_sigma_hat)
                v = (v + v_hat) / 2
                s = (s + s_hat) / 2
                x = self.step_fn(x, v, dt, s=s, w=w)
            else:
                x = self.last_step_fn(x, v, dt, s=s, w=w)
            x_trajs.append(x)
            v_trajs.append(v)
        v_trajs.append(torch.zeros_like(x))
        return x_trajs, v_trajs


class FlowDPMSolverSampler(BaseSampler):
    def __init__(
        self,
        w_scheduler: BaseScheduler = None,
        timeshift=1.0,
        guidance_interval_min: float = 0.0,
        guidance_interval_max: float = 1.0,
        step_fn: Callable = ode_step_fn,
        last_step=None,
        last_step_fn: Callable = ode_step_fn,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.step_fn = step_fn
        self.last_step = last_step
        self.last_step_fn = last_step_fn
        self.w_scheduler = w_scheduler
        self.timeshift = timeshift
        self.guidance_interval_min = guidance_interval_min
        self.guidance_interval_max = guidance_interval_max

        if self.last_step is None or self.num_steps == 1:
            self.last_step = 1.0 / self.num_steps
        timesteps = torch.linspace(0.0, 1 - self.last_step, self.num_steps)
        timesteps = torch.cat([timesteps, torch.tensor([1.0])], dim=0)
        self.timesteps = shift_respace_fn(timesteps, self.timeshift)

    def _impl_sampling(self, net, noise, condition, uncondition):
        batch_size = noise.shape[0]
        steps = self.timesteps.to(noise.device, noise.dtype)
        cfg_condition = torch.cat([uncondition, condition], dim=0)

        x = noise
        x_trajs = [noise]
        v_trajs = []
        v_prev = None
        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            dt = t_next - t_cur
            t_cur = t_cur.repeat(batch_size)

            cfg_x = torch.cat([x, x], dim=0)
            cfg_t = t_cur.repeat(2)
            out = net(cfg_x, cfg_t, cfg_condition)
            if self.guidance_interval_min < t_cur[0] < self.guidance_interval_max:
                guidance = self.guidance
                out = self.guidance_fn(out, guidance)
            else:
                out = self.guidance_fn(out, 1.0)
            v = out

            if v_prev is None:
                x = x + v * dt
            else:
                x = x + dt * (1.5 * v - 0.5 * v_prev)
            v_prev = v
            x_trajs.append(x)
            v_trajs.append(v)
        v_trajs.append(torch.zeros_like(x))
        return x_trajs, v_trajs


def constant(alpha, sigma):
    return 1


def time_shift_fn(t, timeshift=1.0):
    return t / (t + (1 - t) * timeshift)


class DINOv2(nn.Module):
    def __init__(self, model_name: str = "dinov2_vitb14", base_patch_size=16):
        super().__init__()
        self.encoder = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            trust_repo=True,
        )
        self.encoder = self.encoder.to(torch.bfloat16)
        self.pos_embed = copy.deepcopy(self.encoder.pos_embed)
        self.encoder.head = torch.nn.Identity()
        self.patch_size = self.encoder.patch_embed.patch_size
        self.precomputed_pos_embed = dict()
        self.base_patch_size = base_patch_size
        self.encoder.compile()

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(self, x, resize=True):
        b, c, h, w = x.shape
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        if resize:
            x = torch.nn.functional.interpolate(
                x,
                (int(14 * h / self.base_patch_size), int(14 * w / self.base_patch_size)),
                mode="bicubic",
            )
        feature = self.encoder.forward_features(x)["x_norm_patchtokens"]
        feature = feature.to(torch.bfloat16)
        return feature


# DINOv3's SAT-493M weights were pretrained on satellite imagery and do NOT use
# the ImageNet statistics the LVD-1689M (web-image) weights use. Getting this
# wrong silently shifts every feature the alignment loss is aimed at.
# Source: dinov3 README, "For models using the SAT-493M weights".
SAT493M_MEAN = (0.430, 0.411, 0.296)
SAT493M_STD = (0.213, 0.156, 0.143)


class DINOv3(nn.Module):
    """Frozen DINOv3 backbone, as the REPA alignment target.

    Preferred over `DINOv2` for this task: the SAT-493M checkpoint is distilled
    on Maxar imagery at ~0.6 m/px, which is the exact domain of the HR target,
    where DINOv2 is a web-image model being asked to describe nadir satellite
    scenes.

    Loaded from a local source tree and a local .pth rather than torch.hub,
    because the HuggingFace repo is gated and compute nodes may be offline.
    `weights` is the swap point: pass a fine-tuned checkpoint of the same
    architecture to align against that instead.
    """

    def __init__(
        self,
        repo_dir: str = "",  # unused here: see superres.models.dinov3.FrozenDINOv3
        arch: str = "dinov3_vitl16",
        weights: str = "",   # unused here: see superres.models.dinov3.FrozenDINOv3
        base_patch_size: int = 16,
        mean=SAT493M_MEAN,
        std=SAT493M_STD,
        compile_encoder: bool = False,
    ):
        super().__init__()
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        import dinov3.hub.backbones as _backbones

        self.encoder = getattr(_backbones, arch)(pretrained=True, weights=weights)
        self.encoder = self.encoder.to(torch.bfloat16)
        self.encoder.eval()
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.encoder.embed_dim
        self.base_patch_size = base_patch_size
        self.register_buffer("_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        if compile_encoder:
            self.encoder.compile()

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(self, x, resize=True):
        b, c, h, w = x.shape
        x = (x - self._mean.to(x.dtype)) / self._std.to(x.dtype)
        if resize:
            # Identity when base_patch_size == patch_size and the caller already
            # supplied the aligned resolution, which is how REPASRTrainer uses it.
            target = (int(self.patch_size * h / self.base_patch_size),
                      int(self.patch_size * w / self.base_patch_size))
            if target != (h, w):
                x = torch.nn.functional.interpolate(x, target, mode="bicubic")
        feature = self.encoder.forward_features(x)["x_norm_patchtokens"]
        return feature.to(torch.bfloat16)


class SimpleFlowTrainer(BaseTrainer):
    """Plain flow-matching trainer with no auxiliary loss.

    `REPATrainer` adds a DINOv2 feature-alignment loss that is specific to
    natural-image (ImageNet-style) generation and requires an internet
    download of the DINOv2 checkpoint. For a pixel-aligned conditional task
    like super-resolution, that auxiliary loss doesn't apply -- this trainer
    is the same rectified-flow objective (`REPATrainer.fm_loss`) without it.
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        loss_weight_fn: Callable = constant,
        lognorm_t: bool = False,
        timeshift: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.loss_weight_fn = loss_weight_fn
        self.lognorm_t = lognorm_t
        self.timeshift = timeshift

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        batch_size = x.shape[0]
        if self.lognorm_t:
            base_t = torch.randn((batch_size,), device=x.device, dtype=torch.float32).sigmoid()
        else:
            base_t = torch.rand((batch_size,), device=x.device, dtype=torch.float32)
        t = time_shift_fn(base_t, self.timeshift).to(x.dtype)
        noise = torch.randn_like(x)
        alpha = self.scheduler.alpha(t)
        dalpha = self.scheduler.dalpha(t)
        sigma = self.scheduler.sigma(t)
        dsigma = self.scheduler.dsigma(t)

        x_t = alpha * x + noise * sigma
        v_t = dalpha * x + dsigma * noise

        out = net(x_t, t, y)

        weight = self.loss_weight_fn(alpha, sigma)
        fm_loss = weight * (out - v_t) ** 2
        return dict(loss=fm_loss.mean())


class REPATrainer(BaseTrainer):
    def __init__(
        self,
        scheduler: BaseScheduler,
        loss_weight_fn: Callable = constant,
        feat_loss_weight: float = 0.5,
        lognorm_t: bool = False,
        timeshift: float = 1.0,
        encoder: Optional[nn.Module] = None,
        align_layer: int = 8,
        proj_denoiser_dim: int = 256,
        proj_hidden_dim: int = 256,
        proj_encoder_dim: int = 256,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.timeshift = timeshift
        self.loss_weight_fn = loss_weight_fn
        self.feat_loss_weight = feat_loss_weight
        self.align_layer = align_layer
        self.encoder = encoder
        no_grad(self.encoder)

        self.proj = nn.Sequential(
            nn.Sequential(
                nn.Linear(proj_denoiser_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_encoder_dim),
            )
        )

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        raw_images = metadata["raw_image"]
        batch_size, c, height, width = x.shape
        if self.lognorm_t:
            base_t = torch.randn((batch_size), device=x.device, dtype=torch.float32).sigmoid()
        else:
            base_t = torch.rand((batch_size), device=x.device, dtype=torch.float32)
        t = time_shift_fn(base_t, self.timeshift).to(x.dtype)
        noise = torch.randn_like(x)
        alpha = self.scheduler.alpha(t)
        dalpha = self.scheduler.dalpha(t)
        sigma = self.scheduler.sigma(t)
        dsigma = self.scheduler.dsigma(t)

        x_t = alpha * x + noise * sigma
        v_t = dalpha * x + dsigma * noise

        src_feature = []

        def forward_hook(net, input, output):
            feature = output
            if isinstance(feature, tuple):
                feature = feature[0]
            src_feature.append(feature)

        handle = net.patch_blocks[self.align_layer - 1].register_forward_hook(forward_hook)

        out = net(x_t, t, y)
        src_feature = self.proj(src_feature[0])
        handle.remove()

        with torch.no_grad():
            dst_feature = self.encoder(raw_images)

        if dst_feature.shape[1] != src_feature.shape[1]:
            bsz, ls, ch = src_feature.shape
            ld = dst_feature.shape[1]
            hs = int(ls ** 0.5)
            hd = int(ld ** 0.5)
            src_spatial = src_feature.view(bsz, hs, hs, ch).permute(0, 3, 1, 2)
            if ld < ls:
                resized = torch.nn.functional.adaptive_avg_pool2d(src_spatial, (hd, hd))
            else:
                resized = torch.nn.functional.interpolate(src_spatial, size=(hd, hd), mode="bilinear", align_corners=False)
            src_feature = resized.permute(0, 2, 3, 1).reshape(bsz, ld, ch)

        cos_sim = torch.nn.functional.cosine_similarity(src_feature, dst_feature, dim=-1)
        cos_loss = 1 - cos_sim

        weight = self.loss_weight_fn(alpha, sigma)
        fm_loss = weight * (out - v_t) ** 2
        out = dict(
            fm_loss=fm_loss.mean(),
            cos_loss=cos_loss.mean(),
            loss=fm_loss.mean() + self.feat_loss_weight * cos_loss.mean(),
        )
        return out

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        self.proj.state_dict(
            destination=destination,
            prefix=prefix + "proj.",
            keep_vars=keep_vars,
        )


class REPASRTrainer(REPATrainer):
    """`REPATrainer` adapted to the Sentinel-2 -> Maxar SR task.

    Only two things differ from the parent, both forced by this task's shape:

    1. There is no `metadata["raw_image"]`. `S2MaxarSRDataset` yields just a
       filename, and adding a second full-resolution HR copy to every batch
       would put ~30MB/sample back into the dataloader queue that already had
       to be shrunk (prefetch_factor 8 -> 2) to stop a host-RAM OOM. It is not
       needed: the clean HR *is* `x`, in [-1, 1], so (x + 1) / 2 recovers the
       [0, 1] image DINOv2's ImageNet normalisation expects, for free.

    2. The encoder is fed at `encoder_resolution`, not at 1584px. The parent
       lets the encoder do its own resize, which at 1584 would push ~1386px
       through the ViT -> 99x99 = 9801 patch tokens: ~81x the attention cost of
       the aligned case, and the parent would then have to *upsample* the
       denoiser's 33x33 tokens to meet it. Feeding 33 * patch_size instead
       yields exactly 33x33 = 1089 encoder tokens, matching the 33x33 patch
       grid 1:1, so token i of the denoiser aligns with the encoder token
       covering the same ground, and the parent's shape branch is a no-op.

       That means encoder_resolution must track the encoder's patch size:
         DINOv3 vitl16 (patch 16) -> 528, proj_encoder_dim 1024   <- default
         DINOv2 vitb14 (patch 14) -> 462, proj_encoder_dim  768
       Set the encoder's `base_patch_size` equal to its real patch size so its
       internal resize is the identity on what we hand it.
    """

    def __init__(self, *args, encoder_resolution: int = 462, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_resolution = encoder_resolution

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        raw = (x.float() + 1.0) / 2.0
        if raw.shape[-1] != self.encoder_resolution:
            raw = torch.nn.functional.interpolate(
                raw,
                size=(self.encoder_resolution, self.encoder_resolution),
                mode="bicubic",
                align_corners=False,
            )
        metadata = dict(metadata or {})
        metadata["raw_image"] = raw.clamp(0.0, 1.0).to(x.dtype)
        return super()._impl_trainstep(net, ema_net, solver, x, y, metadata)

