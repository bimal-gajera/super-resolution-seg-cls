"""Device-dependent dtype selection.

This project is expected to run on both V100 (PSC's `GPU-shared` majority) and
H100 nodes. Volta has no bfloat16: `torch.autocast(dtype=torch.bfloat16)` on a
V100 either errors or silently falls back, and casting a module with
`.to(torch.bfloat16)` produces very slow emulated kernels.

So pick the widest fast dtype the actual device supports, rather than
hardcoding the one the H100 likes.
"""

from __future__ import annotations

import torch


def amp_dtype(device: str | torch.device = "cuda") -> torch.dtype:
    """bfloat16 where supported (Ampere+), float16 on older GPUs, float32 on CPU."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return torch.float32
    # `torch.cuda.is_bf16_supported()` returns True on V100, where bf16 is only
    # emulated and painfully slow. Gate on the compute capability instead:
    # real bf16 tensor cores start at Ampere (8.0).
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def autocast(device: str | torch.device = "cuda", enabled: bool = True):
    """`torch.autocast` with the right dtype for this device."""
    device = torch.device(device)
    dtype = amp_dtype(device)
    return torch.autocast(
        device_type=device.type, dtype=dtype, enabled=enabled and dtype != torch.float32
    )
