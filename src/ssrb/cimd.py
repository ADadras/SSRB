"""Thin adapter around the frozen official CIMD CVPR 2023 implementation."""

from __future__ import annotations

import sys
import os
import subprocess
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import kl_divergence


CIMD_COMMIT = "a9677afe4eedb163db478cd01ea8228161448b33"
DEFAULT_CIMD_ROOT = Path(os.environ.get("SSRB_CIMD_ROOT", "external/cimd"))


class _NullVisdom:
    """No-network replacement for upstream's unconditional Visdom calls."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def line(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def image(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _install_visdom_stub() -> None:
    if "visdom" in sys.modules:
        return
    module = types.ModuleType("visdom")
    module.Visdom = _NullVisdom  # type: ignore[attr-defined]
    sys.modules["visdom"] = module


def load_official_modules(root: Path = DEFAULT_CIMD_ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    if not (root / "guided_diffusion" / "script_util.py").is_file():
        raise FileNotFoundError(f"missing frozen CIMD checkout: {root}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    if commit != CIMD_COMMIT:
        raise ValueError("CIMD checkout does not match the recorded paper commit")
    _install_visdom_stub()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from guided_diffusion.distribution import AxisAlignedConvGaussian
    from guided_diffusion.script_util import create_gaussian_diffusion, create_model

    return {
        "AxisAlignedConvGaussian": AxisAlignedConvGaussian,
        "create_gaussian_diffusion": create_gaussian_diffusion,
        "create_model": create_model,
    }


class CIMDComponents(nn.Module):
    """Register the diffusion U-Net, ACN, and AMN in one state dictionary."""

    def __init__(
        self,
        *,
        image_size: int = 128,
        num_channels: int = 128,
        num_res_blocks: int = 2,
        num_heads: int = 1,
        attention_resolutions: str = "16",
        channel_mult: str = "",
        latent_dim: int = 6,
        conditioning_channels: int = 4,
        ambiguity_filters: tuple[int, ...] = (32, 64, 128, 192),
        ambiguity_convs_per_block: int = 3,
        cimd_root: Path = DEFAULT_CIMD_ROOT,
    ) -> None:
        super().__init__()
        modules = load_official_modules(cimd_root)
        self.model = modules["create_model"](
            int(image_size),
            int(num_channels),
            int(num_res_blocks),
            channel_mult=str(channel_mult),
            learn_sigma=True,
            class_cond=False,
            use_checkpoint=False,
            attention_resolutions=str(attention_resolutions),
            num_heads=int(num_heads),
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            dropout=0.0,
            resblock_updown=False,
            use_fp16=False,
            use_new_attention_order=False,
        )
        if int(conditioning_channels) != 4:
            first = self.model.input_blocks[0][0]
            if not isinstance(first, nn.Conv2d) or int(first.in_channels) != 5:
                raise TypeError("unexpected official CIMD input layer")
            self.model.input_blocks[0][0] = nn.Conv2d(
                int(conditioning_channels) + 1,
                int(first.out_channels),
                kernel_size=first.kernel_size,
                stride=first.stride,
                padding=first.padding,
                dilation=first.dilation,
                groups=first.groups,
                bias=first.bias is not None,
                padding_mode=first.padding_mode,
            )
        self.conditioning_channels = int(conditioning_channels)
        ambiguity_args = {
            "input_channels": 1,
            "num_filters": list(int(value) for value in ambiguity_filters),
            "no_convs_per_block": int(ambiguity_convs_per_block),
            "latent_dim": int(latent_dim),
            "initializers": {"w": "he_normal", "b": "normal"},
            "posterior": True,
        }
        # Upstream names these prior/posterior. The paper names them the
        # Ambiguity Controlling Network and Ambiguity Modelling Network.
        self.ambiguity_control = modules["AxisAlignedConvGaussian"](**ambiguity_args).float()
        self.ambiguity_modelling = modules["AxisAlignedConvGaussian"](**ambiguity_args).float()


def create_diffusion(
    *,
    steps: int = 1000,
    timestep_respacing: str = "",
    cimd_root: Path = DEFAULT_CIMD_ROOT,
) -> Any:
    modules = load_official_modules(cimd_root)
    return modules["create_gaussian_diffusion"](
        steps=int(steps),
        learn_sigma=True,
        noise_schedule="linear",
        timestep_respacing=str(timestep_respacing),
        rescale_learned_sigmas=False,
        rescale_timesteps=False,
    )


def training_objective(
    components: CIMDComponents,
    diffusion: Any,
    image_four_channel: torch.Tensor,
    target: torch.Tensor,
    timestep: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    vlb_weight: float = 0.001,
    ambiguity_weight: float = 0.001,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Paper-faithful loss without upstream's clean-target in-place mutation."""

    clean = (target > 0.5).float()
    if noise is None:
        noise = torch.randn_like(clean)
    noised = diffusion.q_sample(clean, timestep, noise=noise)
    model_input = torch.cat([image_four_channel, noised], dim=1)
    model_output = components.model(
        model_input, diffusion._scale_timesteps(timestep)
    )
    if model_output.shape[1] != 2:
        raise ValueError("locked CIMD configuration expects learned mean and variance")
    predicted_noise, variance_values = torch.split(model_output, 1, dim=1)
    frozen_output = torch.cat([predicted_noise.detach(), variance_values], dim=1)
    variational = diffusion._vb_terms_bpd(
        model=lambda *_args, value=frozen_output, **_kwargs: value,
        x_start=clean,
        x_t=noised,
        t=timestep,
        clip_denoised=False,
    )["output"]
    mse = (noise - predicted_noise).pow(2).flatten(1).mean(dim=1)
    image_luminance = image_four_channel[:, :1]
    controlling = components.ambiguity_control(image_luminance, predicted_noise)
    modelling = components.ambiguity_modelling(image_luminance, clean)
    ambiguity = kl_divergence(modelling, controlling)
    per_example = mse + float(vlb_weight) * variational + float(ambiguity_weight) * ambiguity
    terms = {
        "mse": mse,
        "vb": variational,
        "ambiguity_kl": ambiguity,
        "loss": per_example,
    }
    return per_example.mean(), terms


@torch.no_grad()
def sample_mask(
    model: nn.Module,
    diffusion: Any,
    image_four_channel: torch.Tensor,
) -> torch.Tensor:
    """Run the official conditional reverse process without shape hard-coding."""

    mask = torch.randn_like(image_four_channel[:, :1])
    batch = int(image_four_channel.shape[0])
    for index in reversed(range(int(diffusion.num_timesteps))):
        timestep = torch.full(
            (batch,), int(index), device=image_four_channel.device, dtype=torch.long
        )
        conditioned = torch.cat([image_four_channel, mask], dim=1)
        mask = diffusion.p_sample(
            model,
            conditioned,
            timestep,
            clip_denoised=True,
            model_kwargs={},
        )["sample"]
    return mask
