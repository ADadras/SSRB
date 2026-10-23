"""The Probabilistic U-Net used in the accepted comparison (authors' implementation)."""
from __future__ import annotations
import torch
from torch import nn
from .models import TinyUNetBackbone, TinyUNetDecoder

class ProbUNet(nn.Module):
    def __init__(self, *, latent_dim: int = 8, base: int = 32) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.encoder = TinyUNetBackbone(3, base)
        self.posterior_encoder = TinyUNetBackbone(4, base)
        self.prior = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base * 8, latent_dim * 2))
        self.posterior = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base * 8, latent_dim * 2))
        self.fuse = nn.Conv2d(base * 8 + latent_dim, base * 8, 1)
        self.decoder = TinyUNetDecoder(1, base, bottleneck_ch=base * 8)

    def encode_prior(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        z, skips = self.encoder(image)
        mu, logvar = self.prior(z).chunk(2, dim=1)
        return mu, logvar, z, skips

    def encode_posterior(self, image: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, _ = self.posterior_encoder(torch.cat([image, target], dim=1))
        return self.posterior(z).chunk(2, dim=1)

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z_img: torch.Tensor, skips: list[torch.Tensor], latent: torch.Tensor) -> torch.Tensor:
        lat = latent[:, :, None, None].expand(-1, -1, z_img.shape[-2], z_img.shape[-1])
        fused = self.fuse(torch.cat([z_img, lat], dim=1))
        return self.decoder(fused, skips)

    def forward(self, image: torch.Tensor, target: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        prior_mu, prior_logvar, z_img, skips = self.encode_prior(image)
        if target is None:
            latent = self.sample(prior_mu, prior_logvar)
            post_mu = prior_mu
            post_logvar = prior_logvar
        else:
            post_mu, post_logvar = self.encode_posterior(image, target)
            latent = self.sample(post_mu, post_logvar)
        logits = self.decode(z_img, skips, latent)
        return {
            "logits": logits,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "post_mu": post_mu,
            "post_logvar": post_logvar,
        }

def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()

def kl_normal(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        logvar_p
        - logvar_q
        + (torch.exp(logvar_q) + (mu_q - mu_p).pow(2)) / torch.exp(logvar_p).clamp_min(1e-8)
        - 1.0
    ).sum(dim=1).mean()
