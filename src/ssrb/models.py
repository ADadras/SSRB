from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNetBackbone(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32) -> None:
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

    @property
    def out_channels(self) -> int:
        return self.e4.net[0].out_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        z = self.e4(self.pool(s3))
        return z, [s1, s2, s3]


class TinyUNetDecoder(nn.Module):
    def __init__(self, out_ch: int, base: int = 32, bottleneck_ch: int | None = None) -> None:
        super().__init__()
        bch = int(bottleneck_ch or base * 8)
        self.u3 = nn.ConvTranspose2d(bch, base * 4, 2, stride=2)
        self.d3 = ConvBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, z: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        s1, s2, s3 = skips
        x = self.u3(z)
        x = self.d3(torch.cat([x, s3], dim=1))
        x = self.u2(x)
        x = self.d2(torch.cat([x, s2], dim=1))
        x = self.u1(x)
        x = self.d1(torch.cat([x, s1], dim=1))
        return self.out(x)


class KHeadValidityUNet(nn.Module):
    def __init__(self, *, heads: int = 8, base: int = 32) -> None:
        super().__init__()
        self.heads = int(heads)
        self.encoder = TinyUNetBackbone(3, base)
        self.decoder = TinyUNetDecoder(heads, base)
        self.validity = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 8, base * 2),
            nn.SiLU(inplace=True),
            nn.Linear(base * 2, heads),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        z, skips = self.encoder(image)
        return {"mask_logits": self.decoder(z, skips), "validity_logits": self.validity(z)}

class DirectMaskBank(nn.Module):
    def __init__(
        self,
        *,
        layers: int,
        dim: int,
        queries: int,
        hidden: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer_mix = nn.Parameter(torch.zeros(int(layers)))
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(int(dim)),
            nn.Linear(int(dim), int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.mask_head = nn.Linear(int(hidden), int(queries))
        self.rank_head = nn.Sequential(
            nn.LayerNorm(int(dim)),
            nn.Linear(int(dim), int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(queries)),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.layer_mix.softmax(dim=0).view(1, -1, 1, 1)
        mixed = (features * weights).sum(dim=1)
        batch, tokens, _ = mixed.shape
        side = int(round(math.sqrt(tokens)))
        if side * side != tokens:
            raise ValueError(f"feature token count {tokens} is not a square grid")
        hidden = self.token_mlp(mixed)
        masks = self.mask_head(hidden).permute(0, 2, 1).reshape(batch, -1, side, side)
        ranks = self.rank_head(mixed.mean(dim=1))
        return masks, ranks

class FrozenDinoRawResidual(nn.Module):
    """Frozen DINO bank plus one zero-initialized raw spatial/rank residual."""

    def __init__(self, anchor: DirectMaskBank, *, heads: int, base: int) -> None:
        super().__init__()
        if anchor.mask_head.out_features != int(heads):
            raise ValueError("semantic and residual head counts differ")
        self.anchor = anchor
        self.raw = KHeadValidityUNet(heads=int(heads), base=int(base))
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)
        self.zero_initialize_residual()
        self.anchor.eval()

    def zero_initialize_residual(self) -> None:
        nn.init.zeros_(self.raw.decoder.out.weight)
        if self.raw.decoder.out.bias is not None:
            nn.init.zeros_(self.raw.decoder.out.bias)
        final_validity = self.raw.validity[-1]
        if not isinstance(final_validity, nn.Linear):
            raise TypeError("raw validity head no longer ends in a Linear layer")
        nn.init.zeros_(final_validity.weight)
        nn.init.zeros_(final_validity.bias)

    def train(self, mode: bool = True) -> "FrozenDinoRawResidual":
        super().train(mode)
        self.anchor.eval()
        return self

    def forward(
        self, features: torch.Tensor, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            anchor_masks, anchor_ranks = self.anchor(features)
            anchor_masks = F.interpolate(
                anchor_masks,
                size=tuple(image.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )
        residual = self.raw(image)
        if anchor_masks.shape != residual["mask_logits"].shape or anchor_ranks.shape != residual["validity_logits"].shape:
            raise ValueError("head-aligned addition requires identical shapes")
        return (
            anchor_masks + residual["mask_logits"],
            anchor_ranks + residual["validity_logits"],
        )
