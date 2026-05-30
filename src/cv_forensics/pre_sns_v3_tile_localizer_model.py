"""Lightweight pre-SNS 512 tile localization model."""

from __future__ import annotations

from typing import Any


def _conv_block(torch: Any, in_channels: int, out_channels: int):
    nn = torch.nn
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class FixedHighPassBranch:
    """Fixed residual filters used as a small forensic cue branch."""

    def __init__(self, torch: Any):
        self.torch = torch
        self.module = torch.nn.Conv2d(3, 9, kernel_size=3, padding=1, groups=3, bias=False)
        kernels = torch.tensor(
            [
                [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        weights = kernels.repeat(3, 1, 1).unsqueeze(1)
        with torch.no_grad():
            self.module.weight.copy_(weights)
        for param in self.module.parameters():
            param.requires_grad = False


def build_pre_sns_v3_tile_localizer(torch: Any, base_channels: int = 8, with_confidence_head: bool = True):
    """Build a small U-Net-like tile localizer.

    The returned module uses no pretrained weights and depends only on the
    caller-provided torch module.
    """

    nn = torch.nn

    class PreSnsV3TileLocalizer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = int(base_channels)
            self.high_pass = FixedHighPassBranch(torch).module
            self.rgb_stem = _conv_block(torch, 3, channels)
            self.hp_stem = _conv_block(torch, 9, channels)
            self.enc1 = _conv_block(torch, channels * 2, channels * 2)
            self.enc2 = _conv_block(torch, channels * 2, channels * 4)
            self.enc3 = _conv_block(torch, channels * 4, channels * 8)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(channels * 8, channels * 4, kernel_size=2, stride=2)
            self.dec2 = _conv_block(torch, channels * 8, channels * 4)
            self.up1 = nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=2, stride=2)
            self.dec1 = _conv_block(torch, channels * 4, channels * 2)
            self.up0 = nn.ConvTranspose2d(channels * 2, channels * 2, kernel_size=2, stride=2)
            self.dec0 = _conv_block(torch, channels * 4, channels * 2)
            self.mask_head = nn.Conv2d(channels * 2, 1, kernel_size=1)
            self.with_confidence_head = bool(with_confidence_head)
            self.confidence_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels * 8, 1))

        def forward(self, images: Any) -> dict[str, Any]:
            hp = self.high_pass(images)
            x0 = self.torch_cat([self.rgb_stem(images), self.hp_stem(hp)], dim=1)
            x1 = self.enc1(self.pool(x0))
            x2 = self.enc2(self.pool(x1))
            x3 = self.enc3(self.pool(x2))
            y2 = self.up2(x3)
            y2 = self._match_size(y2, x2)
            y2 = self.dec2(self.torch_cat([y2, x2], dim=1))
            y1 = self.up1(y2)
            y1 = self._match_size(y1, x1)
            y1 = self.dec1(self.torch_cat([y1, x1], dim=1))
            y0 = self.up0(y1)
            y0 = self._match_size(y0, x0)
            y0 = self.dec0(self.torch_cat([y0, x0], dim=1))
            result = {"mask_logits": self.mask_head(y0)}
            if self.with_confidence_head:
                result["tile_logits"] = self.confidence_head(x3)
            return result

        def torch_cat(self, values: list[Any], dim: int) -> Any:
            return torch.cat(values, dim=dim)

        @staticmethod
        def _match_size(value: Any, reference: Any) -> Any:
            if value.shape[-2:] == reference.shape[-2:]:
                return value
            return nn.functional.interpolate(value, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    return PreSnsV3TileLocalizer()

