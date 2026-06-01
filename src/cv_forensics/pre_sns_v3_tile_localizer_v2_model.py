"""High-resolution forensic tile localizer v2 model."""

from __future__ import annotations

from typing import Any


FEATURE_MODES = {"rgb_only", "rgb_residual", "rgb_edge_residual"}


def _kernel(torch: Any, values: list[list[float]], channels: int, device: Any, dtype: Any) -> Any:
    base = torch.tensor(values, dtype=dtype, device=device).reshape(1, 1, 3, 3)
    return base.repeat(channels, 1, 1, 1)


def grayscale(torch: Any, images: Any) -> Any:
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=images.dtype, device=images.device).reshape(1, 3, 1, 1)
    return (images * weights).sum(dim=1, keepdim=True)


def fixed_forensic_features(torch: Any, images: Any, mode: str = "rgb_edge_residual", rough_mask_prior: Any | None = None) -> Any:
    """Return deterministic RGB/edge/residual forensic features for NCHW tensors."""

    if mode not in FEATURE_MODES:
        raise ValueError(f"unsupported feature mode: {mode}")
    nnf = torch.nn.functional
    features = [images]
    if mode in {"rgb_residual", "rgb_edge_residual"}:
        channels = images.shape[1]
        lap = _kernel(torch, [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], channels, images.device, images.dtype)
        residual_rgb = nnf.conv2d(images, lap, padding=1, groups=channels)
        residual_mag = residual_rgb.abs().mean(dim=1, keepdim=True)
        features.extend([residual_rgb, residual_mag])
    if mode == "rgb_edge_residual":
        gray = grayscale(torch, images)
        sobel_x = _kernel(torch, [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], 1, images.device, images.dtype)
        sobel_y = _kernel(torch, [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], 1, images.device, images.dtype)
        lap_gray = _kernel(torch, [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], 1, images.device, images.dtype)
        sx = nnf.conv2d(gray, sobel_x, padding=1)
        sy = nnf.conv2d(gray, sobel_y, padding=1)
        lap = nnf.conv2d(gray, lap_gray, padding=1)
        features.extend([sx, sy, lap, (sx.square() + sy.square()).sqrt()])
    if rough_mask_prior is not None:
        features.append(rough_mask_prior.to(device=images.device, dtype=images.dtype))
    return torch.cat(features, dim=1)


def feature_channel_count(mode: str, rough_mask_prior: bool = False) -> int:
    if mode == "rgb_only":
        count = 3
    elif mode == "rgb_residual":
        count = 7
    elif mode == "rgb_edge_residual":
        count = 11
    else:
        raise ValueError(f"unsupported feature mode: {mode}")
    return count + (1 if rough_mask_prior else 0)


def _block(torch: Any, in_channels: int, out_channels: int) -> Any:
    nn = torch.nn
    groups = 1 if out_channels < 4 else 4
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.SiLU(inplace=True),
    )


def build_pre_sns_v3_tile_localizer_v2(
    torch: Any,
    *,
    tile_size: int = 768,
    input_feature_mode: str = "rgb_edge_residual",
    base_channels: int = 8,
    boundary_head: bool = True,
    confidence_head: bool = True,
    rough_mask_prior: bool = False,
) -> Any:
    """Build a lightweight high-resolution forensic tile localizer v2."""

    nn = torch.nn
    in_channels = feature_channel_count(input_feature_mode, rough_mask_prior)

    class TileLocalizerV2(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tile_size = int(tile_size)
            self.input_feature_mode = input_feature_mode
            self.rough_mask_prior = bool(rough_mask_prior)
            c = int(base_channels)
            self.stem = _block(torch, in_channels, c)
            self.enc1 = _block(torch, c, c * 2)
            self.enc2 = _block(torch, c * 2, c * 4)
            self.bottleneck = _block(torch, c * 4, c * 8)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
            self.dec2 = _block(torch, c * 8, c * 4)
            self.up1 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
            self.dec1 = _block(torch, c * 4, c * 2)
            self.up0 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
            self.dec0 = _block(torch, c * 2, c)
            self.mask_head = nn.Conv2d(c, 1, 1)
            self.has_boundary_head = bool(boundary_head)
            self.has_confidence_head = bool(confidence_head)
            self.boundary_head = nn.Conv2d(c, 1, 1)
            self.confidence_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c * 8, 1))

        def forward(self, images: Any, rough_mask_prior: Any | None = None) -> dict[str, Any]:
            feats = fixed_forensic_features(torch, images, self.input_feature_mode, rough_mask_prior if self.rough_mask_prior else None)
            x0 = self.stem(feats)
            x1 = self.enc1(self.pool(x0))
            x2 = self.enc2(self.pool(x1))
            x3 = self.bottleneck(self.pool(x2))
            y2 = self._match(self.up2(x3), x2)
            y2 = self.dec2(torch.cat([y2, x2], dim=1))
            y1 = self._match(self.up1(y2), x1)
            y1 = self.dec1(torch.cat([y1, x1], dim=1))
            y0 = self._match(self.up0(y1), x0)
            y0 = self.dec0(torch.cat([y0, x0], dim=1))
            out = {"mask_logits": self.mask_head(y0)}
            if self.has_boundary_head:
                out["boundary_logits"] = self.boundary_head(y0)
            if self.has_confidence_head:
                out["tile_confidence"] = self.confidence_head(x3)
            return out

        @staticmethod
        def _match(value: Any, ref: Any) -> Any:
            if value.shape[-2:] == ref.shape[-2:]:
                return value
            return nn.functional.interpolate(value, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    return TileLocalizerV2()

