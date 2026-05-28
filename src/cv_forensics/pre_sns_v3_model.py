"""Self-contained pre-SNS v3 model.

The v3 model intentionally avoids external weights.  It combines an RGB branch
with a fixed high-pass residual branch, then shares a compact CNN backbone
across class, binary tamper, family, and localization heads.
"""

from __future__ import annotations

from typing import Any

CLASS_LABELS: tuple[str, ...] = ("real", "full_synthetic", "tampered")
FAMILY_LABELS: tuple[str, ...] = ("LatDiff", "PixDiff", "GAN", "Other", "Real-or-N/A")
MEANINGFUL_FAMILY_LABELS: tuple[str, ...] = ("LatDiff", "PixDiff", "GAN", "Other")
TAMPER_BINARY_LABELS: tuple[str, ...] = ("non_tampered", "tampered")


def build_pre_sns_v3_model(torch: Any, base_channels: int = 24):
    """Return a torch module for pre-SNS v3 training/inference."""

    class FixedHighPass(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            kernels = torch.tensor(
                [
                    [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
                    [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
                    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                ],
                dtype=torch.float32,
            )
            weight = kernels[:, None, :, :].repeat(3, 1, 1, 1)
            self.conv = torch.nn.Conv2d(3, 12, kernel_size=3, padding=1, groups=3, bias=False)
            self.conv.weight.requires_grad_(False)
            with torch.no_grad():
                self.conv.weight.copy_(weight)

        def forward(self, x):
            return self.conv(x)

    class ConvBlock(torch.nn.Module):
        def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
            super().__init__()
            self.block = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                torch.nn.BatchNorm2d(out_ch),
                torch.nn.SiLU(inplace=True),
                torch.nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(out_ch),
                torch.nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class PreSnsV3Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c = int(base_channels)
            self.high_pass = FixedHighPass()
            self.stem = ConvBlock(15, c)
            self.stage2 = ConvBlock(c, c * 2, stride=2)
            self.stage3 = ConvBlock(c * 2, c * 4, stride=2)
            self.stage4 = ConvBlock(c * 4, c * 6, stride=2)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.class_head = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Dropout(0.15), torch.nn.Linear(c * 6, len(CLASS_LABELS)))
            self.tamper_binary_head = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Dropout(0.15), torch.nn.Linear(c * 6, 2))
            self.family_head = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Dropout(0.15), torch.nn.Linear(c * 6, len(FAMILY_LABELS)))
            self.up3 = torch.nn.Conv2d(c * 6 + c * 4, c * 4, 3, padding=1)
            self.up2 = torch.nn.Conv2d(c * 4 + c * 2, c * 2, 3, padding=1)
            self.up1 = torch.nn.Conv2d(c * 2 + c, c, 3, padding=1)
            self.mask_head = torch.nn.Conv2d(c, 1, 1)

        def forward(self, images):
            residual = self.high_pass(images)
            x = torch.cat([images, residual], dim=1)
            f1 = self.stem(x)
            f2 = self.stage2(f1)
            f3 = self.stage3(f2)
            f4 = self.stage4(f3)
            pooled = self.pool(f4)
            d3 = torch.nn.functional.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
            d3 = torch.nn.functional.silu(self.up3(torch.cat([d3, f3], dim=1)))
            d2 = torch.nn.functional.interpolate(d3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = torch.nn.functional.silu(self.up2(torch.cat([d2, f2], dim=1)))
            d1 = torch.nn.functional.interpolate(d2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
            d1 = torch.nn.functional.silu(self.up1(torch.cat([d1, f1], dim=1)))
            return {
                "class_logits": self.class_head(pooled),
                "tamper_binary_logits": self.tamper_binary_head(pooled),
                "family_logits": self.family_head(pooled),
                "localization_logits": self.mask_head(d1),
            }

    return PreSnsV3Model()

