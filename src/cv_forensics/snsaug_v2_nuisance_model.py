"""SNSAug V2 nuisance-aware multi-head model."""

from __future__ import annotations

from typing import Any

CLASS_LABELS: tuple[str, ...] = ("real", "synthetic", "tampered")
DEGRADATION_LABELS: tuple[str, ...] = (
    "jpeg",
    "resize",
    "crop",
    "screenshot_resampling",
    "platform_layout",
    "overlay_text",
    "sticker",
    "news_meme",
    "combined_sns",
    "severity_light",
    "severity_medium",
)


def apply_soft_nuisance_gate(features: Any, nuisance_mask: Any, *, alpha: float = 0.5, min_gate: float = 0.25):
    """Reduce tamper features in predicted benign SNS/local overlay regions."""

    import torch.nn.functional as F

    mask = nuisance_mask.float()
    if mask.shape[-2:] != features.shape[-2:]:
        mask = F.interpolate(mask, size=features.shape[-2:], mode="bilinear", align_corners=False)
    gate = (1.0 - float(alpha) * mask).clamp(min=float(min_gate), max=1.0)
    return features * gate


def build_snsaug_v2_nuisance_model(
    torch: Any,
    *,
    base_channels: int = 16,
    degradation_dim: int = len(DEGRADATION_LABELS),
    gating_alpha: float = 0.5,
    reliability_head: bool = True,
):
    """Return a compact SNS/nuisance-aware forensics model with real weights."""

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
            self.conv = torch.nn.Conv2d(3, 12, kernel_size=3, padding=1, groups=3, bias=False)
            self.conv.weight.requires_grad_(False)
            with torch.no_grad():
                self.conv.weight.copy_(kernels[:, None, :, :].repeat(3, 1, 1, 1))

        def forward(self, images):
            return self.conv(images)

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

        def forward(self, value):
            return self.block(value)

    class SNSAugV2NuisanceModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c = int(base_channels)
            self.model_version = "snsaug_aware_nuisance_multihead_forensics_v1"
            self.gating_alpha = float(gating_alpha)
            self.high_pass = FixedHighPass()
            self.stem = ConvBlock(15, c)
            self.stage2 = ConvBlock(c, c * 2, stride=2)
            self.stage3 = ConvBlock(c * 2, c * 4, stride=2)
            self.stage4 = ConvBlock(c * 4, c * 6, stride=2)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.class_head = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Dropout(0.1), torch.nn.Linear(c * 6, len(CLASS_LABELS)))
            self.global_degradation_head = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(c * 6, int(degradation_dim)),
            )
            self.up3 = torch.nn.Conv2d(c * 6 + c * 4, c * 4, 3, padding=1)
            self.up2 = torch.nn.Conv2d(c * 4 + c * 2, c * 2, 3, padding=1)
            self.up1 = torch.nn.Conv2d(c * 2 + c, c, 3, padding=1)
            self.sns_nuisance_mask_head = torch.nn.Conv2d(c, 1, 1)
            self.tamper_mask_head = torch.nn.Conv2d(c, 1, 1)
            self.reliability_head = torch.nn.Conv2d(c, 1, 1) if reliability_head else None

        def _decode(self, f1, f2, f3, f4):
            d3 = torch.nn.functional.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
            d3 = torch.nn.functional.silu(self.up3(torch.cat([d3, f3], dim=1)))
            d2 = torch.nn.functional.interpolate(d3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = torch.nn.functional.silu(self.up2(torch.cat([d2, f2], dim=1)))
            d1 = torch.nn.functional.interpolate(d2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
            return torch.nn.functional.silu(self.up1(torch.cat([d1, f1], dim=1)))

        def forward(self, images, *, sns_nuisance_mask=None, use_teacher_sns_mask: bool = False):
            residual = self.high_pass(images)
            x = torch.cat([images, residual], dim=1)
            f1 = self.stem(x)
            f2 = self.stage2(f1)
            f3 = self.stage3(f2)
            f4 = self.stage4(f3)
            pooled = self.pool(f4)
            decoder = self._decode(f1, f2, f3, f4)
            nuisance_logits = self.sns_nuisance_mask_head(decoder)
            predicted_nuisance = torch.sigmoid(nuisance_logits).detach()
            gate_mask = sns_nuisance_mask if use_teacher_sns_mask and sns_nuisance_mask is not None else predicted_nuisance
            gated_decoder = apply_soft_nuisance_gate(decoder, gate_mask, alpha=self.gating_alpha)
            output = {
                "class_logits": self.class_head(pooled),
                "tamper_mask_logits": self.tamper_mask_head(gated_decoder),
                "localization_logits": self.tamper_mask_head(gated_decoder),
                "sns_nuisance_mask_logits": nuisance_logits,
                "global_degradation_logits": self.global_degradation_head(pooled),
                "gating_mask": gate_mask,
            }
            if self.reliability_head is not None:
                output["reliability_logits"] = self.reliability_head(gated_decoder)
            return output

    return SNSAugV2NuisanceModel()


__all__ = [
    "CLASS_LABELS",
    "DEGRADATION_LABELS",
    "apply_soft_nuisance_gate",
    "build_snsaug_v2_nuisance_model",
]
