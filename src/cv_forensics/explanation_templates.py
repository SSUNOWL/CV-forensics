from __future__ import annotations

from typing import Dict, List, Optional

from .model_output_schema import (
    CLASS_LABEL_REAL,
    CLASS_LABEL_SYNTHETIC,
    CLASS_LABEL_TAMPERED,
    FAMILY_REAL_OR_NA,
    LOCALIZATION_ACTIVATED,
    LOW_CONFIDENCE_THRESHOLD,
)

# Evidence signal ID → Korean short label used in reason generation.
_SIGNAL_ID_KO: Dict[str, str] = {
    "boundary_discontinuity": "경계 불연속성",
    "texture_inconsistency": "텍스처 불일치",
    "local_mask_activation": "로컬 마스크 활성화",
    "provenance_family_signal": "출처 계열 신호",
    "low_confidence": "낮은 신뢰도 신호",
    "social_media_recompression": "소셜 미디어 재압축",
    "screenshot_padding": "스크린샷 패딩",
    "text_overlay_occlusion": "텍스트 오버레이 가림",
    "sticker_overlay_occlusion": "스티커 오버레이 가림",
    "global_synthetic_artifact": "전역 합성 아티팩트",
}

# Per-signal Korean detail sentences appended after the summary signal list.
# Each signal ID that appears in evidence will add its detail sentence to the
# generated reason, making each signal meaningfully influence the output.
_SIGNAL_DETAIL_KO: Dict[str, str] = {
    "boundary_discontinuity": "경계 불연속성이 탐지되어 편집 흔적이 시사됨.",
    "texture_inconsistency": "텍스처 불일치로 부분 합성 가능성이 있음.",
    "local_mask_activation": "로컬 마스크 활성화로 조작 의심 영역이 식별됨.",
    "provenance_family_signal": "출처 계열 신호가 특정 생성 도구 계열을 가리킴.",
    "low_confidence": "신뢰도 신호가 분류 불확실성을 나타냄.",
    "social_media_recompression": "소셜 미디어 재압축 아티팩트가 탐지됨.",
    "screenshot_padding": "스크린샷 패딩 감지 — 원본 이미지 컨텍스트 손실 가능성 있음.",
    "text_overlay_occlusion": "텍스트 오버레이로 이미지 일부가 가려짐.",
    "sticker_overlay_occlusion": "스티커 오버레이로 이미지 일부가 가려짐.",
    "global_synthetic_artifact": "전역 합성 아티팩트가 탐지되어 AI 생성 가능성을 시사함.",
}

# ---------------------------------------------------------------------------
# Template IDs
# ---------------------------------------------------------------------------
TEMPLATE_REAL: str = "real"
TEMPLATE_SYNTHETIC: str = "synthetic"
TEMPLATE_TAMPERED_LOCALIZED: str = "tampered_localized"
TEMPLATE_TAMPERED_NOT_LOCALIZED: str = "tampered_not_localized"
TEMPLATE_LOW_CONFIDENCE: str = "low_confidence"
TEMPLATE_FAMILY_ESTIMATED: str = "family_estimated"
TEMPLATE_FAMILY_UNAVAILABLE: str = "family_unavailable"
TEMPLATE_SOCIAL_MEDIA_PERTURBATION: str = "social_media_perturbation"

TEMPLATE_IDS = (
    TEMPLATE_REAL,
    TEMPLATE_SYNTHETIC,
    TEMPLATE_TAMPERED_LOCALIZED,
    TEMPLATE_TAMPERED_NOT_LOCALIZED,
    TEMPLATE_LOW_CONFIDENCE,
    TEMPLATE_FAMILY_ESTIMATED,
    TEMPLATE_FAMILY_UNAVAILABLE,
    TEMPLATE_SOCIAL_MEDIA_PERTURBATION,
)


def generate_reason(output: Dict[str, object]) -> str:
    """Generate a deterministic Korean explanation from a forensic output dict.

    Consumes the validated fields without any LLM call.
    Family is always described as coarse provenance estimation, not exact attribution.
    Localization is conditional and its absence is explained when relevant.
    """
    class_label = str(output["class"])
    class_conf: Dict[str, float] = output["class_conf"]  # type: ignore[assignment]
    family = str(output["family"])
    family_conf: Dict[str, float] = output["family_conf"]  # type: ignore[assignment]
    localization_head = str(output["localization_head"])
    mask_area_pct = output.get("mask_area_pct")
    threshold_tau = float(output["threshold_tau"])
    tampered_score = float(output["tampered_score"])
    perturbations: List[str] = output.get("perturbations", ["none"])  # type: ignore[assignment]
    evidence_list: List[Dict] = output.get("evidence", [])  # type: ignore[assignment]

    class_confidence = float(class_conf.get(class_label, 0.0))
    family_confidence = float(family_conf.get(family, 0.0))
    max_conf = max((float(v) for v in class_conf.values()), default=0.0)

    signal_ids: List[str] = [
        e.get("signal_id", "")
        for e in evidence_list
        if isinstance(e, dict) and e.get("signal_id")
    ]

    parts: List[str] = []

    # Low-confidence preamble
    if max_conf < LOW_CONFIDENCE_THRESHOLD:
        parts.append(
            f"분류 신뢰도가 낮음 (최고 신뢰도 {max_conf:.0%}). "
            f"결과는 참고용으로만 활용 권장."
        )

    # Family phrase — reused across templates
    def _family_phrase() -> str:
        if family == FAMILY_REAL_OR_NA:
            return "생성 계열 정보 미제공 (Real-or-N/A)."
        return (
            f"생성 계열은 {family}로 추정됨 (신뢰도 {family_confidence:.0%}). "
            f"이는 coarse 계열 추정이며 정확한 모델 귀속이 아님."
        )

    # Main classification block
    if class_label == CLASS_LABEL_REAL:
        parts.append(
            f"실제(real) 이미지로 분류됨 (신뢰도 {class_confidence:.0%}). "
            f"생성 또는 조작 흔적이 탐지되지 않음."
        )
    elif class_label == CLASS_LABEL_SYNTHETIC:
        parts.append(
            f"AI 생성(synthetic) 이미지로 분류됨 (신뢰도 {class_confidence:.0%}). "
            + _family_phrase()
        )
    elif class_label == CLASS_LABEL_TAMPERED:
        if localization_head == LOCALIZATION_ACTIVATED:
            area = float(mask_area_pct) if mask_area_pct is not None else 0.0
            parts.append(
                f"조작(tampered) 이미지로 분류됨 (신뢰도 {class_confidence:.0%}). "
                f"조작 의심 영역이 이미지 면적의 약 {area:.1f}%에서 활성화됨. "
                + _family_phrase()
            )
        else:
            parts.append(
                f"조작(tampered) 이미지로 분류됨 (신뢰도 {class_confidence:.0%}). "
                f"tampered 점수 ({tampered_score:.3f})가 임계값 tau ({threshold_tau:.3f}) 미만으로 "
                f"localization head가 활성화되지 않음. "
                + _family_phrase()
            )
    else:
        parts.append(
            f"이미지 분류: {class_label} (신뢰도 {class_confidence:.0%}). "
            + _family_phrase()
        )

    # Evidence signal IDs — list all signals then add per-signal detail sentences.
    # Each present signal ID influences the reason with a specific Korean sentence.
    if signal_ids:
        ko_signals = [_SIGNAL_ID_KO.get(sid, sid) for sid in signal_ids]
        parts.append(f"탐지된 증거 신호: {', '.join(ko_signals)}.")
        for sid in signal_ids:
            detail = _SIGNAL_DETAIL_KO.get(sid)
            if detail:
                parts.append(detail)

    # Social-media perturbation context
    non_none = [p for p in perturbations if p != "none"]
    if non_none:
        perturbation_str = ", ".join(non_none)
        parts.append(
            f"소셜 미디어 변형 감지됨: {perturbation_str}. "
            f"변형으로 인해 분류 성능이 저하될 수 있음."
        )

    return " ".join(parts)


def build_evidence_summary(output: Dict[str, object]) -> Dict[str, object]:
    """Return a structured evidence summary dict for later reporting."""
    class_label = str(output["class"])
    class_conf: Dict[str, float] = output["class_conf"]  # type: ignore[assignment]
    family = str(output["family"])
    family_conf: Dict[str, float] = output["family_conf"]  # type: ignore[assignment]
    localization_head = str(output["localization_head"])
    mask_area_pct = output.get("mask_area_pct")
    evidence_list: list = output.get("evidence", [])  # type: ignore[assignment]
    perturbations: List[str] = output.get("perturbations", ["none"])  # type: ignore[assignment]

    signal_ids: List[str] = [
        e.get("signal_id", "") for e in evidence_list if isinstance(e, dict) and e.get("signal_id")
    ]
    max_conf = max((float(v) for v in class_conf.values()), default=0.0)

    return {
        "class": class_label,
        "class_confidence": float(class_conf.get(class_label, 0.0)),
        "family": family,
        "family_confidence": float(family_conf.get(family, 0.0)),
        "localization_activated": localization_head == LOCALIZATION_ACTIVATED,
        "mask_area_pct": mask_area_pct,
        "signal_ids": signal_ids,
        "perturbations": [p for p in perturbations if p != "none"],
        "is_low_confidence": max_conf < LOW_CONFIDENCE_THRESHOLD,
    }
