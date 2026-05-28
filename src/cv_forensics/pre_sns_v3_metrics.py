"""Metrics for pre-SNS v3 validation and threshold calibration."""

from __future__ import annotations

import statistics
from typing import Any

from .pre_sns_v3_model import CLASS_LABELS, FAMILY_LABELS, MEANINGFUL_FAMILY_LABELS

DEFAULT_TAU_VALUES: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90)


def _zero_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: tuple[str, ...] = CLASS_LABELS) -> dict[str, Any]:
    idx = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, pred in zip(y_true, y_pred):
        if truth in idx and pred in idx:
            matrix[idx[truth]][idx[pred]] += 1
    return {"labels": list(labels), "matrix": matrix}


def per_label_metrics(y_true: list[str], y_pred: list[str], labels: tuple[str, ...] = CLASS_LABELS) -> dict[str, dict[str, float]]:
    cm = confusion_matrix(y_true, y_pred, labels)["matrix"]
    rows: dict[str, dict[str, float]] = {}
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(labels)) if r != i)
        fn = sum(cm[i][c] for c in range(len(labels)) if c != i)
        precision = _zero_div(tp, tp + fp)
        recall = _zero_div(tp, tp + fn)
        f1 = _zero_div(2 * precision * recall, precision + recall)
        rows[label] = {"precision": precision, "recall": recall, "f1": f1}
    return rows


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {"count": len(values), "min": min(values), "max": max(values), "mean": sum(values) / len(values), "median": statistics.median(values)}


def _mask_iou(pred: Any, truth: Any) -> float:
    p = [1 if bool(v) else 0 for v in _flatten(pred)]
    t = [1 if bool(v) else 0 for v in _flatten(truth)]
    inter = sum(1 for a, b in zip(p, t) if a and b)
    union = sum(1 for a, b in zip(p, t) if a or b)
    return _zero_div(inter, union) if union else 1.0


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]


def threshold_sweep(
    predictions: list[dict[str, Any]],
    tau_values: tuple[float, ...] | list[float] = DEFAULT_TAU_VALUES,
) -> list[dict[str, float]]:
    tampered_total = sum(1 for row in predictions if row["gt_class"] == "tampered")
    non_total = len(predictions) - tampered_total
    rows: list[dict[str, float]] = []
    for tau in tau_values:
        true_active = 0
        false_active = 0
        active_iou: list[float] = []
        for row in predictions:
            active = float(row["tampered_score"]) >= float(tau)
            if row["gt_class"] == "tampered" and active:
                true_active += 1
                if row.get("localization_iou") is not None:
                    active_iou.append(float(row["localization_iou"]))
            elif row["gt_class"] != "tampered" and active:
                false_active += 1
        rows.append(
            {
                "tau": float(tau),
                "localization_activation_recall": _zero_div(true_active, tampered_total),
                "false_activation_rate": _zero_div(false_active, non_total),
                "localization_mean_iou_when_activated": sum(active_iou) / len(active_iou) if active_iou else 0.0,
            }
        )
    return rows


def select_tau(sweep: list[dict[str, float]], max_false_activation_rate: float = 0.20) -> dict[str, Any]:
    if not sweep:
        raise ValueError("threshold sweep is empty")
    eligible = [row for row in sweep if row["false_activation_rate"] <= max_false_activation_rate]
    if eligible:
        best = sorted(
            eligible,
            key=lambda row: (-row["localization_activation_recall"], -row["localization_mean_iou_when_activated"], row["false_activation_rate"], row["tau"]),
        )[0]
        return {"selected_tau": float(best["tau"]), "fallback_selected_tau": False, "selected_row": best}
    best = sorted(sweep, key=lambda row: (row["false_activation_rate"], -row["localization_activation_recall"], row["tau"]))[0]
    return {"selected_tau": float(best["tau"]), "fallback_selected_tau": True, "selected_row": best}


def compute_metrics(
    predictions: list[dict[str, Any]],
    max_false_activation_rate: float = 0.20,
    tau_values: tuple[float, ...] | list[float] = DEFAULT_TAU_VALUES,
) -> dict[str, Any]:
    if not predictions:
        raise ValueError("predictions must not be empty")
    y_true = [str(row["gt_class"]) for row in predictions]
    y_pred = [str(row["pred_class"]) for row in predictions]
    per_class = per_label_metrics(y_true, y_pred)
    accuracy = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred) / len(y_true)
    macro_f1 = sum(row["f1"] for row in per_class.values()) / len(CLASS_LABELS)

    binary_true = ["tampered" if label == "tampered" else "non_tampered" for label in y_true]
    binary_pred = ["tampered" if label == "tampered" else "non_tampered" for label in y_pred]
    binary = per_label_metrics(binary_true, binary_pred, ("non_tampered", "tampered"))["tampered"]

    family_rows = [row for row in predictions if row.get("gt_family") in MEANINGFUL_FAMILY_LABELS and row.get("pred_family") in FAMILY_LABELS]
    family_accuracy = sum(1 for row in family_rows if row.get("gt_family") == row.get("pred_family")) / len(family_rows) if family_rows else None

    ious: list[float] = []
    mask_areas: list[float] = []
    for row in predictions:
        if row.get("mask_area_pct") is not None:
            mask_areas.append(float(row["mask_area_pct"]))
        iou = row.get("localization_iou")
        if iou is None and row.get("pred_mask") is not None and row.get("gt_mask") is not None:
            iou = _mask_iou(row["pred_mask"], row["gt_mask"])
        if row["gt_class"] == "tampered" and iou is not None:
            ious.append(float(iou))
            row["localization_iou"] = float(iou)

    sweep = threshold_sweep(predictions, tau_values)
    selected = select_tau(sweep, max_false_activation_rate)
    selected_row = selected["selected_row"]
    per_source: dict[str, Any] = {}
    per_family: dict[str, Any] = {}
    for key, target in (("source_dataset", per_source), ("gt_family", per_family)):
        values = sorted({str(row.get(key, "unknown")) for row in predictions})
        for value in values:
            rows = [row for row in predictions if str(row.get(key, "unknown")) == value]
            target[value] = confusion_matrix([str(row["gt_class"]) for row in rows], [str(row["pred_class"]) for row in rows])

    return {
        "class_accuracy": accuracy,
        "class_macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "real_recall": per_class["real"]["recall"],
        "tampered_precision": binary["precision"],
        "tampered_recall": binary["recall"],
        "tampered_f1": binary["f1"],
        "family_accuracy": family_accuracy,
        "family_samples_evaluated": len(family_rows),
        "tampered_score_summary_by_gt_class": {label: _summary([float(row["tampered_score"]) for row in predictions if row["gt_class"] == label]) for label in CLASS_LABELS},
        "threshold_sweep": sweep,
        "selected_tau": selected["selected_tau"],
        "fallback_selected_tau": selected["fallback_selected_tau"],
        "max_false_activation_rate": float(max_false_activation_rate),
        "localization_activation_recall": selected_row["localization_activation_recall"],
        "false_activation_rate": selected_row["false_activation_rate"],
        "localization_mean_iou": sum(ious) / len(ious) if ious else None,
        "localization_median_iou": statistics.median(ious) if ious else None,
        "mask_area_pct_summary": _summary(mask_areas),
        "per_source_dataset_confusion_matrix": per_source,
        "per_family_confusion_matrix": per_family,
        "samples_evaluated": len(predictions),
    }

