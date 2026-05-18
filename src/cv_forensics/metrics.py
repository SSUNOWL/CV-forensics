"""Pure-Python metric calculators for the multi-head image forensics prototype.

Task 0009 — metrics-only dry-run module.

All functions operate on toy labels, toy nested-list binary masks,
toy localization states, toy latency values, and toy perturbation groups.
Uses only the Python standard library. No files are read, no images loaded,
no datasets accessed, no checkpoints touched, and no network access occurs.
All metric functions are deterministic.
"""
from __future__ import annotations

import math
import re as _re
from typing import Any, Dict, List, Optional, Tuple

from .model_output_schema import (
    CLASS_LABELS,
    FAMILY_LABELS,
    FAMILY_REAL_OR_NA,
    LOCALIZATION_ACTIVATED,
    LOCALIZATION_SKIPPED_BELOW_THRESHOLD,
    LOCALIZATION_STATES,
)

# ---------------------------------------------------------------------------
# Valid perturbation tags for robustness metrics (excludes "none")
# ---------------------------------------------------------------------------
ROBUSTNESS_PERTURBATION_TAGS: frozenset = frozenset({
    "jpeg",
    "resize",
    "crop",
    "rotation",
    "padding",
    "shear",
    "screenshot",
    "text_overlay",
    "sticker_overlay",
    "recompression_chain",
})

# Fixed label order for classification metrics
_CLASS_ORDER: Tuple[str, ...] = CLASS_LABELS  # ("real", "synthetic", "tampered")

# ---------------------------------------------------------------------------
# Config safety helpers
# ---------------------------------------------------------------------------
_PROTECTED_DIRS = frozenset({"secrets", "data", "datasets", "outputs", "checkpoints"})
_PROTECTED_URL_PREFIXES = ("http://", "https://", "s3://", "gs://", "hf://")
_PROTECTED_ABS_PREFIXES = ("/home/", "/mnt/", "/root/", "/Users/")
_SECRET_TOKEN_KEYWORDS = frozenset({"token", "password", "secret", "credential", "auth", "bearer"})


def _normalized_tokens(value: str) -> List[str]:
    return [t for t in _re.split(r"[^a-z0-9]+", value.lower()) if t]


def _is_unsafe_string(value: str) -> Optional[str]:
    if not value:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    lower = stripped.lower()
    for prefix in _PROTECTED_URL_PREFIXES:
        if lower.startswith(prefix):
            return f"URL-like reference: {value!r}"
    for prefix in _PROTECTED_ABS_PREFIXES:
        if stripped.startswith(prefix):
            return f"absolute machine path: {value!r}"
    # Windows drive paths: C:\ or C:/
    if len(stripped) >= 3 and stripped[0].isalpha() and stripped[1] == ":" and stripped[2] in ("\\", "/"):
        return f"Windows drive path: {value!r}"
    # .env files
    if stripped == ".env" or stripped.startswith(".env."):
        return f".env file reference: {value!r}"
    # Protected directory as a path segment
    parts = _re.split(r"[/\\]+", stripped)
    for part in parts:
        if part and part in _PROTECTED_DIRS:
            return f"protected directory {part!r} in path {value!r}"
    # Secret-looking string content (tokenize to avoid false positives)
    tokens = _normalized_tokens(lower)
    for kw in _SECRET_TOKEN_KEYWORDS:
        if kw in tokens:
            return f"secret-looking value (contains token {kw!r}): {value!r}"
    if "api" in tokens and "key" in tokens:
        return f"secret-looking value (contains api key tokens): {value!r}"
    return None


def _has_secret_key(key: str) -> bool:
    tokens = _normalized_tokens(key.lower())
    if any(t in _SECRET_TOKEN_KEYWORDS for t in tokens):
        return True
    return "api" in tokens and "key" in tokens


def _check_config_node(node: Any, path: str) -> None:
    """Recursively walk a config node and raise ValueError on the first unsafe finding."""
    if isinstance(node, dict):
        for k, v in node.items():
            child = f"{path}.{k}"
            if isinstance(k, str) and _has_secret_key(k):
                raise ValueError(f"{child}: secret-looking key name {k!r}")
            _check_config_node(v, child)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _check_config_node(item, f"{path}[{i}]")
    elif isinstance(node, str):
        reason = _is_unsafe_string(node)
        if reason:
            raise ValueError(f"{path}: {reason}")


def check_toy_metric_config_safety(
    config: Dict[str, Any],
    context: str = "toy_metric_config",
) -> None:
    """Recursively validate a toy metric config for protected paths, URLs, and secret-like values.

    Rejects:
    - .env, .env.* references
    - secrets, data, datasets, outputs, checkpoints path components
    - absolute local paths (/home/, /mnt/, /root/, /Users/)
    - Windows drive paths
    - http://, https://, s3://, gs://, hf:// URLs
    - secret-looking keys or values (token, api_key, password, secret, credential, auth, bearer)

    Raises ValueError with an actionable message on the first violation found.
    """
    _check_config_node(config, context)


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _validate_class_labels(labels: List[str], name: str) -> None:
    for i, lbl in enumerate(labels):
        if lbl not in CLASS_LABELS:
            raise ValueError(
                f"{name}[{i}]: invalid class label {lbl!r}. "
                f"Must be one of {CLASS_LABELS}."
            )


def _validate_family_labels(labels: List[str], name: str) -> None:
    for i, lbl in enumerate(labels):
        if lbl not in FAMILY_LABELS:
            raise ValueError(
                f"{name}[{i}]: invalid family label {lbl!r}. "
                f"Must be one of {FAMILY_LABELS}."
            )


def _validate_localization_states(states: List[str], name: str) -> None:
    for i, state in enumerate(states):
        if state not in LOCALIZATION_STATES:
            raise ValueError(
                f"{name}[{i}]: invalid localization state {state!r}. "
                f"Must be one of {LOCALIZATION_STATES}."
            )


def _validate_mask(mask: List[List[Any]], name: str) -> Tuple[int, int]:
    """Validate a nested-list binary mask. Returns (n_rows, n_cols)."""
    if not isinstance(mask, list) or len(mask) == 0:
        raise ValueError(f"{name}: mask must be a non-empty nested list.")
    n_cols: Optional[int] = None
    for i, row in enumerate(mask):
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError(f"{name}[{i}]: each row must be a non-empty list.")
        if n_cols is None:
            n_cols = len(row)
        elif len(row) != n_cols:
            raise ValueError(
                f"{name}: ragged mask detected. Row {i} has {len(row)} columns, "
                f"expected {n_cols}."
            )
        for j, val in enumerate(row):
            if val not in (0, 1, False, True):
                raise ValueError(
                    f"{name}[{i}][{j}]: invalid mask value {val!r}. "
                    f"Must be 0, 1, False, or True."
                )
    return len(mask), n_cols  # type: ignore[return-value]


def _validate_latency(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name}: latency must be numeric, got bool.")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name}: latency must be numeric, got {type(value).__name__}.")
    fval = float(value)
    if math.isnan(fval):
        raise ValueError(f"{name}: latency must not be NaN.")
    if math.isinf(fval):
        raise ValueError(f"{name}: latency must not be infinite.")
    if fval <= 0.0:
        raise ValueError(f"{name}: latency must be positive, got {fval}.")
    return fval


def _validate_perturbation_tag(tag: str, name: str) -> None:
    if tag not in ROBUSTNESS_PERTURBATION_TAGS:
        raise ValueError(
            f"{name}: unknown perturbation tag {tag!r}. "
            f"Must be one of {sorted(ROBUSTNESS_PERTURBATION_TAGS)}."
        )


def _validate_numeric_score(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name}: score must be numeric, got bool.")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name}: score must be numeric, got {type(value).__name__}.")
    fval = float(value)
    if math.isnan(fval):
        raise ValueError(f"{name}: score must not be NaN.")
    if math.isinf(fval):
        raise ValueError(f"{name}: score must not be infinite.")
    return fval


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Any]:
    """Compute a confusion matrix for the three class labels.

    Label order is fixed: real, synthetic, tampered.

    Returns a dict with:
        - labels: fixed label order list
        - matrix: nested list [true_idx][pred_idx] of counts
        - n: total sample count
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length "
            f"(got {len(y_true)} and {len(y_pred)})."
        )
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    _validate_class_labels(y_true, "y_true")
    _validate_class_labels(y_pred, "y_pred")

    n = len(_CLASS_ORDER)
    idx = {lbl: i for i, lbl in enumerate(_CLASS_ORDER)}
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        matrix[idx[t]][idx[p]] += 1

    return {
        "labels": list(_CLASS_ORDER),
        "matrix": matrix,
        "n": len(y_true),
    }


def classification_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """Compute overall classification accuracy."""
    cm = classification_confusion_matrix(y_true, y_pred)
    matrix = cm["matrix"]
    correct = sum(matrix[i][i] for i in range(len(_CLASS_ORDER)))
    return correct / cm["n"]


def per_class_precision(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Per-class precision. Returns 0.0 for classes with no predicted positives."""
    cm = classification_confusion_matrix(y_true, y_pred)
    matrix = cm["matrix"]
    n = len(_CLASS_ORDER)
    result = {}
    for j, lbl in enumerate(_CLASS_ORDER):
        tp = matrix[j][j]
        col_sum = sum(matrix[i][j] for i in range(n))
        result[lbl] = tp / col_sum if col_sum > 0 else 0.0
    return result


def per_class_recall(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Per-class recall. Returns 0.0 for classes with no actual positives."""
    cm = classification_confusion_matrix(y_true, y_pred)
    matrix = cm["matrix"]
    result = {}
    for i, lbl in enumerate(_CLASS_ORDER):
        tp = matrix[i][i]
        row_sum = sum(matrix[i])
        result[lbl] = tp / row_sum if row_sum > 0 else 0.0
    return result


def per_class_f1(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Per-class F1 score. Returns 0.0 for undefined cases (zero-division safe)."""
    prec = per_class_precision(y_true, y_pred)
    rec = per_class_recall(y_true, y_pred)
    result = {}
    for lbl in _CLASS_ORDER:
        p, r = prec[lbl], rec[lbl]
        result[lbl] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return result


def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    """Macro-averaged F1 over all three class labels."""
    f1s = per_class_f1(y_true, y_pred)
    return sum(f1s.values()) / len(f1s)


def classification_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Any]:
    """Full classification metrics bundle (accuracy, per-class P/R/F1, Macro-F1)."""
    cm = classification_confusion_matrix(y_true, y_pred)
    acc = classification_accuracy(y_true, y_pred)
    prec = per_class_precision(y_true, y_pred)
    rec = per_class_recall(y_true, y_pred)
    f1 = per_class_f1(y_true, y_pred)
    mf1 = macro_f1(y_true, y_pred)
    return {
        "confusion_matrix": cm,
        "accuracy": acc,
        "per_class_precision": prec,
        "per_class_recall": rec,
        "per_class_f1": f1,
        "macro_f1": mf1,
    }


# ---------------------------------------------------------------------------
# Family / provenance metrics
# ---------------------------------------------------------------------------

def family_accuracy(
    y_true: List[str],
    y_pred: List[str],
    ignore_real_or_na: bool = False,
) -> Dict[str, Any]:
    """Compute generator-family coarse provenance accuracy.

    Args:
        y_true: Ground-truth family labels.
        y_pred: Predicted family labels.
        ignore_real_or_na: If True, samples where y_true == 'Real-or-N/A' are
            excluded from evaluation. This mirrors the SID-Set family-label
            masking strategy where family supervision is unavailable.

    Returns:
        dict with n_evaluated, n_correct, n_ignored, accuracy, ignore_real_or_na.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length "
            f"(got {len(y_true)} and {len(y_pred)})."
        )
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    _validate_family_labels(y_true, "y_true")
    _validate_family_labels(y_pred, "y_pred")

    n_ignored = 0
    n_correct = 0
    n_evaluated = 0
    for t, p in zip(y_true, y_pred):
        if ignore_real_or_na and t == FAMILY_REAL_OR_NA:
            n_ignored += 1
            continue
        n_evaluated += 1
        if t == p:
            n_correct += 1

    acc = n_correct / n_evaluated if n_evaluated > 0 else 0.0
    return {
        "n_evaluated": n_evaluated,
        "n_correct": n_correct,
        "n_ignored": n_ignored,
        "accuracy": acc,
        "ignore_real_or_na": ignore_real_or_na,
    }


# ---------------------------------------------------------------------------
# Localization metrics (toy nested-list masks)
# ---------------------------------------------------------------------------

def mask_iou(
    pred_mask: List[List[Any]],
    gt_mask: List[List[Any]],
) -> float:
    """Compute binary mask IoU from two rectangular nested-list masks.

    Toy convention: if both masks are all-zero, IoU is defined as 1.0
    (both agree there is no tampered region). This avoids undefined 0/0
    division and is documented as a toy-metric choice.

    Args:
        pred_mask: Predicted binary mask as a nested list of 0/1/False/True.
        gt_mask: Ground-truth binary mask in the same format.

    Returns:
        IoU float in [0.0, 1.0].
    """
    rows_p, cols_p = _validate_mask(pred_mask, "pred_mask")
    rows_g, cols_g = _validate_mask(gt_mask, "gt_mask")
    if rows_p != rows_g or cols_p != cols_g:
        raise ValueError(
            f"pred_mask and gt_mask must have the same shape. "
            f"Got ({rows_p}, {cols_p}) and ({rows_g}, {cols_g})."
        )
    intersection = 0
    union = 0
    for i in range(rows_p):
        for j in range(cols_p):
            p = 1 if pred_mask[i][j] else 0
            g = 1 if gt_mask[i][j] else 0
            intersection += p & g
            union += p | g

    if union == 0:
        # Both all-zero: agree on no tampered region → perfect toy IoU
        return 1.0
    return intersection / union


def localization_activation_recall(
    gt_class_labels: List[str],
    localization_states: List[str],
) -> Dict[str, Any]:
    """Compute localization head activation recall for tampered examples.

    For each true tampered example:
        - 'activated': counted as a hit.
        - 'skipped_below_threshold': counted as missed activation (false negative
          caused by threshold_tau being too high).
        - 'not_applicable' or 'unavailable': counted in n_other (edge cases).

    activation_recall = n_activated / n_tampered (0.0 if n_tampered == 0).

    Returns:
        dict with n_tampered, n_activated, n_skipped_below_threshold,
        n_other, activation_recall.
    """
    if len(gt_class_labels) != len(localization_states):
        raise ValueError(
            f"gt_class_labels and localization_states must have the same length "
            f"(got {len(gt_class_labels)} and {len(localization_states)})."
        )
    if len(gt_class_labels) == 0:
        raise ValueError("Inputs must not be empty.")
    _validate_class_labels(gt_class_labels, "gt_class_labels")
    _validate_localization_states(localization_states, "localization_states")

    n_tampered = 0
    n_activated = 0
    n_skipped = 0
    n_other = 0

    for cls, state in zip(gt_class_labels, localization_states):
        if cls != "tampered":
            continue
        n_tampered += 1
        if state == LOCALIZATION_ACTIVATED:
            n_activated += 1
        elif state == LOCALIZATION_SKIPPED_BELOW_THRESHOLD:
            n_skipped += 1
        else:
            n_other += 1

    recall = n_activated / n_tampered if n_tampered > 0 else 0.0
    return {
        "n_tampered": n_tampered,
        "n_activated": n_activated,
        "n_skipped_below_threshold": n_skipped,
        "n_other": n_other,
        "activation_recall": recall,
    }


# ---------------------------------------------------------------------------
# Runtime metrics
# ---------------------------------------------------------------------------

def latency_summary(latency_ms_list: List[Any]) -> Dict[str, Any]:
    """Compute mean, min, max, and count of latency values in milliseconds.

    Args:
        latency_ms_list: Non-empty list of positive numeric latency values (ms).

    Returns:
        dict with mean_ms, min_ms, max_ms, count.
    """
    if not isinstance(latency_ms_list, list) or len(latency_ms_list) == 0:
        raise ValueError("latency_ms_list must be a non-empty list.")
    values = [
        _validate_latency(v, f"latency_ms_list[{i}]")
        for i, v in enumerate(latency_ms_list)
    ]
    n = len(values)
    return {
        "mean_ms": sum(values) / n,
        "min_ms": min(values),
        "max_ms": max(values),
        "count": n,
    }


def fps_summary(latency_ms_list: List[Any]) -> Dict[str, Any]:
    """Compute FPS statistics derived from latency values.

    FPS = 1000.0 / latency_ms for each sample.

    Args:
        latency_ms_list: Non-empty list of positive numeric latency values (ms).

    Returns:
        dict with mean_fps, min_fps, max_fps, count.
    """
    if not isinstance(latency_ms_list, list) or len(latency_ms_list) == 0:
        raise ValueError("latency_ms_list must be a non-empty list.")
    fps_values = [
        1000.0 / _validate_latency(v, f"latency_ms_list[{i}]")
        for i, v in enumerate(latency_ms_list)
    ]
    n = len(fps_values)
    return {
        "mean_fps": sum(fps_values) / n,
        "min_fps": min(fps_values),
        "max_fps": max(fps_values),
        "count": n,
    }


# ---------------------------------------------------------------------------
# Robustness drop metrics
# ---------------------------------------------------------------------------

def perturbation_robustness_drop(
    clean_score: Any,
    perturbed_score: Any,
    perturbation_tag: str,
) -> Dict[str, Any]:
    """Compute score drop for a single perturbation tag.

    This is a metric CALCULATOR only. SNS augmentation is not implemented here.
    This function accepts pre-computed toy clean and perturbed scores and
    serves as a placeholder for later social-media robustness evaluation.

    Drop = clean_score - perturbed_score (positive value means degradation).

    Args:
        clean_score: Numeric score (e.g. F1 or IoU) measured on clean input.
        perturbed_score: Numeric score measured on perturbed input.
        perturbation_tag: One of the supported SNS perturbation tag strings.

    Returns:
        dict with perturbation_tag, clean_score, perturbed_score, drop.
    """
    _validate_perturbation_tag(perturbation_tag, "perturbation_tag")
    clean = _validate_numeric_score(clean_score, "clean_score")
    perturbed = _validate_numeric_score(perturbed_score, "perturbed_score")
    return {
        "perturbation_tag": perturbation_tag,
        "clean_score": clean,
        "perturbed_score": perturbed,
        "drop": clean - perturbed,
    }


def robustness_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-perturbation and aggregate robustness drop summary.

    Each record must contain:
        - perturbation_tag: str (one of the supported SNS perturbation tags)
        - clean_score: numeric
        - perturbed_score: numeric

    Returns:
        dict with per_perturbation (tag → drop result), mean_drop, max_drop, n.
    """
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError("records must be a non-empty list.")
    drops: List[float] = []
    per_perturbation: Dict[str, Any] = {}
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise TypeError(f"records[{i}]: expected dict, got {type(rec).__name__}.")
        tag = rec.get("perturbation_tag", "")
        result = perturbation_robustness_drop(
            rec.get("clean_score"),
            rec.get("perturbed_score"),
            tag,
        )
        per_perturbation[tag] = result
        drops.append(result["drop"])

    return {
        "per_perturbation": per_perturbation,
        "mean_drop": sum(drops) / len(drops),
        "max_drop": max(drops),
        "n": len(drops),
    }
