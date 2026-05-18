#!/usr/bin/env python3
"""Run all toy metric calculators from a toy metrics config and print JSON summary.

Usage:
    python3 scripts/agent/run_toy_metrics.py configs/metrics/toy_metrics.example.json

Reads the toy config, runs all metric calculators, and prints a JSON summary
to stdout. Does not write any files.
"""
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

_REPO_ROOT = next(
    (p for p in _Path(__file__).resolve().parents if (p / "src" / "cv_forensics").is_dir()),
    None,
)
if _REPO_ROOT is not None:
    _SRC_ROOT = _REPO_ROOT / "src"
    if str(_SRC_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_SRC_ROOT))

import json
import sys
from typing import Any, Dict, List


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <toy_metrics_config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"Cannot load config {config_path!r}: {exc}")

    # Safety check
    try:
        from cv_forensics.metrics import (
            check_toy_metric_config_safety,
            classification_metrics,
            family_accuracy,
            fps_summary,
            latency_summary,
            mask_iou,
            localization_activation_recall,
            robustness_summary,
        )
    except ImportError as exc:
        _fail(f"Cannot import cv_forensics.metrics: {exc}")

    try:
        check_toy_metric_config_safety(cfg)
    except ValueError as exc:
        _fail(f"Config failed safety check: {exc}")

    summary: Dict[str, Any] = {}

    # Classification metrics
    cls_cfg = cfg.get("class_labels", {})
    summary["classification"] = classification_metrics(
        cls_cfg["y_true"], cls_cfg["y_pred"]
    )

    # Family metrics
    fam_cfg = cfg.get("family_labels", {})
    summary["family"] = family_accuracy(
        fam_cfg["y_true"],
        fam_cfg["y_pred"],
        ignore_real_or_na=fam_cfg.get("ignore_real_or_na", False),
    )

    # Mask IoU
    masks = cfg.get("masks", [])
    iou_results: List[Dict[str, Any]] = []
    for mask_pair in masks:
        iou_val = mask_iou(mask_pair["pred_mask"], mask_pair["gt_mask"])
        iou_results.append({
            "note": mask_pair.get("_comment", ""),
            "iou": iou_val,
        })
    summary["mask_iou"] = iou_results

    # Localization activation recall
    loc_cfg = cfg.get("localization", {})
    summary["localization_activation_recall"] = localization_activation_recall(
        loc_cfg["gt_class_labels"],
        loc_cfg["localization_states"],
    )

    # Latency and FPS
    lat_values = cfg.get("latency_ms", [])
    summary["latency"] = latency_summary(lat_values)
    summary["fps"] = fps_summary(lat_values)

    # Robustness drop
    perturb_records = cfg.get("perturbation_records", [])
    summary["robustness"] = robustness_summary(perturb_records)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
