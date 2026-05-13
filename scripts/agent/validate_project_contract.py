#!/usr/bin/env python3
"""Validate the project contract JSON against required schema."""
import json
import sys

REQUIRED_TOP_LEVEL_KEYS = [
    "project_title",
    "final_goal",
    "target_outputs",
    "datasets",
    "architecture",
    "implementation_stages",
    "evaluation_metrics",
    "first_risks",
    "guardrails",
]

REQUIRED_OUTPUT_KEYS = ["class", "mask", "family", "reason"]

REQUIRED_DATASETS = ["Community Forensics-Small", "SID-Set"]

REQUIRED_METRIC_SUBSTRINGS = [
    "classification",
    "macro-f1",
    "iou",
    "family",
    "robustness",
    "latency",
    "fps",
    "recall",
]


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def check_top_level_keys(contract):
    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in contract]
    if missing:
        fail(f"Missing top-level keys: {missing}")


def check_target_outputs(contract):
    outputs = contract.get("target_outputs", {})
    missing = [k for k in REQUIRED_OUTPUT_KEYS if k not in outputs]
    if missing:
        fail(f"target_outputs missing required keys: {missing}")


def check_datasets(contract):
    datasets = contract.get("datasets", {})
    missing = [d for d in REQUIRED_DATASETS if d not in datasets]
    if missing:
        fail(f"datasets missing required entries: {missing}")


def check_metrics(contract):
    metrics = contract.get("evaluation_metrics", [])
    metric_text = " ".join(
        (m.get("name", "") + " " + m.get("purpose", "")).lower()
        for m in metrics
    )
    missing = [s for s in REQUIRED_METRIC_SUBSTRINGS if s not in metric_text]
    if missing:
        fail(f"evaluation_metrics missing coverage for: {missing}")


def main():
    if len(sys.argv) != 2:
        print("Usage: validate_project_contract.py <path/to/contract.json>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            contract = json.load(f)
    except FileNotFoundError:
        fail(f"File not found: {path}")
    except json.JSONDecodeError as e:
        fail(f"Invalid JSON: {e}")

    check_top_level_keys(contract)
    check_target_outputs(contract)
    check_datasets(contract)
    check_metrics(contract)

    print(f"OK: {path} passes all contract validation checks.")
    sys.exit(0)


if __name__ == "__main__":
    main()
