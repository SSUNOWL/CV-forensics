"""Pre-SNS training preflight helpers.

This module validates readiness for the final dry-run before a guarded
training entrypoint. It does not write files, download data, scan directories,
or perform real training.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List

from .pre_sns_manifest import CLASS_LABELS, FAMILY_LABELS, validate_sample


@dataclasses.dataclass(frozen=True)
class PreflightIssue:
    field: str
    message: str


def _issue(field: str, message: str) -> PreflightIssue:
    return PreflightIssue(field=field, message=message)


def summarize_manifest_readiness(manifest: Dict[str, Any], max_samples: int, cpu_only: bool) -> Dict[str, Any]:
    """Validate unified manifest coverage and return a dry-run readiness summary."""
    issues: List[PreflightIssue] = []
    if cpu_only is not True:
        issues.append(_issue("tiny_preflight_limits.cpu_only", "cpu_only must be true"))
    if not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples < 1:
        issues.append(_issue("tiny_preflight_limits.max_samples", "max_samples must be a positive integer"))

    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        issues.append(_issue("manifest.samples", "manifest must contain a non-empty samples list"))
        samples = []
    if max_samples > 0 and len(samples) > max_samples:
        issues.append(_issue("manifest.samples", "sample count exceeds max_samples"))

    seen_classes: set[str] = set()
    seen_families: set[str] = set()
    tampered_with_mask = 0
    class_loss_count = 0
    family_loss_count = 0
    localization_loss_count = 0
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            issues.append(_issue(f"manifest.samples[{index}]", "sample must be an object"))
            continue
        for sample_issue in validate_sample(sample, index):
            issues.append(_issue(sample_issue.path, sample_issue.message))
        class_label = sample.get("class_label")
        if class_label in CLASS_LABELS:
            seen_classes.add(class_label)
            class_loss_count += 1
        family_label = sample.get("family_label")
        if family_label in FAMILY_LABELS:
            seen_families.add(family_label)
            family_loss_count += 1
        tasks = sample.get("tasks_available")
        routing = sample.get("loss_routing")
        if isinstance(tasks, dict) and isinstance(routing, dict):
            if routing.get("class_loss") is not bool(tasks.get("class")):
                issues.append(_issue(f"manifest.samples[{index}].loss_routing.class_loss", "does not match tasks_available.class"))
            if routing.get("family_loss") is not bool(tasks.get("family")):
                issues.append(_issue(f"manifest.samples[{index}].loss_routing.family_loss", "does not match tasks_available.family"))
            if routing.get("localization_loss") is not bool(tasks.get("localization")):
                issues.append(_issue(f"manifest.samples[{index}].loss_routing.localization_loss", "does not match tasks_available.localization"))
            if routing.get("localization_loss") is True:
                localization_loss_count += 1
        else:
            issues.append(_issue(f"manifest.samples[{index}].loss_routing", "tasks_available and loss_routing are required"))
        if class_label == "tampered" and isinstance(sample.get("mask_path"), str):
            tampered_with_mask += 1

    for label in CLASS_LABELS:
        if label not in seen_classes:
            issues.append(_issue("class_coverage", f"missing class coverage: {label}"))
    if not seen_families:
        issues.append(_issue("family_coverage", "missing family coverage"))
    if tampered_with_mask < 1:
        issues.append(_issue("tampered_mask_coverage", "missing tampered sample with mask"))
    if family_loss_count < 1:
        issues.append(_issue("loss_routing.family_loss", "no family loss samples available"))
    if localization_loss_count < 1:
        issues.append(_issue("loss_routing.localization_loss", "no localization loss samples available"))

    return {
        "ready": not issues,
        "issues": [dataclasses.asdict(issue) for issue in issues],
        "sample_count": len(samples),
        "class_labels_seen": sorted(seen_classes),
        "family_labels_seen": sorted(seen_families),
        "tampered_with_mask_count": tampered_with_mask,
        "class_loss_count": class_loss_count,
        "family_loss_count": family_loss_count,
        "localization_loss_count": localization_loss_count,
        "cpu_only": cpu_only,
        "max_samples": max_samples,
    }


def ensure_ready(summary: Dict[str, Any]) -> None:
    """Raise ValueError if the readiness summary is not ready."""
    if not summary.get("ready"):
        messages = [f"{issue['field']}: {issue['message']}" for issue in summary.get("issues", [])]
        raise ValueError("preflight readiness failed:\n" + "\n".join(messages))
