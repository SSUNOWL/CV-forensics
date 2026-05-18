"""Pure-Python dry-run training loop skeleton for task 0010.

Connects config schema, dataset manifests, fake inference, and toy metrics into a
simulated training/evaluation flow. No real training, no real data, no file writes
beyond reading the config JSON. All behavior is deterministic for a fixed config.
"""
from __future__ import annotations

import dataclasses
import json
import re as _re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Allowed enumerations (mirrors config_schema constants)
# ---------------------------------------------------------------------------

ALLOWED_METRIC_NAMES = (
    "accuracy_3way",
    "macro_f1",
    "tampered_mask_iou",
    "generator_family_accuracy",
    "perturbation_robustness_drop",
    "latency_ms",
    "fps",
    "localization_activation_recall",
)

ALLOWED_STAGES = (
    "dry_run",
    "cf_small_binary_backbone",
    "provenance_head",
    "sid_set_multihead_finetune",
    "social_media_robustness",
)

ALLOWED_FAMILY_POLICIES = (
    "available",
    "unavailable",
    "freeze",
    "mask_out",
    "mixed_cf_small_batch",
)

# ---------------------------------------------------------------------------
# Safety constants
# ---------------------------------------------------------------------------

_PROTECTED_DIRS = frozenset({"secrets", "data", "datasets", "outputs", "checkpoints"})
_PROTECTED_URL_PREFIXES = ("http://", "https://", "ftp://", "s3://", "gs://", "hf://")
_PROTECTED_ABS_PREFIXES = ("/home/", "/mnt/", "/root/", "/Users/")
_SECRET_TOKEN_KEYWORDS = frozenset(
    {"token", "password", "secret", "credential", "auth", "bearer"}
)
_SECRET_KEY_KEYWORDS = frozenset(
    {"token", "api_key", "password", "secret", "credential", "auth", "bearer"}
)


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
    if len(stripped) >= 3 and stripped[0].isalpha() and stripped[1] == ":" and stripped[2] in ("\\", "/"):
        return f"Windows drive path: {value!r}"
    if stripped == ".env" or stripped.startswith(".env."):
        return f".env file reference: {value!r}"
    parts = _re.split(r"[/\\]+", stripped)
    for part in parts:
        if part and part in _PROTECTED_DIRS:
            return f"protected directory {part!r} in path {value!r}"
    tokens = _normalized_tokens(lower)
    for kw in _SECRET_TOKEN_KEYWORDS:
        if kw in tokens:
            return f"secret-looking value (contains token {kw!r}): {value!r}"
    if "api" in tokens and "key" in tokens:
        return f"secret-looking value (contains api key tokens): {value!r}"
    return None


def _has_secret_key(key: str) -> bool:
    lower = key.lower()
    if lower in _SECRET_KEY_KEYWORDS:
        return True
    tokens = _normalized_tokens(lower)
    if any(t in _SECRET_TOKEN_KEYWORDS for t in tokens):
        return True
    return "api" in tokens and "key" in tokens


def _check_safety_node(node: Any, path: str) -> None:
    """Recursively walk a config node and raise ValueError on the first unsafe finding."""
    if isinstance(node, dict):
        for k, v in node.items():
            child = f"{path}.{k}"
            if isinstance(k, str) and _has_secret_key(k):
                raise ValueError(f"{child}: secret-looking key name {k!r}")
            _check_safety_node(v, child)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _check_safety_node(item, f"{path}[{i}]")
    elif isinstance(node, str):
        reason = _is_unsafe_string(node)
        if reason:
            raise ValueError(f"{path}: {reason}")


def check_dry_run_training_config_safety(
    raw_config: Dict[str, Any],
    context: str = "dry_run_training_config",
) -> None:
    """Recursively validate a dry-run training config for unsafe references.

    Raises ValueError with an actionable message if any protected path, URL,
    secret-looking key, or secret-looking value is found.
    """
    _check_safety_node(raw_config, context)


# ---------------------------------------------------------------------------
# Config and result dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DryRunTrainingConfig:
    schema_version: str
    dry_run: bool
    no_download: bool
    no_training: bool
    no_network: bool
    no_outputs: bool
    no_checkpoints: bool
    stage: str
    epochs: int
    batch_size: int
    seed: int
    threshold_tau: float
    fake_input_config_ref: str
    toy_metrics_config_ref: str
    dataset_manifest_refs: List[str]
    metric_names: List[str]
    family_policy: str
    dataset_plan: Optional[str] = None


@dataclasses.dataclass
class DryRunEpochSummary:
    epoch: int
    n_batches: int
    n_samples: int
    fake_inference_count: int
    classification_metrics: Dict[str, Any]
    family_metrics: Dict[str, Any]
    localization_metrics: Dict[str, Any]
    latency_metrics: Dict[str, Any]


@dataclasses.dataclass
class DryRunTrainingResult:
    config_stage: str
    config_seed: int
    config_epochs: int
    config_batch_size: int
    config_family_policy: str
    config_threshold_tau: float
    metric_names: List[str]
    epoch_summaries: List[DryRunEpochSummary]
    final_metrics: Dict[str, Any]
    dry_run: bool
    no_training: bool
    no_outputs: bool
    no_checkpoints: bool


# ---------------------------------------------------------------------------
# Config loading and validation
# ---------------------------------------------------------------------------

def load_dry_run_training_config(path: str) -> DryRunTrainingConfig:
    """Load and validate a dry-run training config JSON from *path*.

    Raises FileNotFoundError, json.JSONDecodeError, or ValueError.
    """
    with open(path) as f:
        raw = json.load(f)
    check_dry_run_training_config_safety(raw)
    return validate_dry_run_training_config(raw)


def validate_dry_run_training_config(raw: Dict[str, Any]) -> DryRunTrainingConfig:
    """Validate a raw config dict and return a DryRunTrainingConfig.

    Raises ValueError with actionable messages on any violation.
    """
    errors: List[str] = []

    def _require_bool_true(key: str) -> None:
        val = raw.get(key)
        if val is not True:
            errors.append(f"{key!r} must be true (got {val!r}).")

    def _require_key(key: str) -> None:
        if key not in raw:
            errors.append(f"Missing required key: {key!r}.")

    _require_key("schema_version")
    _require_bool_true("dry_run")
    _require_bool_true("no_download")
    _require_bool_true("no_training")
    _require_bool_true("no_network")
    _require_bool_true("no_outputs")
    _require_bool_true("no_checkpoints")
    _require_key("stage")
    _require_key("epochs")
    _require_key("batch_size")
    _require_key("seed")
    _require_key("threshold_tau")
    _require_key("fake_input_config_ref")
    _require_key("toy_metrics_config_ref")
    _require_key("dataset_manifest_refs")
    _require_key("metric_names")
    _require_key("family_policy")

    stage = raw.get("stage")
    if stage is not None and stage not in ALLOWED_STAGES:
        errors.append(f"Unknown stage {stage!r}. Must be one of {ALLOWED_STAGES}.")

    family_policy = raw.get("family_policy")
    if family_policy is not None and family_policy not in ALLOWED_FAMILY_POLICIES:
        errors.append(
            f"Unknown family_policy {family_policy!r}. "
            f"Must be one of {ALLOWED_FAMILY_POLICIES}."
        )

    metric_names = raw.get("metric_names", [])
    if not isinstance(metric_names, list):
        errors.append("'metric_names' must be a list.")
    else:
        for m in metric_names:
            if m not in ALLOWED_METRIC_NAMES:
                errors.append(
                    f"Unknown metric {m!r}. Must be one of {ALLOWED_METRIC_NAMES}."
                )

    dataset_manifest_refs = raw.get("dataset_manifest_refs", [])
    if not isinstance(dataset_manifest_refs, list):
        errors.append("'dataset_manifest_refs' must be a list.")

    epochs = raw.get("epochs")
    if epochs is not None and (not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1):
        errors.append(f"'epochs' must be a positive integer (got {epochs!r}).")

    batch_size = raw.get("batch_size")
    if batch_size is not None and (not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1):
        errors.append(f"'batch_size' must be a positive integer (got {batch_size!r}).")

    if errors:
        raise ValueError(
            "DryRunTrainingConfig validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return DryRunTrainingConfig(
        schema_version=str(raw["schema_version"]),
        dry_run=True,
        no_download=True,
        no_training=True,
        no_network=True,
        no_outputs=True,
        no_checkpoints=True,
        stage=str(raw["stage"]),
        epochs=int(raw["epochs"]),
        batch_size=int(raw["batch_size"]),
        seed=int(raw["seed"]),
        threshold_tau=float(raw["threshold_tau"]),
        fake_input_config_ref=str(raw["fake_input_config_ref"]),
        toy_metrics_config_ref=str(raw["toy_metrics_config_ref"]),
        dataset_manifest_refs=list(raw["dataset_manifest_refs"]),
        metric_names=list(raw["metric_names"]),
        family_policy=str(raw["family_policy"]),
        dataset_plan=raw.get("dataset_plan"),
    )


# ---------------------------------------------------------------------------
# Fake batch builder
# ---------------------------------------------------------------------------

# Fixed rotation of canonical fake scenarios (no randomness beyond seed offset)
_FAKE_BATCH_SCENARIOS = (
    {
        "scenario": "real_clean",
        "class_hint": "real",
        "family_hint": "Real-or-N/A",
        "tampered_score": 0.05,
        "mask_area_hint_pct": None,
        "perturbations": ["none"],
        "evidence_hints": [],
    },
    {
        "scenario": "synthetic_latdiff",
        "class_hint": "synthetic",
        "family_hint": "LatDiff",
        "tampered_score": 0.12,
        "mask_area_hint_pct": None,
        "perturbations": ["jpeg"],
        "evidence_hints": ["global_synthetic_artifact", "provenance_family_signal"],
    },
    {
        "scenario": "tampered_localized",
        "class_hint": "tampered",
        "family_hint": "LatDiff",
        "tampered_score": 0.82,
        "mask_area_hint_pct": 11.2,
        "perturbations": ["jpeg", "recompression_chain"],
        "evidence_hints": ["boundary_discontinuity", "texture_inconsistency"],
    },
    {
        "scenario": "tampered_below_threshold",
        "class_hint": "tampered",
        "family_hint": "GAN",
        "tampered_score": 0.30,
        "mask_area_hint_pct": None,
        "perturbations": ["resize", "crop"],
        "evidence_hints": ["boundary_discontinuity"],
    },
)

_N_SCENARIOS = len(_FAKE_BATCH_SCENARIOS)


def build_fake_batches(
    config: DryRunTrainingConfig,
    n_batches: int,
) -> List[List[Dict[str, Any]]]:
    """Build a deterministic list of fake batches for a dry-run epoch.

    Each batch is a list of fake input dicts. Records rotate through the canonical
    scenarios using config.seed as the base offset. No randomness, no file I/O.
    """
    batches: List[List[Dict[str, Any]]] = []
    for batch_idx in range(n_batches):
        batch: List[Dict[str, Any]] = []
        for sample_idx in range(config.batch_size):
            scenario_idx = (
                config.seed + batch_idx * config.batch_size + sample_idx
            ) % _N_SCENARIOS
            template = _FAKE_BATCH_SCENARIOS[scenario_idx]
            record: Dict[str, Any] = dict(template)
            record["input_id"] = (
                f"seed{config.seed}_b{batch_idx}_s{sample_idx}"
                f"_{template['scenario']}"
            )
            record["threshold_tau"] = config.threshold_tau
            batch.append(record)
        batches.append(batch)
    return batches


# ---------------------------------------------------------------------------
# Dry-run epoch runner
# ---------------------------------------------------------------------------

def run_dry_epoch(
    config: DryRunTrainingConfig,
    epoch: int,
    n_batches: int,
) -> DryRunEpochSummary:
    """Simulate one epoch of dry-run training/evaluation.

    Calls fake inference and toy metrics. No file I/O, no real optimization,
    no real data access.
    """
    from .inference_stub import FakeInput, run_fake_batch
    from .metrics import (
        classification_metrics,
        family_accuracy,
        localization_activation_recall,
        mask_iou,
        latency_summary,
    )

    batches = build_fake_batches(config, n_batches)

    all_outputs: List[Dict[str, Any]] = []
    for batch in batches:
        fake_inputs = [FakeInput.from_dict(r) for r in batch]
        outputs = run_fake_batch(fake_inputs)
        all_outputs.extend(outputs)

    n_samples = len(all_outputs)

    y_true_class: List[str] = []
    y_pred_class: List[str] = []
    y_true_family: List[str] = []
    y_pred_family: List[str] = []
    gt_class_labels: List[str] = []
    localization_states: List[str] = []
    latency_ms_list: List[float] = []

    for i, output in enumerate(all_outputs):
        batch_idx = i // config.batch_size
        sample_idx = i % config.batch_size
        scenario_idx = (
            config.seed + batch_idx * config.batch_size + sample_idx
        ) % _N_SCENARIOS
        template = _FAKE_BATCH_SCENARIOS[scenario_idx]

        gt_class = template["class_hint"]
        y_true_class.append(gt_class)
        y_pred_class.append(output["class"])

        y_true_family.append(template["family_hint"])
        y_pred_family.append(output["family"])

        gt_class_labels.append(gt_class)
        localization_states.append(output["localization_head"])

        # Deterministic fake latency in ms (always positive)
        lat = 10.0 + float((config.seed + epoch + i) % 10)
        latency_ms_list.append(lat)

    cls_result = classification_metrics(y_true_class, y_pred_class)
    fam_result = family_accuracy(y_true_family, y_pred_family, ignore_real_or_na=True)
    loc_result = localization_activation_recall(gt_class_labels, localization_states)
    lat_result = latency_summary(latency_ms_list)

    # Toy mask IoU: fixed pair per epoch (deterministic)
    toy_pred = [[1, 0], [0, 1]]
    toy_gt = [[1, 1], [0, 0]]
    iou_val = mask_iou(toy_pred, toy_gt)

    cls_summary: Dict[str, Any] = {
        "accuracy": cls_result["accuracy"],
        "macro_f1": cls_result["macro_f1"],
        "per_class_f1": cls_result["per_class_f1"],
    }
    fam_summary: Dict[str, Any] = {
        "family_accuracy": fam_result["accuracy"],
        "n_evaluated": fam_result["n_evaluated"],
    }
    loc_summary: Dict[str, Any] = {
        "localization_activation_recall": loc_result["activation_recall"],
        "n_tampered": loc_result["n_tampered"],
        "tampered_mask_iou": iou_val,
    }

    return DryRunEpochSummary(
        epoch=epoch,
        n_batches=n_batches,
        n_samples=n_samples,
        fake_inference_count=n_samples,
        classification_metrics=cls_summary,
        family_metrics=fam_summary,
        localization_metrics=loc_summary,
        latency_metrics=lat_result,
    )


# ---------------------------------------------------------------------------
# Top-level dry-run training
# ---------------------------------------------------------------------------

_N_BATCHES_PER_EPOCH = 4


def run_dry_training(config: DryRunTrainingConfig) -> DryRunTrainingResult:
    """Run the full dry-run training simulation.

    Simulates config.epochs epochs with a fixed number of fake batches each.
    No real training, no file writes, no real data access.
    """
    epoch_summaries: List[DryRunEpochSummary] = []
    for epoch in range(1, config.epochs + 1):
        summary = run_dry_epoch(config, epoch, _N_BATCHES_PER_EPOCH)
        epoch_summaries.append(summary)

    final_metrics = summarize_dry_run(epoch_summaries, config)

    return DryRunTrainingResult(
        config_stage=config.stage,
        config_seed=config.seed,
        config_epochs=config.epochs,
        config_batch_size=config.batch_size,
        config_family_policy=config.family_policy,
        config_threshold_tau=config.threshold_tau,
        metric_names=list(config.metric_names),
        epoch_summaries=epoch_summaries,
        final_metrics=final_metrics,
        dry_run=True,
        no_training=True,
        no_outputs=True,
        no_checkpoints=True,
    )


def summarize_dry_run(
    epoch_summaries: List[DryRunEpochSummary],
    config: DryRunTrainingConfig,
) -> Dict[str, Any]:
    """Aggregate epoch summaries into a final metrics dict keyed by metric name.

    Averages numeric metrics across epochs. Returns all metric_names from config.
    """
    if not epoch_summaries:
        return {"_dry_run": True, "_no_real_training": True, "_epochs": 0}

    n = len(epoch_summaries)
    avg_accuracy = sum(e.classification_metrics["accuracy"] for e in epoch_summaries) / n
    avg_macro_f1 = sum(e.classification_metrics["macro_f1"] for e in epoch_summaries) / n
    avg_family_acc = sum(e.family_metrics["family_accuracy"] for e in epoch_summaries) / n
    avg_loc_recall = sum(
        e.localization_metrics["localization_activation_recall"]
        for e in epoch_summaries
    ) / n
    avg_iou = sum(e.localization_metrics["tampered_mask_iou"] for e in epoch_summaries) / n
    avg_latency = sum(e.latency_metrics["mean_ms"] for e in epoch_summaries) / n
    avg_fps = sum(1000.0 / e.latency_metrics["mean_ms"] for e in epoch_summaries) / n

    _metric_values: Dict[str, Any] = {
        "accuracy_3way": avg_accuracy,
        "macro_f1": avg_macro_f1,
        "generator_family_accuracy": avg_family_acc,
        "localization_activation_recall": avg_loc_recall,
        "tampered_mask_iou": avg_iou,
        "latency_ms": avg_latency,
        "fps": avg_fps,
        "perturbation_robustness_drop": None,
    }

    summary: Dict[str, Any] = {}
    for metric_name in config.metric_names:
        summary[metric_name] = _metric_values.get(metric_name)

    summary["_dry_run"] = True
    summary["_no_real_training"] = True
    summary["_no_real_data"] = True
    summary["_epochs"] = n
    summary["_n_samples_total"] = sum(e.n_samples for e in epoch_summaries)
    return summary


def dry_run_result_to_dict(result: DryRunTrainingResult) -> Dict[str, Any]:
    """Convert a DryRunTrainingResult to a JSON-serializable dict."""
    return dataclasses.asdict(result)
