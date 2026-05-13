"""Config schema constants and validation helpers for datasets and experiments."""

# Label constants

CLASS_LABELS = ("real", "synthetic", "tampered")

FAMILY_LABELS = ("LatDiff", "PixDiff", "GAN", "Other", "Real-or-N/A")

DATASET_IDS = ("community_forensics_small", "sid_set")

STAGES = (
    "cf_small_binary_backbone",
    "provenance_head",
    "sid_set_multihead_finetune",
    "social_media_robustness",
    "dry_run",
)

METRICS = (
    "accuracy_3way",
    "macro_f1",
    "tampered_mask_iou",
    "generator_family_accuracy",
    "perturbation_robustness_drop",
    "latency_ms",
    "fps",
    "localization_activation_recall",
)

# Patterns that must not appear in any path value inside a config.
_PROTECTED_PATH_PATTERNS = (
    "/home/",
    "/mnt/",
    "/root/",
    "/data/",
    "/datasets/",
    "/outputs/",
    "/checkpoints/",
    "data/",
    "datasets/",
    "outputs/",
    "checkpoints/",
    ".env",
    "secrets",
    "http://",
    "https://",
    "ftp://",
    "s3://",
    "gs://",
)


# Helpers

def _collect_string_values(obj):
    """Recursively yield all string leaf values from a nested dict/list."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _collect_string_values(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _collect_string_values(item)


def contains_protected_path(value: str) -> bool:
    """Return True if *value* matches any protected path pattern."""
    lowered = value.lower()
    for pat in _PROTECTED_PATH_PATTERNS:
        if pat.lower() in lowered:
            return True
    return False


def check_protected_paths(config: dict) -> list:
    """Return list of error strings for any protected paths found in *config*."""
    errors = []
    for s in _collect_string_values(config):
        if contains_protected_path(s):
            errors.append(f"Protected path detected in config value: {s!r}")
    return errors


# Dataset config validation

_DATASET_REQUIRED_KEYS = ("id", "display_name", "primary_roles", "root_placeholder")


def validate_dataset_entry(entry: dict) -> list:
    """Validate a single dataset entry dict. Returns list of error strings."""
    errors = []
    for key in _DATASET_REQUIRED_KEYS:
        if key not in entry:
            errors.append(f"Dataset entry missing required key: {key!r}")
    if "id" in entry and entry["id"] not in DATASET_IDS:
        errors.append(
            f"Unknown dataset id {entry['id']!r}. Expected one of {DATASET_IDS}."
        )
    supported_labels = entry.get("supported_labels")
    if isinstance(supported_labels, dict):
        family_labels = supported_labels.get("family")
        if family_labels is not None and tuple(family_labels) != FAMILY_LABELS:
            errors.append(
                f"Dataset family labels must be exactly {FAMILY_LABELS}."
            )
    required_targets = entry.get("required_target_fields")
    if isinstance(required_targets, dict):
        class_labels = required_targets.get("class_label")
        if class_labels is not None and tuple(class_labels) != CLASS_LABELS:
            errors.append(
                f"Dataset class labels must be exactly {CLASS_LABELS}."
            )
    errors.extend(check_protected_paths(entry))
    return errors


def validate_datasets_config(config: dict) -> list:
    """Validate a full datasets config dict. Returns list of error strings."""
    errors = []
    if "datasets" not in config:
        errors.append("Datasets config missing top-level 'datasets' key.")
        return errors

    datasets = config["datasets"]
    if not isinstance(datasets, list):
        errors.append("'datasets' must be a list.")
        return errors

    seen_ids = []
    for i, entry in enumerate(datasets):
        entry_errors = validate_dataset_entry(entry)
        errors.extend(f"[dataset #{i}] {e}" for e in entry_errors)
        if "id" in entry:
            if entry["id"] in seen_ids:
                errors.append(f"Duplicate dataset id: {entry['id']!r}")
            else:
                seen_ids.append(entry["id"])

    return errors


def extract_dataset_ids(config: dict) -> list:
    """Return list of dataset ids defined in a datasets config dict."""
    return [e["id"] for e in config.get("datasets", []) if "id" in e]


# Experiment config validation

_EXPERIMENT_REQUIRED_KEYS = (
    "schema_version",
    "experiment_id",
    "stage",
    "dry_run",
    "seed",
)


def validate_experiment_config(config: dict, known_dataset_ids: list = None) -> list:
    """Validate an experiment config dict. Returns list of error strings."""
    errors = []

    for key in _EXPERIMENT_REQUIRED_KEYS:
        if key not in config:
            errors.append(f"Experiment config missing required key: {key!r}")

    if "stage" in config and config["stage"] not in STAGES:
        errors.append(
            f"Unknown stage {config['stage']!r}. Expected one of {STAGES}."
        )

    metrics = config.get("metrics")
    if metrics is None:
        errors.append("Experiment config missing required key: 'metrics'")
    elif not isinstance(metrics, list):
        errors.append("Experiment 'metrics' must be a list.")
    else:
        for metric in metrics:
            if metric not in METRICS:
                errors.append(
                    f"Unknown metric {metric!r}. Expected one of {METRICS}."
                )

    heads = config.get("heads")
    if isinstance(heads, dict):
        classification_labels = heads.get("classification", {}).get("labels")
        if (
            classification_labels is not None
            and tuple(classification_labels) != CLASS_LABELS
        ):
            errors.append(
                f"Classification head labels must be exactly {CLASS_LABELS}."
            )
        provenance_labels = heads.get("provenance", {}).get("labels")
        if provenance_labels is not None and tuple(provenance_labels) != FAMILY_LABELS:
            errors.append(
                f"Provenance head labels must be exactly {FAMILY_LABELS}."
            )

    if "dry_run" in config and config["dry_run"] is not True:
        errors.append(
            "Experiment config has 'dry_run' set to false. "
            "Only dry_run=true is permitted in example/smoke configs."
        )

    if known_dataset_ids is not None:
        for ref in config.get("dataset_refs", []):
            if isinstance(ref, dict):
                ref_id = ref.get("id")
            else:
                ref_id = ref
            if ref_id and ref_id not in known_dataset_ids:
                errors.append(
                    f"Experiment references unknown dataset id {ref_id!r}."
                )

    errors.extend(check_protected_paths(config))
    return errors


# Dry-run safety check

def is_dry_run_safe(experiment_config: dict) -> bool:
    """Return True only if the experiment is marked dry_run=true."""
    return experiment_config.get("dry_run") is True
