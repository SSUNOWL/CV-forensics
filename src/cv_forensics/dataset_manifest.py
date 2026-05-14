"""Dataset manifest schema constants and validation helpers.

Manifests are planning and validation artifacts only. They must not point to
real data, trigger downloads, or require actual dataset presence on disk.
"""

import json
import re


# Constants

KNOWN_DATASET_IDS = frozenset({
    "community_forensics_small",
    "sid_set",
    "combined_smoke",
})

CLASS_LABELS = frozenset({"real", "synthetic", "tampered"})

FAMILY_LABELS = frozenset({"LatDiff", "PixDiff", "GAN", "Other", "Real-or-N/A"})

MANIFEST_KINDS = frozenset({"dataset_manifest", "combined_manifest"})

SPLIT_ROLES = frozenset({
    "train",
    "validation",
    "eval",
    "generator_holdout",
    "model_name_holdout",
    "compeval_eval",
    "smoke",
})

DATASET_TASKS = frozenset({
    "binary_real_fake",
    "coarse_provenance",
    "three_way_classification",
    "tampered_mask_localization",
    "social_media_robustness",
})

FAMILY_SUPERVISION_POLICIES = frozenset({
    "available",
    "unavailable",
    "freeze",
    "mask_out",
    "mixed_cf_small_batch",
})

PERTURBATIONS = frozenset({
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

STAGES = frozenset({
    "cf_small_binary_backbone",
    "provenance_head",
    "sid_set_multihead_finetune",
    "social_media_robustness",
    "dry_run",
})


# Guardrail patterns

_PROTECTED_PATH_PATTERNS = (
    "/home/",
    "/mnt/",
    "/root/",
    "/users/",
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
)

_URL_PREFIXES = (
    "http://",
    "https://",
    "s3://",
    "gs://",
    "hf://",
    "ftp://",
)

_SECRET_KEY_PATTERN = re.compile(
    r"(token|api_key|password|secret|credential|passwd|auth_key)",
    re.IGNORECASE,
)

# Matches long base64-like opaque strings (potential embedded tokens/keys).
_SECRET_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9+/]{20,}={0,2}$")

# Matches a standalone placeholder token such as <CF_SMALL_ROOT>.
_PLACEHOLDER_PATTERN = re.compile(r"^<[A-Z0-9_]+>$")

# Matches Windows absolute paths at the start of a string.
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:[/\\]")


def _is_placeholder(value: str) -> bool:
    return bool(_PLACEHOLDER_PATTERN.match(value))


# JSON loading

def load_manifest(path: str) -> dict:
    """Load and parse a JSON manifest file. Raises on missing file or bad JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# String/key collectors

def _iter_string_leaves(obj, _parent_key=""):
    """Yield (parent_key, value) for every string leaf in a nested dict/list."""
    if isinstance(obj, str):
        yield _parent_key, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_string_leaves(v, _parent_key=k)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_string_leaves(item, _parent_key=_parent_key)


def _iter_all_keys(obj):
    """Yield every dict key in a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _iter_all_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_all_keys(item)


# Individual safety checks

def check_protected_paths(manifest: dict) -> list:
    """Return errors for any protected or absolute local path patterns in values."""
    errors = []
    for _, val in _iter_string_leaves(manifest):
        if _is_placeholder(val):
            continue
        lower = val.lower()
        flagged = False
        for pat in _PROTECTED_PATH_PATTERNS:
            if pat in lower:
                errors.append(f"Protected path detected in value: {val!r}")
                flagged = True
                break
        if not flagged and _WINDOWS_DRIVE_PATTERN.match(val):
            errors.append(f"Absolute Windows path detected in value: {val!r}")
    return errors


def check_urls(manifest: dict) -> list:
    """Return errors for URL values found in the manifest."""
    errors = []
    for _, val in _iter_string_leaves(manifest):
        lower = val.lower()
        for prefix in _URL_PREFIXES:
            if prefix in lower:
                errors.append(f"URL detected in value: {val!r}")
                break
    return errors


def check_secret_keys(manifest: dict) -> list:
    """Return errors for secret-looking key names."""
    errors = []
    for key in _iter_all_keys(manifest):
        if _SECRET_KEY_PATTERN.search(str(key)):
            errors.append(f"Secret-looking key name detected: {key!r}")
    return errors


def check_secret_values(manifest: dict) -> list:
    """Return errors for secret-looking values (long opaque base64-like strings)."""
    errors = []
    for _, val in _iter_string_leaves(manifest):
        if _is_placeholder(val):
            continue
        if len(val) >= 20 and _SECRET_VALUE_PATTERN.match(val):
            errors.append(f"Secret-looking value detected: {val!r}")
    return errors


def check_guardrail_flags(manifest: dict) -> list:
    """Return errors if any of dry_run/no_download/no_training/no_network are false."""
    errors = []
    for flag in ("dry_run", "no_download", "no_training", "no_network"):
        val = manifest.get(flag)
        if val is False:
            errors.append(f"Guardrail flag {flag!r} must be true but is false.")
        elif flag in manifest and val is not True:
            errors.append(
                f"Guardrail flag {flag!r} must be boolean true, got {val!r}."
            )
    return errors


# CF-Small-specific validation

_CF_SMALL_REQUIRED_METADATA = ("architecture", "model_name", "subset")


def _validate_cf_small(manifest: dict) -> list:
    errors = []
    fields = manifest.get("expected_metadata_fields", [])
    for f in _CF_SMALL_REQUIRED_METADATA:
        if f not in fields:
            errors.append(
                f"CF-Small manifest missing expected_metadata_field: {f!r}"
            )
    family_labels = manifest.get("family_labels", [])
    if family_labels:
        unknown = set(family_labels) - FAMILY_LABELS
        if unknown:
            errors.append(f"Unknown family labels in CF-Small manifest: {sorted(unknown)}")
    class_labels = manifest.get("class_labels", [])
    if class_labels:
        unknown = set(class_labels) - CLASS_LABELS
        if unknown:
            errors.append(f"Unknown class labels in CF-Small manifest: {sorted(unknown)}")
    split_plan = manifest.get("split_plan", {})
    if isinstance(split_plan, dict):
        for role in split_plan:
            if role not in SPLIT_ROLES:
                errors.append(f"Unknown split role in CF-Small split_plan: {role!r}")
    tasks = manifest.get("tasks", [])
    for task in tasks:
        if task not in DATASET_TASKS:
            errors.append(f"Unknown task in CF-Small manifest: {task!r}")
    perturbation_notes = manifest.get("perturbation_notes", [])
    if isinstance(perturbation_notes, list):
        for p in perturbation_notes:
            if p not in PERTURBATIONS:
                errors.append(
                    f"Unknown perturbation in CF-Small perturbation_notes: {p!r}"
                )
    return errors


# SID-Set-specific validation

_SID_SET_REQUIRED_TARGET_FIELDS = (
    "image_ref",
    "class_label",
    "tampered_mask_ref",
    "split",
)


def _validate_sid_set(manifest: dict) -> list:
    errors = []
    target_fields = manifest.get("target_fields", [])
    for f in _SID_SET_REQUIRED_TARGET_FIELDS:
        if f not in target_fields:
            errors.append(f"SID-Set manifest missing target_field: {f!r}")
    class_labels = manifest.get("class_labels", [])
    if class_labels:
        unknown = set(class_labels) - CLASS_LABELS
        if unknown:
            errors.append(f"Unknown class labels in SID-Set manifest: {sorted(unknown)}")
    tasks = manifest.get("tasks", [])
    for task in tasks:
        if task not in DATASET_TASKS:
            errors.append(f"Unknown task in SID-Set manifest: {task!r}")
    family_sup = manifest.get("family_supervision", {})
    if isinstance(family_sup, dict) and family_sup:
        status = family_sup.get("status")
        if status is not None and status not in FAMILY_SUPERVISION_POLICIES:
            errors.append(f"Unknown family_supervision status: {status!r}")
        for p in family_sup.get("allowed_policies", []):
            if p not in FAMILY_SUPERVISION_POLICIES:
                errors.append(f"Unknown family_supervision policy: {p!r}")
    # Reject CF-Small-compatible family labels without an explicit supervision policy.
    family_labels = manifest.get("family_labels", [])
    if family_labels:
        cf_compatible = set(family_labels) & FAMILY_LABELS
        if cf_compatible and not family_sup:
            errors.append(
                "SID-Set manifest claims CF-Small compatible family labels "
                "but has no explicit family_supervision policy."
            )
    return errors


# Dataset manifest validation

_DATASET_MANIFEST_REQUIRED = (
    "schema_version",
    "manifest_kind",
    "dataset_id",
    "dry_run",
    "no_download",
    "no_training",
    "no_network",
    "root_placeholder",
)


def validate_dataset_manifest(manifest: dict) -> list:
    """Validate a dataset_manifest kind manifest. Returns list of error strings."""
    errors = []
    for key in _DATASET_MANIFEST_REQUIRED:
        if key not in manifest:
            errors.append(f"Dataset manifest missing required key: {key!r}")
    kind = manifest.get("manifest_kind")
    if kind is not None and kind not in MANIFEST_KINDS:
        errors.append(f"Unknown manifest_kind: {kind!r}")
    elif kind is not None and kind != "dataset_manifest":
        errors.append(f"Expected manifest_kind 'dataset_manifest', got {kind!r}")
    dataset_id = manifest.get("dataset_id")
    if dataset_id is not None and dataset_id not in KNOWN_DATASET_IDS:
        errors.append(f"Unknown dataset_id: {dataset_id!r}")
    elif dataset_id == "combined_smoke":
        errors.append(
            "dataset_id 'combined_smoke' is only valid with manifest_kind 'combined_manifest'."
        )
    errors.extend(check_guardrail_flags(manifest))
    errors.extend(check_protected_paths(manifest))
    errors.extend(check_urls(manifest))
    errors.extend(check_secret_keys(manifest))
    errors.extend(check_secret_values(manifest))
    if dataset_id == "community_forensics_small":
        errors.extend(_validate_cf_small(manifest))
    elif dataset_id == "sid_set":
        errors.extend(_validate_sid_set(manifest))
    return errors


# Combined manifest validation

_COMBINED_MANIFEST_REQUIRED = (
    "schema_version",
    "manifest_kind",
    "dataset_id",
    "dry_run",
    "no_download",
    "no_training",
    "no_network",
    "root_placeholder",
    "referenced_datasets",
)


def validate_combined_manifest(manifest: dict, known_ids: set = None) -> list:
    """Validate a combined_manifest kind manifest. Returns list of error strings."""
    errors = []
    if known_ids is None:
        known_ids = KNOWN_DATASET_IDS
    for key in _COMBINED_MANIFEST_REQUIRED:
        if key not in manifest:
            errors.append(f"Combined manifest missing required key: {key!r}")
    kind = manifest.get("manifest_kind")
    if kind is not None and kind not in MANIFEST_KINDS:
        errors.append(f"Unknown manifest_kind: {kind!r}")
    elif kind is not None and kind != "combined_manifest":
        errors.append(f"Expected manifest_kind 'combined_manifest', got {kind!r}")
    dataset_id = manifest.get("dataset_id")
    if dataset_id is not None and dataset_id not in KNOWN_DATASET_IDS:
        errors.append(f"Unknown dataset_id: {dataset_id!r}")
    for ref_id in manifest.get("referenced_datasets", []):
        if ref_id not in known_ids:
            errors.append(
                f"Combined manifest references unknown dataset id: {ref_id!r}"
            )
    for p in manifest.get("perturbations", []):
        if p not in PERTURBATIONS:
            errors.append(f"Unknown perturbation: {p!r}")
    for role in manifest.get("validation_strategy", []):
        if role not in SPLIT_ROLES:
            errors.append(f"Unknown validation_strategy role: {role!r}")
    for p in manifest.get("sid_family_policy", []):
        if p not in FAMILY_SUPERVISION_POLICIES:
            errors.append(f"Unknown sid_family_policy entry: {p!r}")
    errors.extend(check_guardrail_flags(manifest))
    errors.extend(check_protected_paths(manifest))
    errors.extend(check_urls(manifest))
    errors.extend(check_secret_keys(manifest))
    errors.extend(check_secret_values(manifest))
    return errors


# Dispatch

def validate_manifest(manifest: dict, known_ids: set = None) -> list:
    """Dispatch to the appropriate validator based on manifest_kind."""
    kind = manifest.get("manifest_kind")
    if kind == "combined_manifest":
        return validate_combined_manifest(manifest, known_ids=known_ids)
    elif kind == "dataset_manifest":
        return validate_dataset_manifest(manifest)
    else:
        return [f"Unknown or missing manifest_kind: {kind!r}"]


def validate_manifests(manifests: list) -> dict:
    """Validate a list of manifest dicts with cross-manifest constraint checks.

    Returns dict mapping list index to list of error strings.
    Cross-manifest checks: duplicate dataset_ids, combined manifest references.
    """
    results = {i: [] for i in range(len(manifests))}
    seen_dataset_ids: dict = {}
    known_ids: set = set(KNOWN_DATASET_IDS)

    for i, manifest in enumerate(manifests):
        dataset_id = manifest.get("dataset_id")
        kind = manifest.get("manifest_kind")
        errors = validate_manifest(manifest, known_ids=known_ids)
        results[i].extend(errors)
        if kind == "dataset_manifest" and dataset_id:
            if dataset_id in seen_dataset_ids:
                results[i].append(
                    f"Duplicate dataset_id {dataset_id!r} "
                    f"(already seen at index {seen_dataset_ids[dataset_id]})."
                )
            else:
                seen_dataset_ids[dataset_id] = i

    return results
