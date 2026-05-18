"""Local data readiness gate for task 0011.

Validates symbolic readiness configuration only. Does not inspect dataset
directories, download data, train models, write outputs, or write checkpoints.
"""
from __future__ import annotations

import dataclasses
import json
import re as _re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Safety constants (mirrors training_dry_run safety approach)
# ---------------------------------------------------------------------------

_PROTECTED_DIRS = frozenset({"secrets", "data", "datasets", "outputs", "checkpoints"})
_PROTECTED_URL_PREFIXES = ("http://", "https://", "ftp://", "s3://", "gs://", "hf://")
_PROTECTED_ABS_PREFIXES = ("/home/", "/mnt/", "/root/", "/users/")
# Token-based keyword sets: use normalized tokens so that 'authentication' or
# 'authoritative' do NOT match, but standalone 'auth' does.
_SECRET_TOKEN_KEYWORDS = frozenset(
    {"token", "password", "secret", "credential", "auth", "bearer"}
)
_SECRET_KEY_KEYWORDS = frozenset(
    {"token", "api_key", "password", "secret", "credential", "auth", "bearer"}
)

# protected_path_exclusions is a declarative field that names what is excluded.
# Its values (e.g. ".env", "secrets") are NOT path references, so the safety
# checker skips this top-level key.
_SKIP_SAFETY_TOP_KEYS = frozenset({"protected_path_exclusions"})


def _normalized_tokens(value: str) -> List[str]:
    """Split value into lowercase alphanumeric tokens."""
    return [t for t in _re.split(r"[^a-z0-9]+", value.lower()) if t]


def _is_unsafe_string(value: str) -> Optional[str]:
    """Return a reason string if *value* is unsafe, else None."""
    if not value:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    lower = stripped.lower()
    for prefix in _PROTECTED_URL_PREFIXES:
        if prefix in lower:
            return f"URL-like reference: {value!r}"
    for prefix in _PROTECTED_ABS_PREFIXES:
        if lower.startswith(prefix):
            return f"absolute machine path: {value!r}"
    if (
        len(stripped) >= 3
        and stripped[0].isalpha()
        and stripped[1] == ":"
        and stripped[2] in ("\\", "/")
    ):
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
    """Return True if *key* is a secret-looking name."""
    lower = key.lower()
    if lower in _SECRET_KEY_KEYWORDS:
        return True
    tokens = _normalized_tokens(lower)
    if any(t in _SECRET_TOKEN_KEYWORDS for t in tokens):
        return True
    return "api" in tokens and "key" in tokens


def _check_safety_node(node: Any, path: str) -> None:
    """Recursively walk a node and raise ValueError on the first unsafe finding."""
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


def check_local_data_readiness_config_safety(
    raw_config: Dict[str, Any],
    context: str = "local_data_readiness_config",
) -> None:
    """Recursively validate a readiness config for unsafe references.

    Raises ValueError with an actionable message if any protected path, URL,
    secret-looking key, or secret-looking value is found.

    The 'protected_path_exclusions' top-level field is skipped because it is
    declarative (it names what is excluded), not a real path reference.
    """
    if not isinstance(raw_config, dict):
        _check_safety_node(raw_config, context)
        return
    for k, v in raw_config.items():
        if k in _SKIP_SAFETY_TOP_KEYS:
            continue
        child = f"{context}.{k}"
        if isinstance(k, str) and _has_secret_key(k):
            raise ValueError(f"{child}: secret-looking key name {k!r}")
        _check_safety_node(v, child)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ReadinessIssue:
    field: str
    message: str
    severity: str  # "error" or "warning"


@dataclasses.dataclass
class LocalDataReadinessConfig:
    schema_version: str
    dry_run: bool
    no_download: bool
    no_training: bool
    no_network: bool
    no_outputs: bool
    no_checkpoints: bool
    dataset_plan: Any
    manifest_refs: List[str]
    local_path_policy: Any
    output_policy: Any
    checkpoint_policy: Any
    approval: Any
    protected_path_exclusions: List[str]


@dataclasses.dataclass
class LocalDataReadinessResult:
    ready: bool
    issues: List[ReadinessIssue]
    dry_run_safe: bool
    has_approval_field: bool
    has_protected_path_exclusions: bool
    summary: str


# ---------------------------------------------------------------------------
# Config loading and validation
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = (
    "schema_version",
    "dry_run",
    "no_download",
    "no_training",
    "no_network",
    "no_outputs",
    "no_checkpoints",
    "dataset_plan",
    "manifest_refs",
    "local_path_policy",
    "output_policy",
    "checkpoint_policy",
    "approval",
    "protected_path_exclusions",
)

_GUARDRAIL_FLAGS = (
    "dry_run",
    "no_download",
    "no_training",
    "no_network",
    "no_outputs",
    "no_checkpoints",
)


def load_local_data_readiness_config(path: str) -> LocalDataReadinessConfig:
    """Load and validate a readiness config JSON from *path*.

    Raises FileNotFoundError, json.JSONDecodeError, or ValueError.
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    check_local_data_readiness_config_safety(raw)
    return validate_local_data_readiness_config(raw)


def validate_local_data_readiness_config(
    raw: Dict[str, Any],
) -> LocalDataReadinessConfig:
    """Validate a raw config dict and return a LocalDataReadinessConfig.

    Raises ValueError with actionable messages on any violation.
    Does NOT re-run the safety check; call check_local_data_readiness_config_safety
    separately if the raw dict has not been safety-checked yet.
    """
    errors: List[str] = []

    for key in _REQUIRED_KEYS:
        if key not in raw:
            errors.append(f"Missing required key: {key!r}.")

    for flag in _GUARDRAIL_FLAGS:
        val = raw.get(flag)
        if val is not True:
            errors.append(f"{flag!r} must be true (got {val!r}).")

    if errors:
        raise ValueError(
            "LocalDataReadinessConfig validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return LocalDataReadinessConfig(
        schema_version=str(raw["schema_version"]),
        dry_run=True,
        no_download=True,
        no_training=True,
        no_network=True,
        no_outputs=True,
        no_checkpoints=True,
        dataset_plan=raw["dataset_plan"],
        manifest_refs=list(raw.get("manifest_refs", [])),
        local_path_policy=raw["local_path_policy"],
        output_policy=raw["output_policy"],
        checkpoint_policy=raw["checkpoint_policy"],
        approval=raw["approval"],
        protected_path_exclusions=list(raw.get("protected_path_exclusions", [])),
    )


# ---------------------------------------------------------------------------
# Readiness evaluation
# ---------------------------------------------------------------------------

def evaluate_local_data_readiness(
    config: LocalDataReadinessConfig,
) -> LocalDataReadinessResult:
    """Evaluate readiness state from a validated config.

    Deterministic: same config always produces the same result.
    No file I/O, no training, no dataset inspection, no outputs written.
    """
    issues: List[ReadinessIssue] = []

    # Guardrail flag checks
    for flag in _GUARDRAIL_FLAGS:
        if not getattr(config, flag, False):
            issues.append(ReadinessIssue(
                field=flag,
                message=f"Guardrail flag {flag!r} must be true.",
                severity="error",
            ))

    dry_run_safe = all(getattr(config, f, False) for f in _GUARDRAIL_FLAGS)

    # Approval field
    has_approval_field = bool(config.approval)
    if not has_approval_field:
        issues.append(ReadinessIssue(
            field="approval",
            message="approval field is missing or empty.",
            severity="error",
        ))

    # Protected path exclusions
    has_protected_path_exclusions = bool(config.protected_path_exclusions)
    if not has_protected_path_exclusions:
        issues.append(ReadinessIssue(
            field="protected_path_exclusions",
            message="protected_path_exclusions must be present and non-empty.",
            severity="error",
        ))

    # Manifest refs
    if not config.manifest_refs:
        issues.append(ReadinessIssue(
            field="manifest_refs",
            message="manifest_refs is empty; at least one manifest reference is recommended.",
            severity="warning",
        ))

    # Real local path policy: if policy indicates real paths, approval must be granted
    if isinstance(config.local_path_policy, dict):
        policy_val = str(config.local_path_policy.get("policy", ""))
        uses_real_paths = policy_val not in ("symbolic_only", "no_paths", "", "none")
        if uses_real_paths:
            approval_granted = (
                isinstance(config.approval, dict)
                and config.approval.get("local_data_approved") is True
            )
            if not approval_granted:
                issues.append(ReadinessIssue(
                    field="local_path_policy",
                    message=(
                        f"local_path_policy.policy={policy_val!r} indicates real paths "
                        "but approval.local_data_approved is not true. "
                        "Explicit user approval is required before real data access."
                    ),
                    severity="error",
                ))

    ready = not any(i.severity == "error" for i in issues)
    summary = _build_summary(ready, issues)

    return LocalDataReadinessResult(
        ready=ready,
        issues=issues,
        dry_run_safe=dry_run_safe,
        has_approval_field=has_approval_field,
        has_protected_path_exclusions=has_protected_path_exclusions,
        summary=summary,
    )


def summarize_local_data_readiness(result: LocalDataReadinessResult) -> str:
    """Return a human-readable summary string for *result*."""
    return result.summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_summary(ready: bool, issues: List[ReadinessIssue]) -> str:
    if ready:
        warnings = [i.message for i in issues if i.severity == "warning"]
        if warnings:
            return (
                "LOCAL_DATA_READINESS_OK: Symbolic readiness config is valid and "
                "dry-run safe. Warnings: " + "; ".join(warnings)
            )
        return (
            "LOCAL_DATA_READINESS_OK: Symbolic readiness config is valid and "
            "dry-run safe. No real datasets, training, outputs, or checkpoints."
        )
    errors = [i.message for i in issues if i.severity == "error"]
    return "LOCAL_DATA_READINESS_FAIL: " + "; ".join(errors)
