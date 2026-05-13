"""pytest-style tests for config_schema validation helpers."""

import os
import sys

# Path setup
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_TESTS_DIR, ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")

for _p in (_REPO_ROOT, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cv_forensics.config_schema import (  # noqa: E402
    CLASS_LABELS,
    FAMILY_LABELS,
    DATASET_IDS,
    STAGES,
    METRICS,
    validate_datasets_config,
    validate_experiment_config,
    extract_dataset_ids,
    contains_protected_path,
    is_dry_run_safe,
)


# Constants

def test_class_labels():
    assert set(CLASS_LABELS) == {"real", "synthetic", "tampered"}


def test_family_labels():
    assert set(FAMILY_LABELS) == {"LatDiff", "PixDiff", "GAN", "Other", "Real-or-N/A"}


def test_dataset_ids():
    assert "community_forensics_small" in DATASET_IDS
    assert "sid_set" in DATASET_IDS


def test_stages_include_dry_run():
    assert "dry_run" in STAGES


def test_metrics_complete():
    expected = {
        "accuracy_3way",
        "macro_f1",
        "tampered_mask_iou",
        "generator_family_accuracy",
        "perturbation_robustness_drop",
        "latency_ms",
        "fps",
        "localization_activation_recall",
    }
    assert expected.issubset(set(METRICS))


# Protected path detection

def test_protected_path_home():
    assert contains_protected_path("/home/user/data")


def test_protected_path_mnt():
    assert contains_protected_path("/mnt/nvme/datasets")


def test_protected_path_env():
    assert contains_protected_path(".env")


def test_protected_path_secrets():
    assert contains_protected_path("secrets/api_key.txt")


def test_protected_path_http():
    assert contains_protected_path("http://example.com/data.zip")


def test_protected_path_placeholder_ok():
    assert not contains_protected_path("<CF_SMALL_ROOT>/train")


# Datasets config validation

def _valid_datasets_config():
    return {
        "datasets": [
            {
                "id": "community_forensics_small",
                "display_name": "Community Forensics-Small",
                "primary_roles": ["backbone training"],
                "root_placeholder": "<CF_SMALL_ROOT>",
            },
            {
                "id": "sid_set",
                "display_name": "SID-Set",
                "primary_roles": ["3-way classification"],
                "root_placeholder": "<SID_SET_ROOT>",
            },
        ]
    }


def test_valid_datasets_config_passes():
    errors = validate_datasets_config(_valid_datasets_config())
    assert errors == [], f"Unexpected errors: {errors}"


def test_datasets_config_missing_key():
    cfg = {"datasets": [{"id": "community_forensics_small", "display_name": "CF"}]}
    errors = validate_datasets_config(cfg)
    assert any("primary_roles" in e or "root_placeholder" in e for e in errors)


def test_datasets_config_unknown_id():
    cfg = {
        "datasets": [
            {
                "id": "unknown_dataset",
                "display_name": "Unknown",
                "primary_roles": [],
                "root_placeholder": "<X>",
            }
        ]
    }
    errors = validate_datasets_config(cfg)
    assert any("unknown_dataset" in e for e in errors)


def test_datasets_config_protected_path_rejected():
    cfg = {
        "datasets": [
            {
                "id": "community_forensics_small",
                "display_name": "CF",
                "primary_roles": [],
                "root_placeholder": "/home/user/data/cf_small",
            }
        ]
    }
    errors = validate_datasets_config(cfg)
    assert any("Protected path" in e for e in errors)


def test_datasets_config_family_labels_rejected_when_wrong():
    cfg = _valid_datasets_config()
    cfg["datasets"][0]["supported_labels"] = {"family": ["GAN"]}
    errors = validate_datasets_config(cfg)
    assert any("family labels" in e for e in errors)


def test_datasets_config_duplicate_id():
    cfg = {
        "datasets": [
            {
                "id": "community_forensics_small",
                "display_name": "CF",
                "primary_roles": [],
                "root_placeholder": "<CF>",
            },
            {
                "id": "community_forensics_small",
                "display_name": "CF2",
                "primary_roles": [],
                "root_placeholder": "<CF2>",
            },
        ]
    }
    errors = validate_datasets_config(cfg)
    assert any("Duplicate" in e for e in errors)


# Experiment config validation

def _valid_experiment_config():
    return {
        "schema_version": "0.1",
        "experiment_id": "test_exp",
        "stage": "dry_run",
        "dry_run": True,
        "seed": 42,
        "dataset_refs": [{"id": "community_forensics_small"}],
        "metrics": ["accuracy_3way", "macro_f1"],
    }


def _known_ids():
    return list(DATASET_IDS)


def test_valid_experiment_config_passes():
    errors = validate_experiment_config(_valid_experiment_config(), _known_ids())
    assert errors == [], f"Unexpected errors: {errors}"


def test_experiment_config_missing_dry_run():
    cfg = {
        "schema_version": "0.1",
        "experiment_id": "exp",
        "stage": "dry_run",
        "seed": 0,
    }
    errors = validate_experiment_config(cfg)
    assert any("dry_run" in e for e in errors)


def test_experiment_config_dry_run_false_rejected():
    cfg = {**_valid_experiment_config(), "dry_run": False}
    errors = validate_experiment_config(cfg, _known_ids())
    assert any("dry_run" in e for e in errors)


def test_experiment_config_unknown_dataset_ref():
    cfg = {**_valid_experiment_config(), "dataset_refs": [{"id": "nonexistent"}]}
    errors = validate_experiment_config(cfg, _known_ids())
    assert any("nonexistent" in e for e in errors)


def test_experiment_config_unknown_stage():
    cfg = {**_valid_experiment_config(), "stage": "nonexistent_stage"}
    errors = validate_experiment_config(cfg, _known_ids())
    assert any("stage" in e.lower() or "nonexistent_stage" in e for e in errors)


def test_experiment_config_unknown_metric():
    cfg = {**_valid_experiment_config(), "metrics": ["not_a_metric"]}
    errors = validate_experiment_config(cfg, _known_ids())
    assert any("not_a_metric" in e for e in errors)


def test_experiment_config_missing_metrics():
    cfg = _valid_experiment_config()
    del cfg["metrics"]
    errors = validate_experiment_config(cfg, _known_ids())
    assert any("metrics" in e for e in errors)


def test_experiment_config_head_labels_rejected_when_wrong():
    cfg = {
        **_valid_experiment_config(),
        "heads": {"classification": {"labels": ["real", "fake"]}},
    }
    errors = validate_experiment_config(cfg, _known_ids())
    assert any("Classification head labels" in e for e in errors)


def test_experiment_config_protected_path_rejected():
    cfg = {**_valid_experiment_config(), "output_dir": "/home/user/outputs"}
    errors = validate_experiment_config(cfg, _known_ids())
    assert any("Protected path" in e for e in errors)


# Dry-run safety

def test_is_dry_run_safe_true():
    assert is_dry_run_safe({"dry_run": True})


def test_is_dry_run_safe_false():
    assert not is_dry_run_safe({"dry_run": False})


def test_is_dry_run_safe_missing():
    assert not is_dry_run_safe({})


# Example JSON files

def test_example_datasets_json():
    import json
    path = os.path.join(_REPO_ROOT, "configs", "datasets.example.json")
    assert os.path.isfile(path), f"Missing {path}"
    with open(path) as f:
        cfg = json.load(f)
    errors = validate_datasets_config(cfg)
    assert errors == [], f"Example datasets.json has errors: {errors}"


def test_example_smoke_experiment_json():
    import json
    path = os.path.join(_REPO_ROOT, "configs", "experiments", "smoke_baseline.json")
    assert os.path.isfile(path), f"Missing {path}"
    with open(path) as f:
        cfg = json.load(f)

    ds_path = os.path.join(_REPO_ROOT, "configs", "datasets.example.json")
    with open(ds_path) as f:
        ds_cfg = json.load(f)
    known_ids = extract_dataset_ids(ds_cfg)

    errors = validate_experiment_config(cfg, known_ids)
    assert errors == [], f"Example smoke_baseline.json has errors: {errors}"


def test_smoke_experiment_is_dry_run():
    import json
    path = os.path.join(_REPO_ROOT, "configs", "experiments", "smoke_baseline.json")
    with open(path) as f:
        cfg = json.load(f)
    assert is_dry_run_safe(cfg), "smoke_baseline.json must have dry_run=true"


def test_smoke_experiment_metrics_present():
    import json
    path = os.path.join(_REPO_ROOT, "configs", "experiments", "smoke_baseline.json")
    with open(path) as f:
        cfg = json.load(f)
    listed = set(cfg.get("metrics", []))
    required = {"accuracy_3way", "macro_f1", "tampered_mask_iou"}
    missing = required - listed
    assert not missing, f"smoke_baseline.json missing metrics: {missing}"


if __name__ == "__main__":
    import traceback
    failed = 0
    tests = {k: v for k, v in globals().items() if k.startswith("test_") and callable(v)}
    for name, fn in tests.items():
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    if failed:
        print(f"\n{failed}/{len(tests)} tests failed.")
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} tests passed.")
