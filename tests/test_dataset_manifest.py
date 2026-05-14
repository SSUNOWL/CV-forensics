"""pytest-style tests for dataset_manifest validation helpers."""

import json
import os
import sys

# Path setup
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_TESTS_DIR, ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")

for _p in (_REPO_ROOT, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cv_forensics.dataset_manifest import (  # noqa: E402
    KNOWN_DATASET_IDS,
    CLASS_LABELS,
    FAMILY_LABELS,
    MANIFEST_KINDS,
    SPLIT_ROLES,
    DATASET_TASKS,
    FAMILY_SUPERVISION_POLICIES,
    PERTURBATIONS,
    validate_dataset_manifest,
    validate_combined_manifest,
    validate_manifests,
    check_protected_paths,
    check_urls,
    check_secret_keys,
    check_guardrail_flags,
)

_MANIFESTS_DIR = os.path.join(_REPO_ROOT, "configs", "manifests")


# Constants

def test_known_dataset_ids():
    assert "community_forensics_small" in KNOWN_DATASET_IDS
    assert "sid_set" in KNOWN_DATASET_IDS
    assert "combined_smoke" in KNOWN_DATASET_IDS


def test_class_labels():
    assert CLASS_LABELS == {"real", "synthetic", "tampered"}


def test_family_labels():
    assert FAMILY_LABELS == {"LatDiff", "PixDiff", "GAN", "Other", "Real-or-N/A"}


def test_manifest_kinds():
    assert "dataset_manifest" in MANIFEST_KINDS
    assert "combined_manifest" in MANIFEST_KINDS


def test_split_roles_complete():
    expected = {
        "train", "validation", "eval",
        "generator_holdout", "model_name_holdout", "compeval_eval", "smoke",
    }
    assert expected.issubset(SPLIT_ROLES)


def test_perturbations_complete():
    expected = {
        "jpeg", "resize", "crop", "rotation", "padding", "shear",
        "screenshot", "text_overlay", "sticker_overlay", "recompression_chain",
    }
    assert expected.issubset(PERTURBATIONS)


def test_family_supervision_policies():
    expected = {"available", "unavailable", "freeze", "mask_out", "mixed_cf_small_batch"}
    assert expected.issubset(FAMILY_SUPERVISION_POLICIES)


# Helper builders

def _cf_small_manifest():
    return {
        "schema_version": "1.0",
        "manifest_kind": "dataset_manifest",
        "dataset_id": "community_forensics_small",
        "display_name": "CF-Small",
        "dry_run": True,
        "no_download": True,
        "no_training": True,
        "no_network": True,
        "root_placeholder": "<CF_SMALL_ROOT>",
        "expected_metadata_fields": ["architecture", "model_name", "subset"],
        "class_labels": ["real", "synthetic"],
        "family_labels": ["LatDiff", "PixDiff", "GAN", "Other", "Real-or-N/A"],
        "split_plan": {
            "train": "main training split",
            "generator_holdout": "held-out generator",
        },
        "tasks": ["binary_real_fake", "coarse_provenance"],
    }


def _sid_set_manifest():
    return {
        "schema_version": "1.0",
        "manifest_kind": "dataset_manifest",
        "dataset_id": "sid_set",
        "display_name": "SID-Set",
        "dry_run": True,
        "no_download": True,
        "no_training": True,
        "no_network": True,
        "root_placeholder": "<SID_SET_ROOT>",
        "class_labels": ["real", "synthetic", "tampered"],
        "target_fields": ["image_ref", "class_label", "tampered_mask_ref", "split"],
        "tasks": ["three_way_classification", "tampered_mask_localization"],
        "family_supervision": {
            "status": "unavailable",
            "allowed_policies": ["freeze", "mask_out", "mixed_cf_small_batch"],
        },
    }


def _combined_manifest():
    return {
        "schema_version": "1.0",
        "manifest_kind": "combined_manifest",
        "dataset_id": "combined_smoke",
        "dry_run": True,
        "no_download": True,
        "no_training": True,
        "no_network": True,
        "root_placeholder": "<MANIFEST_ONLY_NO_DATA>",
        "referenced_datasets": ["community_forensics_small", "sid_set"],
        "perturbations": ["jpeg", "resize"],
        "validation_strategy": ["generator_holdout"],
        "sid_family_policy": ["freeze"],
    }


# Valid manifests pass

def test_valid_cf_small_passes():
    errors = validate_dataset_manifest(_cf_small_manifest())
    assert errors == [], f"Unexpected errors: {errors}"


def test_valid_sid_set_passes():
    errors = validate_dataset_manifest(_sid_set_manifest())
    assert errors == [], f"Unexpected errors: {errors}"


def test_valid_combined_manifest_passes():
    errors = validate_combined_manifest(_combined_manifest())
    assert errors == [], f"Unexpected errors: {errors}"


# Example JSON files pass

def _load_example(filename):
    path = os.path.join(_MANIFESTS_DIR, filename)
    assert os.path.isfile(path), f"Missing example manifest: {path}"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_example_cf_small_manifest_passes():
    manifest = _load_example("community_forensics_small.example.json")
    errors = validate_dataset_manifest(manifest)
    assert errors == [], f"Example CF-Small manifest errors: {errors}"


def test_example_sid_set_manifest_passes():
    manifest = _load_example("sid_set.example.json")
    errors = validate_dataset_manifest(manifest)
    assert errors == [], f"Example SID-Set manifest errors: {errors}"


def test_example_combined_manifest_passes():
    manifest = _load_example("combined_smoke_manifest.example.json")
    errors = validate_combined_manifest(manifest)
    assert errors == [], f"Example combined manifest errors: {errors}"


def test_all_three_example_manifests_pass_together():
    manifests = [
        _load_example("community_forensics_small.example.json"),
        _load_example("sid_set.example.json"),
        _load_example("combined_smoke_manifest.example.json"),
    ]
    results = validate_manifests(manifests)
    all_errors = [e for errs in results.values() for e in errs]
    assert all_errors == [], f"Cross-manifest validation errors: {all_errors}"


# Combined manifest references CF-Small and SID-Set

def test_combined_manifest_references_cf_small_and_sid_set():
    manifest = _load_example("combined_smoke_manifest.example.json")
    refs = set(manifest.get("referenced_datasets", []))
    assert "community_forensics_small" in refs
    assert "sid_set" in refs


# Protected paths rejected

def test_protected_path_home_rejected():
    m = {**_cf_small_manifest(), "root_placeholder": "/home/user/cf_small"}
    errors = check_protected_paths(m)
    assert any("Protected path" in e for e in errors), errors


def test_protected_path_mnt_rejected():
    m = {**_cf_small_manifest(), "root_placeholder": "/mnt/nvme/datasets"}
    errors = check_protected_paths(m)
    assert any("Protected path" in e for e in errors), errors


def test_absolute_local_path_rejected():
    m = {**_cf_small_manifest(), "root_placeholder": "/root/projects/cf"}
    errors = check_protected_paths(m)
    assert any("Protected path" in e for e in errors), errors


def test_windows_drive_path_rejected():
    m = {**_cf_small_manifest(), "root_placeholder": "C:/Users/user/data"}
    errors = check_protected_paths(m)
    assert any("Windows" in e or "Protected" in e for e in errors), errors


def test_data_path_rejected():
    m = {**_cf_small_manifest(), "root_placeholder": "data/cf_small"}
    errors = check_protected_paths(m)
    assert any("Protected path" in e for e in errors), errors


def test_datasets_path_rejected():
    m = {**_cf_small_manifest(), "root_placeholder": "datasets/sid"}
    errors = check_protected_paths(m)
    assert any("Protected path" in e for e in errors), errors


def test_secrets_path_rejected():
    m = {**_cf_small_manifest(), "root_placeholder": "path/secrets/token"}
    errors = check_protected_paths(m)
    assert any("Protected path" in e for e in errors), errors


def test_env_path_rejected():
    m = {**_cf_small_manifest(), "root_placeholder": ".env"}
    errors = check_protected_paths(m)
    assert any("Protected path" in e for e in errors), errors


def test_placeholder_root_allowed():
    m = _cf_small_manifest()
    errors = check_protected_paths(m)
    assert errors == [], f"Placeholder should be allowed, got: {errors}"


# URLs rejected

def test_http_url_rejected():
    m = {**_cf_small_manifest(), "root_placeholder": "http://example.com/data.zip"}
    errors = check_urls(m)
    assert any("URL" in e for e in errors), errors


def test_https_url_rejected():
    m = {**_cf_small_manifest(), "root_placeholder": "https://example.com/cf.tar.gz"}
    errors = check_urls(m)
    assert any("URL" in e for e in errors), errors


def test_s3_url_rejected():
    m = {**_cf_small_manifest(), "root_placeholder": "s3://bucket/cf_small"}
    errors = check_urls(m)
    assert any("URL" in e for e in errors), errors


def test_gs_url_rejected():
    m = {**_cf_small_manifest(), "root_placeholder": "gs://bucket/sid_set"}
    errors = check_urls(m)
    assert any("URL" in e for e in errors), errors


def test_hf_url_rejected():
    m = {**_cf_small_manifest(), "root_placeholder": "hf://some-repo/dataset"}
    errors = check_urls(m)
    assert any("URL" in e for e in errors), errors


# Guardrail flags rejected when false

def test_dry_run_false_rejected():
    m = {**_cf_small_manifest(), "dry_run": False}
    errors = check_guardrail_flags(m)
    assert any("dry_run" in e for e in errors), errors


def test_no_download_false_rejected():
    m = {**_cf_small_manifest(), "no_download": False}
    errors = check_guardrail_flags(m)
    assert any("no_download" in e for e in errors), errors


def test_no_training_false_rejected():
    m = {**_cf_small_manifest(), "no_training": False}
    errors = check_guardrail_flags(m)
    assert any("no_training" in e for e in errors), errors


def test_no_network_false_rejected():
    m = {**_cf_small_manifest(), "no_network": False}
    errors = check_guardrail_flags(m)
    assert any("no_network" in e for e in errors), errors


def test_dry_run_false_rejected_via_validate():
    m = {**_cf_small_manifest(), "dry_run": False}
    errors = validate_dataset_manifest(m)
    assert any("dry_run" in e for e in errors), errors


def test_no_download_false_rejected_via_validate():
    m = {**_cf_small_manifest(), "no_download": False}
    errors = validate_dataset_manifest(m)
    assert any("no_download" in e for e in errors), errors


def test_no_training_false_rejected_via_validate():
    m = {**_cf_small_manifest(), "no_training": False}
    errors = validate_dataset_manifest(m)
    assert any("no_training" in e for e in errors), errors


def test_no_network_false_rejected_via_validate():
    m = {**_cf_small_manifest(), "no_network": False}
    errors = validate_dataset_manifest(m)
    assert any("no_network" in e for e in errors), errors


# Unknown dataset ids rejected

def test_unknown_dataset_id_rejected():
    m = {**_cf_small_manifest(), "dataset_id": "unknown_dataset"}
    errors = validate_dataset_manifest(m)
    assert any("unknown_dataset" in e for e in errors), errors


def test_unknown_referenced_dataset_id_rejected():
    m = {**_combined_manifest(), "referenced_datasets": ["unknown_dataset"]}
    errors = validate_combined_manifest(m)
    assert any("unknown_dataset" in e for e in errors), errors


# Duplicate dataset ids rejected across manifests

def test_duplicate_dataset_ids_rejected():
    m1 = _cf_small_manifest()
    m2 = _cf_small_manifest()
    results = validate_manifests([m1, m2])
    all_errors = [e for errs in results.values() for e in errs]
    assert any("Duplicate" in e or "duplicate" in e for e in all_errors), all_errors


# SID-Set family label claims rejected without explicit policy

def test_sid_set_cf_family_labels_without_policy_rejected():
    m = _sid_set_manifest()
    # Add CF-Small compatible family labels but remove the supervision policy.
    m["family_labels"] = ["LatDiff", "GAN"]
    m.pop("family_supervision", None)
    errors = validate_dataset_manifest(m)
    assert any("family" in e.lower() for e in errors), errors


def test_sid_set_cf_family_labels_with_policy_allowed():
    m = _sid_set_manifest()
    # family_supervision is already present in _sid_set_manifest(), no error expected
    # even if family_labels were added.
    errors = validate_dataset_manifest(m)
    assert errors == [], f"Unexpected errors: {errors}"


# Required CF-Small metadata fields present

def test_cf_small_missing_architecture_rejected():
    m = _cf_small_manifest()
    m["expected_metadata_fields"] = ["model_name", "subset"]  # missing architecture
    errors = validate_dataset_manifest(m)
    assert any("architecture" in e for e in errors), errors


def test_cf_small_missing_model_name_rejected():
    m = _cf_small_manifest()
    m["expected_metadata_fields"] = ["architecture", "subset"]
    errors = validate_dataset_manifest(m)
    assert any("model_name" in e for e in errors), errors


def test_cf_small_missing_subset_rejected():
    m = _cf_small_manifest()
    m["expected_metadata_fields"] = ["architecture", "model_name"]
    errors = validate_dataset_manifest(m)
    assert any("subset" in e for e in errors), errors


# Required SID-Set target fields present

def test_sid_set_missing_image_ref_rejected():
    m = _sid_set_manifest()
    m["target_fields"] = ["class_label", "tampered_mask_ref", "split"]
    errors = validate_dataset_manifest(m)
    assert any("image_ref" in e for e in errors), errors


def test_sid_set_missing_class_label_rejected():
    m = _sid_set_manifest()
    m["target_fields"] = ["image_ref", "tampered_mask_ref", "split"]
    errors = validate_dataset_manifest(m)
    assert any("class_label" in e for e in errors), errors


def test_sid_set_missing_tampered_mask_ref_rejected():
    m = _sid_set_manifest()
    m["target_fields"] = ["image_ref", "class_label", "split"]
    errors = validate_dataset_manifest(m)
    assert any("tampered_mask_ref" in e for e in errors), errors


def test_sid_set_missing_split_rejected():
    m = _sid_set_manifest()
    m["target_fields"] = ["image_ref", "class_label", "tampered_mask_ref"]
    errors = validate_dataset_manifest(m)
    assert any("split" in e for e in errors), errors


# Secret key detection

def test_secret_key_token_rejected():
    m = {**_cf_small_manifest(), "api_token": "some_value"}
    errors = check_secret_keys(m)
    assert any("token" in e.lower() for e in errors), errors


def test_secret_key_password_rejected():
    m = {**_cf_small_manifest(), "password": "some_value"}
    errors = check_secret_keys(m)
    assert any("password" in e.lower() for e in errors), errors


if __name__ == "__main__":
    import traceback
    failed = 0
    tests = {k: v for k, v in globals().items() if k.startswith("test_") and callable(v)}
    for name, fn in sorted(tests.items()):
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
