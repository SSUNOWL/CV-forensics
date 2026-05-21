#!/usr/bin/env python3
"""Standalone tests for pre-SNS training artifact policy and validation."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.pre_sns_training_artifacts import (  # noqa: E402
    ArtifactPolicyError,
    REQUIRED_ARTIFACT_FILES,
    prepare_artifact_root,
    validate_artifact_root,
    write_training_artifacts,
)
from scripts.agent.validate_pre_sns_training_artifacts import (  # noqa: E402
    OK_MARKER,
    parse_result_json,
    validate,
)


TEMP_ROOT = Path.home() / ".codex" / "memories"


def temporary_root():
    return tempfile.TemporaryDirectory(dir=str(TEMP_ROOT))


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def assert_raises(expected, func, *args, **kwargs) -> None:
    try:
        func(*args, **kwargs)
    except expected:
        return
    raise AssertionError(f"expected {expected.__name__}")


def base_config(run_root: str, checkpoint_root: str) -> dict:
    return {
        "config_kind": "approved_pre_sns_baseline_training",
        "no_write_dry_run": False,
        "approved_run_root": run_root,
        "approved_checkpoint_root": checkpoint_root,
    }


def base_result(run_root: str, checkpoint_root: str, artifact_manifest_path: str, checkpoint_path: str) -> dict:
    return {
        "marker": "PRE_SNS_BASELINE_TRAINING_RUN_OK",
        "entrypoint_marker": "PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK",
        "training_started": True,
        "training_completed": True,
        "total_loss_finite": True,
        "no_download": True,
        "no_network": True,
        "no_sns_augmentation": True,
        "no_write_dry_run": False,
        "approved_run_root": run_root,
        "approved_checkpoint_root": checkpoint_root,
        "artifact_manifest_path": artifact_manifest_path,
        "checkpoint_path": checkpoint_path,
    }


def create_valid_artifact_set(root: str) -> tuple[dict, dict, Path]:
    run_root = Path(root) / "cvf_run"
    checkpoint_root = Path(root) / "cvf_weights"
    checkpoint_root.mkdir()
    checkpoint_path = checkpoint_root / "tiny.pt"
    checkpoint_path.write_bytes(b"tiny checkpoint")
    config = base_config(str(run_root), str(checkpoint_root))
    result = base_result(str(run_root), str(checkpoint_root), "", str(checkpoint_path))
    info = write_training_artifacts(
        run_root=run_root,
        checkpoint_root=checkpoint_root,
        config_snapshot=config,
        manifest_snapshot={"samples": []},
        metrics_summary={"total_loss_finite": True},
        run_summary=result,
        checkpoint_path=checkpoint_path,
        overwrite=True,
    )
    result["artifact_manifest_path"] = info["artifact_manifest_path"]
    write_json(run_root / "run_summary.json", result)
    return config, result, run_root


def test_artifact_roots_outside_repo_are_accepted() -> None:
    with temporary_root() as tmp:
        root = validate_artifact_root(os.path.join(tmp, "cvf_run"))
        assert str(root).startswith(os.path.realpath(tmp))


def test_repo_outputs_and_checkpoints_roots_are_rejected() -> None:
    assert_raises(ArtifactPolicyError, validate_artifact_root, REPO_ROOT / "outputs" / "x")
    assert_raises(ArtifactPolicyError, validate_artifact_root, REPO_ROOT / "checkpoints" / "x")


def test_non_empty_run_root_is_rejected_unless_explicitly_allowed() -> None:
    with temporary_root() as tmp:
        root = Path(tmp) / "run"
        root.mkdir()
        (root / "existing.txt").write_text("occupied", encoding="utf-8")
        assert_raises(ArtifactPolicyError, prepare_artifact_root, root)
        accepted = prepare_artifact_root(root, overwrite=True)
        assert accepted == Path(os.path.realpath(root))


def test_empty_pre_existing_run_root_is_accepted() -> None:
    with temporary_root() as tmp:
        root = Path(tmp) / "run"
        root.mkdir()
        accepted = prepare_artifact_root(root)
        assert accepted == Path(os.path.realpath(root))


def test_artifact_manifest_contains_required_files() -> None:
    with temporary_root() as tmp:
        _config, result, _run_root = create_valid_artifact_set(tmp)
        manifest = json.loads(Path(result["artifact_manifest_path"]).read_text(encoding="utf-8"))
        for filename in REQUIRED_ARTIFACT_FILES:
            assert filename in manifest["files"], filename


def test_validator_rejects_missing_run_summary() -> None:
    with temporary_root() as tmp:
        config, result, run_root = create_valid_artifact_set(tmp)
        (run_root / "run_summary.json").unlink()
        errors = validate(config, result)
        assert "run_summary.json" in "\n".join(errors), errors


def test_validator_rejects_missing_checkpoint_for_actual_run() -> None:
    with temporary_root() as tmp:
        config, result, _run_root = create_valid_artifact_set(tmp)
        Path(result["checkpoint_path"]).unlink()
        errors = validate(config, result)
        assert "checkpoint_path" in "\n".join(errors), errors


def test_validator_accepts_valid_small_artifact_set() -> None:
    with temporary_root() as tmp:
        config, result, _run_root = create_valid_artifact_set(tmp)
        errors = validate(config, result)
        assert not errors, "\n".join(errors)


def test_parsing_result_json_with_leading_log_text_works() -> None:
    with temporary_root() as tmp:
        path = Path(tmp) / "result.log"
        payload = {"marker": "PRE_SNS_BASELINE_TRAINING_RUN_OK", "total_loss_finite": True}
        path.write_text("log line before json\n" + json.dumps(payload), encoding="utf-8")
        parsed = parse_result_json(path)
        assert parsed == payload


def test_no_write_dry_run_result_does_not_require_artifacts() -> None:
    config = {"no_write_dry_run": True}
    result = {
        "marker": "PRE_SNS_BASELINE_TRAINING_DRY_RUN_OK",
        "entrypoint_marker": "PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK",
        "training_started": True,
        "training_completed": True,
        "no_write_dry_run": True,
        "total_loss_finite": True,
    }
    assert validate(config, result) == []


def test_actual_training_result_requires_artifacts() -> None:
    with temporary_root() as tmp:
        config = base_config(os.path.join(tmp, "run"), os.path.join(tmp, "weights"))
        result = base_result(config["approved_run_root"], config["approved_checkpoint_root"], "", "")
        errors = validate(config, result)
        joined = "\n".join(errors)
        assert "artifact_manifest_path" in joined
        assert "checkpoint_path" in joined


def test_no_write_dry_run_requires_completed_training_semantics() -> None:
    config = {"no_write_dry_run": True}
    result = {
        "marker": "PRE_SNS_BASELINE_TRAINING_DRY_RUN_OK",
        "entrypoint_marker": "PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK",
        "training_started": True,
        "training_completed": True,
        "no_write_dry_run": True,
        "total_loss_finite": True,
    }
    assert validate(config, result) == []
    bad = dict(result)
    bad["training_started"] = False
    assert "training_started" in "\n".join(validate(config, bad))
    bad = dict(result)
    bad["marker"] = "PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK"
    assert "DRY_RUN" in "\n".join(validate(config, bad))


def test_actual_training_requires_run_semantics_and_artifact_references() -> None:
    with temporary_root() as tmp:
        config, result, _run_root = create_valid_artifact_set(tmp)
        assert validate(config, result) == []
        bad = dict(result)
        bad["training_started"] = False
        assert "training_started" in "\n".join(validate(config, bad))
        bad = dict(result)
        bad["marker"] = "PRE_SNS_BASELINE_TRAINING_DRY_RUN_OK"
        assert "actual run marker" in "\n".join(validate(config, bad))


def main() -> int:
    tests = [
        test_artifact_roots_outside_repo_are_accepted,
        test_repo_outputs_and_checkpoints_roots_are_rejected,
        test_non_empty_run_root_is_rejected_unless_explicitly_allowed,
        test_empty_pre_existing_run_root_is_accepted,
        test_artifact_manifest_contains_required_files,
        test_validator_rejects_missing_run_summary,
        test_validator_rejects_missing_checkpoint_for_actual_run,
        test_validator_accepts_valid_small_artifact_set,
        test_parsing_result_json_with_leading_log_text_works,
        test_no_write_dry_run_result_does_not_require_artifacts,
        test_actual_training_result_requires_artifacts,
        test_no_write_dry_run_requires_completed_training_semantics,
        test_actual_training_requires_run_semantics_and_artifact_references,
    ]
    for test in tests:
        test()
    print(f"{OK_MARKER}_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
