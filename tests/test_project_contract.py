"""pytest tests for the project contract JSON and validator."""
import json
import subprocess
import sys
from pathlib import Path

CONTRACT_PATH = Path(__file__).parent.parent / "configs" / "project_contract.json"
VALIDATOR_PATH = Path(__file__).parent.parent / "scripts" / "agent" / "validate_project_contract.py"

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


def load_contract():
    with open(CONTRACT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_contract_file_exists():
    assert CONTRACT_PATH.exists(), f"Contract file not found: {CONTRACT_PATH}"


def test_contract_is_valid_json():
    contract = load_contract()
    assert isinstance(contract, dict)


def test_top_level_keys_present():
    contract = load_contract()
    for key in REQUIRED_TOP_LEVEL_KEYS:
        assert key in contract, f"Missing top-level key: {key}"


def test_target_outputs_contain_required_keys():
    contract = load_contract()
    outputs = contract["target_outputs"]
    for key in ["class", "mask", "family", "reason"]:
        assert key in outputs, f"target_outputs missing: {key}"


def test_datasets_contain_required_entries():
    contract = load_contract()
    datasets = contract["datasets"]
    assert "Community Forensics-Small" in datasets
    assert "SID-Set" in datasets


def test_evaluation_metrics_coverage():
    contract = load_contract()
    metrics = contract["evaluation_metrics"]
    metric_text = " ".join(
        (m.get("name", "") + " " + m.get("purpose", "")).lower()
        for m in metrics
    )
    for substring in ["classification", "macro-f1", "iou", "family", "robustness", "latency", "fps", "recall"]:
        assert substring in metric_text, f"evaluation_metrics missing coverage for: {substring}"


def test_implementation_stages_count():
    contract = load_contract()
    stages = contract["implementation_stages"]
    assert len(stages) >= 4, "Expected at least 4 implementation stages"


def test_guardrails_not_empty():
    contract = load_contract()
    assert len(contract["guardrails"]) > 0


def test_validator_script_passes():
    result = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), str(CONTRACT_PATH)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Validator failed:\n{result.stdout}\n{result.stderr}"
