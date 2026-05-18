# Inference Stub — Task 0008

<!-- INFERENCE_STUB_OK -->

## Purpose

This module provides a **fake, deterministic inference stub** for the lightweight multi-head image forensics system. It exercises the model output schema and template explanation pipeline from task 0007 using fully symbolic fake inputs. It is **not** real model inference and does **not** perform training, dataset access, image loading, mask loading, or any network activity.

## What This Is Not

- Not real model inference
- Not a training loop
- Not a dataset loader
- Not an SNS augmentation implementation (SNS perturbation tags are represented as schema metadata only)

## Mapping to the Proposal Architecture

The stub mirrors the inference flow described in `docs/project_brief.md` §9 using fake functions:

```
FakeInput (symbolic)
  -> fake_backbone_summary        # Symbolic backbone feature summary
  -> fake_classification_head     # Deterministic class_conf: real / synthetic / tampered
  -> fake_family_head             # Deterministic family_conf: LatDiff / PixDiff / GAN / Other / Real-or-N/A
  -> fake_localization_head       # Conditional: activated / skipped_below_threshold / not_applicable
  -> fake_evidence_aggregation    # Maps class/family/localization context to evidence signal IDs
  -> generate_reason()            # Deterministic Korean template explanation (task 0007)
  -> validate_forensic_output()   # Task 0007 schema validation
```

Conditional localization follows the proposal rule: `if tampered_score >= threshold_tau` → `activated`, else `skipped_below_threshold`. For non-tampered classes it is `not_applicable`. This preserves the false-negative risk analysis: a high `tau` can suppress localization on genuine tampered images.

## Reuse of Task 0007 Schema

`inference_stub.py` imports directly from:
- `src/cv_forensics/model_output_schema.py` — labels, constants, `validate_forensic_output`, `EvidenceSignal`, `ForensicOutput`
- `src/cv_forensics/explanation_templates.py` — `generate_reason`

All final outputs pass `validate_forensic_output`.

## Supported Scenarios

| Scenario | class_hint | Localization |
|---|---|---|
| `real_clean` | real | not_applicable |
| `synthetic_latdiff` | synthetic | not_applicable |
| `tampered_localized` | tampered + score ≥ tau | activated |
| `tampered_below_threshold` | tampered + score < tau | skipped_below_threshold |
| `low_confidence` | spread evenly | not_applicable |

## Preparation for Future Tasks

- **Task 0009** — toy metric calculators will consume `ForensicOutput` dicts produced by this stub.
- **Task 0010** — dry-run training skeleton will use FakeInput records as a template for real batch records.
- **Later real inference** — replace fake head functions with real model forward passes; the output schema and validation remain unchanged.
- **Later SNS augmentation evaluation** — perturbation tags in `FakeInput.perturbations` align with `PERTURBATION_TAGS` in the schema; real augmentation logic will populate them.

## Fake Input Config Safety

Before any fake inference runs, the full raw fake input config is **recursively validated** by `check_fake_input_config_safety`, including top-level fields and every nested `fake_inputs` record. The check inspects every dict key and every string value at any nesting depth and rejects:

- **Protected paths** — any path segment equal to `secrets`, `data`, `datasets`, `outputs`, or `checkpoints` (e.g. `data/foo.jpg`, `./data/foo.jpg`, `outputs/mask.png`, `checkpoints/model.pt`)
- **Absolute machine paths** — strings starting with `/home/`, `/mnt/`, `/root/`, `/Users/`
- **Windows drive paths** — strings matching `C:\...` or any `<drive>:\...` pattern
- **`.env` file references** — the exact string `.env` or any `.env.*` variant
- **URL-like references** — strings starting with `http://`, `https://`, `s3://`, `gs://`, or `hf://`
- **Secret-looking key names** — dict keys with normalized tokens such as `token`, `api_key`, `password`, `secret`, `credential`, `auth`, or `bearer`
- **Secret-looking string values** — string values with clear secret tokens such as `token`, `api_key`, `password`, `secret`, `credential`, `auth`, or `bearer`

The token-based secret check avoids naive substring false positives in ordinary prose, so text such as `authoritative dry-run note` or `authentication policy is documented` is not rejected merely because a longer word contains the letters `auth`. The existing `configs/inference/fake_inputs.example.json` is verified clean by `test_example_config_passes_safety_check`. Both `validate_inference_stub.py` and `run_fake_inference.py` call this check before running any inference — an unsafe config causes an immediate non-zero exit with an actionable error message.

## SNS Augmentation

SNS augmentation is **not implemented** in task 0008. Perturbation fields such as `"jpeg"`, `"recompression_chain"`, and `"text_overlay"` appear in fake input configs as schema-compatibility metadata only.

## Validation Commands

```bash
python3 scripts/agent/validate_inference_stub.py configs/inference/fake_inputs.example.json
```

```bash
python3 scripts/agent/run_fake_inference.py configs/inference/fake_inputs.example.json
```

```bash
python3 tests/test_inference_stub.py
```

Optional, if pytest is installed:

```bash
pytest -q tests/test_inference_stub.py
```

## Files

| File | Purpose |
|---|---|
| `src/cv_forensics/inference_stub.py` | Core fake inference pipeline |
| `configs/inference/fake_inputs.example.json` | Example fake input config |
| `scripts/agent/run_fake_inference.py` | CLI runner: prints JSON summary |
| `scripts/agent/validate_inference_stub.py` | CLI validator: checks schema and behavior |
| `tests/test_inference_stub.py` | Unit tests (stdlib only, pytest-compatible) |
