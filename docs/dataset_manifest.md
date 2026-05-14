# Dataset Manifest System

## Purpose

Dataset manifests are **planning and validation artifacts only**. They document what a dataset provides, how it should be split, what metadata fields are expected, and what supervision policies apply. They do not point to real data and do not trigger any downloads, file reads, or network access.

Manifests support:

- Community Forensics-Small metadata auditing and holdout planning
- SID-Set split and target-field auditing
- Generator-holdout and model-name-holdout planning
- Multi-dataset dry-run planning
- Family/provenance supervision policy documentation
- Enforcement of no-download, no-training, and no-network guardrails

---

## Manifests Are Planning Artifacts

All manifest JSON files in `configs/manifests/` are example files only.

- They use placeholder roots such as `<CF_SMALL_ROOT>`, `<SID_SET_ROOT>`, `<MANIFEST_ONLY_NO_DATA>`.
- Placeholders are treated as symbolic strings and are never resolved on disk.
- No image paths, mask paths, or metadata CSV paths are checked for existence.
- No real dataset folder is inspected.

---

## Community Forensics-Small Manifest

File: `configs/manifests/community_forensics_small.example.json`

### Key Fields

| Field | Description |
|---|---|
| `expected_metadata_fields` | Must include `architecture`, `model_name`, `subset` |
| `class_labels` | `real`, `synthetic` (binary for backbone stage) |
| `family_labels` | `LatDiff`, `PixDiff`, `GAN`, `Other`, `Real-or-N/A` |
| `split_plan` | Dict of split role names to descriptions |
| `perturbation_notes` | List of perturbation names covered |
| `tasks` | `binary_real_fake`, `coarse_provenance` |

### Metadata Notes

- `architecture`: primary supervision source for coarse family/provenance labels.
- `model_name`: used for label distribution analysis and model-name-holdout planning only.
- `subset`: used for data distribution analysis.
- **Exact model attribution is out of scope.** Provenance is coarse and architecture-based.

### Generator-Holdout Planning

Community Forensics-Small requires generator-holdout or model-name-holdout splits to evaluate unseen generator generalization. Random split risks putting the same generator in both train and validation, inflating apparent generalization performance.

Split roles: `train`, `generator_holdout`, `model_name_holdout`, `compeval_eval`.

---

## SID-Set Manifest

File: `configs/manifests/sid_set.example.json`

### Key Fields

| Field | Description |
|---|---|
| `class_labels` | `real`, `synthetic`, `tampered` (3-way) |
| `target_fields` | `image_ref`, `class_label`, `tampered_mask_ref`, `split` |
| `expected_split_rows` | Planning metadata: train 210K, validation 30K, total 240K |
| `tasks` | `three_way_classification`, `tampered_mask_localization` |
| `family_supervision` | Policy for handling absent family labels |

### Family Supervision Issue

SID-Set may not carry Community Forensics-Small compatible family labels. Applying family/provenance loss to SID-Set samples without verification can corrupt provenance head training.

Three supported policies documented in `family_supervision.allowed_policies`:

| Policy | Description |
|---|---|
| `freeze` | Freeze provenance head during SID-Set fine-tuning |
| `mask_out` | Mask out family loss for SID-Set samples |
| `mixed_cf_small_batch` | Include CF-Small mini-batches to maintain provenance supervision |

The active policy is chosen at experiment config time, not in the manifest.

---

## Combined Smoke Manifest

File: `configs/manifests/combined_smoke_manifest.example.json`

This is a dry-run multi-dataset planning manifest. It documents:

- `referenced_datasets`: `community_forensics_small`, `sid_set`
- `stage_plan`: four-stage project pipeline
- `validation_strategy`: holdout and external evaluation plan
- `perturbations`: all social-media perturbation candidates
- `metrics`: all evaluation metrics

This manifest is for workflow validation and planning review only. No dataset is loaded and no training is run.

---

## Protected Path and No-Network Guardrails

The validator rejects manifests that contain:

- Absolute local machine paths: `/home/`, `/mnt/`, `/root/`, `/Users/`, Windows drive paths
- Protected directory references: `data/`, `datasets/`, `outputs/`, `checkpoints/`
- Environment file references: `.env` or `.env.*`
- Secret directory references: `secrets`
- Network URLs: `http://`, `https://`, `s3://`, `gs://`, `hf://`, `ftp://`
- Secret-looking key names: `token`, `api_key`, `password`, `secret`, `credential`
- Secret-looking values (long opaque base64-like strings)
- Guardrail flags set to false: `dry_run`, `no_download`, `no_training`, `no_network`

Placeholder tokens such as `<CF_SMALL_ROOT>`, `<SID_SET_ROOT>`, `<MANIFEST_ONLY_NO_DATA>` are explicitly allowed and are not resolved on disk.

---

## Validation Commands

Validate all three example manifests together (also checks cross-manifest references):

```bash
python3 scripts/agent/validate_dataset_manifest.py \
    configs/manifests/community_forensics_small.example.json \
    configs/manifests/sid_set.example.json \
    configs/manifests/combined_smoke_manifest.example.json
```

Run pytest tests:

```bash
pytest -q tests/test_dataset_manifest.py
```

---

## Relation to Other Project Documents

| Document | Relation |
|---|---|
| `docs/project_brief.md` | Source of project goals, dataset roles, family labels, and stage plan |
| `configs/project_contract.json` | Machine-readable project contract with target outputs and metrics |
| `docs/config_schema.md` | Dataset and experiment config schema used by training scripts |
| `src/cv_forensics/dataset_manifest.py` | Python module implementing manifest constants and validation logic |
| `src/cv_forensics/config_schema.py` | Config schema module for dataset and experiment configs (separate from manifests) |
