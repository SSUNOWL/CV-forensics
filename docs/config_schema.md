# Config Schema: Datasets and Experiments

## Purpose

This document describes the configuration schema for dataset definitions and experiment definitions used in the cv-forensics project.

The schema is implemented in `src/cv_forensics/config_schema.py` using only the Python standard library.

---

## Relationship to Project Brief and Contract

The schema constants and validation rules directly reflect the project goals described in `docs/project_brief.md` and `configs/project_contract.json`:

| Schema concept | Source |
|---|---|
| `CLASS_LABELS` = `real`, `synthetic`, `tampered` | Section 6.3, Section 19 of project brief |
| `FAMILY_LABELS` = `LatDiff`, `PixDiff`, `GAN`, `Other`, `Real-or-N/A` | Section 6.4 of project brief |
| `DATASET_IDS` = `community_forensics_small`, `sid_set` | Section 4 of project brief |
| `STAGES` | Section 8 of project brief |
| `METRICS` | Section 10 of project brief |

---

## Current Configs Are Examples Only

`configs/datasets.example.json` and `configs/experiments/smoke_baseline.json` are **example configs only**.

- They contain no real data paths.
- All dataset root fields use placeholder tokens such as `<CF_SMALL_ROOT>` and `<SID_SET_ROOT>`.
- They contain no secrets, API keys, tokens, or machine-specific paths.
- The smoke experiment has `dry_run: true` and does not request training or data download.

Production configs that point to real data paths must be created outside of version control and must never be committed.

---

## Community Forensics-Small Role

Community Forensics-Small (`community_forensics_small`) is used for:

- Shared visual backbone pre-training with generator diversity.
- Generator-family provenance head training, supervised by the `architecture` metadata field.
- Unseen generator generalization evaluation via generator-holdout or model_name-holdout splits.

Required metadata fields: `architecture`, `model_name`, `subset`.

---

## SID-Set Role

SID-Set (`sid_set`) is used for:

- 3-way (`real` / `synthetic` / `tampered`) classification fine-tuning.
- Tampered region mask localization.

### Family/Provenance Label Handling

SID-Set samples may not carry Community Forensics-Small family labels. During Stage 3 fine-tuning, one of the following strategies must be applied to avoid corrupting the provenance head:

1. **Freeze** the provenance head and do not update it during SID-Set fine-tuning.
2. **Mask-out** the family loss for SID-Set samples that have no family label.
3. **Mixed-batch**: include CF-Small mini-batches alongside SID-Set mini-batches to maintain provenance supervision.

The chosen strategy must be declared in the experiment config (`provenance.sid_set_strategy`).

---

## Validation Commands

```bash
python3 scripts/agent/validate_config_schema.py \
    configs/datasets.example.json \
    configs/experiments/smoke_baseline.json
```

Optional (if pytest is installed):

```bash
pytest -q tests/test_config_schema.py
```

---

## Forbidden Paths and Safety Guardrails

The validator rejects any config that contains the following in any string value:

| Pattern | Reason |
|---|---|
| `/home/`, `/mnt/`, `/root/` | Machine-specific absolute paths |
| `data/`, `datasets/` | Protected data directories |
| `outputs/`, `checkpoints/` | Protected output directories |
| `.env`, `secrets` | Credential or secret files |
| `http://`, `https://`, `ftp://`, `s3://`, `gs://` | Network URLs |

Placeholder tokens such as `<CF_SMALL_ROOT>` are allowed and are the correct way to represent dataset roots in example configs.

---

## Schema Module Summary

`src/cv_forensics/config_schema.py` provides:

- Constants: `CLASS_LABELS`, `FAMILY_LABELS`, `DATASET_IDS`, `STAGES`, `METRICS`
- `validate_datasets_config(config)` returns list of error strings
- `validate_experiment_config(config, known_dataset_ids)` returns list of error strings
- `extract_dataset_ids(config)` returns list of dataset id strings
- `contains_protected_path(value)` returns bool
- `is_dry_run_safe(experiment_config)` returns bool
