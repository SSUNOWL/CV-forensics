# Local Data Readiness Gate

<!-- LOCAL_DATA_READINESS_OK -->

This document describes the local data readiness gate introduced in task 0011.

## What This Gate Does Not Do

- It does **not** download datasets.
- It does **not** train models.
- It does **not** inspect dataset directories recursively.
- It does **not** read images or masks from disk.
- It does **not** write outputs or checkpoints.
- It does **not** implement SNS augmentation.

All validation is symbolic and operates purely on configuration structure.

## What the Gate Checks

### 1. Symbolic Local-Data Plan

The `dataset_plan` field declares which datasets are planned and their roles (e.g. `backbone_pretraining`, `multihead_finetuning`). The gate validates that this field is present and structurally well-formed. No real dataset directories are accessed.

### 2. Manifest References

The `manifest_refs` field lists symbolic references to dataset manifest config files. The gate verifies that this field is present and that no references contain protected paths, URLs, or secret-like values.

### 3. Approval Policy

The `approval` field declares whether explicit user approval for local data access has been granted. If `local_path_policy.policy` indicates real paths will be used, `approval.local_data_approved` must be `true`. In dry-run mode, `local_data_approved` is `false` and all paths remain symbolic.

### 4. Output and Checkpoint Policy

The `output_policy` and `checkpoint_policy` fields declare what policy governs output and checkpoint directories. In dry-run mode these must be set to `no_outputs` and `no_checkpoints` respectively. No output or checkpoint directories are created by this gate.

### 5. Protected-Path Exclusions

The `protected_path_exclusions` field explicitly lists the path categories that are excluded from any real access. This field is declarative: its entries name what is excluded (e.g. `".env"`, `"secrets"`, `"data"`) and are not interpreted as real path references. The safety checker skips this field during validation.

### 6. Guardrail Flags

The gate enforces that all six guardrail flags are `true`:

| Flag | Meaning |
|---|---|
| `dry_run` | No real training loop is running |
| `no_download` | No dataset download is allowed |
| `no_training` | No model training is allowed |
| `no_network` | No network access is allowed |
| `no_outputs` | No output directories are written |
| `no_checkpoints` | No checkpoint directories are written |

## Safety Validation

The gate recursively checks all config values (except `protected_path_exclusions`) and rejects:

- Absolute local paths: `/home/`, `/mnt/`, `/root/`, `/Users/`
- Windows drive paths: `C:\`, `D:\`
- `.env` and `.env.*` file references
- Protected directory names as path components: `data`, `datasets`, `outputs`, `checkpoints`, `secrets`
- URL schemes: `http://`, `https://`, `s3://`, `gs://`, `hf://`, `ftp://`
- Secret-like key names: `token`, `api_key`, `password`, `secret`, `credential`, `auth`, `bearer`
- Secret-like values containing those tokens as standalone words

Ordinary prose is allowed. For example, `"authoritative guidance"` and `"authentication policy"` are acceptable note values because `auth` appears only as a substring of a longer word, not as a standalone token.

## What Must Happen Before Real Training

Before any real local data access or model training begins, the following must be completed and explicitly approved:

1. **Explicit user approval**: `approval.local_data_approved` must be set to `true` by the user.
2. **Local data readiness approval**: The readiness gate must pass with a valid local-data config that reflects real paths and approved policies.
3. **Validated manifests**: All dataset manifests referenced in `manifest_refs` must pass manifest validation.
4. **Output and checkpoint policy decision**: `output_policy` and `checkpoint_policy` must be updated to reflect the approved real-training configuration.
5. **Small local subset smoke test**: Before full training, a small local subset smoke test should be run to verify the data pipeline end-to-end.

SNS augmentation is not implemented at this stage. It begins only after a pre-SNS baseline evaluation is committed.

## Configuration Format

See `configs/local_data/readiness.example.json` for a complete symbolic dry-run safe example.

Required top-level fields:

```
schema_version          string
dry_run                 true
no_download             true
no_training             true
no_network              true
no_outputs              true
no_checkpoints          true
dataset_plan            object (symbolic dataset roles and statuses)
manifest_refs           array of symbolic manifest references
local_path_policy       object (policy name and notes)
output_policy           object (policy name and notes)
checkpoint_policy       object (policy name and notes)
approval                object (local_data_approved flag and notes)
protected_path_exclusions  array of excluded path category names
```

## Validation Commands

```bash
python3 scripts/agent/validate_local_data_readiness.py configs/local_data/readiness.example.json
```

```bash
python3 tests/test_local_data_gate.py
```

```bash
grep -q LOCAL_DATA_READINESS_OK docs/local_data_readiness.md
```
