# Task 0006: Dataset Manifest Validator, No Data Download

## Role of Claude Code

Claude Code is the implementation worker. Codex is the supervisor and reviewer.

Implement this task exactly as written. Do not reinterpret the project scope beyond this task file. Stop after implementation and validation, then report the requested results.

## Objective

Implement a pure-Python dataset manifest schema and validator for Community Forensics-Small, SID-Set, and a combined dry-run manifest.

The validator must validate manifest files only. It must not download datasets, read real image files, inspect real dataset folders, train models, install packages, or touch protected paths.

The manifest system should support future work for:

- Community Forensics-Small metadata auditing
- SID-Set split and target-field auditing
- generator-holdout or model-name-holdout planning
- multi-dataset dry-run planning
- family/provenance supervision policy documentation
- no-download, no-training, no-network guardrails

## Files Claude May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `docs/config_schema.md`
- `configs/datasets.example.json`
- `configs/experiments/smoke_baseline.json`
- `src/cv_forensics/config_schema.py`
- `tasks/0006-dataset-manifest-validator.md`
- `docs/repo_structure.md`, only if it exists
- `src/cv_forensics/__init__.py`, only if it exists
- `src/cv_forensics/contracts.py`, only if it exists
- `src/cv_forensics/outputs.py`, only if it exists
- `src/cv_forensics/evidence.py`, only if it exists
- Any file Claude is allowed to create or modify in this task, if it already exists

## Files Claude May Modify

- `src/cv_forensics/__init__.py`
- `src/cv_forensics/dataset_manifest.py`
- `configs/manifests/community_forensics_small.example.json`
- `configs/manifests/sid_set.example.json`
- `configs/manifests/combined_smoke_manifest.example.json`
- `scripts/agent/validate_dataset_manifest.py`
- `tests/test_dataset_manifest.py`
- `docs/dataset_manifest.md`

If a parent directory for an allowed file does not exist, Claude may create that parent directory. Claude must not create any other files.

## Forbidden Actions

Claude must not:

- download datasets
- train a model
- install packages
- access network resources from shell commands
- access `.env`, `.env.*`, `secrets`, `data`, `datasets`, `outputs`, or `checkpoints`
- inspect protected directories or protected files recursively
- read real image files
- validate actual image file existence
- walk real dataset directories
- modify unrelated files
- run `git push`
- run `git pull`
- run `git fetch`
- run `curl`, `wget`, `ssh`, `scp`, `rsync`, `pip`, `npm`, `yarn`, `pnpm`, `kaggle`, `huggingface-cli`, or `wandb`
- create large files
- use `rg`
- use `rm -rf`
- use `--permission-mode bypassPermissions`
- use `--dangerously-skip-permissions`

## Important Project Facts to Preserve

- Community Forensics-Small is used for shared visual backbone learning and generator-family provenance learning.
- Community Forensics-Small provenance should be coarse, architecture-based provenance, not exact model attribution.
- Important Community Forensics-Small metadata fields include `architecture`, `model_name`, and `subset`.
- Initial family label candidates are `LatDiff`, `PixDiff`, `GAN`, `Other`, and `Real-or-N/A`.
- SID-Set is used for real / synthetic / tampered 3-way classification and tampered mask localization.
- SID-Set may not have generator-family labels compatible with Community Forensics-Small.
- SID-Set family/provenance handling should support freeze, mask-out, or mixed Community Forensics-Small batch policies.
- The project prefers generator-holdout or model_name-holdout validation over simple random split for unseen generator generalization.
- SID-Set initial implementation should document public split assumptions such as train 210K and validation 30K, total 240K rows, as planning metadata only.
- All manifests in this task are examples and dry-run planning artifacts only. They must not point to real local data.

## Implementation Requirements

### 1. Create or update `src/cv_forensics/dataset_manifest.py`

- Use only Python standard library.
- Do not import third-party packages.
- Do not read files except JSON manifests passed to validator functions.
- Do not access `data`, `datasets`, `outputs`, `checkpoints`, `.env`, or `secrets`.
- Define constants for:
  - known dataset ids:
    - `community_forensics_small`
    - `sid_set`
    - `combined_smoke`
  - class labels:
    - `real`
    - `synthetic`
    - `tampered`
  - family labels:
    - `LatDiff`
    - `PixDiff`
    - `GAN`
    - `Other`
    - `Real-or-N/A`
  - manifest kinds:
    - `dataset_manifest`
    - `combined_manifest`
  - split roles:
    - `train`
    - `validation`
    - `eval`
    - `generator_holdout`
    - `model_name_holdout`
    - `compeval_eval`
    - `smoke`
  - dataset tasks:
    - `binary_real_fake`
    - `coarse_provenance`
    - `three_way_classification`
    - `tampered_mask_localization`
    - `social_media_robustness`
  - family supervision policies:
    - `available`
    - `unavailable`
    - `freeze`
    - `mask_out`
    - `mixed_cf_small_batch`
  - perturbations:
    - `jpeg`
    - `resize`
    - `crop`
    - `rotation`
    - `padding`
    - `shear`
    - `screenshot`
    - `text_overlay`
    - `sticker_overlay`
    - `recompression_chain`

Define validation helpers for:

- loading JSON manifests
- validating a single dataset manifest
- validating a combined manifest
- validating dataset ids
- validating split definitions
- validating expected metadata fields
- validating class labels
- validating family labels
- validating SID-Set family/provenance policy
- validating no-download and no-training flags
- detecting protected paths and unsafe paths
- detecting absolute local machine paths
- detecting URLs
- detecting secret-looking fields or values
- checking cross-manifest dataset references

The validator must reject:

- absolute local paths such as `/home/`, `/mnt/`, `/root/`, `/Users/`, or Windows drive paths
- paths containing `.env` or `.env.*`
- paths containing `secrets`
- paths containing `data/`
- paths containing `datasets/`
- paths containing `outputs/`
- paths containing `checkpoints/`
- URLs such as `http://`, `https://`, `s3://`, `gs://`, `hf://`
- secret-looking keys such as `token`, `api_key`, `password`, `secret`, `credential`
- secret-looking values
- `dry_run: false` in example manifests
- `no_download: false`
- `no_training: false`
- `no_network: false`
- unknown dataset ids
- unknown manifest kinds
- unknown labels
- unknown split roles
- SID-Set manifests that claim Community Forensics-Small compatible family labels without an explicit policy note
- combined manifests that reference unknown dataset ids

The validator must allow placeholder roots such as:

- `<CF_SMALL_ROOT>`
- `<SID_SET_ROOT>`
- `<MANIFEST_ONLY_NO_DATA>`

The validator must treat placeholders as symbolic strings only and must not resolve them on disk.

The validator must not check whether any image, mask, metadata CSV, or dataset folder actually exists.

### 2. Create `configs/manifests/community_forensics_small.example.json`

This must be an example manifest only.

It must not contain real local paths, protected paths, URLs, or secrets.

It should include:

- `schema_version`
- `manifest_kind`: `dataset_manifest`
- `dataset_id`: `community_forensics_small`
- `display_name`
- `dry_run`: `true`
- `no_download`: `true`
- `no_training`: `true`
- `no_network`: `true`
- `root_placeholder`: `<CF_SMALL_ROOT>`
- `primary_roles`:
  - `shared_backbone_pretraining`
  - `coarse_provenance`
  - `generator_holdout_validation`
- `tasks`:
  - `binary_real_fake`
  - `coarse_provenance`
- `expected_metadata_fields`:
  - `architecture`
  - `model_name`
  - `subset`
- `class_labels`:
  - `real`
  - `synthetic`
- `family_labels`:
  - `LatDiff`
  - `PixDiff`
  - `GAN`
  - `Other`
  - `Real-or-N/A`
- `split_plan`:
  - `train`
  - `generator_holdout`
  - `model_name_holdout`
  - `compeval_eval`
- `perturbation_notes`:
  - `jpeg`
  - `resize`
  - `crop`
  - `rotation`
  - `padding`
  - `shear`
- notes explaining that exact model attribution is out of scope

### 3. Create `configs/manifests/sid_set.example.json`

This must be an example manifest only.

It must not contain real local paths, protected paths, URLs, or secrets.

It should include:

- `schema_version`
- `manifest_kind`: `dataset_manifest`
- `dataset_id`: `sid_set`
- `display_name`
- `dry_run`: `true`
- `no_download`: `true`
- `no_training`: `true`
- `no_network`: `true`
- `root_placeholder`: `<SID_SET_ROOT>`
- `primary_roles`:
  - `three_way_classification`
  - `tampered_mask_localization`
  - `mask_supervision`
- `tasks`:
  - `three_way_classification`
  - `tampered_mask_localization`
- `class_labels`:
  - `real`
  - `synthetic`
  - `tampered`
- `target_fields`:
  - `image_ref`
  - `class_label`
  - `tampered_mask_ref`
  - `split`
- `expected_split_rows`:
  - `train`: `210000`
  - `validation`: `30000`
  - `total_initial`: `240000`
- `family_supervision`:
  - `status`: `unavailable`
  - `allowed_policies`:
    - `freeze`
    - `mask_out`
    - `mixed_cf_small_batch`
- notes explaining that SID-Set may lack Community Forensics-Small compatible family labels

### 4. Create `configs/manifests/combined_smoke_manifest.example.json`

This must be a dry-run combined planning manifest only.

It must not contain real local paths, protected paths, URLs, or secrets.

It should include:

- `schema_version`
- `manifest_kind`: `combined_manifest`
- `dataset_id`: `combined_smoke`
- `dry_run`: `true`
- `no_download`: `true`
- `no_training`: `true`
- `no_network`: `true`
- `root_placeholder`: `<MANIFEST_ONLY_NO_DATA>`
- `referenced_datasets`:
  - `community_forensics_small`
  - `sid_set`
- `stage_plan`:
  - `cf_small_binary_backbone`
  - `provenance_head`
  - `sid_set_multihead_finetune`
  - `social_media_robustness`
- `validation_strategy`:
  - `generator_holdout`
  - `model_name_holdout`
  - `compeval_eval`
- `sid_family_policy`:
  - `freeze`
  - `mask_out`
  - `mixed_cf_small_batch`
- `perturbations`:
  - `jpeg`
  - `resize`
  - `crop`
  - `rotation`
  - `padding`
  - `shear`
  - `screenshot`
  - `text_overlay`
  - `sticker_overlay`
  - `recompression_chain`
- `metrics`:
  - `accuracy_3way`
  - `macro_f1`
  - `tampered_mask_iou`
  - `generator_family_accuracy`
  - `perturbation_robustness_drop`
  - `latency_ms`
  - `fps`
  - `localization_activation_recall`
- notes explaining that this manifest is for validation and planning only

### 5. Create `scripts/agent/validate_dataset_manifest.py`

- Use only Python standard library.
- Accept one or more positional manifest JSON paths.
- Validate all provided manifests using `src/cv_forensics/dataset_manifest.py`.
- Be import-safe without package installation.
- It may add the repository root or `src` directory to `sys.path`.
- Print a concise success message on pass.
- Print actionable errors and exit non-zero on fail.
- Do not access actual dataset folders.
- Do not check image or mask file existence.
- Do not download anything.
- Do not train anything.
- Do not access network resources.
- Do not access protected paths.
- When multiple manifests are provided, validate cross-manifest references:
  - `combined_smoke` must reference known dataset ids that were provided or known examples
  - `community_forensics_small` and `sid_set` ids must be unique
  - `referenced_datasets` must not contain unknown ids

### 6. Create `docs/dataset_manifest.md`

Keep it concise but useful.

Explain:

- the purpose of dataset manifests
- that manifests are planning and validation artifacts only
- that manifests do not point to real data and do not trigger downloads
- Community Forensics-Small manifest fields:
  - `architecture`
  - `model_name`
  - `subset`
  - family labels
  - generator-holdout planning
- SID-Set manifest fields:
  - 3-way class label
  - tampered mask target
  - split rows as planning metadata
  - family supervision issue
  - freeze, mask-out, and mixed Community Forensics-Small batch policies
- combined smoke manifest use
- protected path and no-network guardrails
- validation commands
- how this task relates to `docs/project_brief.md`, `configs/project_contract.json`, and `docs/config_schema.md`

### 7. Create `tests/test_dataset_manifest.py`

- Use pytest-style assertions.
- Do not install pytest.
- Use only Python standard library plus pytest-style tests.
- Keep tests lightweight.
- Do not access external resources.
- Do not access protected paths.

Tests should cover:

- valid example manifests pass
- protected paths are rejected
- absolute local paths are rejected
- URLs are rejected
- unknown dataset ids are rejected
- unknown referenced dataset ids are rejected
- duplicate dataset ids are rejected when validating multiple manifests
- SID-Set family label claims are rejected unless policy is explicit and valid
- combined smoke manifest references Community Forensics-Small and SID-Set
- `dry_run: false` is rejected
- `no_download: false` is rejected
- `no_training: false` is rejected
- `no_network: false` is rejected
- required Community Forensics-Small metadata fields are present
- required SID-Set target fields are present

### 8. Update `src/cv_forensics/__init__.py` only if needed

- If the file exists, Claude may add a lightweight export for `dataset_manifest`.
- If the file does not exist, Claude may create a minimal package marker.
- Do not add heavy imports.
- Do not import third-party packages.

## Validation Commands

Each command is intentionally a simple one-command block. Do not combine commands with `&&`, `;`, pipes, redirects, command substitution, or shell variables.

```bash
python3 scripts/agent/validate_dataset_manifest.py configs/manifests/community_forensics_small.example.json configs/manifests/sid_set.example.json configs/manifests/combined_smoke_manifest.example.json
```

```bash
test -f src/cv_forensics/dataset_manifest.py
```

```bash
test -f configs/manifests/community_forensics_small.example.json
```

```bash
test -f configs/manifests/sid_set.example.json
```

```bash
test -f configs/manifests/combined_smoke_manifest.example.json
```

```bash
test -f scripts/agent/validate_dataset_manifest.py
```

```bash
test -f docs/dataset_manifest.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- src/cv_forensics/__init__.py src/cv_forensics/dataset_manifest.py configs/manifests/community_forensics_small.example.json configs/manifests/sid_set.example.json configs/manifests/combined_smoke_manifest.example.json scripts/agent/validate_dataset_manifest.py tests/test_dataset_manifest.py docs/dataset_manifest.md
```

Optional:

```bash
pytest -q tests/test_dataset_manifest.py
```

## Acceptance Criteria

- `python3 scripts/agent/validate_dataset_manifest.py configs/manifests/community_forensics_small.example.json configs/manifests/sid_set.example.json configs/manifests/combined_smoke_manifest.example.json` passes.
- If pytest is available, `pytest -q tests/test_dataset_manifest.py` passes.
- Changed files are limited to:
  - `src/cv_forensics/__init__.py`
  - `src/cv_forensics/dataset_manifest.py`
  - `configs/manifests/community_forensics_small.example.json`
  - `configs/manifests/sid_set.example.json`
  - `configs/manifests/combined_smoke_manifest.example.json`
  - `scripts/agent/validate_dataset_manifest.py`
  - `tests/test_dataset_manifest.py`
  - `docs/dataset_manifest.md`
- The Community Forensics-Small manifest reflects its role in shared backbone learning and coarse provenance metadata planning.
- The SID-Set manifest reflects its role in 3-way classification and tampered mask localization.
- The combined smoke manifest reflects the four-stage project plan and social-media perturbation plan.
- Manifest examples contain no real local paths, no protected paths, no secrets, no network URLs, and no dataset downloads.
- Validator does not inspect actual data, datasets, outputs, checkpoints, or image files.
- Validator does not check real file existence for image or mask references.
- No `.env`, secrets, data, datasets, outputs, or checkpoints are touched.
- No dataset download, model training, package installation, network access, large file creation, or unrelated modification occurs.

## Stop Condition

Stop after implementation and validation. Report:

- files changed
- validation commands run and their results
- any skipped optional validation and why
- confirmation that forbidden paths and actions were not touched
