# Task 0005: Config Schema for Datasets and Experiments

## Role of Claude Code

Claude Code is the implementation worker. Codex is the supervisor and reviewer.

Implement only what is specified in this task file. Do not reinterpret project scope beyond this task.

## Task Objective

Implement a pure-Python, auditable configuration schema for dataset definitions and experiment definitions, without downloading data, training models, installing packages, accessing network resources, or touching protected paths.

This task should help future work define:

- dataset registry entries for Community Forensics-Small and SID-Set
- dry-run experiment configs
- allowed stages and metrics
- protected path guardrails
- validation rules for config files

## Files Claude May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0005-config-schema.md`
- `docs/repo_structure.md` only if it exists
- `src/cv_forensics/__init__.py` only if it exists
- `src/cv_forensics/contracts.py` only if it exists
- `src/cv_forensics/outputs.py` only if it exists
- `src/cv_forensics/evidence.py` only if it exists
- any file Claude is allowed to create or modify in this task, if it already exists

## Files Claude May Modify

- `src/cv_forensics/__init__.py`
- `src/cv_forensics/config_schema.py`
- `configs/datasets.example.json`
- `configs/experiments/smoke_baseline.json`
- `scripts/agent/validate_config_schema.py`
- `tests/test_config_schema.py`
- `docs/config_schema.md`

If a parent directory for an allowed file does not exist, Claude may create that parent directory. Claude must not create any other files.

## Forbidden Actions

Claude must not:

- download datasets
- train a model
- install packages
- access network resources from shell commands
- access `.env`, `.env.*`, `secrets`, `data`, `datasets`, `outputs`, or `checkpoints`
- inspect protected directories or protected files recursively
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

## Implementation Requirements

### 1. Create or update `src/cv_forensics/config_schema.py`

- Use only Python standard library.
- Do not import third-party packages.
- Define constants for:
  - class labels: `real`, `synthetic`, `tampered`
  - family labels: `LatDiff`, `PixDiff`, `GAN`, `Other`, `Real-or-N/A`
  - dataset ids: `community_forensics_small`, `sid_set`
  - implementation stages:
    - `cf_small_binary_backbone`
    - `provenance_head`
    - `sid_set_multihead_finetune`
    - `social_media_robustness`
    - `dry_run`
  - metrics:
    - `accuracy_3way`
    - `macro_f1`
    - `tampered_mask_iou`
    - `generator_family_accuracy`
    - `perturbation_robustness_drop`
    - `latency_ms`
    - `fps`
    - `localization_activation_recall`
- Define validation helpers for:
  - dataset config dictionaries
  - experiment config dictionaries
  - protected path detection
  - uniqueness of dataset ids
  - experiment dataset references
  - dry-run experiment safety
- Keep the module lightweight and deterministic.
- Do not read real data.
- Do not access `data`, `datasets`, `outputs`, `checkpoints`, `.env`, or `secrets`.

### 2. Create `configs/datasets.example.json`

- It must be an example config only.
- It must not contain real absolute local paths.
- It must not contain secrets, tokens, API keys, usernames, or machine-specific paths.
- It should define Community Forensics-Small and SID-Set entries.
- Community Forensics-Small entry should include:
  - `id`
  - `display_name`
  - primary roles
  - placeholder root such as `<CF_SMALL_ROOT>`
  - split placeholders
  - required metadata fields: `architecture`, `model_name`, `subset`
  - supported labels for binary class and family labels
- SID-Set entry should include:
  - `id`
  - `display_name`
  - primary roles
  - placeholder root such as `<SID_SET_ROOT>`
  - split placeholders
  - required target fields for 3-way class and tampered mask localization
  - note that family/provenance labels may be unavailable and should be masked, frozen, or mixed with CF-Small depending on stage
- Include protected path rules as metadata or guardrail notes.

### 3. Create `configs/experiments/smoke_baseline.json`

- It must be a dry-run example only.
- It must not request real training.
- It must not request data download.
- It must not contain real data paths.
- It should reference dataset ids from `configs/datasets.example.json`.
- It should define:
  - `schema_version`
  - `experiment_id`
  - `stage`
  - `dry_run: true`
  - `seed`
  - `image_size`
  - `batch_size`
  - dataset references
  - model/backbone placeholder
  - heads: `classification`, `provenance`, `localization`, `explanation_template`
  - losses as names only
  - metrics list
  - safety flags such as `no_download`, `no_training`, `no_network`, `no_protected_paths`
- It should be suitable for future extension but must remain non-executable.

### 4. Create `scripts/agent/validate_config_schema.py`

- Use only Python standard library.
- Accept exactly two positional arguments:
  - dataset config JSON path
  - experiment config JSON path
- Validate both JSON files using `src/cv_forensics/config_schema.py`.
- Be import-safe without package installation.
- It may add the repository root or `src` directory to `sys.path`.
- Print a concise success message on pass.
- Print actionable errors and exit non-zero on fail.
- Reject configs that include:
  - protected paths
  - absolute local machine paths such as `/home/`, `/mnt/`, `/root/`
  - `.env` or `.env.*`
  - `secrets`
  - `data/`
  - `datasets/`
  - `outputs/`
  - `checkpoints/`
  - network URLs
  - missing required dataset ids
  - experiment references to unknown dataset ids
  - `dry_run` false in the smoke example

### 5. Create `docs/config_schema.md`

- Explain the purpose of the dataset and experiment config schemas.
- Explain how the schema maps to `docs/project_brief.md` and `configs/project_contract.json`.
- Explain that current configs are examples only and do not point to real data.
- Explain the Community Forensics-Small and SID-Set roles.
- Explain why SID-Set family/provenance labels may need freeze, mask-out, or mixed-batch handling.
- Explain validation commands.
- Explain forbidden paths and safety guardrails.
- Keep it concise but useful.

### 6. Create `tests/test_config_schema.py`

- Use pytest-style assertions.
- Do not install pytest.
- Keep tests lightweight.
- Tests should be pure Python and not access external resources.
- Tests should cover:
  - valid example configs pass
  - protected paths are rejected
  - unknown dataset references are rejected
  - smoke experiment must be `dry_run` true
  - required metrics or labels are present
- If pytest is not installed locally, it is acceptable that the pytest command is skipped by the wrapper.

### 7. Update `src/cv_forensics/__init__.py` only if needed

- If the file exists, Claude may add a lightweight export for `config_schema`.
- If the file does not exist, Claude may create a minimal package marker.
- Do not add heavy imports.

## Validation Commands

```bash
python3 scripts/agent/validate_config_schema.py configs/datasets.example.json configs/experiments/smoke_baseline.json
```

```bash
test -f configs/datasets.example.json
```

```bash
test -f configs/experiments/smoke_baseline.json
```

```bash
test -f src/cv_forensics/config_schema.py
```

```bash
test -f scripts/agent/validate_config_schema.py
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- src/cv_forensics/__init__.py src/cv_forensics/config_schema.py configs/datasets.example.json configs/experiments/smoke_baseline.json scripts/agent/validate_config_schema.py tests/test_config_schema.py docs/config_schema.md
```

Optional validation:

```bash
pytest -q tests/test_config_schema.py
```

## Acceptance Criteria

- `python3 scripts/agent/validate_config_schema.py configs/datasets.example.json configs/experiments/smoke_baseline.json` passes.
- If pytest is available, `pytest -q tests/test_config_schema.py` passes.
- Changed files are limited to:
  - `src/cv_forensics/__init__.py`
  - `src/cv_forensics/config_schema.py`
  - `configs/datasets.example.json`
  - `configs/experiments/smoke_baseline.json`
  - `scripts/agent/validate_config_schema.py`
  - `tests/test_config_schema.py`
  - `docs/config_schema.md`
- The configs match `docs/project_brief.md` and `configs/project_contract.json`.
- Config examples contain no real local paths, no secrets, no protected paths, no network URLs, and no dataset downloads.
- The smoke experiment remains dry-run only.
- No `.env`, `secrets`, `data`, `datasets`, `outputs`, or `checkpoints` are touched.
- No dataset download, model training, package installation, network access, large file creation, or unrelated file modification occurs.

## Stop Condition

Stop after implementation and validation. Report:

- files changed
- validation commands run and their results
- any skipped optional validation and why
- confirmation that forbidden paths and actions were not touched
