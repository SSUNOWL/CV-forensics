# Task 0004: Repository Skeleton for Image Forensics Prototype

## Goal

Implement a lightweight repository/module skeleton for the image forensics prototype without downloading datasets, training models, installing packages, or accessing protected paths.

## Purpose

Create a minimal, auditable Python structure that future tasks can extend for:

- target output schema
- class/family/mask/reason result representation
- template-based explanation placeholders
- repository structure documentation
- pure-Python validation

## Files Claude May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0004-repo-skeleton.md`

## Files Claude May Modify

- `src/cv_forensics/__init__.py`
- `src/cv_forensics/contracts.py`
- `src/cv_forensics/outputs.py`
- `src/cv_forensics/evidence.py`
- `scripts/agent/validate_repo_skeleton.py`
- `tests/test_repo_skeleton.py`
- `docs/repo_structure.md`

## Forbidden Actions

- Do not download datasets.
- Do not train a model.
- Do not install packages.
- Do not access `.env`, `.env.*`, `secrets`, `data`, `datasets`, `outputs`, or `checkpoints`.
- Do not run `git push`.
- Do not create large files.
- Do not access network resources from shell commands.
- Do not use `rg`.
- Do not modify files outside `## Files Claude May Modify`.

## Implementation Requirements

- Create a small Python package under `src/cv_forensics`.
- Use only the Python standard library.
- Define lightweight constants or dataclasses for:
  - class labels: `real`, `synthetic`, `tampered`
  - initial family labels: `LatDiff`, `PixDiff`, `GAN`, `Other`, `Real-or-N/A`
  - target output fields: `class`, `class_conf`, `family`, `family_conf`, `localization_head`, `mask_area_pct`, `reason`
- Add a class/family/mask/reason result representation that maps directly to the target output format in `docs/project_brief.md` and `configs/project_contract.json`.
- Add a template-based explanation helper with deterministic placeholder text.
- Add `docs/repo_structure.md` explaining the skeleton and how it maps to `docs/project_brief.md`.
- Add `scripts/agent/validate_repo_skeleton.py` that checks required files and can import the module without package installation.
- Add pytest-style tests in `tests/test_repo_skeleton.py`; they must use only the standard library plus pytest's normal test discovery features and must not require installing any package during the task.
- Keep all files small, text-only, and auditable.

## Validation Commands

```bash
python3 scripts/agent/validate_repo_skeleton.py
pytest -q tests/test_repo_skeleton.py
git status --short --untracked-files=all
git diff -- src/cv_forensics/__init__.py src/cv_forensics/contracts.py src/cv_forensics/outputs.py src/cv_forensics/evidence.py scripts/agent/validate_repo_skeleton.py tests/test_repo_skeleton.py docs/repo_structure.md
```

## Acceptance Criteria

- `python3 scripts/agent/validate_repo_skeleton.py` passes.
- If `pytest` is available, `pytest -q tests/test_repo_skeleton.py` passes; if `pytest` is unavailable, the wrapper may report it as SKIP.
- Changed files are limited to the files listed in `## Files Claude May Modify`.
- The skeleton matches `docs/project_brief.md`, `docs/project_contract.md`, and `configs/project_contract.json`.
- No `.env`, secrets, data, datasets, outputs, or checkpoints are touched.
- No dataset download, model training, package installation, network access, large file creation, or unrelated modification occurs.

## Stop Condition

Stop after implementing the allowed files and running the validation commands. Report changed files, validation results, and any skipped validation command. Do not continue into dataset work, model training, package installation, or any task outside this file.
