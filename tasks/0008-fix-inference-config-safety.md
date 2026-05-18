# Task 0008 Fix: Inference Config Safety Validation

## Role

Claude Code is the implementation worker. Codex is the supervisor and reviewer.

This is a NEEDS_FIX repair task for `tasks/0008-lightweight-inference-stub.md`.

The current implementation has a fake inference flow, reuses the task 0007 model output schema and explanation templates, preserves conditional localization behavior, and passes validation for the clean example config. The remaining issue is that raw fake input configs are not proven to reject protected paths, URL-like references, or secret-looking fields before inference.

## Files Claude May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `docs/model_output_schema.md`
- `docs/inference_stub.md`
- `tasks/0008-lightweight-inference-stub.md`
- `tasks/0008-fix-inference-config-safety.md`
- `src/cv_forensics/model_output_schema.py`
- `src/cv_forensics/explanation_templates.py`
- `src/cv_forensics/inference_stub.py`
- `configs/inference/fake_inputs.example.json`
- `scripts/agent/run_fake_inference.py`
- `scripts/agent/validate_inference_stub.py`
- `tests/test_inference_stub.py`

## Files Claude May Modify

- `src/cv_forensics/__init__.py`
- `src/cv_forensics/inference_stub.py`
- `configs/inference/fake_inputs.example.json`
- `scripts/agent/run_fake_inference.py`
- `scripts/agent/validate_inference_stub.py`
- `tests/test_inference_stub.py`
- `docs/inference_stub.md`

## Files Claude Must Not Modify

- `tasks/0008-lightweight-inference-stub.md`
- `tasks/0008-fix-inference-config-safety.md`
- Any unrelated files

## Forbidden Actions

Claude must not:

- Download datasets
- Train a model
- Install packages
- Access network resources from shell commands
- Access `.env`, `.env.*`, secrets, data, datasets, outputs, or checkpoints
- Inspect protected directories or protected files recursively
- Read real image files
- Read real mask files
- Validate actual image or mask file existence
- Walk real dataset directories
- Write generated predictions to `outputs/`
- Run `git push`
- Run `git pull`
- Run `git fetch`
- Run `curl`, `wget`, `ssh`, `scp`, `rsync`, `pip`, `npm`, `yarn`, `pnpm`, `kaggle`, `huggingface-cli`, or `wandb`
- Create large files
- Use `rg`
- Use `rm -rf`
- Use `--permission-mode bypassPermissions`
- Use `--dangerously-skip-permissions`
- Import `torch`, `torchvision`, `PIL`, `cv2`, `numpy`, `pandas`, `sklearn`, or any third-party package
- Import `pytest`

## Implementation Requirements

### 1. Strengthen raw fake input config validation

- The validator must reject unsafe references inside the raw fake input config before or during fake inference.
- It must not only inspect generated outputs.
- It must recursively inspect config dictionaries and lists for unsafe keys and unsafe string values.
- This recursive check should apply to all fake input records and relevant top-level config fields.

### 2. Protected path and URL detection

The validation must reject string values that look like:

- `.env`
- `.env.*`
- `secrets`
- `secrets/...`
- `data`
- `data/...`
- `./data/...`
- `datasets`
- `datasets/...`
- `outputs`
- `outputs/...`
- `checkpoints`
- `checkpoints/...`
- `/home/...`
- `/mnt/...`
- `/root/...`
- `/Users/...`
- Windows drive paths such as `C:\...`
- `http://...`
- `https://...`
- `s3://...`
- `gs://...`
- `hf://...`

### 3. Secret-looking keys and values

The validation must reject key names or values that indicate secrets, such as:

- `token`
- `api_key`
- `password`
- `secret`
- `credential`
- `auth`
- `bearer`

### 4. Avoid false positives on ordinary prose

- The validator should be reasonably path-aware.
- It may reject URL-like strings and path-like strings aggressively.
- It should not reject ordinary documentation text merely because it mentions safety in prose.
- It should reject explicit unsafe path, URL, or secret-looking values in config fields.
- It should reject suspicious reference-like keys such as `image_ref`, `mask_ref`, `checkpoint_ref`, `url_ref`, `output_ref`, `dataset_ref`, `path`, `root`, `uri`, `url` if their values contain unsafe strings.

### 5. Strengthen `tests/test_inference_stub.py`

Add tests that prove malicious fake configs are rejected.

- The tests must use standard library only.
- Do not import `pytest`.
- The tests must remain runnable with:

```bash
python3 tests/test_inference_stub.py
```

Tests should cover at least:

- Fake input with `image_ref` set to `data/foo.jpg` is rejected
- Fake input with `image_ref` set to `./data/foo.jpg` is rejected
- Fake input with `mask_ref` set to `outputs/mask.png` is rejected
- Fake input with `checkpoint_ref` set to `checkpoints/model.pt` is rejected
- Fake input with `url_ref` set to `https://example.com/image.jpg` is rejected
- Fake input with `api_key` field is rejected
- Fake input with `token` field is rejected
- Fake input with secret-like value is rejected
- Fake input with `/home/example/file.jpg` is rejected
- Fake input with `C:\Users\example\file.jpg` is rejected
- Current `configs/inference/fake_inputs.example.json` still passes

### 6. Strengthen `scripts/agent/validate_inference_stub.py`

- It should fail if the raw fake input config contains unsafe references.
- It should still validate generated final forensic outputs.
- It should print actionable errors.
- It should exit non-zero on unsafe config.
- It should continue to pass for `configs/inference/fake_inputs.example.json`.
- It should not read real image files.
- It should not check real file existence for image, mask, output, dataset, or checkpoint references.

### 7. Strengthen `scripts/agent/run_fake_inference.py` if needed

- It should also reject unsafe fake input configs before running fake inference.
- It should not produce output for unsafe configs.
- It should still print a JSON summary for the valid example config.
- It must not write files.
- It must not create `outputs/`.

### 8. Strengthen `src/cv_forensics/inference_stub.py` if needed

- It may expose reusable config safety helpers if that keeps validation and runner behavior consistent.
- Keep standard library only.
- Keep fake inference deterministic.
- Continue reusing task 0007 model output schema and explanation helpers.
- Do not implement real inference, model loading, image reading, or SNS augmentation.

### 9. Update `docs/inference_stub.md` if useful

- Document that fake input configs are recursively checked for protected paths, URLs, secret-looking fields, and protected references.
- Keep marker string:
  - `INFERENCE_STUB_OK`

## Validation Commands

Run each command separately.

```bash
python3 scripts/agent/check_agent_changes.py tasks/0008-fix-inference-config-safety.md
```

```bash
python3 scripts/agent/validate_inference_stub.py configs/inference/fake_inputs.example.json
```

```bash
python3 scripts/agent/run_fake_inference.py configs/inference/fake_inputs.example.json
```

```bash
python3 tests/test_inference_stub.py
```

```bash
grep -q INFERENCE_STUB_OK docs/inference_stub.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- src/cv_forensics/__init__.py src/cv_forensics/inference_stub.py configs/inference/fake_inputs.example.json scripts/agent/run_fake_inference.py scripts/agent/validate_inference_stub.py tests/test_inference_stub.py docs/inference_stub.md
```

Optional validation command:

```bash
pytest -q tests/test_inference_stub.py
```

## Acceptance Criteria

- `python3 scripts/agent/validate_inference_stub.py configs/inference/fake_inputs.example.json` passes.
- `python3 scripts/agent/run_fake_inference.py configs/inference/fake_inputs.example.json` runs and prints a JSON summary.
- `python3 tests/test_inference_stub.py` passes.
- If `pytest` is available, `pytest -q tests/test_inference_stub.py` passes.
- Changed files are limited to:
  - `src/cv_forensics/__init__.py`
  - `src/cv_forensics/inference_stub.py`
  - `configs/inference/fake_inputs.example.json`
  - `scripts/agent/run_fake_inference.py`
  - `scripts/agent/validate_inference_stub.py`
  - `tests/test_inference_stub.py`
  - `docs/inference_stub.md`
- Raw fake input configs containing protected paths are rejected.
- Raw fake input configs containing URLs are rejected.
- Raw fake input configs containing secret-looking keys or values are rejected.
- The valid `configs/inference/fake_inputs.example.json` still passes.
- The fake inference outputs still validate against the task 0007 model output schema.
- No real images, masks, datasets, outputs, checkpoints, `.env`, or secrets are accessed.
- No output files are written by the fake inference runner.
- No dataset download, model training, package installation, network access, large file creation, or unrelated modification occurs.
- `INFERENCE_STUB_OK` remains present in `docs/inference_stub.md`.

## Stop Condition

Stop after implementation and validation. Report:

- Files changed
- Validation commands run and results
- Proof that malicious fake configs are rejected
- Confirmation that the valid fake input example still passes
- Confirmation that forbidden paths and actions were not touched
- Confirmation that no dataset access, training, package installation, network access, real image reading, output writing, or SNS augmentation occurred
