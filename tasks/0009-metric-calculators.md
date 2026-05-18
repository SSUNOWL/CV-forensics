# Task 0009: Pure-Python Metric Calculators with Toy Arrays

## Role of Claude Code

Claude Code is the implementation worker. Codex is the supervisor, task writer, delegation controller, reviewer, and limited repair manager.

Implement this task exactly as written. Do not reinterpret the project scope beyond this task file. Stop after implementation and validation, then report the requested results.

## Task Objective

Implement pure-Python metric calculators with toy arrays for the lightweight multi-head image forensics prototype.

This task defines lightweight metrics that future dry-run and real evaluation tasks can reuse. It must connect to the task 0007 model output schema and task 0008 fake inference outputs, but it must not use real datasets, real images, real masks, model checkpoints, training loops, package installation, or network access.

This is a metrics-only dry-run task using toy labels, toy nested-list binary masks, toy localization states, toy latency values, and toy perturbation groups.

## Files Claude May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `docs/model_output_schema.md`
- `docs/inference_stub.md`
- `tasks/0009-metric-calculators.md`
- `src/cv_forensics/model_output_schema.py`
- `src/cv_forensics/explanation_templates.py`
- `src/cv_forensics/inference_stub.py`
- `configs/inference/fake_inputs.example.json`
- `scripts/agent/run_fake_inference.py`
- `scripts/agent/validate_inference_stub.py`
- any file Claude is allowed to create or modify in this task, if it already exists

## Files Claude May Modify

- `src/cv_forensics/__init__.py`
- `src/cv_forensics/metrics.py`
- `configs/metrics/toy_metrics.example.json`
- `scripts/agent/validate_metrics.py`
- `scripts/agent/run_toy_metrics.py`
- `tests/test_metrics.py`
- `docs/metrics.md`

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
- read real mask files
- validate actual image or mask file existence
- walk real dataset directories
- write output files
- create `outputs/`
- read or write model checkpoints
- implement real model inference
- implement training loops
- implement SNS augmentation
- modify unrelated files
- run `git push`
- run `git pull`
- run `git fetch`
- run `curl`, `wget`, `ssh`, `scp`, `rsync`, `pip`, `pip3`, `npm`, `yarn`, `pnpm`, `kaggle`, `huggingface-cli`, or `wandb`
- create large files
- use `rg`
- use `rm -rf`
- use `--permission-mode bypassPermissions`
- use `--dangerously-skip-permissions`
- import `numpy`, `pandas`, `sklearn`, `torch`, `torchvision`, `PIL`, `cv2`, `pytest`, or any third-party package

## Important Project Facts

- The final prototype should output class, mask/localization, family/provenance, and reason together.
- The class labels are:
  - `real`
  - `synthetic`
  - `tampered`
- The family/provenance labels are coarse family-level labels, not exact model attribution.
- Initial family labels are:
  - `LatDiff`
  - `PixDiff`
  - `GAN`
  - `Other`
  - `Real-or-N/A`
- Conditional localization activates when `tampered_score >= threshold_tau`.
- Localization activation recall matters because a high `threshold_tau` can cause false negatives.
- Social-media perturbation robustness drop should be representable as a metric, but SNS augmentation itself must not be implemented in this task.
- Task 0007 defines the final forensic output schema.
- Task 0008 defines deterministic fake inference outputs that future metric evaluation can consume.
- This task is metrics-only with toy arrays and fake outputs.

## Implementation Requirements

### 1. Create `src/cv_forensics/metrics.py`

- Use only Python standard library.
- Do not import third-party packages.
- Use plain Python lists, dictionaries, dataclasses, and `math` only.
- Keep all metric behavior deterministic.
- Import label constants from `src/cv_forensics/model_output_schema.py` where practical.
- Do not read files inside the metric functions.
- Do not read images, masks, datasets, outputs, checkpoints, `.env`, or secrets.
- Do not write files.
- Implement robust validation helpers for toy metric inputs.
- Reject invalid labels, invalid mask shapes, invalid localization states, invalid latency values, invalid FPS inputs, and invalid perturbation groups with actionable `ValueError` or `TypeError` messages.
- Implement recursive safety validation for toy metric config dictionaries and lists. It must reject:
  - `.env`
  - `.env.*`
  - `secrets`
  - `data`
  - `datasets`
  - `outputs`
  - `checkpoints`
  - absolute local paths such as `/home/`, `/mnt/`, `/root/`, `/Users/`
  - Windows drive paths
  - URLs such as `http://`, `https://`, `s3://`, `gs://`, `hf://`
  - secret-looking keys or values such as `token`, `api_key`, `password`, `secret`, `credential`, `auth`, or `bearer`

### 2. Classification Metrics

Implement:

- confusion matrix for labels `real`, `synthetic`, `tampered`
- accuracy
- per-class precision
- per-class recall
- per-class F1
- Macro-F1

Requirements:

- Use the fixed label order `real`, `synthetic`, `tampered`.
- Reject unknown ground-truth or predicted class labels.
- Handle zero-division safely by returning `0.0` for undefined precision, recall, or F1.
- Return structured dictionaries suitable for JSON serialization.

### 3. Family Metrics

Implement generator-family accuracy using valid labels:

- `LatDiff`
- `PixDiff`
- `GAN`
- `Other`
- `Real-or-N/A`

Requirements:

- Reject unknown family labels.
- Support a configuration option to ignore or mask `Real-or-N/A` where appropriate.
- Return the number of evaluated examples, number correct, ignored count, and accuracy.
- Treat family labels as coarse provenance only, not exact model attribution.

### 4. Localization Metrics

Implement:

- binary mask IoU for tiny toy masks represented as nested Python lists
- localization activation recall for tampered examples

Requirements:

- Masks must be rectangular nested lists containing only `0`, `1`, `False`, or `True`.
- Reject empty masks, ragged masks, invalid values, or mismatched shapes.
- IoU should be `intersection / union`.
- If both masks are all-zero, define IoU as `1.0` for this toy metric and document that convention.
- Localization states must support at least:
  - `activated`
  - `skipped_below_threshold`
  - `not_applicable`
  - `unavailable`
- Localization activation recall should count true tampered examples and report the fraction with `localization_head == "activated"`.
- It must explicitly account for `skipped_below_threshold` as a missed activation for tampered examples.

### 5. Runtime Metrics

Implement:

- latency summary for `latency_ms` values:
  - mean
  - min
  - max
  - count
- FPS summary computed from latency or explicit timing records

Requirements:

- Reject negative, zero, non-numeric, NaN, or infinite latency values.
- FPS from latency should use `1000.0 / latency_ms`.
- Return structured JSON-serializable dictionaries.

### 6. Robustness Metrics

Implement perturbation robustness drop using toy grouped score records.

Requirements:

- Support perturbation tags:
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
- Do not implement augmentation itself.
- Accept toy clean and perturbed score records.
- Compute drop as `clean_score - perturbed_score`.
- Return per-perturbation and aggregate summaries.
- Reject unknown perturbation tags or non-numeric scores.
- Document that this is a placeholder/calculator for later SNS robustness evaluation.

### 7. Example Config

Create `configs/metrics/toy_metrics.example.json`.

The example must:

- be dry-run only
- contain safety flags such as `dry_run`, `no_download`, `no_training`, and `no_network`
- contain toy class labels and predictions
- contain toy family labels and predictions
- contain tiny toy binary masks as nested lists
- contain toy localization states including `activated` and `skipped_below_threshold`
- contain toy latency values
- contain toy perturbation groups
- include perturbation tags only as metadata for metric calculation
- not contain real paths, URLs, secrets, datasets, outputs, checkpoints, image references, or mask file references
- not point to images or masks
- not write outputs

### 8. Scripts

Create `scripts/agent/validate_metrics.py`.

- Use only Python standard library.
- Be import-safe without package installation.
- Add the repository `src` path to `sys.path` if needed.
- Accept exactly one positional argument: the toy metrics config JSON path.
- Validate the toy metrics config.
- Run all metric calculators on the toy config.
- Validate that metric results are structured, finite, deterministic, and semantically plausible.
- Validate that unsafe config values are rejected by the safety checker.
- Print a concise success message on pass.
- Print actionable errors and exit non-zero on failure.
- Do not write files.

Create `scripts/agent/run_toy_metrics.py`.

- Use only Python standard library.
- Be import-safe without package installation.
- Add the repository `src` path to `sys.path` if needed.
- Accept exactly one positional argument: the toy metrics config JSON path.
- Read and validate the toy config.
- Run all metric calculators.
- Print a JSON summary to stdout.
- Do not write files.

### 9. Tests

Create `tests/test_metrics.py`.

- Use only Python standard library.
- Do not import `pytest`.
- Keep pytest-style test functions using plain `assert`.
- Make the file runnable directly with:

```bash
python3 tests/test_metrics.py
```

- Keep it pytest-compatible if pytest is installed.
- Tests should cover:
  - accuracy
  - Macro-F1
  - zero-division safe precision, recall, and F1
  - mask IoU
  - localization activation recall
  - family accuracy
  - latency and FPS
  - robustness drop
  - invalid labels rejected
  - invalid mask shapes rejected
  - protected path, URL, and secret-like config values rejected

### 10. Documentation

Create `docs/metrics.md`.

The document must explain:

- purpose of metrics
- how metrics map to the project proposal
- how metrics connect to task 0007 model output schema and task 0008 fake inference outputs
- why toy arrays are used before real data
- why SNS robustness drop is a placeholder/calculator before actual augmentation
- that no real data, images, masks, outputs, checkpoints, training, downloads, network access, or package installation are used
- validation commands
- marker string `METRICS_OK`

Keep the document concise but useful.

## Validation Commands

```bash
python3 scripts/agent/validate_metrics.py configs/metrics/toy_metrics.example.json
```

```bash
python3 scripts/agent/run_toy_metrics.py configs/metrics/toy_metrics.example.json
```

```bash
python3 tests/test_metrics.py
```

```bash
test -f src/cv_forensics/metrics.py
```

```bash
test -f configs/metrics/toy_metrics.example.json
```

```bash
test -f scripts/agent/validate_metrics.py
```

```bash
test -f scripts/agent/run_toy_metrics.py
```

```bash
test -f docs/metrics.md
```

```bash
grep -q METRICS_OK docs/metrics.md
```

```bash
git status --short --untracked-files=all
```

```bash
git diff -- src/cv_forensics/__init__.py src/cv_forensics/metrics.py configs/metrics/toy_metrics.example.json scripts/agent/validate_metrics.py scripts/agent/run_toy_metrics.py tests/test_metrics.py docs/metrics.md
```

Optional, if pytest is installed:

```bash
pytest -q tests/test_metrics.py
```

## Acceptance Criteria

- `python3 scripts/agent/validate_metrics.py configs/metrics/toy_metrics.example.json` passes.
- `python3 scripts/agent/run_toy_metrics.py configs/metrics/toy_metrics.example.json` prints a JSON summary to stdout.
- `python3 tests/test_metrics.py` passes.
- If pytest exists, `pytest -q tests/test_metrics.py` also passes.
- Changed files are limited to the allowed task files.
- Metrics are pure Python and deterministic.
- Tests do not import `pytest` or third-party packages.
- No real data, images, masks, outputs, checkpoints, secrets, package installs, training, downloads, or network access are used.
- Robustness drop is represented as a metric calculator only; SNS augmentation is not implemented.
- The example config is dry-run only and contains no protected paths, URLs, secrets, datasets, outputs, checkpoints, or image/mask references.
- `docs/metrics.md` contains `METRICS_OK`.

## Stop Condition

After implementation, Claude must stop and report:

- files changed
- validation commands run and their results
- confirmation that no real datasets, images, masks, checkpoints, outputs, secrets, package installation, training, downloads, network access, or SNS augmentation were used
- any skipped optional validation, with the reason

Claude must not commit changes.
