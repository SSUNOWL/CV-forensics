# Training Dry-Run Skeleton

`TRAINING_DRY_RUN_OK`

## What This Is

This module (`src/cv_forensics/training_dry_run.py`) is a **pure-Python dry-run
skeleton**, not real training. It simulates a training/evaluation loop using fake
inputs and toy metrics. No model weights are updated, no real images are loaded, no
real datasets are accessed, no checkpoints are written, and no output files are
created.

## Purpose

The dry-run skeleton verifies that the project's training loop components are
correctly wired together before real local data is available. It connects:

- **Config schema** (`config_schema.py`): stage, metric names, family policy, seed.
- **Dataset manifests** (`dataset_manifest.py`): symbolic dataset references only
  (no real paths).
- **Fake inference** (`inference_stub.py`): deterministic fake outputs for each
  scenario — real, synthetic, tampered-localized, tampered-below-threshold.
- **Toy metrics** (`metrics.py`): classification accuracy, Macro-F1, mask IoU,
  family accuracy, localization activation recall, latency, FPS.

## What the Dry-Run Loop Does

1. Loads a dry-run training config JSON.
2. Validates all safety flags (`dry_run`, `no_training`, `no_network`, `no_outputs`,
   `no_checkpoints`).
3. Rejects any config containing real paths, URLs, secrets, or protected directories.
4. Simulates `epochs` epochs, each with a fixed number of fake batches.
5. Each batch rotates deterministically through four canonical scenarios using `seed`
   as offset (no randomness, fully reproducible).
6. Calls fake inference for each fake record.
7. Calls toy metrics on the collected fake outputs.
8. Produces an epoch summary and a final metrics dict keyed by `metric_names`.

## What It Does Not Do

- **No real training**: model parameters are never updated.
- **No real data**: no images, no masks, no dataset files are read.
- **No checkpoints**: nothing is written to disk.
- **No network access**: no downloads, no API calls.
- **No SNS augmentation**: social-media perturbation augmentation is not implemented
  here. It begins only after the pre-SNS baseline evaluation is committed.

## What Is Needed Before Real Training

Before switching from dry-run to real training, the following must be confirmed:

1. **Explicit user approval**: real training must be explicitly authorized.
2. **Local data readiness gate**: datasets must exist at the configured local paths.
3. **Validated manifests**: dataset manifests must be reviewed and validated against
   the actual local data layout.
4. **Output and checkpoint policy**: paths for saving checkpoints and predictions must
   be agreed upon and configured.
5. **Small local subset smoke test**: a real training run on a small local subset
   (e.g., 100 samples) must pass before full-scale training begins.

## Key Files

| File | Role |
|---|---|
| `src/cv_forensics/training_dry_run.py` | Core dry-run module |
| `configs/training/dry_run_training.example.json` | Example dry-run config |
| `scripts/agent/run_training_dry_run.py` | CLI runner (prints JSON summary) |
| `scripts/agent/validate_training_dry_run.py` | Validator (exit 0 on pass) |
| `tests/test_training_dry_run.py` | Test suite (pytest-compatible) |

## Validation Commands

```bash
python3 scripts/agent/validate_training_dry_run.py configs/training/dry_run_training.example.json
python3 scripts/agent/run_training_dry_run.py configs/training/dry_run_training.example.json
python3 tests/test_training_dry_run.py
```

## SNS Augmentation Note

Social-media perturbation augmentation (screenshot, text overlay, sticker overlay,
recompression chain) is **not implemented** in this module. The toy metrics module
(`metrics.py`) includes a `perturbation_robustness_drop` calculator as a placeholder,
but actual augmentation is deferred until the pre-SNS baseline is evaluated.
