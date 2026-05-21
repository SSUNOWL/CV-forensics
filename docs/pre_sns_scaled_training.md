# Pre-SNS Scaled Training

`PRE_SNS_SCALED_TRAINING_PLAN_OK`

This phase prepares larger pre-SNS baseline training configs after the pilot
artifact path has been proven. The config generator only writes a guarded JSON
plan; it does not start training, read images or masks, write checkpoints, or
run SNS augmentation.

## Scaling Path

Suggested increments:

- 2560 samples, 1 epoch
- 8192 samples, 2 epochs
- 20000 samples, 3 epochs
- 50000 samples max for this guarded phase

The validator enforces upper bounds for `max_samples`, `epochs`, `batch_size`,
and `max_image_size` so scaled runs stay controlled.

## Artifact Policy

Generated configs point `approved_run_root` under `~/cvf_runs` and
`approved_checkpoint_root` under `~/cvf_checkpoints`. Those are names only
until the user manually runs the existing training entrypoint. Repository
`outputs/` and `checkpoints/` remain forbidden.

## Scope

This is still pre-SNS training. It does not implement SNS augmentation or SNS
perturbation evaluation. Results from this guarded phase are not final
full-dataset performance claims unless separately approved and documented.
