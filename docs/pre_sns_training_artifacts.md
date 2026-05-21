# Pre-SNS Training Artifacts

`PRE_SNS_TRAINING_ARTIFACTS_OK`

This phase writes artifacts only for an explicitly approved pre-SNS baseline
pilot training run. It is not SNS augmentation, not SNS perturbation
evaluation, and not a final full-dataset performance claim.

## Artifact policy

Training artifacts must be written outside the repository. The intended local
locations are:

- `~/cvf_runs` for run summaries and JSON artifacts
- `~/cvf_checkpoints` for small pilot checkpoints

Repository `outputs/` and `checkpoints/` are forbidden. Paths containing
protected segments such as `.env`, secrets, data, datasets, outputs, or
checkpoints are rejected.

## Generated files

Actual write runs produce:

- `config_snapshot.json`
- `manifest_snapshot.json`
- `metrics_summary.json`
- `run_summary.json`
- `artifact_manifest.json`
- a small model checkpoint under the approved checkpoint root

`artifact_manifest.json` records the generated artifact paths and file sizes.

## Dry run vs actual run

When `no_write_dry_run=true`, the training entrypoint validates the config and
manifest and runs a minimal forward/backward smoke loop that reports finite loss
and metric fields. It must not create run artifacts or checkpoints. A successful
dry run reports `marker: PRE_SNS_BASELINE_TRAINING_DRY_RUN_OK`,
`entrypoint_marker: PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK`,
`training_started: true`, and `training_completed: true`.

When `no_write_dry_run=false`, the training entrypoint creates the approved run
and checkpoint roots, rejects non-empty roots unless explicit overwrite is
enabled, writes the JSON artifacts, and writes a small checkpoint outside the
repository. A successful actual run reports
`marker: PRE_SNS_BASELINE_TRAINING_RUN_OK`,
`entrypoint_marker: PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK`,
`artifact_manifest_path`, and `checkpoint_path`.

SNS augmentation is intentionally absent from this phase.
