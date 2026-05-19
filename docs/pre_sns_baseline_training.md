# Pre-SNS Baseline Training Entrypoint

This document describes the guarded final entrypoint before real pre-SNS baseline training.

The entrypoint is not SNS augmentation and is not an SNS evaluation stage. It exists to enforce the last set of local-data and artifact-writing gates before the user starts actual training manually.

Real training requires the exact approval text:

```text
I_APPROVE_PRE_SNS_BASELINE_TRAINING
```

The approved local config must provide:

- a unified pre-SNS manifest path produced by the manifest gate
- an approved run root outside this repository
- an approved weight root outside this repository
- explicit batch, epoch, sample, seed, device, and loss-weight settings

The script refuses to train without explicit approval. It rejects remote references, recursive scan flags, protected repository paths, and repository-local artifact roots named for generated outputs or weights.

The `no_write_dry_run` mode checks the guarded entrypoint without creating run artifacts or weights. It prints a JSON summary instead of writing files.

Actual pre-SNS baseline training should start only after tasks 0020, 0021, 0022, and 0023 are PASS and committed, and after the user creates a local approved config with approved local paths.

Marker: PRE_SNS_BASELINE_TRAINING_ENTRYPOINT_OK
