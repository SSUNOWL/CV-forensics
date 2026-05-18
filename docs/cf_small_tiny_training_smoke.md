# CF-Small Local Tiny Training Smoke

<!-- CF_SMALL_TINY_TRAIN_SMOKE_OK -->

This document describes the Community Forensics-Small local tiny training smoke
workflow for non-SNS baseline verification. It is the first controlled bridge
from dry-run planning into explicitly approved local non-SNS data.

The smoke verifies:

- local manifest connection
- tiny image loading from explicit sample paths
- binary real-vs-synthetic label handling
- minimal model forward/backward
- finite loss
- smoke accuracy plumbing

This is not full training and not a performance claim. It is only a small
connectivity and plumbing check before any full baseline training approval.

No SNS augmentation is used. No SNS perturbation evaluation is run. No outputs
or checkpoints are written by default.

Actual user-run mode requires an untracked local config with:

- `config_kind: approved_local_smoke`
- `approved_real_data_access: true`
- `user_approval_text: I_APPROVE_LOCAL_NON_SNS_TINY_TRAINING_SMOKE`
- explicit `approved_local_roots`
- explicit `sample_manifest` entries

After this passes, the next natural task is SID-Set local tiny multi-head smoke.
