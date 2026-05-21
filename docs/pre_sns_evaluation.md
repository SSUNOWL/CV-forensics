# Pre-SNS Evaluation

`PRE_SNS_EVALUATION_OK`

This runner evaluates an approved local pre-SNS manifest subset with one
explicit trained pilot checkpoint. It is a baseline evaluation path for the
pre-SNS model, not a final full-dataset performance claim.

## Metrics

The summary reports proposal-aligned metrics:

- `accuracy_3way` for `real / full_synthetic / tampered`
- `macro_f1_3way` computed without sklearn
- `family_accuracy` for samples that include a family label
- `mask_iou_mean` for tampered samples with masks
- `localization_activation_recall` for tampered samples
- `latency_ms_mean` and `fps_estimate`
- sample counts by dataset and class

## Scope

SNS augmentation and SNS perturbation robustness are intentionally absent from
this phase. The runner must not download data, access the network, train, write
checkpoints, recursively scan directories, or evaluate SNS perturbations.

## Artifact Policy

New configs should use `config_kind: approved_pre_sns_evaluation`. The legacy
local alias `approved_local_pre_sns_evaluation` is accepted for compatibility
with existing local config generators. In either approved mode, the approval
phrase must be present before local evaluation can run. The runner reads only
the explicit `manifest_path`, the explicit `checkpoint_path`, and explicit
`image_path` or `mask_path` values already listed in the manifest.

When `write_eval_artifact=false`, the runner prints JSON only. When
`write_eval_artifact=true`, it writes `evaluation_summary.json` under
`approved_eval_root`, which must be outside the repository and not under
repository `outputs/` or `checkpoints/`.
