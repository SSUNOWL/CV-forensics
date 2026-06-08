# SNSAug V2 Balanced Hard-Negative Fine-Tune

SNSAUG_V2_BALANCED_HARD_NEGATIVE_FINETUNE_OK

## Purpose

The 0061 real checkpoint comparison path showed that the 30x3 realweights checkpoint recovered tampered activation under SNSAug, but it became over-tampered-biased. The 0062 threshold sweep used balanced records (`real=520`, `synthetic=520`, `tampered=520`) and still found no threshold candidate with `real_fpr <= 0.20` and `tampered_recall >= 0.50`. That means threshold-only calibration is insufficient.

The corrective run must train against the failure directly: suppress `p_tampered` on real and synthetic SNSAug samples while keeping tampered activation recovery for actual tampered samples.

0063 originally added the config, dry-run planner, guardrails, loss helpers, and tests, but deliberately did not enter the real training branch. 0063b adds the guarded actual branch: when the exact approval text is present, the runner loads the pre-SNS v3 bundle, trains the configured small phase schedule, writes external artifacts, and saves real-weight checkpoints.

## Losses

The new hard-negative term is:

```text
L_non_tampered_tampered_suppression =
  max(0, p_tampered - p_tampered_ceiling)^2
```

It applies only to `real` and `synthetic` labels. Defaults are:

```text
p_tampered_ceiling = 0.05
lambda_hardneg = 2.0
```

`L_tampered_score_consistency` applies only to `tampered` labels. It must not push real or synthetic samples toward tampered activation.

Clean/SNS class consistency for real and synthetic pairs compares only real/synthetic probability mass. This preserves non-tampered class behavior without rewarding tampered overactivation.

Training logs and loss breakdowns must include:

- `mean_p_tampered_real_sns`
- `mean_p_tampered_synthetic_sns`
- `mean_p_tampered_tampered_sns`
- `hardneg_loss`
- `tampered_score_consistency_loss`
- `clean_sns_class_consistency_loss`

## Sampling

Hard SNS phases use balanced hard-negative sampling with approximately:

```text
non_tampered_sns : tampered_sns = 2 : 1
```

Training input remains train-only. Validation manifests, fixed-pair roots, oracle diagnostics, outputs, and checkpoints are never training input.

## Best Checkpoint Policy

The primary score is:

```text
balanced_score =
  + 1.0 * snsaug_tampered_recall
  + 1.0 * snsaug_valid_iou
  + 0.8 * synthetic_recall
  - 2.0 * real_fpr
  - 1.0 * non_tampered_high_mask_rate
```

The first corrective run guardrails are:

- `real_fpr <= 0.30`
- `synthetic_recall >= 0.40`
- `clean_macro_f1 >= 0.75`
- `tampered_recall >= 0.50`

## Outputs

Approved real runs must write only to configured external roots:

- `training_log.jsonl`
- `loss_breakdown.json`
- `per_phase_metrics.json`
- `clean_validation_metrics.json`
- `snsaug_0058c_metrics.json`
- `threshold_sweep_after_training.json`
- `artifact_manifest.json`
- `best/` real `.pt` checkpoint
- last real `.pt` checkpoint

Checkpoint payloads must contain real `model_state_dict` values. Proxy-only `trainable_state` checkpoints are invalid. Balanced hard-negative checkpoints use:

```text
checkpoint_kind = snsaug_v2_balanced_hard_negative_real_model_weights
```

They also record `model_version`, `global_step`, `phase`, `metrics`, `optimizer_state_dict`, and `config_digest`.

## Guardrails

The CLI supports `--dry-run`, which validates the config and prints the plan without training or writing checkpoints. Real training and checkpoint writing require:

```text
approval_text = I_APPROVE_SNSAUG_V2_BALANCED_HARD_NEGATIVE_FINETUNE
```

Without that exact text, non-dry-run execution fails before creating output or checkpoint roots.

The config requires `no_network=true`, `no_download=true`, `no_training_from_scratch=true`, external output/checkpoint roots, train-only manifests, and approved evaluation roots used only for validation/evaluation.
