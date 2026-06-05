# Task: 0058e SNSAug V2 Forced Localization Oracle Gate Diagnostic

## Task Title

Add an evaluation-only SNSAug V2 diagnostic comparing normal conditional localization, oracle tampered localization, and tampered-score threshold sweep.

## Role

Codex-only task writer, implementation worker, reviewer, and limited repair manager for an evaluation/reporting-only diagnostic task.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0058-snsaug-v2-small-benchmark-with-geometry-and-postprocess.md`
- `tasks/0058c-snsaug-v2-small-benchmark-class-balanced-and-tampered-mask-fix.md`
- `tasks/0058d-snsaug-v2-export-pred-redmask-for-visual-comparison.md`
- `docs/snsaug_v2_fixed_pairs_eval.md`
- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py`
- `scripts/evaluation/run_snsaug_v2_fixed_pairs_eval.py`
- `tests/test_snsaug_v2_fixed_pairs_eval.py`

## Files Codex May Modify

- `src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py`
- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `src/cv_forensics/snsaug_v2_forced_localization_oracle_gate.py`
- `scripts/evaluation/run_snsaug_v2_forced_localization_oracle_gate.py`
- `tests/test_snsaug_v2_forced_localization_oracle_gate.py`
- `docs/snsaug_v2_forced_localization_oracle_gate.md`

## Forbidden Actions

- Do not train models.
- Do not fine-tune models.
- Do not run `0054`.
- Do not use validation failures for training.
- Do not access network resources from shell commands.
- Do not download datasets, assets, or checkpoints.
- Do not install packages.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- Do not inspect protected directories recursively.
- Do not modify model checkpoints.
- Do not write generated benchmark/evaluation outputs inside the repository.
- Do not run the real diagnostic on the user-provided fixed-pair root unless the user explicitly approves that run.
- Do not modify SNSAug generation code or unrelated model/training code.
- Do not use Claude Code while Claude Code is unavailable.
- Do not use `rg`.
- Do not run `git push`, `git pull`, or `git fetch`.

## Important Project Facts

- `0058c` produced a class-balanced SNSAug V2 small benchmark and made denominator-zero metrics explicit.
- `0058d` exported predicted redmask and overlay visualizations from fixed-pair evaluation.
- Balanced SNSAug V2 evaluation showed strong degradation under SNS/crop/resize/canvas profiles.
- Manual inspection suggests many tampered SNSAug samples suffer tampered-score collapse, so conditional localization is not activated and predicted redmasks are empty.
- Occlusion analysis suggests high local SNS overlay occlusion is not the main cause.
- The diagnostic must determine whether failure is mainly:
  - class/activation bottleneck, or
  - mask decoder / geometry localization bottleneck.
- The real fixed-pair root for later user-approved execution is:
  - `/home/rlatjswo/cvf_runs/snsaug_v2_0058c_small_val_pairs_20pc_balanced_20260605_220707`
- The frozen best model bundle for later user-approved execution is:
  - `.local/pre_sns_current_best_model_bundle.json`
- Implementation tests must use small temporary fixtures only, not the real benchmark root.

## Implementation Requirements

1. Add an evaluation-only diagnostic module:
   - `src/cv_forensics/snsaug_v2_forced_localization_oracle_gate.py`
2. Add a CLI wrapper:
   - `scripts/evaluation/run_snsaug_v2_forced_localization_oracle_gate.py`
3. The CLI must support `--help` without requiring data paths or model bundle paths.
4. The diagnostic must compare these modes:
   - `normal_gate`
   - `oracle_tampered_gate`
   - `threshold_sweep`
5. `normal_gate` must preserve current behavior:
   - localization runs only when the model's normal class/tampered-score gate activates localization.
6. `oracle_tampered_gate` must force localization for records with `content_label == "tampered"` regardless of predicted class or tampered score.
7. For non-tampered records in `oracle_tampered_gate`, keep normal behavior unless explicitly configured.
8. `threshold_sweep` must evaluate activation and mask metrics for thresholds:
   - `0.00`
   - `0.01`
   - `0.05`
   - `0.10`
   - `0.25`
   - `0.50`
9. Threshold sweep must not retrain and must not modify checkpoints.
10. If the existing reporting pipeline cannot force localization without a narrow hook, add only a reporting-layer/config hook in `src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py`.
11. Preserve existing fixed-pair evaluator behavior unless the new diagnostic explicitly requests oracle or sweep behavior.
12. Use the same valid-IoU policy as fixed-pair evaluation:
    - `valid_region = 1 - ignore_mask`
    - SNS overlay pixels must not count as tamper ground truth for valid IoU.
13. Reuse or mirror the 0058d visual export behavior so diagnostic outputs include:
    - `pred_masks/`
    - `pred_red_overlays/`
    - `gt_red_overlays/`
    - `ignore_blue_overlays/`
    - `overlap_overlays/`
14. The diagnostic must write these outputs under the configured external output root:
    - `oracle_gate_eval_records.jsonl`
    - `oracle_gate_eval_comparisons.jsonl`
    - `oracle_gate_per_profile_metrics.json`
    - `oracle_gate_drop_metrics.json`
    - `threshold_sweep_metrics.json`
    - `activation_bottleneck_summary.json`
    - `forced_localization_worst_samples.json`
    - `visual_gallery_manifest.json`
    - `artifact_manifest.json`
15. Required metrics per profile and per mode:
    - `tampered_recall`
    - `localization_activation_recall`
    - `tampered_valid_mean_iou`
    - `tampered_valid_median_iou`
    - `tampered_raw_mean_iou`
    - `mean_p_tampered_on_tampered`
    - `activation_flip_off_rate`
    - `class_flip_rate`
    - `non_tampered_high_mask_rate`
    - `real_fpr`
    - `synthetic_recall`
16. Add interpretation logic:
    - If `oracle_tampered_gate` improves IoU compared with `normal_gate`, classify failure as `class_activation_bottleneck`.
    - If `oracle_tampered_gate` still has near-zero IoU, classify failure as `mask_decoder_geometry_bottleneck`.
    - If lower-threshold sweep improves activation but increases real FPR, report threshold tradeoff.
17. Keep output JSON schemas deterministic and testable.
18. Add tests in `tests/test_snsaug_v2_forced_localization_oracle_gate.py` covering:
    - oracle gate forces localization on tampered rows
    - normal gate behavior remains unchanged
    - threshold sweep computes activation recall
    - valid IoU excludes `ignore_mask`
    - visual masks are exported
    - no training or fine-tuning is triggered
19. Add documentation:
    - `docs/snsaug_v2_forced_localization_oracle_gate.md`
20. Documentation must include marker:
    - `SNSAUG_V2_FORCED_LOCALIZATION_ORACLE_GATE_OK`

## Validation Commands

```bash
python3 tests/test_snsaug_v2_forced_localization_oracle_gate.py
```

```bash
python3 scripts/evaluation/run_snsaug_v2_forced_localization_oracle_gate.py --help
```

```bash
grep -q SNSAUG_V2_FORCED_LOCALIZATION_ORACLE_GATE_OK docs/snsaug_v2_forced_localization_oracle_gate.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0058e-snsaug-v2-forced-localization-oracle-gate-diagnostic.md
```

## Acceptance Criteria

- Diagnostic supports `normal_gate`, `oracle_tampered_gate`, and threshold sweep without training or checkpoint modification.
- Oracle tampered gate forces localization for ground-truth tampered rows while preserving normal behavior for non-tampered rows unless configured otherwise.
- Threshold sweep reports activation recall and real-FPR tradeoffs for the required thresholds.
- Valid-IoU computation excludes ignored SNS nuisance pixels.
- Required diagnostic JSONL/JSON outputs and visual mask/overlay directories are produced under an external output root.
- Tests cover oracle forcing, normal gate preservation, threshold sweep activation recall, ignore-mask-aware valid IoU, visual export, and no-training guardrails.
- Documentation contains `SNSAUG_V2_FORCED_LOCALIZATION_ORACLE_GATE_OK`.
- All changed files are within the allowed modify list.

## Stop Condition

Stop immediately if completing this task would require:

- training or fine-tuning,
- using validation failures for training,
- accessing protected paths such as `.env`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`,
- downloading datasets/assets/checkpoints,
- installing packages,
- using network resources,
- modifying model checkpoints,
- writing generated evaluation outputs inside the repository,
- running the real benchmark diagnostic without explicit user approval,
- changing files outside the allowed modify list,
- or expanding beyond fixed-pair evaluation/oracle-gate diagnostic scope.
