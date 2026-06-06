# Task: 0060b Debug Tampered Score Consistency Loss Zero

## Task Title

Debug and fix the 0060b full-curriculum branch when `tampered_score_consistency_loss` stays zero across all phases.

## Role

Codex-only task writer, implementation worker, reviewer, and limited repair manager for a targeted SNSAug V2 full-curriculum loss/logging repair.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0059-snsaug-v2-train-only-curriculum-manifest-for-activation-and-mask-robustness.md`
- `tasks/0060a-snsaug-v2-activation-mask-robustness-finetune-smoke.md`
- `tasks/0060a-fix-finetune-smoke-actual-training-branch.md`
- `tasks/0060b-snsaug-v2-full-curriculum-finetune.md`
- `tasks/0060b-fix-full-curriculum-actual-training-branch.md`
- `configs/training/snsaug_v2_full_curriculum_finetune.example.json`
- `scripts/agent/validate_snsaug_v2_full_curriculum_finetune_config.py`
- `scripts/training/run_snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_losses.py`
- `src/cv_forensics/snsaug_v2_finetune_runner.py`
- `src/cv_forensics/snsaug_v2_dataset_wrapper.py`
- `src/cv_forensics/snsaug_v2_train_curriculum.py`
- `tests/test_snsaug_v2_full_curriculum_finetune.py`
- `tests/test_snsaug_v2_finetune_smoke.py`
- `docs/snsaug_v2_full_curriculum_finetune.md`

## Files Codex May Modify

- `tasks/0060b-debug-tampered-score-consistency-loss-zero.md`
- `src/cv_forensics/snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_losses.py`
- `tests/test_snsaug_v2_full_curriculum_finetune.py`
- `docs/snsaug_v2_full_curriculum_finetune.md`

## Forbidden Actions

- Do not run long full training.
- Do not use validation samples for training.
- Do not use validation failures directly for training.
- Do not use fixed-pair roots, clean validation manifests, 0058c benchmark images, 0058e oracle outputs, or evaluation outputs as training input.
- Do not access network resources from shell commands.
- Do not download datasets, assets, or checkpoints.
- Do not install packages.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or protected directories.
- Do not inspect protected directories recursively.
- Do not write generated artifacts, logs, reports, or checkpoints inside the repository.
- Do not modify existing model checkpoints.
- Do not run `git push`, `git pull`, or `git fetch`.
- Do not use `rg`.
- Do not use Claude Code.

## Important Project Facts

- The original SNSAug failure mode was:
  - SNS transforms lower `p_tampered`
  - conditional localization gate turns off
  - predicted redmask becomes empty or missing
- 0060b guarded short actual branch now runs and writes outputs.
- Recent 0060b short-run per-phase metrics improved:
  - SNSAug tampered recall: `0.45 -> 0.50 -> 0.55`
  - SNSAug valid IoU: `0.24 -> 0.28 -> 0.32`
  - Real FPR stayed `0.04`
- However `tampered_score_consistency_loss` is `0.0` in every phase.
- This may mean the loss is skipped, no tampered pairs are being counted, or the synthetic SNS logits always satisfy the clean/floor target.
- The documentation marker must remain:
  - `SNSAUG_V2_FULL_CURRICULUM_FINETUNE_OK`

## Implementation Requirements

1. Inspect the 0060b full-curriculum training branch and loss helpers to determine why `tampered_score_consistency_loss` remains zero.
2. Check whether `L_tampered_score_consistency` is implemented or stubbed.
3. Check whether the training branch creates clean/SNS paired logits for the same `base_id`.
4. Check whether tampered samples are included in the paired consistency calculation.
5. Check whether `lambda_score > 0` but the loss is skipped.
6. Check whether the score floor condition is always satisfied.
7. Check whether `p_tampered_clean` and `p_tampered_sns` are available during training.
8. Make `tampered_score_consistency_loss` nonzero when SNS `p_tampered` falls below clean `p_tampered` or the configured floor.
9. Keep the loss zero only when there are no tampered pairs, and log that reason explicitly.
10. Add per-step or per-phase logging fields:
    - `mean_p_tampered_clean`
    - `mean_p_tampered_sns`
    - `tampered_score_consistency_loss`
    - `tampered_pair_count`
    - optional `tampered_score_consistency_skip_reason`
11. Ensure phase metrics aggregate those fields so a completed short run can show whether activation consistency is active.
12. Add tests where synthetic SNS `p_tampered_sns` is lower than clean `p_tampered_clean` and loss must be greater than zero.
13. Add tests that loss remains zero only when there are no tampered pairs and the skip reason is present.
14. Preserve existing dry-run behavior.
15. Preserve all existing output-root and checkpoint-root guardrails.
16. Do not run real long full-curriculum training during validation.

## Validation Commands

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_full_curriculum_finetune.py
```

```bash
python3 scripts/training/run_snsaug_v2_full_curriculum_finetune.py --help
```

```bash
grep -q SNSAUG_V2_FULL_CURRICULUM_FINETUNE_OK docs/snsaug_v2_full_curriculum_finetune.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0060b-debug-tampered-score-consistency-loss-zero.md
```

## Acceptance Criteria

- The root cause of zero `tampered_score_consistency_loss` is identified in the review.
- Tests prove that collapsed SNS tampered probability produces nonzero tampered-score consistency loss.
- Tests prove that a no-tampered-pair case logs an explicit skip reason.
- 0060b actual short branch logs `mean_p_tampered_clean`, `mean_p_tampered_sns`, `tampered_score_consistency_loss`, and `tampered_pair_count`.
- Phase metrics aggregate the same diagnostic fields.
- Existing dry-run behavior remains unchanged.
- No validation data, fixed-pair data, oracle data, or evaluation outputs are used as training input.
- All changed files are within the allowed modify list.

## Stop Condition

Stop immediately if completing this task would require:

- running long full training,
- using validation/test/evaluation data as training data,
- using validation failures directly for training,
- downloading datasets/assets/checkpoints,
- installing packages,
- network access,
- protected path access,
- writing outputs/checkpoints inside the repository,
- changing files outside the allowed modify list,
- modifying existing checkpoints,
- or changing the 0058c benchmark or 0058e analysis outputs.
