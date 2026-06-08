# Task: 0063 SNSAug V2 Balanced Hard-Negative Fine-Tune

## Task Title

Implement guarded balanced hard-negative SNSAug V2 fine-tuning infrastructure to reduce over-tampered bias without running long training.

## Role

Codex-only task writer, implementation worker, reviewer, and limited repair manager. Claude Code is unavailable, so Codex must implement directly only after this task file is committed and the user explicitly says `implement`.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0054-snsaug-v2-aware-finetuning.md`
- `tasks/0059-snsaug-v2-train-only-curriculum-manifest-for-activation-and-mask-robustness.md`
- `tasks/0060b-snsaug-v2-full-curriculum-finetune.md`
- `tasks/0060b-fix-real-model-weight-checkpoints.md`
- `tasks/0061b-fix-checkpoint-comparison-evaluator-real-inference.md`
- `tasks/0061c-fix-empty-checkpoint-comparisons-with-robust-record-join.md`
- `docs/snsaug_v2_full_curriculum_finetune.md`
- `docs/snsaug_v2_checkpoint_comparison_eval.md`
- `configs/training/snsaug_v2_full_curriculum_finetune.example.json`
- `configs/training/snsaug_v2_train_curriculum_manifest.example.json`
- `scripts/agent/validate_snsaug_v2_full_curriculum_finetune_config.py`
- `scripts/training/run_snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_losses.py`
- `src/cv_forensics/snsaug_v2_train_curriculum.py`
- `src/cv_forensics/snsaug_v2_dataset_wrapper.py`
- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `src/cv_forensics/snsaug_v2_checkpoint_comparison_eval.py`
- `src/cv_forensics/pre_sns_v3_model.py`
- `tests/test_snsaug_v2_full_curriculum_finetune.py`
- `tests/test_snsaug_v2_checkpoint_comparison_eval.py`
- `scripts/agent/check_agent_changes.py`

## Files Codex May Modify

- `configs/training/snsaug_v2_balanced_hard_negative_finetune.example.json`
- `scripts/agent/validate_snsaug_v2_balanced_hard_negative_finetune_config.py`
- `scripts/training/run_snsaug_v2_balanced_hard_negative_finetune.py`
- `src/cv_forensics/snsaug_v2_balanced_hard_negative_finetune.py`
- `src/cv_forensics/snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_losses.py`
- `tests/test_snsaug_v2_balanced_hard_negative_finetune.py`
- `docs/snsaug_v2_balanced_hard_negative_finetune.md`

## Forbidden Actions

- Do not implement until this task file is committed and the user explicitly says `implement`.
- Do not modify this task file during implementation.
- Do not run long training.
- Do not start real training without explicit user approval for the real run.
- Do not train from scratch.
- Do not download datasets, assets, or checkpoints.
- Do not install packages.
- Do not access network resources from shell commands.
- Do not use validation samples for training.
- Do not use validation failures directly as training samples.
- Do not use 0058c fixed benchmark images, fixed pair roots, validation pair roots, evaluation roots, oracle diagnostic outputs, or checkpoint comparison outputs as training input.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, `checkpoints/`, or protected directories.
- Do not inspect protected directories recursively.
- Do not write training outputs, logs, or checkpoints inside the repository.
- Do not create large files.
- Do not run `git push`, `git pull`, or `git fetch`.
- Do not use `rg`.
- Do not use `/snap/bin/codex`, `/home/rlatjswo/.npm-global/bin/codex`, `--permission-mode bypassPermissions`, or `--dangerously-skip-permissions`.

## Important Project Facts

- The repaired 0061 checkpoint comparison evaluator now performs real checkpoint inference and robust record joins.
- The 30x3 realweights model fixed the original SNS tampered activation collapse but overpredicts `tampered`.
- The 0062 threshold sweep used balanced records with `real=520`, `synthetic=520`, and `tampered=520`; the failure is not caused by a tampered-heavy evaluation set.
- The threshold sweep had `candidate_count=0` for `real_fpr <= 0.20` and `tampered_recall >= 0.50`; threshold-only calibration is insufficient.
- Observed 30x3 failures include high real FPR, zero synthetic recall, and near-perfect tampered recall across clean and SNS profiles.
- The corrective run must suppress tampered probability on real/synthetic SNSAug samples while preserving tampered activation recovery.
- Training data must remain train-only and must not include validation or fixed-pair evaluation samples.
- The implementation must remain compatible with the lightweight multi-head forensic prototype: class, mask, family, and template-reason pipeline.
- Marker for this task:
  - `SNSAUG_V2_BALANCED_HARD_NEGATIVE_FINETUNE_OK`

## Implementation Requirements

1. Add a balanced hard-negative fine-tuning config:
   - `configs/training/snsaug_v2_balanced_hard_negative_finetune.example.json`
2. Add a config validator:
   - `scripts/agent/validate_snsaug_v2_balanced_hard_negative_finetune_config.py`
3. Add a guarded training CLI:
   - `scripts/training/run_snsaug_v2_balanced_hard_negative_finetune.py`
4. Add or update a training module:
   - Prefer `src/cv_forensics/snsaug_v2_balanced_hard_negative_finetune.py` for task-specific behavior.
   - Reuse existing 0060b full-curriculum helpers where practical.
5. Add or update losses:
   - Implement `L_non_tampered_tampered_suppression`.
   - For labels `real` and `synthetic`, penalize `p_tampered` above a ceiling.
   - Default loss should use margin form:
     ```text
     max(0, p_tampered - p_tampered_ceiling)^2
     ```
   - BCE-to-zero is acceptable only if documented and tested.
   - Defaults:
     ```text
     p_tampered_ceiling = 0.05
     lambda_hardneg = 2.0
     ```
6. Restrict tampered score consistency:
   - Apply `L_tampered_score_consistency` only when `label == tampered`.
   - Do not apply tampered score recovery to `real` or `synthetic`.
7. Add clean/SNS class consistency:
   - For paired clean/SNS views with labels `real` and `synthetic`, preserve the class distribution while avoiding tampered overactivation.
   - This consistency term must not encourage `tampered` for non-tampered labels.
8. Log these values in training logs and loss summaries:
   - `mean_p_tampered_real_sns`
   - `mean_p_tampered_synthetic_sns`
   - `mean_p_tampered_tampered_sns`
   - `hardneg_loss`
   - `tampered_score_consistency_loss`
   - `clean_sns_class_consistency_loss`
9. Add hard-negative sampling mode:
   - Increase real/synthetic SNSAug samples during hard SNS phases.
   - In hard SNS phases, target approximately:
     ```text
     non_tampered_sns : tampered_sns = 2 : 1
     ```
   - Preserve train-only manifest guardrails.
   - Reject eval pair roots or validation roots as training input.
10. Replace best checkpoint primary score with:
    ```text
    balanced_score =
      + 1.0 * snsaug_tampered_recall
      + 1.0 * snsaug_valid_iou
      + 0.8 * synthetic_recall
      - 2.0 * real_fpr
      - 1.0 * non_tampered_high_mask_rate
    ```
11. Add first corrective run checkpoint guardrails:
    - `real_fpr <= 0.30`
    - `synthetic_recall >= 0.40`
    - `clean_macro_f1 >= 0.75`
    - `tampered_recall >= 0.50`
12. Required real-run outputs must be planned and validated:
    - `training_log.jsonl`
    - `loss_breakdown.json`
    - `per_phase_metrics.json`
    - `clean_validation_metrics.json`
    - `snsaug_0058c_metrics.json`
    - `threshold_sweep_after_training.json`
    - `artifact_manifest.json`
    - `best/` real `.pt` checkpoint
    - `last` real `.pt` checkpoint
13. Real checkpoint payloads must contain a real `model_state_dict`; proxy-only `trainable_state` checkpoint payloads are invalid.
14. The CLI must support `--help` without requiring real paths.
15. The CLI must support `--dry-run` that validates config and prints the plan without training or writing checkpoints.
16. Real runs must require exact approval text:
    - `I_APPROVE_SNSAUG_V2_BALANCED_HARD_NEGATIVE_FINETUNE`
17. Config validation must require:
    - `config_kind == "approved_snsaug_v2_balanced_hard_negative_finetune"`
    - `execution_mode == "approved_local_snsaug_v2_balanced_hard_negative_finetune"`
    - `run_kind == "balanced_hard_negative_finetune"`
    - `no_network == true`
    - `no_download == true`
    - `no_training_from_scratch == true`
    - external output root
    - external checkpoint root
    - train-only manifest roots
    - approved evaluation roots for validation/evaluation only
18. Config validation must reject:
    - missing exact approval text for real non-dry-run training
    - any `split != train` row in the training manifest
    - training image paths under approved evaluation roots
    - fixed-pair, validation, benchmark, oracle, output, or checkpoint paths as training input
    - repo-local output or checkpoint roots
    - protected-path roots
    - `no_network != true`
    - `no_download != true`
    - train-from-scratch settings
19. Add CPU-safe tests:
    - hard-negative loss is greater than zero when `p_tampered > p_tampered_ceiling` for `real` or `synthetic`.
    - hard-negative loss is zero when `p_tampered <= p_tampered_ceiling`.
    - hard-negative loss ignores `tampered` labels.
    - tampered score consistency applies only to `tampered` labels.
    - clean/SNS class consistency does not encourage tampered probability for `real` or `synthetic`.
    - hard-negative sampling plan targets approximately 2:1 non-tampered SNS to tampered SNS in hard SNS phases.
    - train split only guard rejects validation/test rows.
    - eval/fixed-pair paths are rejected as training input.
    - `no_network` and `no_download` are required.
    - dry-run starts no training and writes no checkpoints.
    - real checkpoint validation accepts payloads with `model_state_dict`.
    - real checkpoint validation rejects proxy-only `trainable_state`.
    - docs marker is present.
20. Add documentation:
    - `docs/snsaug_v2_balanced_hard_negative_finetune.md`
    - Explain why threshold-only calibration was insufficient.
    - Explain why balanced hard-negative correction is needed.
    - Explain loss terms, sampling ratio, best-checkpoint balanced score, guardrails, required outputs, dry-run behavior, and protected-path restrictions.
    - Include marker `SNSAUG_V2_BALANCED_HARD_NEGATIVE_FINETUNE_OK`.
21. Keep validation commands CPU-safe and small. Do not run any long training in tests.

## Validation Commands

```bash
python3 scripts/agent/validate_snsaug_v2_balanced_hard_negative_finetune_config.py configs/training/snsaug_v2_balanced_hard_negative_finetune.example.json
```

```bash
python3 scripts/training/run_snsaug_v2_balanced_hard_negative_finetune.py --help
```

```bash
python3 tests/test_snsaug_v2_balanced_hard_negative_finetune.py
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0063-snsaug-v2-balanced-hard-negative-finetune.md
```

## Acceptance Criteria

- The task marker `SNSAUG_V2_BALANCED_HARD_NEGATIVE_FINETUNE_OK` appears in the docs and test output.
- Config validation accepts the example config and rejects unsafe or leaky training inputs in tests.
- The CLI `--help` command succeeds without real paths.
- Dry-run behavior starts no training and writes no checkpoints.
- Hard-negative loss behavior matches the required ceiling semantics.
- Tampered score consistency is restricted to `tampered` labels.
- The sampling plan supports approximately 2:1 non-tampered SNS to tampered SNS in hard SNS phases.
- Best-checkpoint selection uses the required `balanced_score` formula and guardrails.
- Required output names are represented in the plan and artifact schema.
- Real checkpoint validation requires `model_state_dict` or equivalent real state dict and rejects proxy-only checkpoints.
- No datasets are downloaded.
- No model training is run during implementation validation.
- No protected paths are accessed or modified.
- `python3 scripts/agent/check_agent_changes.py tasks/0063-snsaug-v2-balanced-hard-negative-finetune.md` passes after implementation.
- Git diff contains only files listed in `## Files Codex May Modify`.

## Stop Condition

Stop after creating this task file. Do not implement until the user commits this task file and explicitly says `implement`.
