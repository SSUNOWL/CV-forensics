# Task: 0060b SNSAug V2 Full Curriculum Fine-Tune

## Task Title

Run guarded full SNSAug V2 curriculum fine-tuning to produce the main SNSAug-aware Multi-head Forensics Model v1 checkpoint.

## Role

Codex-only task writer, implementation worker, reviewer, and limited repair manager for a real full-curriculum fine-tuning task with strict approval, leakage, and output guardrails.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0054-snsaug-v2-aware-finetuning.md`
- `tasks/0059-snsaug-v2-train-only-curriculum-manifest-for-activation-and-mask-robustness.md`
- `tasks/0060a-snsaug-v2-activation-mask-robustness-finetune-smoke.md`
- `configs/training/snsaug_v2_train_curriculum_manifest.example.json`
- `configs/training/snsaug_v2_finetune_smoke.example.json`
- `scripts/agent/validate_snsaug_v2_train_curriculum_config.py`
- `scripts/agent/validate_snsaug_v2_finetune_smoke_config.py`
- `scripts/training/build_snsaug_v2_train_curriculum_manifest.py`
- `scripts/training/run_snsaug_v2_finetune_smoke.py`
- `src/cv_forensics/snsaug_v2_train_curriculum.py`
- `src/cv_forensics/snsaug_v2_dataset_wrapper.py`
- `src/cv_forensics/snsaug_v2_losses.py`
- `src/cv_forensics/snsaug_v2_finetune_runner.py`
- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_model.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_training.py`
- `src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py`
- `scripts/evaluation/run_snsaug_v2_fixed_pairs_eval.py`
- `tests/test_snsaug_v2_train_curriculum_manifest.py`
- `tests/test_snsaug_v2_finetune_smoke.py`
- `tests/test_snsaug_v2_fixed_pairs_eval.py`
- `.local/pre_sns_current_best_model_bundle.json`

## Files Codex May Modify

- `tasks/0060b-snsaug-v2-full-curriculum-finetune.md`
- `configs/training/snsaug_v2_full_curriculum_finetune.example.json`
- `scripts/agent/validate_snsaug_v2_full_curriculum_finetune_config.py`
- `scripts/training/run_snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_full_curriculum_finetune.py`
- `tests/test_snsaug_v2_full_curriculum_finetune.py`
- `docs/snsaug_v2_full_curriculum_finetune.md`

## Forbidden Actions

- Do not train from scratch.
- Do not run full curriculum fine-tuning unless the config contains the exact approval text `I_APPROVE_SNSAUG_V2_FULL_CURRICULUM_FINETUNE`.
- Do not use validation samples for training.
- Do not use validation failures directly for training.
- Do not use evaluation pair images as training images.
- Do not use 0058c fixed benchmark images, fixed pair roots, validation pair roots, or evaluation outputs as training input.
- Do not use optional 0058e oracle diagnostic outputs for training; analysis/reporting only.
- Do not modify the 0058c benchmark after training.
- Do not access network resources from shell commands.
- Do not download datasets, assets, or checkpoints.
- Do not install packages.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or protected directories.
- Do not inspect protected directories recursively.
- Do not write checkpoints, logs, reports, or evaluation outputs inside the repository.
- Do not modify model checkpoints except by writing new full-curriculum checkpoints under the configured external checkpoint root.
- Do not run `git push`, `git pull`, or `git fetch`.
- Do not use `rg`.

## Important Project Facts

- 0059 created a train-only SNSAug V2 curriculum manifest and fixed three-phase schedule.
- 0060a smoke fine-tuning infrastructure passed.
- The target checkpoint is `SNSAug-aware Multi-head Forensics Model v1`.
- Full curriculum fine-tuning must start from the frozen pre-SNS best model bundle:
  - `.local/pre_sns_current_best_model_bundle.json`
- Training data must be the 0059 train-only curriculum manifest outputs:
  - `snsaug_v2_train_curriculum_manifest.jsonl`
  - `snsaug_v2_curriculum_schedule.json`
  - `snsaug_v2_profile_sampling_weights.json`
- Evaluation data:
  - clean validation manifest
  - 0058c fixed balanced SNSAug benchmark
  - optional 0058e oracle diagnostic output for analysis only, never training
- Main optimization targets:
  - maintain clean performance
  - improve SNSAug tampered recall
  - improve SNSAug localization activation recall
  - improve valid IoU under SNSAug
  - avoid increasing real FPR
  - avoid synthetic-to-real collapse and synthetic-to-tampered false positives
- Save best checkpoint by:
  - primary: SNSAug tampered recall plus valid IoU
  - secondary: clean macro-F1
  - guardrail: real FPR must not exceed configured limit

## Implementation Requirements

1. Add full-curriculum fine-tuning config:
   - `configs/training/snsaug_v2_full_curriculum_finetune.example.json`
2. Add config validator:
   - `scripts/agent/validate_snsaug_v2_full_curriculum_finetune_config.py`
3. Add full-curriculum training CLI:
   - `scripts/training/run_snsaug_v2_full_curriculum_finetune.py`
4. Add full-curriculum training module:
   - `src/cv_forensics/snsaug_v2_full_curriculum_finetune.py`
5. Add CPU-safe unit/guardrail tests:
   - `tests/test_snsaug_v2_full_curriculum_finetune.py`
6. Add docs:
   - `docs/snsaug_v2_full_curriculum_finetune.md`
7. Documentation must include marker:
   - `SNSAUG_V2_FULL_CURRICULUM_FINETUNE_OK`
8. Config validation must require:
   - `approval_text == "I_APPROVE_SNSAUG_V2_FULL_CURRICULUM_FINETUNE"` for any real full fine-tuning run
   - `run_kind == "full_curriculum_finetune"`
   - `model_version == "snsaug_aware_multihead_forensics_v1"`
   - `start_from_pre_sns_best == true`
   - base model bundle path under approved model input roots
   - 0059 train manifest, schedule, and profile weights under approved train-manifest roots
   - clean validation manifest under approved evaluation roots
   - 0058c pair root under approved evaluation roots
   - optional 0058e analysis root under approved analysis/evaluation roots
   - output root outside repository
   - checkpoint root outside repository
   - `no_network == true`
   - `no_download == true`
9. Config validation must reject:
   - missing exact approval text for real full training
   - training manifest containing `split != train`
   - validation/test rows in training manifest
   - fixed-pair/evaluation/oracle paths as training input
   - train-from-scratch configs
   - repo-local output or checkpoint roots
   - protected-path roots
   - missing real-FPR guardrail limit
   - missing best-checkpoint selection policy
10. CLI must support `--help` without requiring real paths.
11. CLI must support `--dry-run` that validates config, loads small metadata safely when requested, prints the training/evaluation plan, and does not train or write checkpoints.
12. Real full training must not start unless the exact approval text is present and all guardrails pass.
13. Full training must use the 0059 schedule:
   - Phase 1: activation recovery and safe geometry adaptation
   - Phase 2: geometry plus light platform layout
   - Phase 3: platform layout plus combined SNS
14. Full training must use the SNSAug V2 DatasetWrapper or an equivalent existing path for on-the-fly SNSAug views.
15. Full training must preserve labels after SNSAug:
   - `real`
   - `synthetic`
   - `tampered`
16. Full training must start from the pre-SNS best checkpoint; do not train from scratch.
17. Full training must train/update the intended v1 components:
   - class head
   - tamper localization head
   - optional family head loss only where `family_loss_mask == 1`
18. Full training must preserve the existing architecture as much as practical.
19. Full training loss:
   ```text
   L_total =
     L_class
   + lambda_mask * L_tamper_mask_valid
   + lambda_score * L_tampered_score_consistency
   + lambda_consistency * L_clean_sns_class_consistency
   + lambda_hardneg * L_real_synthetic_sns_hard_negative
   + optional family loss where family_loss_mask == 1
   ```
20. `L_tamper_mask_valid` must use:
   - `valid_region = 1 - ignore_mask`
   - SNS nuisance pixels excluded from tamper localization loss
21. `L_tampered_score_consistency` must target the activation bottleneck found in 0058d/0058e.
22. `L_hardneg` must control real FPR and synthetic-to-tampered false positives.
23. Training must save checkpoints only under the configured external checkpoint root:
   - best checkpoint
   - last checkpoint
24. Training/run artifacts must be written only under the configured external output root:
   - training logs
   - per-phase metrics
   - clean validation metrics
   - 0058c SNSAug fixed benchmark metrics
   - robustness drop metrics
   - comparison vs frozen pre-SNS baseline
   - report markdown
   - artifact manifest
25. Evaluation after full training must use validation/evaluation data only:
   - clean validation manifest
   - 0058c fixed balanced SNSAug benchmark
   - optional 0058e oracle diagnostic for analysis only
26. Metrics must include:
   - clean accuracy
   - clean macro-F1
   - clean tampered recall
   - clean valid IoU
   - SNSAug per-profile accuracy
   - SNSAug tampered recall
   - SNSAug localization activation recall
   - SNSAug valid IoU
   - real FPR
   - synthetic recall
   - synthetic-to-real confusion
   - synthetic-to-tampered confusion
   - non-tampered high mask rate
27. Best checkpoint selection must enforce:
   - primary score: SNSAug tampered recall plus valid IoU
   - secondary tie-break: clean macro-F1
   - guardrail: real FPR must not exceed configured limit
28. Comparison vs frozen pre-SNS baseline must be reported without using baseline failures as training data.
29. CPU-safe tests must cover:
   - approval text requirement
   - train-only manifest enforcement
   - rejection of val/test leakage
   - rejection of fixed-pair/evaluation/oracle paths as training input
   - output/checkpoint root guardrails
   - train-from-scratch rejection
   - best-checkpoint selection policy
   - real-FPR guardrail behavior
   - dry-run starts no training and writes no checkpoints
   - required output schema/planned artifact names
   - docs marker

## Validation Commands

```bash
python3 scripts/agent/validate_snsaug_v2_full_curriculum_finetune_config.py configs/training/snsaug_v2_full_curriculum_finetune.example.json
```

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
python3 scripts/agent/check_agent_changes.py tasks/0060b-snsaug-v2-full-curriculum-finetune.md
```

## Acceptance Criteria

- A guarded full-curriculum SNSAug V2 fine-tuning config, validator, CLI, module, tests, and docs exist.
- The validator rejects missing approval, train-from-scratch, validation leakage, evaluation-pair training input, protected paths, and repo-local output/checkpoint roots.
- The CLI supports `--help` without real paths and `--dry-run` without training.
- Real full training can only run with exact approval text and passing guardrails.
- Training starts from the pre-SNS best checkpoint, not scratch.
- Training uses the 0059 curriculum schedule and train-only manifest.
- Training/evaluation artifacts and checkpoints are written only outside the repository.
- Best/last checkpoint policy, real-FPR guardrail, and pre-SNS baseline comparison are implemented.
- Tests are CPU-safe and do not require real full training, real datasets, network, or downloads.
- Documentation contains `SNSAUG_V2_FULL_CURRICULUM_FINETUNE_OK`.
- All changed files are within the allowed modify list.

## Stop Condition

Stop immediately if completing this task would require:

- running full training without exact approval text,
- training from scratch,
- using validation/test/fixed-pair/evaluation/oracle outputs as training data,
- using validation failures directly for training,
- modifying the 0058c benchmark after training,
- accessing protected paths such as `.env`, `secrets/`, `data/`, `datasets/`, `outputs/`, or protected checkpoint directories,
- downloading datasets/assets/checkpoints,
- installing packages,
- using network resources,
- writing outputs or checkpoints inside the repository,
- changing files outside the allowed modify list,
- or expanding beyond SNSAug-aware v1 full-curriculum fine-tuning.
