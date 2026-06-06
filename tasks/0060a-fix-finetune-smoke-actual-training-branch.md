# Task: 0060a Fix Fine-Tune Smoke Actual Training Branch

## Task Title

Fix the SNSAug V2 fine-tune smoke runner so non-dry-run execution performs a real short smoke training branch.

## Role

Codex-only task writer, implementation worker, reviewer, and limited repair manager for a targeted 0060a smoke-training repair.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0059-snsaug-v2-train-only-curriculum-manifest-for-activation-and-mask-robustness.md`
- `tasks/0060a-snsaug-v2-activation-mask-robustness-finetune-smoke.md`
- `configs/training/snsaug_v2_finetune_smoke.example.json`
- `scripts/agent/validate_snsaug_v2_finetune_smoke_config.py`
- `scripts/training/run_snsaug_v2_finetune_smoke.py`
- `src/cv_forensics/snsaug_v2_finetune_runner.py`
- `src/cv_forensics/snsaug_v2_losses.py`
- `src/cv_forensics/snsaug_v2_dataset_wrapper.py`
- `src/cv_forensics/snsaug_v2_train_curriculum.py`
- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_model.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_training.py`
- `src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py`
- `tests/test_snsaug_v2_finetune_smoke.py`
- `docs/snsaug_v2_finetune_smoke.md`
- `.local/pre_sns_current_best_model_bundle.json`

## Files Codex May Modify

- `tasks/0060a-fix-finetune-smoke-actual-training-branch.md`
- `configs/training/snsaug_v2_finetune_smoke.example.json`
- `scripts/agent/validate_snsaug_v2_finetune_smoke_config.py`
- `scripts/training/run_snsaug_v2_finetune_smoke.py`
- `src/cv_forensics/snsaug_v2_finetune_runner.py`
- `src/cv_forensics/snsaug_v2_losses.py`
- `tests/test_snsaug_v2_finetune_smoke.py`
- `docs/snsaug_v2_finetune_smoke.md`

## Forbidden Actions

- Do not run full training.
- Do not run real smoke training on real user data during validation unless the user explicitly approves that run separately.
- Do not use validation samples for training.
- Do not use validation failures for training.
- Do not use evaluation pair images as training images.
- Do not use fixed-pair roots, validation pair roots, 0058c benchmark images, or evaluation outputs as training input.
- Do not access network resources from shell commands.
- Do not download datasets, assets, or checkpoints.
- Do not install packages.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or protected directories.
- Do not inspect protected directories recursively.
- Do not write checkpoints, logs, reports, or generated artifacts inside the repository.
- Do not modify model checkpoints except by writing new smoke checkpoints under the configured external checkpoint root.
- Do not implement full-curriculum training.
- Do not implement the v2 SNS/Nuisance Mask Head.
- Do not use Claude Code while Claude Code is unavailable.
- Do not use `rg`.
- Do not run `git push`, `git pull`, or `git fetch`.

## Important Project Facts

- 0060a smoke config now validates and manifest paths are correct.
- Dry-run prints the smoke plan correctly.
- Current non-dry-run exits immediately and reports:
  - `checkpoint_written: false`
  - `training_started: false`
- The tmux session disappears because the runner exits immediately.
- The fix must make non-dry-run execute a real short smoke fine-tuning branch when guardrails pass.
- This is still smoke training, not full training.
- The exact approval text for actual smoke training remains:
  - `I_APPROVE_SNSAUG_V2_FINETUNE_SMOKE`

## Implementation Requirements

1. Preserve dry-run behavior:
   - print plan
   - `checkpoint_written=false`
   - `training_started=false`
   - no checkpoint written
2. Non-dry-run must:
   - validate config
   - require approval text `I_APPROVE_SNSAUG_V2_FINETUNE_SMOKE`
   - require `smoke_only=true`
   - require `no_full_training=true`
   - allow actual short smoke training
   - reject full training settings
3. Config validation must reject:
   - train manifest rows where `split != train`
   - evaluation pair root or fixed-pair paths used as training input
   - output root inside repository
   - checkpoint root inside repository
   - `max_steps` above the configured smoke limit
   - missing or false `no_network`
   - missing or false `no_download`
4. Actual smoke training branch must:
   - load the pre-SNS best bundle metadata/path
   - load the train-only curriculum manifest
   - sample a small class-balanced subset according to `samples_per_class` or `max_steps`
   - build SNSAug on-the-fly views according to the curriculum schedule or a safe equivalent for fixture tests
   - train only configured components:
     - `class_head`
     - `tamper_localization_head`
   - respect `family_loss_mask`
   - use `ignore_mask` to exclude SNS overlay from tamper mask loss
   - run for `max_steps <= 300` by default
   - write per-step losses to `smoke_train_log.jsonl`
   - write `smoke_train_summary.json`
   - write `loss_breakdown.json`
   - write `snsaug_sampling_summary.json`
   - write at least one checkpoint under `checkpoint_root`
   - set `training_started=true` and `checkpoint_written=true` in `artifact_manifest.json`
5. Losses must include:
   - `L_class`
   - `L_tamper_mask_valid`
   - `L_tampered_score_consistency`
   - `L_clean_sns_class_consistency`
   - `L_hardneg`
   - optional family loss where `family_loss_mask == 1`
6. Evaluation after smoke must:
   - write `smoke_eval_clean_summary.json`
   - write `smoke_eval_0058c_summary.json`
   - run lightweight subset evaluation if practical
   - otherwise write valid smoke placeholders with `eval_subset_only=true`
   - never pretend full evaluation ran if it did not
7. Tests in `tests/test_snsaug_v2_finetune_smoke.py` must cover:
   - dry-run does not train
   - non-dry-run starts training
   - smoke output files exist
   - `checkpoint_written=true`
   - `training_started=true`
   - manifest split leakage rejected
   - max-steps guardrail
   - losses are finite
8. Docs must update `docs/snsaug_v2_finetune_smoke.md` with:
   - dry-run vs actual smoke run
   - expected output files
   - common issue: `training_started=false` means actual training branch did not run
9. Documentation must keep marker:
   - `SNSAUG_V2_FINETUNE_SMOKE_OK`

## Validation Commands

```bash
python3 scripts/agent/validate_snsaug_v2_finetune_smoke_config.py configs/training/snsaug_v2_finetune_smoke.example.json
```

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_finetune_smoke.py
```

```bash
python3 scripts/training/run_snsaug_v2_finetune_smoke.py --help
```

```bash
grep -q SNSAUG_V2_FINETUNE_SMOKE_OK docs/snsaug_v2_finetune_smoke.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0060a-fix-finetune-smoke-actual-training-branch.md
```

## Acceptance Criteria

- Dry-run remains non-training and writes no checkpoint.
- Non-dry-run with valid smoke approval and guardrails executes the actual short smoke training branch.
- Non-dry-run writes all required smoke outputs outside the repository.
- At least one smoke checkpoint is written under the configured external checkpoint root.
- `artifact_manifest.json` records `training_started=true` and `checkpoint_written=true`.
- Train manifest split leakage is rejected.
- Fixed-pair/evaluation roots cannot be used as training input.
- Smoke max-step guardrails prevent full training.
- Losses are finite in CPU-safe tests.
- Documentation explains dry-run vs actual smoke run and the `training_started=false` failure mode.
- Documentation contains `SNSAUG_V2_FINETUNE_SMOKE_OK`.
- All changed files are within the allowed modify list.

## Stop Condition

Stop immediately if completing this task would require:

- full training,
- using validation/test/evaluation data as training data,
- using validation failures directly for training,
- downloading datasets/assets/checkpoints,
- installing packages,
- network access,
- protected path access,
- writing outputs/checkpoints inside the repository,
- changing files outside the allowed modify list,
- or implementing full-curriculum/v2 nuisance-mask-head scope.
