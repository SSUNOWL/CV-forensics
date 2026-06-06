# Task: 0060b Fix Full Curriculum Actual Training Branch

## Task Title

Fix the SNSAug V2 full-curriculum runner so non-dry-run execution enters an actual guarded training branch instead of returning a plan-only stub.

## Role

Codex-only task writer, implementation worker, reviewer, and limited repair manager for a targeted 0060b full-curriculum training-branch repair.

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
- `configs/training/snsaug_v2_full_curriculum_finetune.example.json`
- `scripts/agent/validate_snsaug_v2_full_curriculum_finetune_config.py`
- `scripts/training/run_snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_losses.py`
- `src/cv_forensics/snsaug_v2_finetune_runner.py`
- `src/cv_forensics/snsaug_v2_train_curriculum.py`
- `src/cv_forensics/snsaug_v2_dataset_wrapper.py`
- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_model.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_training.py`
- `tests/test_snsaug_v2_full_curriculum_finetune.py`
- `tests/test_snsaug_v2_finetune_smoke.py`
- `docs/snsaug_v2_full_curriculum_finetune.md`
- `.local/pre_sns_current_best_model_bundle.json`

## Files Codex May Modify

- `tasks/0060b-fix-full-curriculum-actual-training-branch.md`
- `configs/training/snsaug_v2_full_curriculum_finetune.example.json`
- `scripts/agent/validate_snsaug_v2_full_curriculum_finetune_config.py`
- `scripts/training/run_snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_full_curriculum_finetune.py`
- `tests/test_snsaug_v2_full_curriculum_finetune.py`
- `docs/snsaug_v2_full_curriculum_finetune.md`

## Forbidden Actions

- Do not run long full training during implementation validation.
- Do not train from scratch.
- Do not use validation samples for training.
- Do not use validation failures directly for training.
- Do not use fixed-pair roots, clean validation manifests, 0058c benchmark images, 0058e oracle outputs, or evaluation outputs as training input.
- Do not modify the 0058c benchmark or 0058e analysis outputs.
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

- The 0060b config now validates with real local inputs.
- `--dry-run` correctly prints the plan and reports:
  - `dry_run=true`
  - `training_started=false`
  - `checkpoint_written=false`
- Current non-dry-run returns a plan-like output and reports:
  - `training_started=false`
  - `checkpoint_written=false` or manifest-only checkpoint routing depending on the path
- This means the actual 0060b full-curriculum training branch is not connected.
- Full-curriculum training must start from `.local/pre_sns_current_best_model_bundle.json`.
- Training data must come only from the 0059 train-only curriculum manifest.
- Clean validation, 0058c fixed pairs, and optional 0058e oracle outputs are evaluation or analysis only.
- The target model version remains `snsaug_aware_multihead_forensics_v1`.
- The documentation marker must remain:
  - `SNSAUG_V2_FULL_CURRICULUM_FINETUNE_OK`

## Implementation Requirements

1. Preserve dry-run behavior:
   - validate config
   - print plan
   - `dry_run=true`
   - `training_started=false`
   - `checkpoint_written=false`
   - do not create checkpoint files or checkpoint directories
2. Non-dry-run must actually execute a guarded training branch:
   - validate config
   - require `approval_text == "I_APPROVE_SNSAUG_V2_FULL_CURRICULUM_FINETUNE"`
   - require `no_network=true`
   - require `no_download=true`
   - reject output and checkpoint roots inside the repository
   - reject any train manifest row with `split != train`
   - reject training image or mask paths under clean validation, 0058c evaluation pair root, 0058e oracle root, fixed-pair roots, or evaluation output roots
3. The actual branch must load:
   - base model bundle metadata from `.local/pre_sns_current_best_model_bundle.json`
   - 0059 train-only curriculum manifest
   - 0059 curriculum schedule
   - 0059 profile sampling weights
4. The actual branch must execute three curriculum phases:
   - phase 1: activation recovery plus safe geometry
   - phase 2: geometry plus light platform layout
   - phase 3: platform layout plus combined SNS
5. The phase step controls must support:
   - `max_steps_per_phase`
   - `phase_1_max_steps`
   - `phase_2_max_steps`
   - `phase_3_max_steps`
6. Guardrails must allow tiny local sanity runs with `max_steps_per_phase <= 30`, and reject settings that would launch a long full run during tests unless explicitly configured for real training.
7. The training branch must use existing loss utilities from `src/cv_forensics/snsaug_v2_losses.py`:
   - `L_class`
   - `L_tamper_mask_valid` with `valid_region = 1 - ignore_mask`
   - `L_tampered_score_consistency`
   - `L_clean_sns_class_consistency`
   - `L_real_synthetic_sns_hard_negative`
   - optional family loss only where `family_loss_mask == 1`
8. The implementation may use a CPU-safe lightweight training path for tests, but it must update trainable state, compute finite losses, and produce real checkpoint files.
9. Checkpoints must be written under `checkpoint_root`:
   - `snsaug_aware_multihead_forensics_v1_best.pt`
   - `snsaug_aware_multihead_forensics_v1_last.pt`
10. Outputs must be written under `output_root`:
    - `training_log.jsonl`
    - `per_phase_metrics.json`
    - `clean_validation_metrics.json`
    - `snsaug_0058c_metrics.json`
    - `robustness_drop_metrics.json`
    - `pre_sns_baseline_comparison.json`
    - `full_curriculum_report.md`
    - `artifact_manifest.json`
11. `artifact_manifest.json` must include:
    - `training_started=true`
    - `checkpoint_written=true`
    - `best_checkpoint_path`
    - `last_checkpoint_path`
12. Post-training evaluation summaries may be subset-only in tests, but must be explicitly marked:
    - `eval_subset_only=true`
    - `sample_count`
    - `full_evaluation_ran=false`
13. Do not claim full evaluation ran unless the full evaluator actually ran.
14. Tests must cover:
    - dry-run does not train
    - non-dry-run starts training
    - `training_started=true` for actual run
    - `checkpoint_written=true` for actual run
    - `.pt` best and last checkpoint files are written
    - train split leakage is rejected
    - evaluation roots are not used as training input
    - max-steps-per-phase guardrail works
    - required output files exist
    - loss values are finite
15. Docs must add troubleshooting guidance:
    - if tmux disappears and `training_started=false` appears in the log, the actual training branch is not running
    - validate dry-run separately from actual run
    - check `artifact_manifest.json` fields `training_started` and `checkpoint_written`

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
python3 scripts/agent/check_agent_changes.py tasks/0060b-fix-full-curriculum-actual-training-branch.md
```

## Acceptance Criteria

- Dry-run remains non-training and writes no checkpoints.
- Non-dry-run with a valid guarded config executes the actual training branch.
- Actual branch writes finite per-step or per-phase losses.
- Actual branch writes all required output files under `output_root`.
- Actual branch writes both required `.pt` checkpoint files under `checkpoint_root`.
- `artifact_manifest.json` records `training_started=true`, `checkpoint_written=true`, `best_checkpoint_path`, and `last_checkpoint_path`.
- Train manifest leakage is rejected.
- Clean validation, 0058c fixed-pair, 0058e oracle, fixed-pair, and evaluation-output paths cannot be used as training input.
- Max-steps-per-phase guardrails prevent accidental long training in tests.
- Evaluation summaries clearly distinguish subset-only placeholder evaluation from full evaluation.
- Documentation includes the troubleshooting section and marker.
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
