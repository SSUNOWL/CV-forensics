# Task: 0060a SNSAug V2 Activation Mask Robustness Fine-Tune Smoke

## Task Title

Implement guarded SNSAug V2 activation and mask robustness fine-tuning smoke infrastructure.

## Role

Codex-only task writer, implementation worker, reviewer, and limited repair manager for the first guarded SNSAug-aware smoke fine-tuning task.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0054-snsaug-v2-aware-finetuning.md`
- `tasks/0059-snsaug-v2-train-only-curriculum-manifest-for-activation-and-mask-robustness.md`
- `configs/training/snsaug_v2_train_curriculum_manifest.example.json`
- `scripts/agent/validate_snsaug_v2_train_curriculum_config.py`
- `scripts/training/build_snsaug_v2_train_curriculum_manifest.py`
- `src/cv_forensics/snsaug_v2_train_curriculum.py`
- `src/cv_forensics/snsaug_v2_dataset_wrapper.py`
- `src/cv_forensics/snsaug_v2_training_manifest.py`
- `src/cv_forensics/snsaug_v2_finetune.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_training.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_model.py`
- `src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py`
- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `scripts/evaluation/run_snsaug_v2_fixed_pairs_eval.py`
- `tests/test_snsaug_v2_training_manifest.py`
- `tests/test_snsaug_v2_train_curriculum_manifest.py`
- `tests/test_snsaug_v2_finetune.py`
- `tests/test_snsaug_v2_fixed_pairs_eval.py`
- `.local/pre_sns_current_best_model_bundle.json`

## Files Codex May Modify

- `tasks/0060a-snsaug-v2-activation-mask-robustness-finetune-smoke.md`
- `configs/training/snsaug_v2_finetune_smoke.example.json`
- `scripts/agent/validate_snsaug_v2_finetune_smoke_config.py`
- `scripts/training/run_snsaug_v2_finetune_smoke.py`
- `src/cv_forensics/snsaug_v2_finetune_runner.py`
- `src/cv_forensics/snsaug_v2_losses.py`
- `tests/test_snsaug_v2_finetune_smoke.py`
- `docs/snsaug_v2_finetune_smoke.md`

## Forbidden Actions

- Do not run full training.
- Do not run any real fine-tuning unless the config contains the exact approval text `I_APPROVE_SNSAUG_V2_FINETUNE_SMOKE`.
- Do not exceed smoke constraints when a real smoke run is explicitly approved.
- Do not use validation samples for training.
- Do not use validation failure samples directly for training.
- Do not use the 0058c fixed validation benchmark, fixed pair roots, validation pair roots, or evaluation outputs as training input.
- Do not access network resources from shell commands.
- Do not download datasets, assets, or checkpoints.
- Do not install packages.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or protected directories.
- Do not inspect protected directories recursively.
- Do not modify model checkpoints outside the configured external smoke checkpoint directory.
- Do not write checkpoints, logs, evaluation outputs, or generated artifacts inside the repository.
- Do not implement the SNS/Nuisance Mask Head in this smoke task.
- Do not change model architecture broadly unless required to connect existing heads for the smoke runner.
- Do not use Claude Code while Claude Code is unavailable.
- Do not use `rg`.
- Do not run `git push`, `git pull`, or `git fetch`.

## Important Project Facts

- Task 0059 created a train-only SNSAug V2 curriculum manifest and fixed three-phase profile schedule.
- This is the first actual SNSAug-aware model training task, but it is limited to a short smoke run.
- The target model name is `SNSAug-aware Multi-head Forensics Model v1`.
- Model version v1 must keep the existing architecture as much as possible.
- Model version v1 trains the class head and tamper localization head with SNSAug curriculum.
- Do not implement an SNS/Nuisance Mask Head until v2.
- The base model bundle for later user-approved smoke execution is:
  - `.local/pre_sns_current_best_model_bundle.json`
- The 0059 training manifest outputs needed by the smoke runner are:
  - `snsaug_v2_train_curriculum_manifest.jsonl`
  - `snsaug_v2_curriculum_schedule.json`
  - `snsaug_v2_profile_sampling_weights.json`
- The fixed validation benchmark is evaluation-only:
  - `/home/rlatjswo/cvf_runs/snsaug_v2_0058c_small_val_pairs_20pc_balanced_20260605_220707`
- Smoke outputs must be outside the repository:
  - checkpoint directory under `~/cvf_checkpoints`
  - run directory under `~/cvf_runs`
- Implementation tests must use small temporary fixtures only, not real data, real benchmark roots, or large checkpoints.

## Implementation Requirements

1. Add example config:
   - `configs/training/snsaug_v2_finetune_smoke.example.json`
2. Add config validator:
   - `scripts/agent/validate_snsaug_v2_finetune_smoke_config.py`
3. Add smoke runner CLI:
   - `scripts/training/run_snsaug_v2_finetune_smoke.py`
4. Add runner module:
   - `src/cv_forensics/snsaug_v2_finetune_runner.py`
5. Add loss module:
   - `src/cv_forensics/snsaug_v2_losses.py`
6. Add tests:
   - `tests/test_snsaug_v2_finetune_smoke.py`
7. Add docs:
   - `docs/snsaug_v2_finetune_smoke.md`
8. Config validation must require:
   - `approval_text == "I_APPROVE_SNSAUG_V2_FINETUNE_SMOKE"` for any real smoke training run
   - train manifest paths under approved train-manifest input roots
   - base model bundle under an approved model input root
   - evaluation pair root under an approved evaluation input root
   - output root outside repository
   - checkpoint root outside repository
   - `no_network == true`
   - `no_download == true`
   - `run_kind == "smoke"`
   - `model_version == "snsaug_aware_multihead_forensics_v1"`
9. Config validation must reject:
   - missing approval text for real smoke training
   - train manifests containing `split != train`
   - validation/test rows in train manifest
   - evaluation pair roots as training input
   - repo-local output roots
   - repo-local checkpoint roots
   - protected-path roots
   - `max_steps` outside `1..500`
   - `epochs` greater than `1`
   - unsafe or missing guardrail flags
10. CLI must support `--help` without requiring real paths.
11. CLI must support validation/dry-run mode that performs guardrail checks without training.
12. Real smoke training must not start unless the approval text is present.
13. The smoke runner must be able to load the 0059 curriculum manifest and schedule.
14. The smoke runner must integrate the SNSAug V2 DatasetWrapper or an existing equivalent path for on-the-fly SNSAug views.
15. The smoke runner must keep SNSAug labels unchanged:
    - `real`
    - `synthetic`
    - `tampered`
16. The smoke runner must train or update only smoke-allowed model components:
    - class head
    - tamper localization head
    - optional family loss only where `family_loss_mask == 1`
17. The smoke runner must preserve the existing architecture as much as practical.
18. The smoke runner must write checkpoints only under the configured external checkpoint root.
19. The smoke runner must write run artifacts only under the configured external run output root.
20. The smoke run constraints must be enforced:
    - `max_steps` between `1` and `500`
    - `epochs <= 1`
    - small `samples_per_class`
    - batch size safe for RTX 4090
    - optional mixed precision only if already supported locally
21. Implement `L_class`:
    - cross entropy over `real / synthetic / tampered`
    - SNSAug label equals original label
22. Implement `L_tamper_mask_valid`:
    - `valid_region = 1 - ignore_mask`
    - mask loss applies only to valid regions
    - use BCE plus Dice-style term when possible
    - suppress high mask false positives for non-tampered samples
23. Implement `L_tampered_score_consistency`:
    - for tampered clean/SNS pairs
    - encourage `p_tampered_sns` to remain close to `p_tampered_clean` or above a configured floor
    - target the activation bottleneck found in 0058d/0058e
24. Implement `L_clean_sns_class_consistency`:
    - for same-`base_id` clean/SNS views
    - align class distributions with KL or symmetric KL
25. Implement `L_hardneg`:
    - for real/synthetic SNSAug samples
    - penalize high `p_tampered`
    - control real FPR and synthetic-to-tampered false positives
26. Optional family loss must apply only where `family_loss_mask == 1`.
27. Required output files for a real approved smoke run:
    - `smoke_train_log.jsonl`
    - `smoke_train_summary.json`
    - `loss_breakdown.json`
    - `snsaug_sampling_summary.json`
    - `smoke_eval_clean_summary.json`
    - `smoke_eval_0058c_summary.json`
    - `artifact_manifest.json`
28. Required checkpoint output:
    - checkpoint directory under `~/cvf_checkpoints`
    - at least one smoke checkpoint file when real approved smoke training runs
29. Evaluation after an approved smoke run must cover:
    - clean validation subset
    - 0058c fixed SNSAug benchmark subset
30. Evaluation reports must include:
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
    - non-tampered high mask rate
31. CPU-safe tests must cover:
    - config validator requires approval text for real smoke training
    - config validator rejects train manifests with validation/test rows
    - config validator rejects pair/evaluation roots as training input
    - config validator rejects repo-local output and checkpoint roots
    - smoke limits are enforced
    - loss functions compute finite values without third-party training data
    - valid mask loss excludes `ignore_mask`
    - tampered-score consistency penalizes SNS collapse
    - hard negative loss penalizes high tampered probability for real/synthetic samples
    - runner dry-run writes no checkpoint and starts no training
    - no network/download/training occurs in unit tests
32. Documentation must include marker:
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
python3 scripts/agent/check_agent_changes.py tasks/0060a-snsaug-v2-activation-mask-robustness-finetune-smoke.md
```

## Acceptance Criteria

- A guarded SNSAug V2 fine-tuning smoke config, validator, CLI, runner module, loss module, tests, and docs exist.
- The CLI supports `--help` without real paths.
- The validator rejects unsafe configs and requires `I_APPROVE_SNSAUG_V2_FINETUNE_SMOKE` before any real smoke training can run.
- Train manifests are verified as train-only.
- Evaluation pair roots and validation outputs cannot be used as training input.
- Output and checkpoint roots must be outside the repository.
- Smoke limits prevent full training.
- Loss helpers implement class CE, ignore-mask valid-region mask loss, tampered-score consistency, clean/SNS class consistency, hard negative tampered suppression, and family-loss masking where applicable.
- Unit tests are CPU-safe and do not require real training, real datasets, real benchmark roots, or network.
- Real approved smoke runs write required run artifacts and checkpoints only under external roots.
- Documentation contains `SNSAUG_V2_FINETUNE_SMOKE_OK`.
- All changed files are within the allowed modify list.

## Stop Condition

Stop immediately if completing this task would require:

- full training,
- running real smoke fine-tuning without exact approval text,
- using validation/test/evaluation outputs as training data,
- accessing protected paths such as `.env`, `secrets/`, `data/`, `datasets/`, `outputs/`, or protected checkpoint directories,
- downloading datasets/assets/checkpoints,
- installing packages,
- using network resources,
- writing checkpoints or outputs inside the repository,
- implementing the v2 SNS/Nuisance Mask Head,
- changing files outside the allowed modify list,
- or expanding beyond guarded smoke fine-tuning infrastructure and CPU-safe tests.
