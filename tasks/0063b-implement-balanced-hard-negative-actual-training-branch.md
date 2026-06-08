# Task: 0063b Implement Balanced Hard-Negative Actual Training Branch

## Task Title

Implement the real-run branch for SNSAug V2 balanced hard-negative fine-tuning while keeping dry-run and guardrails intact.

## Role

Codex-only task writer, implementation worker, reviewer, and limited repair manager. Claude Code is unavailable, so Codex must implement directly only after this task file is committed and the user explicitly says `implement`.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0063-snsaug-v2-balanced-hard-negative-finetune.md`
- `docs/snsaug_v2_balanced_hard_negative_finetune.md`
- `docs/snsaug_v2_full_curriculum_finetune.md`
- `configs/training/snsaug_v2_balanced_hard_negative_finetune.example.json`
- `configs/training/snsaug_v2_full_curriculum_finetune.example.json`
- `scripts/agent/validate_snsaug_v2_balanced_hard_negative_finetune_config.py`
- `scripts/agent/validate_snsaug_v2_full_curriculum_finetune_config.py`
- `scripts/training/run_snsaug_v2_balanced_hard_negative_finetune.py`
- `scripts/training/run_snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_balanced_hard_negative_finetune.py`
- `src/cv_forensics/snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_losses.py`
- `src/cv_forensics/snsaug_v2_train_curriculum.py`
- `src/cv_forensics/snsaug_v2_dataset_wrapper.py`
- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `src/cv_forensics/pre_sns_v3_model.py`
- `src/cv_forensics/pre_sns_v3_report.py`
- `tests/test_snsaug_v2_balanced_hard_negative_finetune.py`
- `tests/test_snsaug_v2_full_curriculum_finetune.py`
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
- Do not just delete the real-run guard; replace it with a guarded real training branch.
- Do not run long training.
- Do not run real dataset training during implementation validation.
- Do not download datasets, assets, or checkpoints.
- Do not install packages.
- Do not access network resources from shell commands.
- Do not use validation samples for training.
- Do not use validation failures directly as training samples.
- Do not use 0058c fixed benchmark images, fixed pair roots, validation pair roots, evaluation roots, oracle diagnostic outputs, or checkpoint comparison outputs as training input.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, `checkpoints/`, or protected directories.
- Do not inspect protected directories recursively.
- Do not write outputs or checkpoints inside the repository.
- Do not create large files.
- Do not run `git push`, `git pull`, or `git fetch`.
- Do not use `rg`.
- Do not use `/snap/bin/codex`, `/home/rlatjswo/.npm-global/bin/codex`, `--permission-mode bypassPermissions`, or `--dangerously-skip-permissions`.

## Important Project Facts

- 0063 added balanced hard-negative validation, dry-run planning, loss helpers, docs, and tests.
- The current real-run branch in `src/cv_forensics/snsaug_v2_balanced_hard_negative_finetune.py` unconditionally raises:
  ```text
  real balanced hard-negative training is guarded; request explicit real-run approval before writing outputs or checkpoints
  ```
- The 0063b goal is to implement an actual guarded real-run branch for small approved runs, not to run long training.
- The approved real-run text is:
  ```text
  I_APPROVE_SNSAUG_V2_BALANCED_HARD_NEGATIVE_FINETUNE
  ```
- Dry-run behavior must remain unchanged:
  - validate config
  - build plan
  - `training_started=false`
  - `checkpoint_written=false`
  - no checkpoint writes
- Training data must remain train-only and must never include validation/evaluation/fixed-pair roots.
- Real-run outputs and checkpoints must be written only under configured external roots.
- Marker to keep:
  - `SNSAUG_V2_BALANCED_HARD_NEGATIVE_FINETUNE_OK`

## Implementation Requirements

1. Preserve dry-run behavior exactly.
2. For non-dry-run, require:
   - `approval_text == "I_APPROVE_SNSAUG_V2_BALANCED_HARD_NEGATIVE_FINETUNE"`
   - `no_network == true`
   - `no_download == true`
   - train-only manifest validation
   - eval roots rejected as training input
   - external `output_root`
   - external `checkpoint_root`
3. Create `output_root` and `checkpoint_root` only after validation and approval pass.
4. Load the pre-SNS base bundle from config.
5. Instantiate the same SNSAug-aware multihead model architecture used by the 0060b real-weight checkpoint path.
6. Load initial weights from the pre-SNS best bundle or configured base bundle.
7. Train for configured phase steps:
   - `phase_1_max_steps`
   - `phase_2_max_steps`
   - `phase_3_max_steps`
   - Keep tests tiny and CPU-safe.
8. Use existing losses from `src/cv_forensics/snsaug_v2_losses.py`:
   - `L_class`: cross entropy for `real/synthetic/tampered`
   - `L_tamper_mask_valid`: valid region `1 - ignore_mask`
   - `L_non_tampered_tampered_suppression`: applies only to `real/synthetic`, default `p_tampered_ceiling=0.05`, default `lambda_hardneg=2.0`
   - `L_tampered_score_consistency`: applies only to `tampered`, default lambda `0.25`
   - `L_clean_sns_class_consistency`: applies to real/synthetic paired clean/SNS views and excludes tampered overactivation
9. Implement balanced hard-negative sampling:
   - train split only
   - hard phases 2 and 3
   - approximately `non_tampered_sns : tampered_sns = 2 : 1`
   - never use eval pair root for training
10. Write `training_log.jsonl` with per-step or periodic rows containing:
    - `phase`
    - `step`
    - `total_loss`
    - `class_loss`
    - `hardneg_loss`
    - `tampered_score_consistency_loss`
    - `clean_sns_class_consistency_loss`
    - `mask_loss`
    - `mean_p_tampered_real_sns`
    - `mean_p_tampered_synthetic_sns`
    - `mean_p_tampered_tampered_sns`
11. Write these files under `output_root`:
    - `artifact_manifest.json`
    - `training_log.jsonl`
    - `loss_breakdown.json`
    - `per_phase_metrics.json`
    - `clean_validation_metrics.json`
    - `snsaug_0058c_metrics.json`
    - `threshold_sweep_after_training.json`
    - `balanced_hard_negative_report.md` or `full_curriculum_report.md`
12. Write these checkpoints under `checkpoint_root`:
    - `snsaug_aware_multihead_forensics_v1_best.pt`
    - `snsaug_aware_multihead_forensics_v1_last.pt`
13. Both checkpoints must contain real model weights:
    - `model_state_dict` or `state_dict`
    - `optimizer_state_dict` if available
    - `checkpoint_kind = "snsaug_v2_balanced_hard_negative_real_model_weights"`
    - `model_version`
    - `global_step`
    - `phase`
    - `metrics`
    - `config_digest`
14. Best checkpoint policy must use:
    ```text
    balanced_score =
      + 1.0 * snsaug_tampered_recall
      + 1.0 * snsaug_valid_iou
      + 0.8 * synthetic_recall
      - 2.0 * real_fpr
      - 1.0 * non_tampered_high_mask_rate
    ```
15. First corrective run guardrails:
    - `real_fpr <= 0.30`
    - `synthetic_recall >= 0.40`
    - `clean_macro_f1 >= 0.75`
    - `tampered_recall >= 0.50`
16. If no checkpoint satisfies guardrails:
    - still write last checkpoint
    - write best checkpoint as fallback from last
    - mark `best_checkpoint_selected_by = "fallback_last_no_guardrail_pass"` in artifact metadata
17. Update tests in `tests/test_snsaug_v2_balanced_hard_negative_finetune.py`:
    - dry-run does not train
    - real-run with approval starts training
    - real-run without approval fails
    - real-run writes `training_log.jsonl`
    - real-run writes best and last `.pt` checkpoints
    - checkpoints include `model_state_dict`
    - train split leakage is rejected
    - eval pair root cannot be used as training input
    - hard-negative loss applies only to `real/synthetic`
    - tampered score consistency applies only to `tampered`
    - loss values are finite
18. Update docs in `docs/snsaug_v2_balanced_hard_negative_finetune.md`:
    - explain that earlier 0063 was guardrails-only
    - explain the actual training branch
    - explain the hard-negative correction purpose
    - explain real-run approval and outputs
    - keep `SNSAUG_V2_BALANCED_HARD_NEGATIVE_FINETUNE_OK`

## Validation Commands

```bash
python3 scripts/agent/validate_snsaug_v2_balanced_hard_negative_finetune_config.py configs/training/snsaug_v2_balanced_hard_negative_finetune.example.json
```

```bash
python3 tests/test_snsaug_v2_balanced_hard_negative_finetune.py
```

```bash
python3 scripts/training/run_snsaug_v2_balanced_hard_negative_finetune.py --help
```

```bash
grep -q SNSAUG_V2_BALANCED_HARD_NEGATIVE_FINETUNE_OK docs/snsaug_v2_balanced_hard_negative_finetune.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0063b-implement-balanced-hard-negative-actual-training-branch.md
```

## Acceptance Criteria

- Dry-run remains unchanged and writes no checkpoints.
- Non-dry-run without exact approval fails before writing outputs/checkpoints.
- Non-dry-run with exact approval starts tiny CPU-safe training in tests.
- Real-run branch writes all required output files under external `output_root`.
- Real-run branch writes best and last `.pt` checkpoints under external `checkpoint_root`.
- Checkpoints contain real `model_state_dict` or `state_dict`.
- Checkpoint metadata includes required checkpoint kind, model version, global step, phase, metrics, and config digest.
- Training log contains required loss and p-tampered fields.
- Loss values are finite in tests.
- Best checkpoint selection uses balanced score and records fallback when guardrails do not pass.
- Train split leakage and eval-root training input are rejected.
- Docs explain that 0063 was guardrails-only and 0063b adds the actual branch.
- The marker `SNSAUG_V2_BALANCED_HARD_NEGATIVE_FINETUNE_OK` remains present.
- No long training, downloads, network access, package installs, protected-path access, or repo-local output/checkpoint writes occur.
- `python3 scripts/agent/check_agent_changes.py tasks/0063b-implement-balanced-hard-negative-actual-training-branch.md` passes.

## Stop Condition

Stop after creating this task file. Do not implement until the user commits this task file and explicitly says `implement`.
