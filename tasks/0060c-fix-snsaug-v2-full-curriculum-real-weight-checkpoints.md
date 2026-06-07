# 0060c Fix SNSAug V2 Full Curriculum Real Weight Checkpoints

## Title

Fix SNSAug v2 full-curriculum fine-tuning to write real PyTorch model checkpoints.

## Role

Codex-only implementation worker, reviewer, and limited repair manager.

Claude Code is unavailable. Codex must implement directly only after this task file is committed and the user explicitly says `implement`.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0060b-snsaug-v2-full-curriculum-finetune.md`
- `tasks/0060b-fix-full-curriculum-actual-training-branch.md`
- `tasks/0060b-relax-full-curriculum-step-guardrail-after-guarded-short-pass.md`
- `tasks/0060b-debug-tampered-score-consistency-loss-zero.md`
- `tasks/0061b-fix-checkpoint-comparison-evaluator-real-inference.md`
- `tasks/0060c-fix-snsaug-v2-full-curriculum-real-weight-checkpoints.md`
- `configs/training/snsaug_v2_full_curriculum_finetune.example.json`
- `scripts/agent/validate_snsaug_v2_full_curriculum_finetune_config.py`
- `scripts/training/run_snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_losses.py`
- `src/cv_forensics/pre_sns_v3_model.py`
- `src/cv_forensics/pre_sns_v3_report.py`
- `src/cv_forensics/pre_sns_v3_sns_robustness_eval.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_model.py`
- `src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py`
- `src/cv_forensics/snsaug_v2_checkpoint_comparison_eval.py`
- `tests/test_snsaug_v2_full_curriculum_finetune.py`
- `tests/test_snsaug_v2_checkpoint_comparison_eval.py`
- `docs/snsaug_v2_full_curriculum_finetune.md`
- `.local/pre_sns_current_best_model_bundle.json`

## Files Codex May Modify

- `configs/training/snsaug_v2_full_curriculum_finetune.example.json`
- `scripts/agent/validate_snsaug_v2_full_curriculum_finetune_config.py`
- `scripts/training/run_snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_full_curriculum_finetune.py`
- `tests/test_snsaug_v2_full_curriculum_finetune.py`
- `docs/snsaug_v2_full_curriculum_finetune.md`

## Forbidden Actions

- Do not run long training in this implementation task.
- Do not use validation samples for training.
- Do not use evaluation pair roots as training input.
- Do not use validation failures for training.
- Do not use network.
- Do not download assets, datasets, models, or checkpoints.
- Do not install packages.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or checkpoint directories recursively.
- Do not write generated outputs inside the repository, except small test fixtures under test temporary directories.
- Do not modify existing model checkpoints outside test temporary directories.
- Do not replace real inference or real checkpoints with proxy metrics.
- Do not run `git push`, `git pull`, or `git fetch`.
- Do not commit changes unless the user explicitly says `commit this`.

## Important Project Facts

- 0061b correctly rejects the current 0060b checkpoints because they contain only scalar `trainable_state` proxy values and no real tensor weights.
- 0060b full-curriculum non-dry-run must produce checkpoints usable by the 0061b real-inference evaluator.
- The real inference architecture is the pre-SNS v3 model used by `pre_sns_v3_report.py` and bundled by `.local/pre_sns_current_best_model_bundle.json`.
- This task may implement real short training code, but validation must not run long training.
- Output and checkpoint roots must remain outside the repository.

## Implementation Requirements

1. Load the real base model.
   - Start from `.local/pre_sns_current_best_model_bundle.json` or the configured base bundle path.
   - Load the actual pre-SNS model architecture used for classification and localization inference.
   - Apply base model weights from the bundle.
   - Fail loudly if the exact architecture loader is unavailable.
   - Do not silently fall back to `trainable_state` or scalar proxy state.

2. Make the actual non-dry-run branch perform real PyTorch updates.
   - When `scripts/training/run_snsaug_v2_full_curriculum_finetune.py` runs without `--dry-run`, update real `nn.Module` parameters.
   - Train configured components:
     - `class_head`
     - `tamper_localization_head`
   - Keep non-trainable components frozen if configured.
   - Use the 0059 train-only curriculum manifest.
   - Reject any train row with `split != train`.
   - Do not use clean validation manifest, 0058c pair root, or 0058e oracle root as training data.
   - Keep tiny local validation runs bounded by the existing step guardrails.

3. Preserve the required losses.
   - Use existing loss utilities where practical:
     - `L_class`
     - `L_tamper_mask_valid` with `valid_region = 1 - ignore_mask`
     - `L_tampered_score_consistency`
     - `L_clean_sns_class_consistency`
     - `L_real_synthetic_sns_hard_negative`
     - optional family loss only where `family_loss_mask == 1`

4. Write real checkpoint files.
   - Required files under `checkpoint_root`:
     - `snsaug_aware_multihead_forensics_v1_best.pt`
     - `snsaug_aware_multihead_forensics_v1_last.pt`
   - Each checkpoint must be a real PyTorch checkpoint containing:
     - `checkpoint_format: snsaug_v2_real_state_dict_v1`
     - `model_version: snsaug_aware_multihead_forensics_v1`
     - `model_state_dict: model.state_dict()`
     - `optimizer_state_dict: optimizer.state_dict()`
     - optional `scheduler_state_dict`
     - `global_step: int`
     - `phase: int`
     - sanitized `config`
     - `metrics`
     - `trainable_components`
   - Do not write checkpoint files that contain only:
     - `trainable_state`
     - proxy weights
     - scalar-only synthetic state

5. Add a checkpoint validation helper.
   - Add `validate_real_weight_checkpoint(path)`.
   - It must fail if:
     - `model_state_dict` is missing
     - tensor count is too small
     - tensor numel is too small
     - checkpoint contains only `trainable_state`
     - no trainable parameters changed from the base model when a base bundle or base state is provided by the caller

6. Update artifact manifest output.
   - `artifact_manifest.json` must include:
     - `training_started: true`
     - `checkpoint_written: true`
     - `best_checkpoint_path`
     - `last_checkpoint_path`
     - `best_checkpoint_sha256`
     - `last_checkpoint_sha256`
     - `checkpoint_format: snsaug_v2_real_state_dict_v1`
     - `real_weight_checkpoint: true`

7. Maintain 0061b compatibility.
   - The 0061b evaluator must be able to load new checkpoints and run real inference.
   - Do not weaken 0061b proxy checkpoint rejection.

8. Update tests in `tests/test_snsaug_v2_full_curriculum_finetune.py`.
   - Non-dry-run writes `.pt` files with `model_state_dict`.
   - Proxy-only `trainable_state` checkpoint is rejected.
   - `validate_real_weight_checkpoint` passes on a real checkpoint.
   - `validate_real_weight_checkpoint` fails on a proxy checkpoint.
   - No val/test leakage.
   - No eval root is used as training data.
   - Dry-run remains plan-only.

9. Update docs in `docs/snsaug_v2_full_curriculum_finetune.md`.
   - Explain the real checkpoint format.
   - Explain why proxy `trainable_state` is invalid for real inference.
   - Add troubleshooting note for:
     - `fine-tuned checkpoint has only trainable_state proxy values`
   - Keep marker `SNSAUG_V2_FULL_CURRICULUM_FINETUNE_OK`.

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
python3 scripts/agent/check_agent_changes.py tasks/0060c-fix-snsaug-v2-full-curriculum-real-weight-checkpoints.md
```

## Acceptance Criteria

- Non-dry-run 0060b full-curriculum writes real `.pt` checkpoints with `model_state_dict` and optimizer state.
- Proxy-only checkpoint payloads are rejected by `validate_real_weight_checkpoint`.
- Real checkpoint validation confirms tensor count, tensor numel, format marker, and trainable parameter changes.
- Artifact manifest records checkpoint hashes, checkpoint format, and `real_weight_checkpoint=true`.
- Dry-run remains plan-only and writes no checkpoint.
- Train-only and eval-root leakage guardrails remain intact.
- Documentation clearly explains why scalar `trainable_state` checkpoints are invalid.
- All validation commands pass.
- `python3 scripts/agent/check_agent_changes.py tasks/0060c-fix-snsaug-v2-full-curriculum-real-weight-checkpoints.md` passes.

## Stop Condition

After creating this task file, stop and report the task path, short summary, current git status, and exact git add / git commit commands. Do not implement until the user commits this task file and explicitly says `implement`.
