# 0060b Fix Real Model Weight Checkpoints

## Title

Fix 0060b SNSAug v2 full-curriculum fine-tuning to write real model-weight checkpoints.

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
- `tasks/0060c-fix-snsaug-v2-full-curriculum-real-weight-checkpoints.md`
- `tasks/0061b-fix-checkpoint-comparison-evaluator-real-inference.md`
- `tasks/0060b-fix-real-model-weight-checkpoints.md`
- `configs/training/snsaug_v2_full_curriculum_finetune.example.json`
- `configs/evaluation/snsaug_v2_checkpoint_comparison_eval.example.json`
- `scripts/agent/validate_snsaug_v2_full_curriculum_finetune_config.py`
- `scripts/agent/validate_snsaug_v2_checkpoint_comparison_eval_config.py`
- `scripts/training/run_snsaug_v2_full_curriculum_finetune.py`
- `scripts/evaluation/run_snsaug_v2_checkpoint_comparison_eval.py`
- `src/cv_forensics/snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_checkpoint_comparison_eval.py`
- `src/cv_forensics/snsaug_v2_losses.py`
- `src/cv_forensics/pre_sns_v3_model.py`
- `src/cv_forensics/pre_sns_v3_report.py`
- `src/cv_forensics/pre_sns_v3_sns_robustness_eval.py`
- `src/cv_forensics/pre_sns_v3_tile_localizer_v2_model.py`
- `src/cv_forensics/pre_sns_v3_v2_policy_gated_report.py`
- `tests/test_snsaug_v2_full_curriculum_finetune.py`
- `tests/test_snsaug_v2_checkpoint_comparison_eval.py`
- `docs/snsaug_v2_full_curriculum_finetune.md`
- `docs/snsaug_v2_checkpoint_comparison_eval.md`
- `.local/pre_sns_current_best_model_bundle.json`

## Files Codex May Modify

- `configs/training/snsaug_v2_full_curriculum_finetune.example.json`
- `configs/evaluation/snsaug_v2_checkpoint_comparison_eval.example.json`
- `scripts/agent/validate_snsaug_v2_full_curriculum_finetune_config.py`
- `scripts/agent/validate_snsaug_v2_checkpoint_comparison_eval_config.py`
- `scripts/training/run_snsaug_v2_full_curriculum_finetune.py`
- `scripts/evaluation/run_snsaug_v2_checkpoint_comparison_eval.py`
- `src/cv_forensics/snsaug_v2_full_curriculum_finetune.py`
- `src/cv_forensics/snsaug_v2_checkpoint_comparison_eval.py`
- `tests/test_snsaug_v2_full_curriculum_finetune.py`
- `tests/test_snsaug_v2_checkpoint_comparison_eval.py`
- `docs/snsaug_v2_full_curriculum_finetune.md`
- `docs/snsaug_v2_checkpoint_comparison_eval.md`

## Forbidden Actions

- Do not run long training in this implementation task.
- Do not use validation samples for training.
- Do not use evaluation pair roots as training input.
- Do not use validation failures for training.
- Do not use network.
- Do not download assets, datasets, models, or checkpoints.
- Do not install packages.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or checkpoint directories recursively.
- Do not modify existing checkpoint files outside test temporary directories.
- Do not write generated outputs inside the repository, except small test fixtures under test temporary directories.
- Do not replace real inference with proxy metrics.
- Do not write `.pt` files that pretend proxy scalar state is real model state.
- Do not run `git push`, `git pull`, or `git fetch`.
- Do not commit changes unless the user explicitly says `commit this`.

## Important Project Facts

- 0061b correctly rejects old 0060b checkpoints because they contain only proxy `trainable_state` values.
- Existing proxy checkpoints are not valid for real inference comparison.
- 0060b full-curriculum non-dry-run must write checkpoints that 0061b can load for real fixed-pair inference.
- If the full-curriculum runner cannot instantiate a real `torch.nn.Module`, it must fail or write a clearly named proxy artifact, not a fake `.pt` model checkpoint.
- This task may implement or refine real short training code, but validation must stay tiny and must not run long training.

## Implementation Requirements

1. Locate checkpoint saving logic in:
   - `src/cv_forensics/snsaug_v2_full_curriculum_finetune.py`
   - `scripts/training/run_snsaug_v2_full_curriculum_finetune.py`

2. Replace proxy-only checkpoint saving with real model-weight saving.
   - A valid checkpoint must contain at least one of:
     - `model_state_dict`
     - `state_dict`
     - `class_head_state_dict` plus `tamper_localization_head_state_dict`
   - Preferred checkpoint payload:
     - `schema_version: "1.0"`
     - `checkpoint_kind: "snsaug_v2_real_model_weights"`
     - `model_version: "snsaug_aware_multihead_forensics_v1"`
     - `model_state_dict: model.state_dict()`
     - `optimizer_state_dict: optimizer.state_dict()` if available
     - `global_step`
     - `phase`
     - `base_model_bundle_path`
     - `trainable_components: ["class_head", "tamper_localization_head"]`
     - `metrics`
     - `config_digest`

3. If the current 0060b runner is still proxy-only:
   - Do not write `.pt` files pretending to be real checkpoints.
   - Either write `proxy_checkpoint.json` and fail when real checkpoint output is requested, or implement the minimal actual model loading/training path from the pre-SNS bundle.
   - Prefer implementing the minimal actual model loading/training path when feasible.

4. Update 0061b loader compatibility if needed.
   - `pre_sns_bundle` loads `.local/pre_sns_current_best_model_bundle.json`.
   - `snsaug_finetuned_checkpoint` loads pre-SNS base model architecture, then applies `model_state_dict` or `state_dict` from the `.pt`.
   - Do not weaken rejection of `trainable_state`-only checkpoints.

5. Add or refine checkpoint realness validation.
   - Require `tensor_count > 0`.
   - Require `tensor_total_numel > 1000`.
   - Require `model_state_dict` or `state_dict`, or separate class/localization head state dicts.
   - Reject `trainable_state`-only checkpoints.

6. Update tests in `tests/test_snsaug_v2_full_curriculum_finetune.py`.
   - Non-dry-run writes `.pt` with `model_state_dict`.
   - Checkpoint contains tensors.
   - Proxy-only checkpoint is rejected for real inference.
   - Best and last checkpoints both exist.
   - `artifact_manifest.json` points to real checkpoints.

7. Update tests in `tests/test_snsaug_v2_checkpoint_comparison_eval.py`.
   - Fine-tuned checkpoint must have `model_state_dict` or `state_dict`.
   - `trainable_state`-only checkpoint fails.
   - Real checkpoint loads and produces records with `p_tampered`.

8. Update docs.
   - `docs/snsaug_v2_full_curriculum_finetune.md`
   - `docs/snsaug_v2_checkpoint_comparison_eval.md`
   - Add troubleshooting:
     - If 0061b says `trainable_state proxy values`, rerun 0060b after the real checkpoint saving fix.
     - Existing proxy checkpoints are not valid for real inference comparison.
   - Keep markers:
     - `SNSAUG_V2_FULL_CURRICULUM_FINETUNE_OK`
     - `SNSAUG_V2_CHECKPOINT_COMPARISON_EVAL_OK`

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
python3 scripts/agent/validate_snsaug_v2_checkpoint_comparison_eval_config.py configs/evaluation/snsaug_v2_checkpoint_comparison_eval.example.json
```

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_checkpoint_comparison_eval.py
```

```bash
python3 scripts/evaluation/run_snsaug_v2_checkpoint_comparison_eval.py --help
```

```bash
grep -q SNSAUG_V2_FULL_CURRICULUM_FINETUNE_OK docs/snsaug_v2_full_curriculum_finetune.md
```

```bash
grep -q SNSAUG_V2_CHECKPOINT_COMPARISON_EVAL_OK docs/snsaug_v2_checkpoint_comparison_eval.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0060b-fix-real-model-weight-checkpoints.md
```

## Acceptance Criteria

- 0060b non-dry-run writes real `.pt` checkpoint payloads with model tensors.
- 0060b does not write proxy scalar `trainable_state` payloads as model checkpoints.
- Checkpoint realness validation rejects `trainable_state`-only payloads.
- 0061b can load the new real checkpoint format without weakening proxy rejection.
- Real checkpoint fixture evaluation produces records containing `p_tampered`.
- Best and last checkpoints exist and are referenced by the artifact manifest.
- Documentation clearly says existing proxy checkpoints are invalid for real inference.
- All validation commands pass.
- `python3 scripts/agent/check_agent_changes.py tasks/0060b-fix-real-model-weight-checkpoints.md` passes.

## Stop Condition

After creating this task file, stop and report the task path, short summary, current git status, and exact git add / git commit commands. Do not implement until the user commits this task file and explicitly says `implement`.
