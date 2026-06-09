# Task 0064c: Fix Nuisance Checkpoint Loader and Probability Records

## Context

0064b probability adapter implementation validation passed, but the actual 4-model comparison still fails.

Observed failure:

- output root exists
- model_eval_records.jsonl exists
- model_eval_comparisons.jsonl is missing
- per_model_per_profile_metrics.json is missing
- checkpoint_comparison_summary.json is missing
- records: 780
- record model_id: only snsaug_0064_nuisance_30x3
- records_missing_p_tampered: 780
- probability_adapter_diagnostics.json exists

Important diagnostic clue:

The diagnostics list p_real / p_synthetic / p_tampered as available keys, but records still miss p_tampered. The diagnostics also list an error key. Therefore this may be a nuisance-model load/inference error, not just a probability key mapping issue.

## Goal

Make the checkpoint comparison evaluator correctly evaluate 0064 nuisance checkpoints and produce standard records with p_real, p_synthetic, p_tampered, and pred_class.

## Files Codex May Modify

- `tasks/0064c-fix-nuisance-eval-loader-and-probability-records.md`
- `src/cv_forensics/snsaug_v2_checkpoint_comparison_eval.py`
- `tests/test_snsaug_v2_checkpoint_comparison_eval.py`
- `docs/snsaug_v2_checkpoint_comparison_eval.md`
- `configs/evaluation/snsaug_v2_checkpoint_comparison_eval.example.json`
- `scripts/evaluation/run_snsaug_v2_checkpoint_comparison_eval.py`
- `scripts/agent/validate_snsaug_v2_checkpoint_comparison_eval_config.py`

## Files Claude May Modify

- `tasks/0064c-fix-nuisance-eval-loader-and-probability-records.md`
- `src/cv_forensics/snsaug_v2_checkpoint_comparison_eval.py`
- `tests/test_snsaug_v2_checkpoint_comparison_eval.py`
- `docs/snsaug_v2_checkpoint_comparison_eval.md`
- `configs/evaluation/snsaug_v2_checkpoint_comparison_eval.example.json`
- `scripts/evaluation/run_snsaug_v2_checkpoint_comparison_eval.py`
- `scripts/agent/validate_snsaug_v2_checkpoint_comparison_eval_config.py`

## Required Behavior

1. Diagnose nuisance checkpoint type from checkpoint metadata and state_dict keys.

If checkpoint_kind/model_version/state_dict indicates nuisance model, load the nuisance model architecture instead of the older snsaug_aware_multihead_forensics_v1 architecture.

Likely indicators:
- checkpoint_kind contains nuisance
- model_version contains nuisance
- state_dict has sns_nuisance_mask_head / global_degradation_head / reliability_head / nuisance keys

2. Import/use the 0064 nuisance model implementation if available.

Expected module:
- src/cv_forensics/snsaug_v2_nuisance_model.py

The evaluator may import it, but do not modify nuisance training files in this task.

3. Normalize model output probabilities.

Every record must include:
- p_real
- p_synthetic
- p_tampered
- pred_class

Support outputs:
- class_logits
- logits
- classification_logits
- class_head_logits
- class_probs
- probs
- probabilities
- classification.logits
- classification.probs
- existing p_real/p_synthetic/p_tampered fields
- tampered_score plus class logits/probs if available

4. Do not hide inference errors.

If a per-record inference error occurs:
- write model_eval_record_errors.jsonl
- write probability_adapter_diagnostics.json
- include traceback/error message
- fail clearly unless continue_on_record_error=true

Do not emit 780 rows with null p_tampered and only a generic probability error.

5. Preserve existing behavior.

Existing baseline / 0060b / 0063b comparison must still pass and records_missing_p_tampered must remain 0.

6. Partial output behavior.

If evaluation fails, write partial model_eval_records.jsonl, model_eval_record_errors.jsonl, and probability_adapter_diagnostics.json before failing.

7. Tests.

Update tests/test_snsaug_v2_checkpoint_comparison_eval.py:

- old checkpoint output with p_tampered passes
- nuisance checkpoint detection from state_dict keys works
- nuisance output with class_logits produces p_real/p_synthetic/p_tampered
- nuisance output with class_probs produces p_real/p_synthetic/p_tampered
- inference error writes model_eval_record_errors.jsonl and diagnostics
- final records never miss p_tampered for successful run
- 2-model baseline comparison still passes

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_checkpoint_comparison_eval_config.py configs/evaluation/snsaug_v2_checkpoint_comparison_eval.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_checkpoint_comparison_eval.py
- python3 scripts/evaluation/run_snsaug_v2_checkpoint_comparison_eval.py --help
- grep -q SNSAUG_V2_CHECKPOINT_COMPARISON_EVAL_OK docs/snsaug_v2_checkpoint_comparison_eval.md
- python3 scripts/agent/check_agent_changes.py tasks/0064c-fix-nuisance-eval-loader-and-probability-records.md

## Marker

SNSAUG_V2_CHECKPOINT_COMPARISON_EVAL_OK
