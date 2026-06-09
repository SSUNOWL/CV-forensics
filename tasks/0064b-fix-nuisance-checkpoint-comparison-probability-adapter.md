# Task 0064b: Fix Nuisance Checkpoint Comparison Probability Adapter

## Context

0064 nuisance 30x3 training completed and wrote a real checkpoint.
The 4-model checkpoint comparison config validates and all input checkpoints exist.
However, checkpoint comparison evaluation fails before writing outputs.

Observed failure:

- model snsaug_0064_nuisance_30x3 produced records without p_tampered
- output root exists, but artifact_manifest.json, model_eval_records.jsonl, model_eval_comparisons.jsonl, per_model_per_profile_metrics.json, and checkpoint_comparison_summary.json are missing.

Existing 0061/0063b evaluations work and write records with p_real, p_synthetic, p_tampered.
Therefore this is not a training failure. It is an evaluator output-schema adapter failure for the 0064 nuisance model.

## Goal

Update the checkpoint comparison evaluator so that nuisance-model records always include:

- p_real
- p_synthetic
- p_tampered
- pred_class

The evaluator must normalize class probabilities from 0064 nuisance model outputs.

## Files Codex May Modify

- `tasks/0064b-fix-nuisance-checkpoint-comparison-probability-adapter.md`
- `src/cv_forensics/snsaug_v2_checkpoint_comparison_eval.py`
- `tests/test_snsaug_v2_checkpoint_comparison_eval.py`
- `docs/snsaug_v2_checkpoint_comparison_eval.md`
- `configs/evaluation/snsaug_v2_checkpoint_comparison_eval.example.json`
- `scripts/evaluation/run_snsaug_v2_checkpoint_comparison_eval.py`
- `scripts/agent/validate_snsaug_v2_checkpoint_comparison_eval_config.py`

## Files Claude May Modify

- `tasks/0064b-fix-nuisance-checkpoint-comparison-probability-adapter.md`
- `src/cv_forensics/snsaug_v2_checkpoint_comparison_eval.py`
- `tests/test_snsaug_v2_checkpoint_comparison_eval.py`
- `docs/snsaug_v2_checkpoint_comparison_eval.md`
- `configs/evaluation/snsaug_v2_checkpoint_comparison_eval.example.json`
- `scripts/evaluation/run_snsaug_v2_checkpoint_comparison_eval.py`
- `scripts/agent/validate_snsaug_v2_checkpoint_comparison_eval_config.py`

## Required Behavior

1. Add a robust class probability normalization helper in src/cv_forensics/snsaug_v2_checkpoint_comparison_eval.py.

It should support likely output forms:

- output['class_logits']
- output['logits']
- output['classification_logits']
- output['class_head_logits']
- output['class_probs']
- output['probs']
- output['probabilities']
- output['classification']['logits']
- output['classification']['probs']
- record fields that already contain p_real / p_synthetic / p_tampered

If logits are found, apply softmax over labels [real, synthetic, tampered].
If probabilities are found, normalize or clamp if necessary.

2. Every model_eval_records row must include p_real, p_synthetic, p_tampered, and pred_class.

3. If probabilities cannot be extracted, write a probability_adapter_diagnostics.json before failing.

Diagnostics should include:

- model_id
- model_kind
- checkpoint path
- raw output keys
- record keys
- available nested keys
- sample base_id/profile/content_label

4. Do not silently set p_tampered to 0 unless the model produced an explicit probability vector.

5. Keep strict sanity guard: final records missing p_tampered should still fail, but only after diagnostics and partial records are written.

6. Support the 0064 nuisance checkpoint as model_kind snsaug_finetuned_checkpoint unless validator already supports a separate kind.

7. Do not train. Do not download. Do not use network.

## Tests

Update tests/test_snsaug_v2_checkpoint_comparison_eval.py:

- old model output with p_tampered already present still passes
- nuisance output with class_logits produces p_real/p_synthetic/p_tampered
- nuisance output with class_probs produces p_real/p_synthetic/p_tampered
- missing probability output writes diagnostics and fails clearly
- final model_eval_records.jsonl contains p_tampered for all rows

## Validation Commands

- python3 scripts/agent/validate_snsaug_v2_checkpoint_comparison_eval_config.py configs/evaluation/snsaug_v2_checkpoint_comparison_eval.example.json
- CUDA_VISIBLE_DEVICES='' python3 tests/test_snsaug_v2_checkpoint_comparison_eval.py
- python3 scripts/evaluation/run_snsaug_v2_checkpoint_comparison_eval.py --help
- grep -q SNSAUG_V2_CHECKPOINT_COMPARISON_EVAL_OK docs/snsaug_v2_checkpoint_comparison_eval.md

## Marker

SNSAUG_V2_CHECKPOINT_COMPARISON_EVAL_OK
