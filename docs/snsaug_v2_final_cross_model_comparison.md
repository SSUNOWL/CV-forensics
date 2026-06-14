# SNSAug V2 Final Cross-model Comparison Matrix

SNSAUG_V2_0077_FINAL_CROSS_MODEL_COMPARISON_OK

This task generates final report-ready comparison artifacts from the 0076 unified SIDA-7B versus mixed-gate comparison and the 0072 final mixed-gate output.

The main report table contains only:

- `SIDA-7B`
- `mixed_feature_gate`

The ablation/context table contains:

- `pre_sns_baseline`
- `failed_single_model_finetune`
- `mixed_feature_gate`

The generated report must keep the interpretation narrow: SIDA-7B and `mixed_feature_gate` are not identical systems, SIDA emits 3-way labels and masks, and `mixed_feature_gate` is a deployable policy-gated recovery system. The gate is not a drop-in replacement for SIDA.

Run config validation with:

```bash
python3 scripts/agent/validate_snsaug_v2_final_cross_model_comparison_config.py configs/evaluation/snsaug_v2_final_cross_model_comparison.example.json
```
