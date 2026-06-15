# SNSAug V2 Fill Final Cross-model Ablation Metrics

SNSAUG_V2_0077A_FILL_FINAL_CROSS_MODEL_ABLATION_METRICS_OK

This task fills the 0077 final cross-model ablation/context rows from the earlier `final_decision_report.md` Markdown table.

It preserves the main `SIDA-7B` vs `mixed_feature_gate` comparison and only fills:

- `pre_sns_baseline`
- `failed_single_model_finetune`

The selected failed fine-tune model prefers `0064g` hybrid when present, then `0064f`, `0063b`, and `0060b`. Missing profile groups remain `NA`/`null` rather than being invented.

Run config validation with:

```bash
python3 scripts/agent/validate_snsaug_v2_fill_final_cross_model_ablation_metrics_config.py configs/evaluation/snsaug_v2_fill_final_cross_model_ablation_metrics.example.json
```
