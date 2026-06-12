# SNSAug v2 Policy Gate Comparison

SNSAUG_V2_POLICY_GATE_COMPARISON_OK

0071 compares no-training residual-vs-geometry policy gates after 0070a reports a mixed low-level and geometry signal.

Outputs:

- `policy_gate_records.jsonl`
- `policy_gate_metrics.json`
- `policy_gate_oracle_gap_summary.json`
- `policy_gate_report.md`
- `policy_gate_comparison_report.md`
- `artifact_manifest.json`

`artifact_manifest.json` lists both report paths. The two report files may contain identical Markdown; both names are written so audits and downstream scripts can use the stable comparison-report filename.

Each `policy_gate_records.jsonl` row includes selector provenance:

- `selector`
- `chosen_policy`
- `selector_reason`
- `diagnostic_only_selector`
- `geometry_score`
- `residual_score`
- `selected_by_profile_family`

`fixed_original` always records `chosen_policy=original`. `oracle_best_policy_diagnostic` is marked with `diagnostic_only_selector=true`.

Decisions:

- `deployable_policy_gate_promising`
- `diagnostic_only_policy_gate`
- `policy_gate_not_sufficient`
