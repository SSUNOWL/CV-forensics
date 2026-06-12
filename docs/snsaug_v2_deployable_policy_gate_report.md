# SNSAug V2 Deployable Policy Gate Report

SNSAUG_V2_DEPLOYABLE_POLICY_GATE_REPORT_OK

0072 freezes the best non-diagnostic deployable policy gate from 0071a and writes final report-ready artifacts. It is a reporting and artifact-freeze step only: no training, no fine-tuning, no network access, no downloads, and no checkpoint changes.

## Inputs

The runner reads a configured 0071a policy gate output root containing:

- `policy_gate_records.jsonl`
- `policy_gate_metrics.json`
- `policy_gate_oracle_gap_summary.json`
- `policy_gate_report.md`
- `policy_gate_comparison_report.md`
- `artifact_manifest.json`

The input artifact must carry `SNSAUG_V2_POLICY_GATE_COMPARISON_OK`.

## Freeze Rule

The freezer selects the highest-score candidate from `decision.deployable_candidates` in `policy_gate_metrics.json`. It rejects diagnostic-only candidates and rejects final exported records whose `chosen_policy` is oracle-only. The selected gate is therefore a practical deployable policy, not an oracle diagnostic.

## Outputs

Successful actual runs write:

- `deployable_policy_gate_spec.json`
- `final_policy_gate_records.jsonl`
- `final_policy_gate_metrics.json`
- `final_policy_gate_report.md`
- `final_policy_gate_notion_summary.md`
- `final_policy_gate_tables.tsv`
- `artifact_manifest.json`

`deployable_policy_gate_spec.json` records the selected selector, score, good profiles, gains, oracle gap closures, synthetic/real side effects, chosen-policy distribution, feature-gate score summary, and `diagnostic_only=false`.

`final_policy_gate_records.jsonl` contains only rows from the selected selector. Each row includes `selector`, `chosen_policy`, `profile`, `content_label`, `pred_class`, `p_real`, `p_synthetic`, `p_tampered`, `valid_iou`, and `diagnostic_only_selector=false`.

## Report Interpretation

The final report explains why naive SNSAug fine-tuning failed, why local nuisance masking was insufficient, why global geometry/degradation dominated the failure, how the frozen gate combines geometry and residual/DCT signals, which profiles recovered, which profiles remain unresolved, and why the frozen policy is deployable rather than oracle-only.
