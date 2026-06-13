# SNSAug V2 SIDA-7B Diagnostic Baseline

SNSAUG_V2_SIDA7B_DIAGNOSTIC_BASELINE_OK

0074 prepares and evaluates an offline SIDA-7B/SIDA-style diagnostic baseline for SNSAug fixed pairs. It does not train, fine-tune, download, access the network, or modify checkpoints.

## Modes

- `export_sida_eval_subset_mode`: exports a balanced SNSAug subset, prompt list, external runner template, and cached-output schema. SIDA is not run, so no performance conclusion is allowed.
- `cached_sida_output_mode`: reads a cached SIDA output JSONL produced outside `cv-forensics` and computes diagnostics.

## Exported Files

Export mode writes:

- `sida_eval_subset_manifest.jsonl`
- `sida_prompt_list.jsonl`
- `run_sida_external_template.sh`
- `sida_external_output_schema.md`

Prompts request SIDA-style `[CLS]` classification and `[SEG]` mask output when the class is tampered.

## Cached Evaluation

Cached rows may include `image_id`, `image_path`, `base_id`, `profile`, `profile_family`, `content_label`, `sida_text_output`, `sida_pred_class`, optional probabilities, optional `sida_mask_path`, `parse_error`, and `mask_missing`.

When cached outputs exist, the analyzer writes:

- `sida7b_diagnostic_records.jsonl`
- `sida7b_per_profile_metrics.json`
- `sida7b_type_a_type_b_summary.json`
- `sida7b_vs_mixed_gate_comparison.json`
- `sida7b_diagnostic_report.md`
- `sida7b_raw_output_audit.json`
- `sida7b_corrected_type_a_type_b_summary.json`
- `sida7b_corrected_vs_mixed_gate_comparison.json`
- `sida7b_corrected_diagnostic_report.md`
- `artifact_manifest.json`

The corrected report separates SIDA model behavior from wrapper limitations. If raw text contains `[SEG]` but no mask path was cached, this is reported as `mask_extraction_failed`. If raw text does not contain `[SEG]`, this is reported as `mask_not_requested_or_not_generated`. In both cases localization is treated as unavailable for the cached run; the report does not claim localization failure when masks are unavailable.

The raw audit records `[CLS]`, `[SEG]`, real, synthetic, fully synthetic, generated, tampered, and manipulated term counts from `sida_text_output`. The parser prefers the explicit `[CLS]` sentence when present and treats ambiguous class terms as unknown.

## Failure Types

- Type A: local overlay / nuisance profiles such as platform UI, story/shorts/tiktok overlays, and meme/news overlays.
- Type B: global geometry/degradation profiles such as canvas, resize/crop, JPEG, screenshot recapture, and recompression.

Decisions are:

- `sida_handles_both_types`
- `sida_fails_local_overlay_only`
- `sida_fails_global_degradation_only`
- `sida_fails_both_types`
- `sida_cached_outputs_missing`

Corrected cached reports may also include multiple decision findings:

- `sida_classification_collapse_synthetic`
- `sida_local_overlay_worse_than_global`
- `sida_global_degradation_partially_failed`
- `sida_localization_unavailable_mask_missing`
- `sida_vs_mixed_gate_incomplete`
- `sida_vs_mixed_gate_available`

The report explicitly distinguishes export-only artifacts from cached-output evaluations.
