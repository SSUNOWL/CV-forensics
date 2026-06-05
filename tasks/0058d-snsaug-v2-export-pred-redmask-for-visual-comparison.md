# Task: 0058d SNSAug V2 Export Pred Redmask For Visual Comparison

## Task Title

Export predicted redmask and overlay visualizations from SNSAug V2 fixed-pair evaluation for report-quality visual comparison.

## Role

Codex-only task implementation worker, reviewer, and limited repair manager for an evaluation/reporting-only visualization export task.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0058-snsaug-v2-small-benchmark-with-geometry-and-postprocess.md`
- `tasks/0058c-snsaug-v2-small-benchmark-class-balanced-and-tampered-mask-fix.md`
- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `scripts/evaluation/run_snsaug_v2_fixed_pairs_eval.py`
- `configs/evaluation/snsaug_v2_fixed_pairs_eval.example.json`
- `tests/test_snsaug_v2_fixed_pairs_eval.py`
- `docs/snsaug_v2_fixed_pairs_eval.md`

## Files Codex May Modify

- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `tests/test_snsaug_v2_fixed_pairs_eval.py`
- `docs/snsaug_v2_fixed_pairs_eval.md`

## Forbidden Actions

- Do not train models.
- Do not fine-tune models.
- Do not run `0054`.
- Do not access network resources from shell commands.
- Do not download datasets, assets, or checkpoints.
- Do not install packages.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- Do not inspect protected directories recursively.
- Do not modify model checkpoints.
- Do not write generated benchmark/evaluation outputs inside the repository.
- Do not modify unrelated model, training, dataset, or SNSAug generation code.
- Do not use Claude Code while Claude Code is unavailable.
- Do not use `rg`.
- Do not run `git push`, `git pull`, or `git fetch`.

## Important Project Facts

- SNSAug V2 fixed-pair evaluation is evaluation/reporting-only.
- The evaluator already produces records, comparisons, per-profile metrics, robustness drops, and benchmark summaries.
- Report figures need side-by-side visual artifacts:
  - clean image
  - clean GT tamper mask
  - clean predicted redmask
  - SNSAug image
  - transformed GT tamper mask
  - ignore mask / SNS nuisance mask
  - overlap between tamper and ignore masks
  - SNS predicted redmask
- Existing IoU behavior must remain unchanged: valid-region evaluation masks out `ignore_mask`.
- Visual export must not imply model training, fine-tuning, checkpoint modification, or dataset generation.

## Implementation Requirements

1. Update `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py` so fixed-pair evaluation exports visualization PNGs under the configured external `output_root`.
2. Save predicted binary mask PNGs under:
   - `output_root/pred_masks/`
3. Save predicted red overlay images under:
   - `output_root/pred_red_overlays/`
4. Save GT red overlay images under:
   - `output_root/gt_red_overlays/`
5. Save ignore-mask blue overlay images under:
   - `output_root/ignore_blue_overlays/`
6. Save tamper-ignore overlap visualizations under:
   - `output_root/overlap_overlays/`
7. Update every record written to `snsaug_v2_eval_records.jsonl` with:
   - `pred_mask_path`
   - `pred_red_overlay_path`
   - `gt_red_overlay_path`
   - `ignore_blue_overlay_path`
   - `overlap_overlay_path`
   - `pred_mask_available`
   - `localization_activated`
8. Preserve the existing `localization_activated` boolean meaning.
9. Add config handling for `write_empty_pred_mask`.
10. If localization is not activated and `write_empty_pred_mask` is true:
    - write an all-zero predicted mask PNG
    - write corresponding predicted red overlay image
    - still populate `pred_mask_path`
    - set `pred_mask_available` according to whether the predicted mask file exists
    - record `localization_activated=false`
11. If localization is not activated and `write_empty_pred_mask` is false:
    - do not fabricate an empty predicted mask
    - keep path fields `null` where no predicted artifact is written
    - record `localization_activated=false`
12. Keep valid IoU computation unchanged:
    - `valid_region = 1 - ignore_mask`
13. Overlay dimensions must match the evaluated image dimensions.
14. File names must be deterministic and safe for repeated runs.
15. Keep all visualization helpers local to the evaluator module unless an existing local helper already fits.
16. Update tests in `tests/test_snsaug_v2_fixed_pairs_eval.py` to cover:
    - predicted mask files are written when localization is activated
    - empty predicted mask is written when localization is off and `write_empty_pred_mask=true`
    - records include all new path fields
    - red overlay dimensions match image dimensions
    - no training/fine-tuning flags remain true and no checkpoint paths are modified
17. Update `docs/snsaug_v2_fixed_pairs_eval.md` with the new visualization export behavior.
18. Documentation must include marker:
    - `SNSAUG_V2_PRED_REDMASK_EXPORT_OK`

## Validation Commands

```bash
python3 tests/test_snsaug_v2_fixed_pairs_eval.py
```

```bash
grep -q SNSAUG_V2_PRED_REDMASK_EXPORT_OK docs/snsaug_v2_fixed_pairs_eval.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0058d-snsaug-v2-export-pred-redmask-for-visual-comparison.md
```

## Acceptance Criteria

- Fixed-pair evaluation exports predicted mask PNGs and report-ready overlays under the configured external `output_root`.
- `snsaug_v2_eval_records.jsonl` includes all new visualization path and availability fields.
- Localization-off records can still produce deterministic all-zero predicted masks when `write_empty_pred_mask=true`.
- Existing valid-IoU semantics are unchanged.
- Tests verify activated and inactive localization export behavior.
- Documentation contains `SNSAUG_V2_PRED_REDMASK_EXPORT_OK`.
- All changed files are within the task allowed modify list.
- No training, fine-tuning, network, dataset download, protected path access, checkpoint modification, or repo-local generated output occurs.

## Stop Condition

Stop immediately if completing this task would require:

- real training or fine-tuning,
- accessing protected paths such as `.env`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`,
- downloading datasets/assets/checkpoints,
- installing packages,
- using network resources,
- modifying model checkpoints,
- writing generated evaluation outputs inside the repository,
- changing files outside the allowed modify list,
- or expanding beyond fixed-pair evaluator visualization export scope.
