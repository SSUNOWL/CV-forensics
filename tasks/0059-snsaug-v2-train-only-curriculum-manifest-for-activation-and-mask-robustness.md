# Task: 0059 SNSAug V2 Train-Only Curriculum Manifest For Activation And Mask Robustness

## Task Title

Create a train-only SNSAug V2 curriculum manifest for SNS-aware fine-tuning focused on activation and mask robustness.

## Role

Codex-only task writer, implementation worker, reviewer, and limited repair manager for a manifest-generation-only task.

## Files Codex May Read

- `AGENTS.md`
- `CLAUDE.md`
- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0054-snsaug-v2-aware-finetuning.md`
- `tasks/0058c-snsaug-v2-small-benchmark-class-balanced-and-tampered-mask-fix.md`
- `tasks/0058d-snsaug-v2-export-pred-redmask-for-visual-comparison.md`
- `tasks/0058e-snsaug-v2-forced-localization-oracle-gate-diagnostic.md`
- `src/cv_forensics/snsaug_v2_dataset_wrapper.py`
- `src/cv_forensics/snsaug_v2_training_manifest.py`
- `scripts/training/build_snsaug_v2_training_manifest.py`
- `configs/training/snsaug_v2_training_manifest.example.json`
- `scripts/agent/validate_snsaug_v2_training_manifest_config.py`
- `tests/test_snsaug_v2_training_manifest.py`

## Files Codex May Modify

- `configs/training/snsaug_v2_train_curriculum_manifest.example.json`
- `scripts/agent/validate_snsaug_v2_train_curriculum_config.py`
- `scripts/training/build_snsaug_v2_train_curriculum_manifest.py`
- `src/cv_forensics/snsaug_v2_train_curriculum.py`
- `tests/test_snsaug_v2_train_curriculum_manifest.py`
- `docs/snsaug_v2_train_curriculum_manifest.md`

## Forbidden Actions

- Do not train models.
- Do not fine-tune models.
- Do not run `0054`.
- Do not use validation failures for training.
- Do not use validation samples for training.
- Do not use validation failure samples directly for training.
- Do not access network resources from shell commands.
- Do not download datasets, assets, or checkpoints.
- Do not install packages.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- Do not inspect protected directories recursively.
- Do not modify model checkpoints.
- Do not write generated outputs inside the repository.
- Do not run the builder against the real train manifest unless the user explicitly approves that run.
- Do not read optional evaluation result roots recursively unless the user explicitly approves that run.
- Do not modify training loops or launch training.
- Do not use Claude Code while Claude Code is unavailable.
- Do not use `rg`.
- Do not run `git push`, `git pull`, or `git fetch`.

## Important Project Facts

- SNSAug V2 generation, fixed-pair evaluation, redmask export, occlusion analysis, and forced-localization/oracle-gate diagnostics are complete.
- Key findings:
  - The frozen pre-SNS model performs well on clean samples.
  - Under SNSAug profiles, `p_tampered` often collapses.
  - Conditional localization is gated on `p_tampered`, so predicted redmask becomes empty or missing when the gate is off.
  - Occlusion analysis suggests high local overlay occlusion is not the main cause.
  - Forced localization/oracle gate shows that `resize_crop_pad`, `screenshot_recapture_light`, and `zoom_crop` recover some IoU when localization is forced.
  - `tiktok_like`, `instagram_story_like`, `youtube_shorts_like`, and `combined_sns_realistic` remain difficult even under oracle gate.
- The next model must improve both:
  - tampered-score / activation robustness
  - mask decoder robustness under geometry/canvas/platform-layout shift
- The real train manifest for later user-approved execution is:
  - `/home/rlatjswo/cvf_local_store/pre_sns_manifests_0035_clean/pre_sns_large_manifest_train_clean.local.json`
- Optional evaluation roots for later user-approved reference-only execution are:
  - `/home/rlatjswo/cvf_eval_outputs/snsaug_v2_0058c_balanced_benchmark_eval_*`
  - `/home/rlatjswo/cvf_eval_outputs/snsaug_v2_0058d_balanced_eval_with_pred_redmask_*`
  - `/home/rlatjswo/cvf_eval_outputs/snsaug_v2_0058e_forced_localization_oracle_gate_*`
- Recommended output root pattern for later user-approved execution:
  - `~/cvf_runs/snsaug_v2_0059_train_curriculum_manifest_YYYYMMDD_HHMMSS`
- Implementation tests must use small temporary fixtures only, not protected data paths or real manifests.

## Implementation Requirements

1. Add config example:
   - `configs/training/snsaug_v2_train_curriculum_manifest.example.json`
2. Add validator:
   - `scripts/agent/validate_snsaug_v2_train_curriculum_config.py`
3. Add builder CLI:
   - `scripts/training/build_snsaug_v2_train_curriculum_manifest.py`
4. Add implementation module:
   - `src/cv_forensics/snsaug_v2_train_curriculum.py`
5. Add tests:
   - `tests/test_snsaug_v2_train_curriculum_manifest.py`
6. Add docs:
   - `docs/snsaug_v2_train_curriculum_manifest.md`
7. The builder must be manifest-generation-only:
   - no training
   - no fine-tuning
   - no checkpoint writes
   - no dataset download
   - no network
8. The builder must read a training manifest and write outputs only under an external output root.
9. Required outputs:
   - `snsaug_v2_train_curriculum_manifest.jsonl`
   - `snsaug_v2_curriculum_schedule.json`
   - `snsaug_v2_profile_sampling_weights.json`
   - `snsaug_v2_class_balance_summary.json`
   - `snsaug_v2_training_guardrails.json`
   - `snsaug_v2_training_intent_and_references.md`
   - `artifact_manifest.json`
10. Profile groups must be represented exactly:
    - `clean`
    - `geometry_light`
    - `postprocess_light`
    - `screenshot_light`
    - `overlay_light`
    - `platform_layout`
    - `combined`
11. Profile group definitions:
    - `clean`: no augmentation
    - `geometry_light`: `canvas_9x16_only`, `resize_crop_pad`, `zoom_crop` with low probability and light severity
    - `postprocess_light`: `recompression_light`, `resize_jpeg` with light severity early
    - `screenshot_light`: `screenshot_recapture_light`
    - `overlay_light`: `platform_ui_same_size`, `news_meme_overlay`
    - `platform_layout`: `tiktok_like`, `instagram_story_like`, `youtube_shorts_like`
    - `combined`: `combined_sns_realistic`
12. Curriculum schedule must contain exactly:
    - Phase 1:
      - `clean`: `0.45`
      - `geometry_light`: `0.25`
      - `postprocess_light`: `0.15`
      - `overlay_light`: `0.10`
      - `screenshot_light`: `0.05`
      - `platform_layout`: `0.00`
      - `combined`: `0.00`
    - Phase 2:
      - `clean`: `0.35`
      - `geometry_light`: `0.25`
      - `postprocess_light`: `0.15`
      - `screenshot_light`: `0.10`
      - `overlay_light`: `0.05`
      - `platform_layout`: `0.10`
      - `combined`: `0.00`
    - Phase 3:
      - `clean`: `0.25`
      - `geometry_light`: `0.25`
      - `postprocess_light`: `0.15`
      - `screenshot_light`: `0.10`
      - `overlay_light`: `0.05`
      - `platform_layout`: `0.15`
      - `combined`: `0.05`
13. Each phase's profile group weights must sum to `1.0` within a small floating-point tolerance.
14. Each manifest JSONL record must include:
    - `base_id`
    - `source_dataset`
    - `split`
    - `image_path`
    - `tamper_mask_path` if available
    - `content_label`
    - `family_label` if available
    - `family_loss_mask`
    - `allowed_profile_groups`
    - `profile_sampling_weights_by_phase`
    - `severity_schedule`
    - `sampling_weight`
    - `label_preserved`
    - `training_notes`
15. `content_label` must be normalized to:
    - `real`
    - `synthetic`
    - `tampered`
16. Reject records whose split is not `train`.
17. Reject validation pair roots as training input.
18. Reject evaluation result files as training images.
19. `family_loss_mask` must be `0` when family label is missing.
20. `family_loss_mask` must be `1` when family label is present and non-empty.
21. `label_preserved` must always be `true`.
22. Guardrails output must include:
    - `no_training: true`
    - `no_finetune: true`
    - `no_network: true`
    - `no_download: true`
    - `train_only: true`
    - `output_root_outside_repository: true`
    - `validation_samples_rejected: true`
    - `evaluation_outputs_rejected_as_training_images: true`
23. The curriculum manifest must be compatible with existing guarded fine-tuning and SNSAug V2 dataset-wrapper expectations by supporting:
    - on-the-fly SNSAug generation
    - ignore-mask-based valid-region localization loss
    - clean/SNS class consistency
    - tampered-score consistency
    - real/synthetic SNS hard negatives
    - class-balanced sampling
24. The builder must generate a class balance summary containing counts by `content_label` and record totals.
25. The builder must generate a profile sampling weights JSON containing profile groups, profile lists, per-phase group weights, and severity schedule.
26. The builder must generate `snsaug_v2_training_intent_and_references.md`.
27. The training intent document must explain:
    - why this training is needed
    - baseline failure mode:
      - `p_tampered` collapse under SNSAug
      - conditional localization gate off
      - mask decoder weakness under platform layout
    - why train-only split is required
    - why validation/test fixed pairs are evaluation-only
    - why `ignore_mask` is excluded from tamper loss
    - references:
      - Community Forensics CVPR 2025
      - SIDA CVPR 2025
      - Ignoring the Decoy WACV 2026 Workshop
      - Degradation-consistent paired training / clean-degraded consistency
      - B-Free / bias-free paired detector training
28. Config validation must reject unsafe examples, including:
    - repo-local output root
    - non-train split policy
    - missing safety flags
    - protected-path output roots
    - evaluation-output roots as training manifest paths
29. Builder CLI must support `--help` without requiring real paths.
30. Documentation must include marker:
    - `SNSAUG_V2_TRAIN_CURRICULUM_MANIFEST_OK`

## Validation Commands

```bash
python3 scripts/agent/validate_snsaug_v2_train_curriculum_config.py configs/training/snsaug_v2_train_curriculum_manifest.example.json
```

```bash
python3 tests/test_snsaug_v2_train_curriculum_manifest.py
```

```bash
python3 scripts/training/build_snsaug_v2_train_curriculum_manifest.py --help
```

```bash
grep -q SNSAUG_V2_TRAIN_CURRICULUM_MANIFEST_OK docs/snsaug_v2_train_curriculum_manifest.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0059-snsaug-v2-train-only-curriculum-manifest-for-activation-and-mask-robustness.md
```

## Acceptance Criteria

- A train-only curriculum manifest builder exists and writes the required manifest, schedule, weights, class-balance, guardrails, intent, and artifact files under an external output root.
- No training, fine-tuning, checkpoint modification, download, network access, or repo-local generated output occurs.
- Non-train records are rejected.
- Validation/test/evaluation outputs are rejected as training sources.
- Profile group weights sum to `1.0` per phase.
- Manifest records conform to the required schema.
- `family_loss_mask` is correct for present/missing family labels.
- The training intent document explains the SNSAug activation and mask robustness rationale and required references.
- Tests cover train-only split enforcement, no validation leakage, profile weight sums, curriculum schedule parsing, class balance summary, family loss mask handling, manifest schema validation, and output-root guardrails.
- Documentation contains `SNSAUG_V2_TRAIN_CURRICULUM_MANIFEST_OK`.
- All changed files are within the allowed modify list.

## Stop Condition

Stop immediately if completing this task would require:

- training or fine-tuning,
- running `0054`,
- using validation failures for training,
- using validation samples for training,
- accessing protected paths such as `.env`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`,
- downloading datasets/assets/checkpoints,
- installing packages,
- using network resources,
- modifying model checkpoints,
- writing generated outputs inside the repository,
- running the builder on the real train manifest without explicit user approval,
- reading optional evaluation roots recursively without explicit user approval,
- changing files outside the allowed modify list,
- or expanding beyond train-only SNSAug V2 curriculum manifest generation scope.
