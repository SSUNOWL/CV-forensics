## Task Title
0055e SNSAug V2 Overlay Placement Collision Guard

## Role
Codex-only task writer and later implementation worker/reviewer for a narrow SNSAug v2 placement-collision refactor. This task adds deterministic overlay collision avoidance without training, fine-tuning, or touching unrelated model code.

## Files Codex May Read
- AGENTS.md
- CLAUDE.md
- docs/project_brief.md
- docs/project_contract.md
- configs/project_contract.json
- docs/agent_workflow.md
- tasks/0051-snsaug-v2-realistic-overlay-module-and-pair-generator.md
- tasks/0055-snsaug-v2-source-manifest-audit-overlay-generator-and-tiny-fixed-pairs.md
- tasks/0055d-snsaug-v2-layout-only-and-precise-ignore-mask-refactor.md
- docs/snsaug_v2.md
- docs/snsaug_v2_generation.md
- scripts/snsaug_v2/preview_snsaug_v2_overlays.py
- scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py
- src/cv_forensics/snsaug_v2/__init__.py
- src/cv_forensics/snsaug_v2/configs.py
- src/cv_forensics/snsaug_v2/transforms.py
- src/cv_forensics/snsaug_v2/ui_renderers.py
- src/cv_forensics/snsaug_v2/sticker_packs.py
- src/cv_forensics/snsaug_v2/platform_templates.py
- src/cv_forensics/snsaug_v2/sns_augmentor.py
- src/cv_forensics/snsaug_v2/pair_generator.py
- tests/test_snsaug_v2_generation.py
- tests/test_snsaug_v2_layout_only_and_precise_ignore.py
- relevant validators, scripts, and current git status

## Files Codex May Modify
- tasks/0055e-snsaug-v2-overlay-placement-collision-guard.md
- src/cv_forensics/snsaug_v2/__init__.py
- src/cv_forensics/snsaug_v2/configs.py
- src/cv_forensics/snsaug_v2/ui_renderers.py
- src/cv_forensics/snsaug_v2/sticker_packs.py
- src/cv_forensics/snsaug_v2/platform_templates.py
- src/cv_forensics/snsaug_v2/sns_augmentor.py
- src/cv_forensics/snsaug_v2/pair_generator.py
- src/cv_forensics/snsaug_v2/placement.py
- tests/test_snsaug_v2_generation.py
- tests/test_snsaug_v2_layout_only_and_precise_ignore.py
- tests/test_snsaug_v2_overlay_collision_guard.py
- docs/snsaug_v2_generation.md

## Forbidden Actions
- Do not train.
- Do not fine-tune.
- Do not run 0054.
- Do not use network.
- Do not download assets.
- Do not install packages.
- Do not use exact platform logos.
- Do not write generated outputs inside the repository.
- Do not modify unrelated model code.
- Do not modify scripts or files outside the allowed list.

## Important Project Facts
- This project uses Community Forensics-Small and SID-Set for a lightweight multi-head image forensics system covering 3-way classification, tampered mask localization, family provenance when available, and template-based explanation.
- SNSAug v2 simulates social-media perturbations such as platform UI, text overlay, stickers, news/meme banners, screenshot frames, and recompression.
- Current accepted SNSAug behavior:
  - `tiktok_like`, `instagram_story_like`, `youtube_shorts_like`, and `news_meme_overlay` are layout-only by default
  - quality degradation is separated from layout-only profiles
  - outline overlays use pixel-accurate alpha-based `ignore_mask`
  - overlays go to `ignore_mask`, not `tamper_mask`
  - same seed must be deterministic
- Remaining issue:
  - sticker and `text_block` overlays can overlap
  - variable overlays need deterministic collision avoidance before larger pair generation or fine-tuning

## Implementation Requirements
- Implement a narrow placement collision guard within the allowed files only.

- Part A: Add placement utility
  - Add `src/cv_forensics/snsaug_v2/placement.py`.
  - Implement a utility such as `PlacementManager` that can:
    - track placed overlay elements
    - track their alpha masks and coarse boxes
    - test candidate placements before committing
    - reject candidates when overlap exceeds threshold
    - retry placement up to `max_attempts`
    - if no valid placement is found:
      - skip optional overlays, or
      - place in a safe fallback region
      - record the decision in metadata
  - Suggested dataclasses:
    - `PlacedOverlay`
    - `PlacementDecision`
  - `PlacedOverlay` should include:
    - `element_id`
    - `element_type`
    - `bbox`
    - `alpha_mask`
    - `candidate_region`
    - `metadata`
  - `PlacementDecision` should include:
    - `accepted`
    - `bbox`
    - `attempt_count`
    - `rejected_reason`
    - `max_overlap_ratio`
    - `metadata`

- Part B: Alpha-mask based collision
  - Use alpha-mask overlap, not only bbox IoU.
  - For each candidate overlay:
    - render candidate to a temporary transparent RGBA layer
    - extract candidate alpha mask
    - compare against existing placed overlay alpha masks
    - compute:
      - `intersection_area`
      - `candidate_area`
      - `existing_area`
      - `candidate_overlap_ratio = intersection_area / candidate_area`
      - `existing_overlap_ratio = intersection_area / existing_area`
    - reject when thresholds are exceeded

- Part C: Default overlap thresholds
  - Add config values or constants for:
    - `text_vs_sticker_max_overlap = 0.0`
    - `text_vs_text_max_overlap = 0.02`
    - `sticker_vs_sticker_max_overlap = 0.05`
    - `badge_vs_text_max_overlap = 0.0`
    - `variable_vs_fixed_ui_max_overlap = 0.0`
    - `max_placement_attempts = 30`
    - `placement_margin_px = 8`
    - `max_ignore_mask_area_pct_medium = 0.30` or current project equivalent
  - Use dilation or expanded bbox or dilated alpha mask to enforce a small safety margin between text and stickers.

- Part D: Fixed UI vs variable overlays
  - Structural UI remains fixed:
    - progress bars
    - account rows
    - right action bars
    - bottom navigation
    - bottom message input
    - channel/title areas
  - Variable overlays must avoid fixed UI regions unless explicitly allowed:
    - `text_block`
    - AI badge
    - news banner
    - red rectangle
    - red circle
    - arrow
    - emoji
    - speech bubble
    - poll/question/location sticker

- Part E: Placement diversity must remain
  - Do not make positions fully fixed just to avoid overlap.
  - Preserve deterministic seed-based diversity:
    - same seed -> exact same result
    - different seed -> placement may change
    - variable overlays still choose among candidate regions with jitter and size variation

- Part F: Metadata requirements
  - For each overlay element, record:
    - `element_id`
    - `element_type`
    - `candidate_region`
    - `final_bbox`
    - `alpha_area_px`
    - `attempt_count`
    - `accepted`
    - `skipped_due_to_overlap`
    - `rejected_candidates`
    - `max_overlap_ratio`
    - `overlapped_with`
    - `placement_margin_px`
    - `placement_policy_version`
  - For the whole augmentation meta, record:
    - `placement_collision_guard_enabled = true`
    - `placement_retries_total`
    - `placement_skipped_count`
    - `final_ignore_mask_area_pct`

- Part G: Mask policy
  - Do not change current mask policy:
    - overlay pixels must never be added to `tamper_mask`
    - UI/text/sticker/badge pixels go to `ignore_mask`
    - geometric transforms apply to image and `tamper_mask` together
    - JPEG/recompression/blur/color shift apply to image only
    - masks use nearest-neighbor interpolation only

- Part H: Compatibility
  - Do not break:
    - `scripts/snsaug_v2/preview_snsaug_v2_overlays.py`
    - `scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py`
    - existing generation tests

- Part I: Tests
  - Add `tests/test_snsaug_v2_overlay_collision_guard.py`.
  - Cover:
    - `text_block` and sticker do not overlap for the same sample
    - `text_block` and AI badge do not overlap
    - sticker and red rectangle overlap remains below configured threshold
    - fixed UI regions are avoided by variable overlays
    - same seed produces identical placement metadata and identical image hash
    - different seed can produce different placement metadata
    - when all candidate regions are blocked, optional overlay is skipped and metadata records `skipped_due_to_overlap`
    - `ignore_mask` is still generated from alpha-based rendered pixels
    - `tamper_mask` is not contaminated by overlay pixels
    - preview script still runs
    - tiny fixed pair script still runs

- Part J: Documentation
  - Update `docs/snsaug_v2_generation.md`.
  - Add sections covering:
    - placement collision guard
    - overlap thresholds
    - deterministic retry behavior
    - metadata fields
    - when optional overlays are skipped
  - Include marker:
    - `SNSAUG_V2_OVERLAY_COLLISION_GUARD_OK`

- Part K: Real preview rerun
  - After implementation, rerun real dataset preview again on the same sample with at least two seeds:
    - `3407`
    - `9999`
  - Expected result:
    - no visible sticker / `text_block` overlap
    - same seed reproducible
    - different seed may change text/sticker locations
    - `ignore_mask` still marks overlay pixels
    - `tamper_mask` unchanged except geometric transform
  - If the same real sample path is not available from local task context, stop short of rerun and report that blocker explicitly rather than guessing a path.

## Validation Commands
```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_generation.py
```

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_layout_only_and_precise_ignore.py
```

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_overlay_collision_guard.py
```

```bash
python3 scripts/snsaug_v2/preview_snsaug_v2_overlays.py --help
```

```bash
python3 scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py --help
```

```bash
grep -q SNSAUG_V2_OVERLAY_COLLISION_GUARD_OK docs/snsaug_v2_generation.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0055e-snsaug-v2-overlay-placement-collision-guard.md
```

## Acceptance Criteria
- Variable overlays no longer overlap each other beyond configured thresholds.
- Fixed UI regions are protected from variable overlay overlap unless explicitly allowed.
- Deterministic seed-based diversity is preserved.
- Placement retries and skipped overlays are recorded in metadata.
- `ignore_mask` remains alpha-derived and `tamper_mask` remains uncontaminated by overlay pixels.
- Preview and tiny-pair scripts remain compatible.
- Tests, docs marker, and agent change checks all pass.

## Stop Condition
- Stop after implementing only the allowed files, running the listed validation commands, and reviewing the result as PASS or NEEDS_FIX.
- If implementation would require training, fine-tuning, network access, downloading assets, repo-local output writing, or modifying unrelated model code, stop and report the blocker.
