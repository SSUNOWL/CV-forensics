# Task: 0058c SNSAug V2 Small Benchmark Class-Balanced And Tampered Mask Fix

## Task Title

Fix SNSAug V2 small benchmark class balance, tampered-mask inclusion, and evaluator denominator sanity

## Role

Codex-only implementation worker, reviewer, and limited repair manager for an evaluation/generation-only benchmark-fix task.

## Files Codex May Read

- `docs/project_brief.md`
- `docs/project_contract.md`
- `configs/project_contract.json`
- `docs/agent_workflow.md`
- `tasks/0058-snsaug-v2-small-benchmark-with-geometry-and-postprocess.md`
- `docs/snsaug_v2_generation.md`
- `docs/snsaug_v2_small_benchmark_eval.md`
- `configs/snsaug_v2/snsaug_v2_tiny_fixed_pairs.example.json`
- `configs/evaluation/snsaug_v2_fixed_pairs_eval.example.json`
- `scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py`
- `scripts/evaluation/run_snsaug_v2_fixed_pairs_eval.py`
- `scripts/agent/validate_snsaug_v2_generation_config.py`
- `scripts/agent/validate_snsaug_v2_fixed_pairs_eval_config.py`
- `src/cv_forensics/snsaug_v2/configs.py`
- `src/cv_forensics/snsaug_v2/transforms.py`
- `src/cv_forensics/snsaug_v2/platform_templates.py`
- `src/cv_forensics/snsaug_v2/sns_augmentor.py`
- `src/cv_forensics/snsaug_v2/pair_generator.py`
- `src/cv_forensics/snsaug_v2/source_manifest_audit.py`
- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `tests/test_snsaug_v2_generation.py`
- `tests/test_snsaug_v2_fixed_pairs_eval.py`
- `tests/test_snsaug_v2_small_benchmark_eval.py`

## Files Codex May Modify

- `tasks/0058c-snsaug-v2-small-benchmark-class-balanced-and-tampered-mask-fix.md`
- `configs/snsaug_v2/snsaug_v2_tiny_fixed_pairs.example.json`
- `configs/evaluation/snsaug_v2_fixed_pairs_eval.example.json`
- `scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py`
- `scripts/snsaug_v2/check_snsaug_v2_pair_dataset.py`
- `scripts/evaluation/run_snsaug_v2_fixed_pairs_eval.py`
- `src/cv_forensics/snsaug_v2/pair_generator.py`
- `src/cv_forensics/snsaug_v2/source_manifest_audit.py`
- `src/cv_forensics/snsaug_v2_fixed_pairs_eval.py`
- `tests/test_snsaug_v2_generation.py`
- `tests/test_snsaug_v2_fixed_pairs_eval.py`
- `tests/test_snsaug_v2_small_benchmark_class_balance.py`
- `docs/snsaug_v2_small_benchmark_eval.md`
- `docs/snsaug_v2_generation.md`

## Forbidden Actions

- Do not train.
- Do not fine-tune.
- Do not run `0054`.
- Do not use network.
- Do not download assets or datasets.
- Do not install packages.
- Do not write generated outputs inside the repository.
- Do not use validation failures for training.
- Do not access `.env`, `.env.*`, `secrets/`, `data/`, `datasets/`, `outputs/`, or `checkpoints/`.
- Do not modify unrelated model or training code.

## Important Project Facts

- `0058` produced a structurally valid small benchmark with `780` rows but semantically invalid label balance for tampered evaluation.
- The user-provided audit showed:
  - `base count: 60`
  - `base labels: real 50, synthetic 10`
  - `tampered base count: 0`
  - `tampered with valid mask: 0`
- Because the generated benchmark had zero tampered bases, all tampered/localization metrics being `NA` is expected and not a model conclusion.
- This task must fix generation-time class balancing and mask sanity, and make evaluator denominator reporting explicit.
- This task remains evaluation/generation only.

## Implementation Requirements

1. Update `scripts/snsaug_v2/build_snsaug_v2_tiny_fixed_pairs.py` and/or `src/cv_forensics/snsaug_v2/pair_generator.py` so that:
   - `--max-samples-per-class` is explicit and required for small benchmark mode
   - selection is class-balanced across:
     - `real`
     - `synthetic`
     - `tampered`
2. For tampered samples:
   - require valid `tamper_mask_path` by default
   - only allow missing tamper masks when `--allow-tampered-without-mask` is explicitly set
3. If a class has fewer than requested samples:
   - emit a warning
   - record actual selected count
4. If tampered count is zero:
   - fail by default
   - only continue when `--allow-missing-tampered` is explicitly set
5. If `tampered_with_valid_mask_count` is zero:
   - fail by default
   - only continue when explicitly overridden
6. After pair generation, write:
   - `class_balance_summary.json`
   - `mask_availability_summary.json`
7. `class_balance_summary.json` must include:
   - `requested_per_class`
   - `selected_per_class`
   - `base_count`
   - `rows_per_profile`
   - `profile_count`
   - `expected_record_count`
   - `actual_record_count`
8. `mask_availability_summary.json` must include:
   - `tampered_base_count`
   - `tampered_with_valid_mask_count`
   - `tampered_missing_mask_count`
   - `examples_missing_mask`
9. Add standalone sanity checker:
   - `scripts/snsaug_v2/check_snsaug_v2_pair_dataset.py`
10. The sanity checker input is `pair_root` and must verify:
   - `meta.jsonl` exists
   - `pair_index.json` exists
   - clean profile exists
   - no duplicate clean rows
   - each `base_id` has all required profiles
   - label distribution includes `real`, `synthetic`, `tampered`
   - tampered samples have valid `tamper_mask_path`
   - `image_path` exists
   - `ignore_mask_path` exists
   - `label_preserved` is true
   - expected rows = `base_count × profile_count`
11. Update the fixed-pair evaluator to report denominators:
   - `class_count_real`
   - `class_count_synthetic`
   - `class_count_tampered`
   - `tampered_mask_eval_count`
   - `localization_activation_denominator`
12. If a denominator is zero:
   - corresponding metric must be `null` / `NA`
   - emit a warning
   - do not silently output `0.0`
13. Ensure summary/report formatting handles `None` safely and renders `NA`.
14. Add or update tests covering:
   - class-balanced selection
   - missing tampered class fails
   - tampered without mask fails by default
   - pair dataset sanity checker
   - evaluator denominator reporting
   - denominator-zero metrics are `NA`, not `0`
15. Update docs:
   - `docs/snsaug_v2_small_benchmark_eval.md` or `docs/snsaug_v2_generation.md`
16. Documentation must include marker:
   - `SNSAUG_V2_SMALL_BENCHMARK_CLASS_BALANCED_OK`

## Validation Commands

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_generation.py
```

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_fixed_pairs_eval.py
```

```bash
CUDA_VISIBLE_DEVICES="" python3 tests/test_snsaug_v2_small_benchmark_class_balance.py
```

```bash
python3 scripts/snsaug_v2/check_snsaug_v2_pair_dataset.py --help
```

```bash
grep -q SNSAUG_V2_SMALL_BENCHMARK_CLASS_BALANCED_OK docs/snsaug_v2_small_benchmark_eval.md
```

```bash
python3 scripts/agent/check_agent_changes.py tasks/0058c-snsaug-v2-small-benchmark-class-balanced-and-tampered-mask-fix.md
```

## Acceptance Criteria

- Small benchmark pair generation is class-balanced by base sample selection across `real`, `synthetic`, and `tampered`.
- Tampered samples without valid masks are rejected by default.
- Pair generation writes class-balance and mask-availability summaries.
- Pair dataset sanity checker catches duplicate clean rows, missing profiles, missing tampered masks, and missing required classes.
- Evaluator reports denominator counts and keeps denominator-zero metrics as `NA`/`null`, not `0.0`.
- Tests cover the class-balance and denominator edge cases.
- Documentation explains the benchmark sanity requirements.

## Stop Condition

Stop immediately if completing this task would require:
- training or fine-tuning,
- writing outputs inside the repository,
- using protected data or checkpoint paths beyond the explicitly allowed generation/evaluation scope,
- changing unrelated model code,
- or expanding beyond benchmark sanity-fix scope.
