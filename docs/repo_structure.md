# Repository Skeleton: Task 0004

This document describes the lightweight Python skeleton created in task 0004 and explains how each file maps to `docs/project_brief.md`.

---

## Directory Layout

```
src/
  cv_forensics/
    __init__.py        — package entry point; re-exports public API
    contracts.py       — canonical label constants and output field names
    outputs.py         — ForensicsResult dataclass (target output representation)
    evidence.py        — template-based explanation builder (build_reason)

scripts/
  agent/
    validate_repo_skeleton.py   — pure-Python checks: file existence + import smoke test

tests/
  test_repo_skeleton.py         — pytest-style tests for the skeleton

docs/
  repo_structure.md             — this file
```

---

## Mapping to `docs/project_brief.md`

| Skeleton module | Project brief section |
|---|---|
| `contracts.CLASS_LABELS` | §5 Target Outputs — `class` field values (`real / synthetic / tampered`) |
| `contracts.FAMILY_LABELS` | §4.1 Initial Family Label Candidates (`LatDiff / PixDiff / GAN / Other / Real-or-N/A`) |
| `contracts.OUTPUT_FIELDS` | §5 Target Outputs — all seven output fields |
| `outputs.ForensicsResult` | §5 Target Output JSON structure |
| `evidence.build_reason` | §6.6 Evidence Aggregation and Template Explanation |

### Target Output Alignment

`ForensicsResult.to_dict()` produces a dict with exactly these keys, matching the example in §5:

```json
{
  "class": "tampered",
  "class_conf": {"real": 0.03, "synthetic": 0.08, "tampered": 0.89},
  "family": "LatDiff",
  "family_conf": {"LatDiff": 0.78, "PixDiff": 0.14, "GAN": 0.05, "Other": 0.03},
  "localization_head": "activated",
  "mask_area_pct": 11.2,
  "reason": "..."
}
```

### Template Explanation Alignment

`build_reason` implements the deterministic placeholder described in §6.6:

- `tampered` + localization activated → notes area and family
- `tampered` + localization skipped → notes skipped localization and family
- `synthetic` → notes AI-generated and family
- `real` → notes no artifacts

---

## Design Constraints

- **Standard library only.** No external dependencies are imported.
- **No data, checkpoints, or secrets accessed.** The skeleton contains no I/O beyond in-memory computation.
- **Small and auditable.** Every file is under 100 lines.
- **Validates without installation.** `validate_repo_skeleton.py` inserts `src/` into `sys.path` at runtime; no `pip install` step is needed.

---

## How Future Tasks Should Extend This Skeleton

| Future task | Extension point |
|---|---|
| Stage 1 — binary backbone | Add `src/cv_forensics/backbone.py`; import `CLASS_LABELS` from `contracts` |
| Stage 2 — provenance head | Add `src/cv_forensics/provenance.py`; import `FAMILY_LABELS` from `contracts` |
| Stage 3 — localization | Extend `ForensicsResult` with mask tensor field or subclass |
| Stage 4 — perturbations + full pipeline | Extend `build_reason` templates; wire all heads into an inference function |

All extensions must remain within the allowed file list of the corresponding task file.
