# Project Contract: Social-Media-Robust Lightweight Multi-Head Image Forensics

> This contract summarizes the agreed project goal based on `docs/project_brief.md`.
> It is the binding reference for all implementation tasks.

---

## 1. Project Goal

Build a **lightweight multi-head image forensics research prototype** that detects AI-generated and partially manipulated images circulating on social media.

The system must produce four outputs for every input image:

| Output | Description |
|--------|-------------|
| **Class** | `real / synthetic / tampered` — 3-way classification |
| **Mask** | Tampered region localization mask (activated conditionally) |
| **Family** | Coarse generator-family provenance (`LatDiff / PixDiff / GAN / Other / Real-or-N/A`) |
| **Reason** | Template-based evidence explanation |

Hardware constraint: all experiments must fit on a single RTX 4090.

---

## 2. Target Output Format

```json
{
  "class": "tampered",
  "class_conf": {"real": 0.03, "synthetic": 0.08, "tampered": 0.89},
  "family": "LatDiff",
  "family_conf": {"LatDiff": 0.78, "PixDiff": 0.14, "GAN": 0.05, "Other": 0.03},
  "localization_head": "activated",
  "mask_area_pct": 11.2,
  "reason": "Local boundary discontinuity and texture mismatch detected. Generator family estimated as LatDiff."
}
```

---

## 3. Datasets

### 3.1 Community Forensics-Small (CF-Small)

**Role:** Shared visual backbone pre-training and generator-family provenance head training.

- Binary real/fake detector reproduction
- Generator-diversity-driven backbone training
- Unseen generator generalization evaluation (generator-holdout validation)
- Coarse provenance label construction from `architecture` metadata

Initial family labels: `LatDiff / PixDiff / GAN / Other / Real-or-N/A`

### 3.2 SID-Set

**Role:** 3-way classification and tampered mask localization fine-tuning.

- `real / synthetic / tampered` 3-way classification
- Tampered region localization with mask supervision
- Initial scope: train 210K + validation 30K (240K rows)

**Note:** SID-Set lacks generator-family labels. Handle via provenance head freeze, family loss mask-out, or mixed-batch strategy.

---

## 4. Architecture

```
Input Image
  -> Shared Visual Backbone (ConvNeXt-S or CLIP-ViT-S)
      -> 3-way Classification Head  (real / synthetic / tampered)
      -> Generator-Family Provenance Head  (LatDiff / PixDiff / GAN / Other / Real-or-N/A)
      -> Conditional Localization Head  (if tampered_score >= tau)
      -> Evidence Aggregation Module
          -> Template-Based Explanation
```

The localization head is activated only when `tampered_score >= tau`. The threshold `tau` is chosen to prioritize tampered recall, minimizing false negatives.

---

## 5. Implementation Stages

| Stage | Name | Goal |
|-------|------|------|
| 1 | CF-Small Binary Backbone | Reproduce binary real/fake detector; train shared backbone with generator-holdout validation |
| 2 | Generator-Family Provenance Head | Add coarse provenance head using CF-Small `architecture` metadata |
| 3 | SID-Set Multi-Head Fine-Tuning | 3-way classification + tampered mask localization; handle missing family labels |
| 4 | Social-Media Robustness and Evidence Template | Add screenshot/text-overlay/sticker/recompression; measure robustness drop; output template explanation |

---

## 6. Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| 3-way Classification Accuracy | Overall correct rate for real/synthetic/tampered |
| Macro-F1 | Class-imbalance-aware 3-way average performance |
| Tampered Mask IoU | Overlap between predicted and ground-truth masks |
| Generator-Family Accuracy | Coarse provenance label accuracy |
| Perturbation Robustness Drop | Delta F1 / Delta IoU before and after social-media perturbations |
| Latency (ms/image) | Inference time on RTX 4090 |
| FPS | Batch-size-1 inference FPS on RTX 4090 |
| Localization Activation Recall | Fraction of true tampered images where localization head activated |

---

## 7. First Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| **Data Leakage** — same generator in train and validation | Use generator-holdout or model_name-holdout validation; avoid random split |
| **Missing Family Labels in SID-Set** | Freeze provenance head, mask-out family loss, or use mixed-batch during SID-Set fine-tuning |
| **Conditional Localization False Negatives** — tau too high | Set tau based on tampered recall; track IoU / activation recall / latency trade-off |
| **Perturbation Robustness** — social-media transforms degrade performance | Config-controlled perturbation pipeline; report Delta F1 and Delta IoU |
| **Reproducibility** | Config-ize seed, dataset paths, output paths, and all perturbation switches |

---

## 8. Guardrails

The following are forbidden in all agent tasks unless explicitly approved:

- Dataset download
- Model training
- Package installation without approval
- Access to `.env`, `secrets/`, `data/`, `datasets/`, `checkpoints/`, `outputs/`
- `git push`
- Large file creation
- Unrelated file modification
- External network access from shell commands

All agent tasks must be small and reversible.

---

## 9. Validation

```bash
# Validate this contract
python3 scripts/agent/validate_project_contract.py configs/project_contract.json

# If pytest is available
pytest -q tests/test_project_contract.py
```
