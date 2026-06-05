#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def safe_float(v):
    if v is None:
        return None
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def safe_mean(xs):
    xs = [safe_float(x) for x in xs]
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def safe_median(xs):
    xs = [safe_float(x) for x in xs]
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return median(xs)


def ratio(num, den):
    return None if den == 0 else num / den


def fmt(v, nd=3):
    if v is None:
        return "NA"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def load_eval_root(args):
    if args.eval_root:
        return Path(args.eval_root).expanduser().resolve()

    if args.eval_config:
        cfg = json.loads(Path(args.eval_config).expanduser().read_text(encoding="utf-8"))
        return Path(cfg["output_root"]).expanduser().resolve()

    raise SystemExit("Either --eval-root or --eval-config is required.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=str, default=None)
    ap.add_argument("--eval-config", type=str, default=None)
    ap.add_argument("--output-root", type=str, required=True)
    ap.add_argument("--top-n", type=int, default=60)
    args = ap.parse_args()

    eval_root = load_eval_root(args)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records_path = eval_root / "snsaug_v2_eval_records.jsonl"
    comparisons_path = eval_root / "snsaug_v2_eval_comparisons.jsonl"
    metrics_path = eval_root / "snsaug_v2_per_profile_metrics.json"
    drops_path = eval_root / "snsaug_v2_robustness_drop_metrics.json"
    summary_path = eval_root / "snsaug_v2_eval_summary.json"

    records = read_jsonl(records_path)
    comparisons = read_jsonl(comparisons_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    drops = json.loads(drops_path.read_text(encoding="utf-8")) if drops_path.exists() else {}
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    record_by = {
        (str(r.get("base_id")), str(r.get("profile"))): r
        for r in records
    }

    tampered = [r for r in comparisons if r.get("content_label") == "tampered"]

    by_profile = defaultdict(list)
    for r in tampered:
        by_profile[str(r.get("profile"))].append(r)

    profile_rows = []

    for profile, rows in sorted(by_profile.items()):
        n = len(rows)

        clean_correct = sum(1 for r in rows if r.get("clean_correct") is True)
        sns_correct = sum(1 for r in rows if r.get("sns_correct") is True)

        clean_loc_on = sum(1 for r in rows if r.get("clean_localization_activated") is True)
        sns_loc_on = sum(1 for r in rows if r.get("sns_localization_activated") is True)
        activation_flip_off = sum(1 for r in rows if r.get("activation_flip_off") is True or r.get("fragile_activation_flip") is True)

        class_flip = sum(1 for r in rows if r.get("pred_flip") is True)
        correct_to_wrong = sum(1 for r in rows if r.get("correct_to_wrong") is True)
        fragile_conf = sum(1 for r in rows if r.get("fragile_confidence_drop") is True)
        fragile_mask = sum(1 for r in rows if r.get("fragile_mask_drop") is True)

        clean_p = [safe_float(r.get("clean_p_tampered")) for r in rows]
        sns_p = [safe_float(r.get("sns_p_tampered")) for r in rows]
        p_drop = [safe_float(r.get("p_tampered_drop")) for r in rows]
        clean_iou = [safe_float(r.get("clean_valid_iou")) for r in rows]
        sns_iou = [safe_float(r.get("sns_valid_iou")) for r in rows]
        iou_drop = [safe_float(r.get("valid_iou_drop")) for r in rows]

        sns_p_values = [x for x in sns_p if x is not None]

        score_lt_001 = sum(1 for x in sns_p_values if x < 0.01)
        score_lt_005 = sum(1 for x in sns_p_values if x < 0.05)
        score_lt_010 = sum(1 for x in sns_p_values if x < 0.10)
        score_lt_025 = sum(1 for x in sns_p_values if x < 0.25)
        score_lt_050 = sum(1 for x in sns_p_values if x < 0.50)

        pred_mask_present = 0
        pred_red_present = 0
        empty_pred_mask_like = 0

        for r in rows:
            sns_rec = record_by.get((str(r.get("base_id")), profile), {})
            pm = sns_rec.get("pred_mask_path")
            pr = sns_rec.get("pred_red_overlay_path")
            if pm and Path(pm).exists():
                pred_mask_present += 1
            if pr and Path(pr).exists():
                pred_red_present += 1
            if sns_rec.get("localization_activated") is False:
                empty_pred_mask_like += 1

        row = {
            "profile": profile,
            "tampered_comparison_count": n,
            "clean_correct_rate": ratio(clean_correct, n),
            "sns_correct_rate": ratio(sns_correct, n),
            "class_flip_rate": ratio(class_flip, n),
            "correct_to_wrong_rate": ratio(correct_to_wrong, n),
            "clean_localization_activation_rate": ratio(clean_loc_on, n),
            "sns_localization_activation_rate": ratio(sns_loc_on, n),
            "activation_flip_off_rate": ratio(activation_flip_off, n),
            "fragile_confidence_drop_rate": ratio(fragile_conf, n),
            "fragile_mask_drop_rate": ratio(fragile_mask, n),
            "mean_clean_p_tampered": safe_mean(clean_p),
            "mean_sns_p_tampered": safe_mean(sns_p),
            "median_sns_p_tampered": safe_median(sns_p),
            "mean_p_tampered_drop": safe_mean(p_drop),
            "median_p_tampered_drop": safe_median(p_drop),
            "sns_score_lt_0_01_rate": ratio(score_lt_001, n),
            "sns_score_lt_0_05_rate": ratio(score_lt_005, n),
            "sns_score_lt_0_10_rate": ratio(score_lt_010, n),
            "sns_score_lt_0_25_rate": ratio(score_lt_025, n),
            "sns_score_lt_0_50_rate": ratio(score_lt_050, n),
            "mean_clean_valid_iou": safe_mean(clean_iou),
            "mean_sns_valid_iou": safe_mean(sns_iou),
            "mean_valid_iou_drop": safe_mean(iou_drop),
            "pred_mask_present_rate": ratio(pred_mask_present, n),
            "pred_red_overlay_present_rate": ratio(pred_red_present, n),
            "localization_off_empty_mask_rate": ratio(empty_pred_mask_like, n),
        }

        if (
            (row["activation_flip_off_rate"] or 0) >= 0.7
            and (row["mean_p_tampered_drop"] or 0) >= 0.5
        ):
            row["diagnosis"] = "classification_activation_bottleneck"
        elif (row["mean_sns_valid_iou"] or 0) < 0.1 and (row["sns_localization_activation_rate"] or 0) > 0.5:
            row["diagnosis"] = "mask_decoder_or_geometry_bottleneck"
        else:
            row["diagnosis"] = "mixed_or_mild"

        profile_rows.append(row)

    # top cases
    def sort_key(r):
        return (
            1 if r.get("fragile_activation_flip") or r.get("activation_flip_off") else 0,
            safe_float(r.get("p_tampered_drop")) or 0,
            safe_float(r.get("valid_iou_drop")) or 0,
        )

    top_cases = sorted(tampered, key=sort_key, reverse=True)[: args.top_n]

    # write JSON
    result = {
        "marker": "SNSAUG_ACTIVATION_BOTTLENECK_SUMMARY_OK",
        "eval_root": str(eval_root),
        "summary": summary,
        "record_count": len(records),
        "comparison_count": len(comparisons),
        "tampered_comparison_count": len(tampered),
        "record_label_counts": dict(Counter(r.get("content_label") for r in records)),
        "comparison_label_counts": dict(Counter(r.get("content_label") for r in comparisons)),
        "profile_summary": profile_rows,
        "top_cases": top_cases,
    }

    (output_root / "activation_bottleneck_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # write TSV
    headers = [
        "profile",
        "tampered_comparison_count",
        "sns_correct_rate",
        "sns_localization_activation_rate",
        "activation_flip_off_rate",
        "mean_clean_p_tampered",
        "mean_sns_p_tampered",
        "mean_p_tampered_drop",
        "sns_score_lt_0_10_rate",
        "mean_sns_valid_iou",
        "mean_valid_iou_drop",
        "pred_red_overlay_present_rate",
        "diagnosis",
    ]

    tsv_path = output_root / "activation_bottleneck_summary.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for row in profile_rows:
            f.write("\t".join(
                fmt(row.get(h)) if isinstance(row.get(h), (float, int)) or row.get(h) is None else str(row.get(h))
                for h in headers
            ) + "\n")

    # write Markdown
    md_path = output_root / "activation_bottleneck_report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# SNSAug Activation Bottleneck Report\n\n")
        f.write(f"- Eval root: `{eval_root}`\n")
        f.write(f"- Records: {len(records)}\n")
        f.write(f"- Comparisons: {len(comparisons)}\n")
        f.write(f"- Tampered comparisons: {len(tampered)}\n\n")

        f.write("## Profile Summary\n\n")
        f.write("| Profile | SNS Tampered Recall | SNS Loc. Act. | Activation Flip Off | Mean SNS p_tampered | p_tampered Drop | SNS IoU | Diagnosis |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in profile_rows:
            f.write(
                f"| {row['profile']} "
                f"| {fmt(row.get('sns_correct_rate'))} "
                f"| {fmt(row.get('sns_localization_activation_rate'))} "
                f"| {fmt(row.get('activation_flip_off_rate'))} "
                f"| {fmt(row.get('mean_sns_p_tampered'))} "
                f"| {fmt(row.get('mean_p_tampered_drop'))} "
                f"| {fmt(row.get('mean_sns_valid_iou'))} "
                f"| {row.get('diagnosis')} |\n"
            )

        f.write("\n## Interpretation\n\n")
        f.write(
            "If activation_flip_off_rate and mean_p_tampered_drop are high, "
            "the main failure is not missing redmask rendering but class-score collapse before localization. "
            "Training should prioritize tampered-score robustness and clean-SNS class consistency.\n"
        )

    artifact = {
        "marker": "SNSAUG_ACTIVATION_BOTTLENECK_SUMMARY_OK",
        "eval_root": str(eval_root),
        "output_root": str(output_root),
        "summary_json": str(output_root / "activation_bottleneck_summary.json"),
        "summary_tsv": str(tsv_path),
        "report_md": str(md_path),
    }
    (output_root / "artifact_manifest.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(artifact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
