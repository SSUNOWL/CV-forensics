#!/usr/bin/env python3
"""
Analyze whether SNSAug performance drop is caused by:
1) local SNS overlay occluding the original tampered region
2) geometric crop/resize/canvas shrinking or moving the tampered region
3) model class flip / localization activation flip
4) predicted red-mask disappearance

Inputs:
- pair_root/meta.jsonl from SNSAug fixed-pair generation
- eval_root/snsaug_v2_eval_records.jsonl
- eval_root/snsaug_v2_eval_comparisons.jsonl

Outputs:
- occlusion_records.jsonl
- occlusion_profile_summary.json
- occlusion_profile_summary.tsv
- top_cases.json
- galleries/*.jpg
- artifact_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PRED_MASK_KEYS = [
    "pred_mask_path",
    "predicted_mask_path",
    "final_mask_path",
    "final_mask_png_path",
    "final_binary_mask_path",
    "final_mask_binary_path",
    "mask_output_path",
    "output_mask_path",
    "red_mask_path",
    "redmask_path",
]

IMAGE_KEYS = ["image_path", "transformed_image_path", "source_image_path"]
TAMPER_MASK_KEYS = ["tamper_mask_path", "transformed_tamper_mask_path", "source_mask_path", "mask_path"]
IGNORE_MASK_KEYS = ["ignore_mask_path", "ignore_path"]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def safe_path(v: Any) -> Optional[Path]:
    if v in (None, "", "null"):
        return None
    p = Path(str(v)).expanduser()
    return p if p.exists() else None


def first_existing_path(row: Dict[str, Any], keys: List[str]) -> Optional[Path]:
    for k in keys:
        p = safe_path(row.get(k))
        if p is not None:
            return p
    return None


def get_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def get_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return default


def load_rgb(path: Optional[Path], fallback_size: Tuple[int, int] = (512, 512)) -> Image.Image:
    if path is None or not path.exists():
        return Image.new("RGB", fallback_size, (235, 235, 235))
    return Image.open(path).convert("RGB")


def load_mask_bool(path: Optional[Path], size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    im = Image.open(path).convert("L")
    if size is not None and im.size != size:
        im = im.resize(size, Image.Resampling.NEAREST)
    arr = np.asarray(im)
    return arr > 127


def zero_mask(size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.zeros((h, w), dtype=bool)


def mask_area(mask: Optional[np.ndarray]) -> int:
    if mask is None:
        return 0
    return int(mask.astype(bool).sum())


def safe_div(a: float, b: float) -> Optional[float]:
    if b == 0:
        return None
    return float(a) / float(b)


def mean(xs: List[float]) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if not xs:
        return None
    return sum(xs) / len(xs)


def median(xs: List[float]) -> Optional[float]:
    xs = sorted(float(x) for x in xs if x is not None and not math.isnan(float(x)))
    if not xs:
        return None
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) / 2


def fmt(v: Optional[float], nd: int = 3) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def overlay_color(
    img: Image.Image,
    mask: Optional[np.ndarray],
    color: Tuple[int, int, int],
    alpha: int = 120,
) -> Image.Image:
    base = img.convert("RGBA")
    if mask is None:
        return base.convert("RGB")
    m = mask.astype(bool)
    if m.shape[:2] != (base.height, base.width):
        mask_im = Image.fromarray((m.astype(np.uint8) * 255), mode="L")
        mask_im = mask_im.resize(base.size, Image.Resampling.NEAREST)
        m = np.asarray(mask_im) > 127

    overlay = Image.new("RGBA", base.size, (*color, 0))
    arr = np.asarray(overlay).copy()
    arr[m, 0] = color[0]
    arr[m, 1] = color[1]
    arr[m, 2] = color[2]
    arr[m, 3] = alpha
    overlay = Image.fromarray(arr, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def overlay_three_way(
    img: Image.Image,
    tamper: Optional[np.ndarray],
    ignore: Optional[np.ndarray],
) -> Image.Image:
    """
    Red = transformed tamper GT
    Blue = ignore / SNS nuisance mask
    Magenta = overlap(tamper, ignore)
    """
    base = img.convert("RGBA")
    size = base.size

    if tamper is None:
        tamper = zero_mask(size)
    if ignore is None:
        ignore = zero_mask(size)

    def resize_bool(m: np.ndarray) -> np.ndarray:
        if m.shape[:2] == (size[1], size[0]):
            return m.astype(bool)
        im = Image.fromarray((m.astype(np.uint8) * 255), mode="L").resize(size, Image.Resampling.NEAREST)
        return np.asarray(im) > 127

    t = resize_bool(tamper)
    i = resize_bool(ignore)
    o = t & i

    rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    rgba[t] = [255, 0, 0, 110]
    rgba[i] = [0, 90, 255, 100]
    rgba[o] = [255, 0, 255, 180]

    layer = Image.fromarray(rgba, mode="RGBA")
    return Image.alpha_composite(base, layer).convert("RGB")


def text_panel(text: str, size: Tuple[int, int] = (420, 360)) -> Image.Image:
    im = Image.new("RGB", size, (245, 245, 245))
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    lines: List[str] = []
    for part in text.split("\n"):
        cur = ""
        for word in part.split(" "):
            if len(cur) + len(word) + 1 > 48:
                lines.append(cur)
                cur = word
            else:
                cur = (cur + " " + word).strip()
        if cur:
            lines.append(cur)
    y = 20
    for line in lines[:18]:
        d.text((20, y), line, fill=(40, 40, 40), font=font)
        y += 18
    return im


def fit_to_cell(img: Image.Image, cell_size: Tuple[int, int] = (360, 360)) -> Image.Image:
    w, h = img.size
    cw, ch = cell_size
    scale = min(cw / max(w, 1), ch / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", cell_size, (250, 250, 250))
    canvas.paste(resized, ((cw - nw) // 2, (ch - nh) // 2))
    return canvas


def make_cell(title: str, img: Image.Image, cell_size: Tuple[int, int] = (360, 410)) -> Image.Image:
    cw, ch = cell_size
    title_h = 50
    canvas = Image.new("RGB", cell_size, (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    d.rectangle([0, 0, cw, title_h], fill=(235, 235, 235))
    d.text((10, 17), title[:58], fill=(20, 20, 20), font=font)
    body = fit_to_cell(img, (cw, ch - title_h))
    canvas.paste(body, (0, title_h))
    return canvas


def make_grid(cells: List[Tuple[str, Image.Image]], out_path: Path, cols: int = 4) -> None:
    cell_w, cell_h = 360, 410
    rows = int(math.ceil(len(cells) / cols))
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    for idx, (title, img) in enumerate(cells):
        cell = make_cell(title, img, (cell_w, cell_h))
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        grid.paste(cell, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, quality=95)


def build_indices(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        bid = str(r.get("base_id"))
        prof = str(r.get("profile"))
        if bid and prof:
            out[(bid, prof)] = r
    return out


def profile_summary(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_profile: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_profile[str(r["profile"])].append(r)

    out: Dict[str, Dict[str, Any]] = {}
    for prof, rs in sorted(by_profile.items()):
        vals = lambda k: [x[k] for x in rs if x.get(k) is not None]
        out[prof] = {
            "count": len(rs),
            "mean_occlusion_ratio": mean(vals("occlusion_ratio")),
            "median_occlusion_ratio": median(vals("occlusion_ratio")),
            "max_occlusion_ratio": max(vals("occlusion_ratio")) if vals("occlusion_ratio") else None,
            "mean_visible_tamper_ratio": mean(vals("visible_tamper_ratio")),
            "mean_tamper_area_pct_ratio": mean(vals("tamper_area_pct_ratio")),
            "mean_ignore_area_pct": mean(vals("ignore_area_pct")),
            "mean_sns_valid_iou": mean(vals("sns_valid_iou")),
            "mean_clean_valid_iou": mean(vals("clean_valid_iou")),
            "mean_valid_iou_drop": mean(vals("valid_iou_drop")),
            "mean_p_tampered_drop": mean(vals("p_tampered_drop")),
            "class_flip_rate": mean([1.0 if x.get("fragile_class_flip") else 0.0 for x in rs]),
            "activation_flip_rate": mean([1.0 if x.get("fragile_activation_flip") else 0.0 for x in rs]),
            "high_occlusion_case_count": sum(1 for x in rs if (x.get("occlusion_ratio") or 0) >= 0.3),
            "low_visible_tamper_case_count": sum(1 for x in rs if (x.get("visible_tamper_ratio") is not None and x["visible_tamper_ratio"] <= 0.5)),
            "low_area_ratio_case_count": sum(1 for x in rs if (x.get("tamper_area_pct_ratio") is not None and x["tamper_area_pct_ratio"] <= 0.5)),
        }
    return out


def save_profile_summary_tsv(path: Path, summary: Dict[str, Dict[str, Any]]) -> None:
    keys = [
        "profile",
        "count",
        "mean_occlusion_ratio",
        "median_occlusion_ratio",
        "max_occlusion_ratio",
        "mean_visible_tamper_ratio",
        "mean_tamper_area_pct_ratio",
        "mean_ignore_area_pct",
        "mean_sns_valid_iou",
        "mean_clean_valid_iou",
        "mean_valid_iou_drop",
        "mean_p_tampered_drop",
        "class_flip_rate",
        "activation_flip_rate",
        "high_occlusion_case_count",
        "low_visible_tamper_case_count",
        "low_area_ratio_case_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        w.writeheader()
        for prof, s in sorted(summary.items()):
            row = {"profile": prof, **s}
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-root", required=True, type=Path)
    ap.add_argument("--eval-root", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--profiles", nargs="*", default=None)
    ap.add_argument("--case-base-ids", nargs="*", default=None)
    args = ap.parse_args()

    pair_root = args.pair_root.expanduser().resolve()
    eval_root = args.eval_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    meta_path = pair_root / "meta.jsonl"
    records_path = eval_root / "snsaug_v2_eval_records.jsonl"
    comparisons_path = eval_root / "snsaug_v2_eval_comparisons.jsonl"

    if not meta_path.exists():
        raise SystemExit(f"missing meta.jsonl: {meta_path}")
    if not records_path.exists():
        raise SystemExit(f"missing eval records: {records_path}")
    if not comparisons_path.exists():
        raise SystemExit(f"missing eval comparisons: {comparisons_path}")

    meta_rows = read_jsonl(meta_path)
    record_rows = read_jsonl(records_path)
    comparison_rows = read_jsonl(comparisons_path)

    meta_by = build_indices(meta_rows)
    record_by = build_indices(record_rows)

    selected_profiles = set(args.profiles) if args.profiles else None
    selected_base_ids = set(args.case_base_ids) if args.case_base_ids else None

    out_rows: List[Dict[str, Any]] = []

    for c in comparison_rows:
        if c.get("content_label") != "tampered":
            continue

        bid = str(c.get("base_id"))
        profile = str(c.get("profile"))

        if selected_profiles and profile not in selected_profiles:
            continue
        if selected_base_ids and bid not in selected_base_ids:
            continue

        clean_meta = meta_by.get((bid, "clean"))
        sns_meta = meta_by.get((bid, profile))

        if clean_meta is None or sns_meta is None:
            continue

        clean_img_path = first_existing_path(clean_meta, IMAGE_KEYS)
        sns_img_path = first_existing_path(sns_meta, IMAGE_KEYS)

        clean_img = load_rgb(clean_img_path)
        sns_img = load_rgb(sns_img_path)

        clean_mask_path = first_existing_path(clean_meta, TAMPER_MASK_KEYS)
        sns_mask_path = first_existing_path(sns_meta, TAMPER_MASK_KEYS)
        ignore_path = first_existing_path(sns_meta, IGNORE_MASK_KEYS)

        clean_mask = load_mask_bool(clean_mask_path, clean_img.size)
        sns_mask = load_mask_bool(sns_mask_path, sns_img.size)
        ignore_mask = load_mask_bool(ignore_path, sns_img.size)
        if ignore_mask is None:
            ignore_mask = zero_mask(sns_img.size)

        clean_area = mask_area(clean_mask)
        sns_area = mask_area(sns_mask)
        ignore_area = mask_area(ignore_mask)

        clean_image_area = clean_img.size[0] * clean_img.size[1]
        sns_image_area = sns_img.size[0] * sns_img.size[1]

        clean_area_pct = safe_div(clean_area, clean_image_area)
        sns_area_pct = safe_div(sns_area, sns_image_area)
        ignore_area_pct = safe_div(ignore_area, sns_image_area)

        if sns_mask is None:
            overlap_area = 0
            visible_tamper_area = 0
            occlusion_ratio = None
            visible_tamper_ratio = None
        else:
            overlap = sns_mask & ignore_mask
            overlap_area = int(overlap.sum())
            visible_tamper_area = int((sns_mask & (~ignore_mask)).sum())
            occlusion_ratio = safe_div(overlap_area, sns_area)
            visible_tamper_ratio = safe_div(visible_tamper_area, sns_area)

        tamper_area_pct_ratio = None
        if clean_area_pct not in (None, 0) and sns_area_pct is not None:
            tamper_area_pct_ratio = safe_div(sns_area_pct, clean_area_pct)

        rec = {
            "base_id": bid,
            "profile": profile,
            "content_label": c.get("content_label"),
            "clean_image_path": str(clean_img_path) if clean_img_path else None,
            "sns_image_path": str(sns_img_path) if sns_img_path else None,
            "clean_tamper_mask_path": str(clean_mask_path) if clean_mask_path else None,
            "sns_tamper_mask_path": str(sns_mask_path) if sns_mask_path else None,
            "ignore_mask_path": str(ignore_path) if ignore_path else None,
            "clean_image_size": list(clean_img.size),
            "sns_image_size": list(sns_img.size),
            "clean_tamper_area_px": clean_area,
            "sns_tamper_area_px": sns_area,
            "ignore_area_px": ignore_area,
            "overlap_area_px": overlap_area,
            "visible_tamper_area_px": visible_tamper_area,
            "clean_tamper_area_pct": clean_area_pct,
            "sns_tamper_area_pct": sns_area_pct,
            "ignore_area_pct": ignore_area_pct,
            "occlusion_ratio": occlusion_ratio,
            "visible_tamper_ratio": visible_tamper_ratio,
            "tamper_area_pct_ratio": tamper_area_pct_ratio,
            "clean_pred_class": c.get("clean_pred_class"),
            "sns_pred_class": c.get("sns_pred_class"),
            "clean_p_tampered": c.get("clean_p_tampered"),
            "sns_p_tampered": c.get("sns_p_tampered"),
            "p_tampered_drop": c.get("p_tampered_drop"),
            "clean_valid_iou": c.get("clean_valid_iou"),
            "sns_valid_iou": c.get("sns_valid_iou"),
            "valid_iou_drop": c.get("valid_iou_drop"),
            "clean_localization_activated": c.get("clean_localization_activated"),
            "sns_localization_activated": c.get("sns_localization_activated"),
            "fragile_class_flip": get_bool(c.get("fragile_class_flip")),
            "fragile_activation_flip": get_bool(c.get("fragile_activation_flip")),
            "fragile_confidence_drop": get_bool(c.get("fragile_confidence_drop")),
            "fragile_mask_drop": get_bool(c.get("fragile_mask_drop")),
        }

        if occlusion_ratio is not None and occlusion_ratio >= 0.3:
            rec["likely_failure_factor"] = "local_overlay_occlusion"
        elif tamper_area_pct_ratio is not None and tamper_area_pct_ratio <= 0.5:
            rec["likely_failure_factor"] = "geometry_crop_or_scale_reduced_tamper_region"
        elif get_bool(c.get("fragile_activation_flip")):
            rec["likely_failure_factor"] = "classification_or_activation_flip_without_large_local_occlusion"
        else:
            rec["likely_failure_factor"] = "mixed_or_unclear"

        out_rows.append(rec)

    # Save detailed records
    write_jsonl(output_root / "occlusion_records.jsonl", out_rows)

    # Profile summary
    prof_sum = profile_summary(out_rows)
    (output_root / "occlusion_profile_summary.json").write_text(
        json.dumps(prof_sum, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_profile_summary_tsv(output_root / "occlusion_profile_summary.tsv", prof_sum)

    # Select cases for gallery:
    # prioritize high IoU drop, activation flip, then high occlusion.
    def sort_key(r: Dict[str, Any]) -> Tuple[float, float, float, float]:
        return (
            1.0 if r.get("fragile_activation_flip") else 0.0,
            get_float(r.get("valid_iou_drop"), 0.0) or 0.0,
            get_float(r.get("p_tampered_drop"), 0.0) or 0.0,
            get_float(r.get("occlusion_ratio"), 0.0) or 0.0,
        )

    top_rows = sorted(out_rows, key=sort_key, reverse=True)[: args.top_n]

    gallery_dir = output_root / "galleries"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    gallery_manifest: List[Dict[str, Any]] = []

    for idx, r in enumerate(top_rows):
        bid = r["base_id"]
        profile = r["profile"]

        clean_meta = meta_by.get((bid, "clean"))
        sns_meta = meta_by.get((bid, profile))
        clean_record = record_by.get((bid, "clean"), {})
        sns_record = record_by.get((bid, profile), {})

        if clean_meta is None or sns_meta is None:
            continue

        clean_img_path = safe_path(r.get("clean_image_path"))
        sns_img_path = safe_path(r.get("sns_image_path"))
        clean_img = load_rgb(clean_img_path)
        sns_img = load_rgb(sns_img_path)

        clean_gt = load_mask_bool(safe_path(r.get("clean_tamper_mask_path")), clean_img.size)
        sns_gt = load_mask_bool(safe_path(r.get("sns_tamper_mask_path")), sns_img.size)
        ignore = load_mask_bool(safe_path(r.get("ignore_mask_path")), sns_img.size)
        if ignore is None:
            ignore = zero_mask(sns_img.size)

        clean_pred_path = first_existing_path(clean_record, PRED_MASK_KEYS)
        sns_pred_path = first_existing_path(sns_record, PRED_MASK_KEYS)
        clean_pred = load_mask_bool(clean_pred_path, clean_img.size)
        sns_pred = load_mask_bool(sns_pred_path, sns_img.size)

        clean_gt_overlay = overlay_color(clean_img, clean_gt, (255, 0, 0), 130)
        sns_gt_overlay = overlay_color(sns_img, sns_gt, (255, 0, 0), 130)
        sns_ignore_overlay = overlay_color(sns_img, ignore, (0, 90, 255), 115)
        sns_overlap_overlay = overlay_three_way(sns_img, sns_gt, ignore)

        if clean_pred is None:
            clean_pred_overlay = text_panel(
                "Clean predicted mask not found in eval records.\n"
                "Evaluator may need to export final mask PNG."
            )
        else:
            clean_pred_overlay = overlay_color(clean_img, clean_pred, (255, 0, 0), 130)

        if sns_pred is None:
            sns_pred_overlay = text_panel(
                "SNS predicted mask not found in eval records.\n"
                "If localization activation is off, predicted mask may be empty."
            )
        else:
            sns_pred_overlay = overlay_color(sns_img, sns_pred, (255, 0, 0), 130)

        stats_text = (
            f"base_id: {bid}\n"
            f"profile: {profile}\n"
            f"clean_pred: {r.get('clean_pred_class')} / sns_pred: {r.get('sns_pred_class')}\n"
            f"clean_p_tampered: {fmt(get_float(r.get('clean_p_tampered')))}\n"
            f"sns_p_tampered: {fmt(get_float(r.get('sns_p_tampered')))}\n"
            f"p_tampered_drop: {fmt(get_float(r.get('p_tampered_drop')))}\n"
            f"clean IoU: {fmt(get_float(r.get('clean_valid_iou')))}\n"
            f"SNS IoU: {fmt(get_float(r.get('sns_valid_iou')))}\n"
            f"IoU drop: {fmt(get_float(r.get('valid_iou_drop')))}\n"
            f"occlusion_ratio: {fmt(get_float(r.get('occlusion_ratio')))}\n"
            f"visible_tamper_ratio: {fmt(get_float(r.get('visible_tamper_ratio')))}\n"
            f"tamper_area_pct_ratio: {fmt(get_float(r.get('tamper_area_pct_ratio')))}\n"
            f"factor: {r.get('likely_failure_factor')}\n"
            f"clean_loc_on: {r.get('clean_localization_activated')}\n"
            f"sns_loc_on: {r.get('sns_localization_activated')}"
        )
        stats_panel = text_panel(stats_text, (420, 360))

        cells = [
            ("Clean image", clean_img),
            ("Clean GT tamper red", clean_gt_overlay),
            ("Clean pred redmask", clean_pred_overlay),
            ("Stats", stats_panel),
            ("SNS image", sns_img),
            ("SNS GT tamper red", sns_gt_overlay),
            ("SNS ignore mask blue", sns_ignore_overlay),
            ("Tamper red / Ignore blue / Overlap magenta", sns_overlap_overlay),
            ("SNS pred redmask", sns_pred_overlay),
        ]

        out_img = gallery_dir / f"{idx:03d}_{bid}__{profile}.jpg"
        make_grid(cells, out_img, cols=4)

        gallery_manifest.append({
            **r,
            "gallery_path": str(out_img),
            "clean_pred_mask_path": str(clean_pred_path) if clean_pred_path else None,
            "sns_pred_mask_path": str(sns_pred_path) if sns_pred_path else None,
        })

    (output_root / "top_cases.json").write_text(
        json.dumps(gallery_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifact = {
        "marker": "SNSAUG_TAMPER_OCCLUSION_REDMASK_ANALYSIS_OK",
        "pair_root": str(pair_root),
        "eval_root": str(eval_root),
        "output_root": str(output_root),
        "occlusion_records_jsonl": str(output_root / "occlusion_records.jsonl"),
        "occlusion_profile_summary_json": str(output_root / "occlusion_profile_summary.json"),
        "occlusion_profile_summary_tsv": str(output_root / "occlusion_profile_summary.tsv"),
        "top_cases_json": str(output_root / "top_cases.json"),
        "gallery_dir": str(gallery_dir),
        "case_count": len(out_rows),
        "gallery_count": len(gallery_manifest),
        "pred_mask_available_count": sum(
            1
            for g in gallery_manifest
            if g.get("clean_pred_mask_path") or g.get("sns_pred_mask_path")
        ),
    }
    (output_root / "artifact_manifest.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(artifact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
