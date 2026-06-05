#!/usr/bin/env python3
"""Sanity-check an SNSAug V2 fixed-pair dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CLASS_LABELS = ("real", "synthetic", "tampered")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"meta.jsonl line {line_number} is not a JSON object")
            rows.append(value)
    return rows


def _valid_path(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and Path(value).is_file()


def check_pair_dataset(pair_root: str | Path) -> tuple[dict[str, Any], list[str], list[str]]:
    root = Path(pair_root).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    meta_path = root / "meta.jsonl"
    pair_index_path = root / "pair_index.json"
    if not meta_path.is_file():
        return {"pair_root": str(root)}, [f"meta.jsonl missing: {meta_path}"], warnings
    if not pair_index_path.is_file():
        return {"pair_root": str(root)}, [f"pair_index.json missing: {pair_index_path}"], warnings

    rows = _load_jsonl(meta_path)
    pair_index = json.loads(pair_index_path.read_text(encoding="utf-8"))
    if not isinstance(pair_index, dict) or not isinstance(pair_index.get("pairs"), list):
        errors.append("pair_index.json must contain a pairs list")

    profiles = sorted({str(row.get("profile") or "") for row in rows if row.get("profile")})
    required_profiles = set(profiles)
    if "clean" not in required_profiles:
        errors.append("clean profile missing")
    profile_count = len(required_profiles)

    labels = {str(row.get("content_label") or "") for row in rows}
    for label in CLASS_LABELS:
        if label not in labels:
            errors.append(f"required label missing: {label}")

    by_base: dict[str, list[dict[str, Any]]] = {}
    clean_seen: dict[str, int] = {}
    for row_index, row in enumerate(rows):
        base_id = str(row.get("base_id") or "")
        if not base_id:
            errors.append(f"row {row_index}: base_id missing")
            continue
        by_base.setdefault(base_id, []).append(row)
        profile = str(row.get("profile") or "")
        view = str(row.get("view") or "")
        label = str(row.get("content_label") or "")
        if view == "clean" or profile == "clean":
            clean_seen[base_id] = clean_seen.get(base_id, 0) + 1
            if view != "clean" or profile != "clean":
                errors.append(f"{base_id}: clean row must use view=clean and profile=clean")
        if not _valid_path(row.get("image_path")):
            errors.append(f"{base_id}/{profile}: image_path missing or unreadable")
        if not _valid_path(row.get("ignore_mask_path")):
            errors.append(f"{base_id}/{profile}: ignore_mask_path missing or unreadable")
        if row.get("label_preserved") is not True:
            errors.append(f"{base_id}/{profile}: label_preserved must be true")
        if label == "tampered" and not _valid_path(row.get("tamper_mask_path")):
            errors.append(f"{base_id}/{profile}: tampered row missing readable tamper_mask_path")

    for base_id, count in clean_seen.items():
        if count != 1:
            errors.append(f"{base_id}: expected exactly one clean row, found {count}")
    for base_id in by_base:
        if base_id not in clean_seen:
            errors.append(f"{base_id}: clean row missing")

    for base_id, base_rows in by_base.items():
        found_profiles = {str(row.get("profile") or "") for row in base_rows}
        missing = sorted(required_profiles - found_profiles)
        extra_count = len(base_rows) - len(found_profiles)
        if missing:
            errors.append(f"{base_id}: missing profiles {missing}")
        if extra_count:
            errors.append(f"{base_id}: duplicate profile rows found")

    base_count = len(by_base)
    expected_rows = base_count * profile_count
    if len(rows) != expected_rows:
        errors.append(f"expected rows = base_count x profile_count = {expected_rows}, found {len(rows)}")

    summary = {
        "pair_root": str(root),
        "base_count": base_count,
        "profile_count": profile_count,
        "profiles": profiles,
        "record_count": len(rows),
        "expected_record_count": expected_rows,
        "label_counts": {label: sum(1 for row in rows if row.get("content_label") == label) for label in CLASS_LABELS},
        "pair_index_pair_count": len(pair_index.get("pairs", [])) if isinstance(pair_index, dict) and isinstance(pair_index.get("pairs"), list) else None,
        "ok": not errors,
    }
    return summary, errors, warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check an SNSAug V2 fixed-pair dataset for class balance and mask sanity.")
    parser.add_argument("pair_root", nargs="?", help="Pair dataset root containing meta.jsonl and pair_index.json")
    parser.add_argument("--pair-root", dest="pair_root_flag", help="Pair dataset root containing meta.jsonl and pair_index.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pair_root = args.pair_root_flag or args.pair_root
    if not pair_root:
        parser.print_help()
        return 0
    try:
        summary, errors, warnings = check_pair_dataset(pair_root)
    except Exception as exc:
        print(json.dumps({"ok": False, "errors": [str(exc)]}, ensure_ascii=True, indent=2), file=sys.stderr)
        return 2
    payload = {"summary": summary, "warnings": warnings, "errors": errors}
    stream = sys.stdout if not errors else sys.stderr
    print(json.dumps(payload, ensure_ascii=True, indent=2), file=stream)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
