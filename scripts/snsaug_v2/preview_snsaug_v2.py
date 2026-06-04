#!/usr/bin/env python3
"""Preview SNSAug V2 augmentations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2 import SNSAugV2Augmentor, SNSAugV2Config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview deterministic SNSAug V2 overlays.")
    parser.add_argument("--image", required=False, help="Input image path")
    parser.add_argument("--mask", required=False, help="Optional tamper mask path")
    parser.add_argument("--profile", default="combined_sns_realistic", help="SNSAug V2 profile")
    parser.add_argument("--severity", default="medium", help="Severity level")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed")
    parser.add_argument("--output-root", required=False, help="External output directory")
    parser.add_argument("--font-path", required=False, help="Optional local font path")
    return parser


def _save_grid(original: Image.Image, augmented: Image.Image, tamper_mask: Image.Image | None, ignore_mask: Image.Image, path: Path) -> None:
    width, height = augmented.size
    grid = Image.new("RGB", (width * 2, height * 2), (18, 18, 18))
    grid.paste(original.resize((width, height), Image.Resampling.BILINEAR), (0, 0))
    grid.paste(augmented, (width, 0))
    if tamper_mask is not None:
        grid.paste(tamper_mask.convert("RGB"), (0, height))
    else:
        grid.paste(Image.new("RGB", (width, height), (0, 0, 0)), (0, height))
    grid.paste(ignore_mask.convert("RGB"), (width, height))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.image or not args.output_root:
        parser.print_help()
        return 0
    output_root = Path(args.output_root)
    with Image.open(args.image) as image_handle:
        image = image_handle.convert("RGB")
    mask = None
    if args.mask:
        with Image.open(args.mask) as mask_handle:
            mask = mask_handle.convert("L")
    config = SNSAugV2Config(
        profile=args.profile,
        severity=args.severity,
        seed=args.seed,
        font_path=args.font_path,
    )
    result = SNSAugV2Augmentor(config)(image, mask, base_id=Path(args.image).stem, seed=args.seed)
    output_root.mkdir(parents=True, exist_ok=True)
    augmented_path = output_root / "augmented_image.png"
    tamper_mask_path = output_root / "transformed_tamper_mask.png"
    ignore_mask_path = output_root / "ignore_mask.png"
    metadata_path = output_root / "metadata.json"
    grid_path = output_root / "preview_grid.png"
    result.image.save(augmented_path)
    if result.tamper_mask is not None:
        result.tamper_mask.save(tamper_mask_path)
    result.ignore_mask.save(ignore_mask_path)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(result.meta, handle, ensure_ascii=True, indent=2)
    _save_grid(image, result.image, result.tamper_mask, result.ignore_mask, grid_path)
    print(json.dumps({"augmented_image": str(augmented_path), "ignore_mask": str(ignore_mask_path), "preview_grid": str(grid_path)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
