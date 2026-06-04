#!/usr/bin/env python3
"""Preview SNSAug V2 overlays across one or more profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_forensics.snsaug_v2 import SNSAugV2Augmentor, SNSAugV2Config  # noqa: E402
from cv_forensics.snsaug_v2.ui_renderers import draw_overlay_debug  # noqa: E402
from cv_forensics.pre_sns_training_artifacts import validate_artifact_root  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview deterministic SNSAug V2 overlay profiles.")
    parser.add_argument("--image", required=False, help="Input image path")
    parser.add_argument("--mask", required=False, help="Optional tamper mask path")
    parser.add_argument("--profiles", nargs="+", default=["combined_sns_realistic"], help="Profiles to preview")
    parser.add_argument("--severity", default="medium", help="Severity level")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed")
    parser.add_argument("--output-root", required=False, help="External output directory")
    parser.add_argument("--font-path", required=False, help="Optional local font path")
    parser.add_argument("--use_demo_image", action="store_true", help="Use a synthetic demo image when no input image is given")
    return parser


def _demo_image() -> tuple[Image.Image, Image.Image]:
    image = Image.new("RGB", (320, 224), (52, 66, 88))
    draw = ImageDraw.Draw(image)
    draw.rectangle((96, 52, 224, 180), fill=(220, 52, 52))
    draw.ellipse((28, 24, 88, 84), fill=(64, 160, 220))
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((96, 52, 224, 180), fill=255)
    return image, mask


def _save_grid(original: Image.Image, augmented: Image.Image, tamper_mask: Image.Image | None, ignore_mask: Image.Image, debug_image: Image.Image, path: Path) -> None:
    width, height = augmented.size
    grid = Image.new("RGB", (width * 3, height * 2), (18, 18, 18))
    grid.paste(original.resize((width, height), Image.Resampling.BILINEAR), (0, 0))
    grid.paste(augmented, (width, 0))
    grid.paste(debug_image, (width * 2, 0))
    grid.paste(tamper_mask.convert("RGB") if tamper_mask is not None else Image.new("RGB", (width, height), (0, 0, 0)), (0, height))
    grid.paste(ignore_mask.convert("RGB"), (width, height))
    grid.paste(debug_image, (width * 2, height))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.output_root:
        parser.print_help()
        return 0
    output_root = validate_artifact_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.image:
        with Image.open(args.image) as image_handle:
            image = image_handle.convert("RGB")
        mask = None
        if args.mask:
            with Image.open(args.mask) as mask_handle:
                mask = mask_handle.convert("L")
    elif args.use_demo_image:
        image, mask = _demo_image()
    else:
        parser.print_help()
        return 0

    manifest: list[dict[str, str]] = []
    for index, profile in enumerate(args.profiles):
        profile_root = output_root / profile
        profile_root.mkdir(parents=True, exist_ok=True)
        config = SNSAugV2Config(profile=profile, severity=args.severity, seed=args.seed + index, font_path=args.font_path)
        result = SNSAugV2Augmentor(config)(image, mask, base_id="preview_sample", seed=config.seed)
        debug_image = draw_overlay_debug(result.image, result.meta.get("overlay_boxes", []))
        augmented_path = profile_root / "augmented_image.png"
        tamper_mask_path = profile_root / "transformed_tamper_mask.png"
        ignore_mask_path = profile_root / "ignore_mask.png"
        metadata_path = profile_root / "metadata.json"
        debug_path = profile_root / "overlay_debug.png"
        grid_path = profile_root / "preview_grid.png"
        result.image.save(augmented_path)
        if result.tamper_mask is not None:
            result.tamper_mask.save(tamper_mask_path)
        result.ignore_mask.save(ignore_mask_path)
        debug_image.save(debug_path)
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(result.meta, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
        _save_grid(image, result.image, result.tamper_mask, result.ignore_mask, debug_image, grid_path)
        manifest.append(
            {
                "profile": profile,
                "augmented_image": str(augmented_path),
                "tamper_mask": str(tamper_mask_path) if result.tamper_mask is not None else "",
                "ignore_mask": str(ignore_mask_path),
                "metadata": str(metadata_path),
                "overlay_debug": str(debug_path),
                "preview_grid": str(grid_path),
            }
        )
    print(json.dumps({"profiles": manifest, "output_root": str(output_root)}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
