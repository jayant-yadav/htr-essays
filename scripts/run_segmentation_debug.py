"""Run line segmentation debug pipeline on a few images without full model inference.

Per image, this writes:
    - 00_original.png
    - 01_deskewed.png
    - 02_binary.png
    - 03_ruled_lines_mask.png
    - 04_binary_clean.png
    - 05_projection_bboxes.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from htr_essays.inference.segment import LineSegmenter


VALID_EXTS = {".png", ".jpg", ".jpeg"}


def _default_image_dir() -> Path:
    return PROJECT_ROOT.parent / "200essays" / "2000" / "png"


def _collect_images(single_image: Path | None, image_dir: Path | None, max_images: int) -> List[Path]:
    if single_image is not None:
        return [single_image]

    if image_dir is None:
        return []

    candidates = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )
    return candidates[:max_images]


def run_debug(
    images: List[Path],
    output_dir: Path,
    min_line_height: int,
    max_line_height: int,
    min_line_width: int,
    margin_pixels: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    segmenter = LineSegmenter(
        min_line_height=min_line_height,
        max_line_height=max_line_height,
        min_line_width=min_line_width,
        margin_pixels=margin_pixels,
    )

    summary = []

    for idx, image_path in enumerate(images):
        img = Image.open(image_path).convert("RGB")

        image_slug = image_path.stem
        debug_dir = output_dir / f"{idx:03d}_{image_slug}"

        boxes = segmenter.segment(
            img,
            debug_dir=str(debug_dir),
        )

        summary.append(
            {
                "index": idx,
                "image": str(image_path),
                "debug_dir": str(debug_dir),
                "num_boxes": len(boxes),
                "method": "projection",
            }
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Processed images: {len(images)}")
    print(f"Output dir: {output_dir}")
    print(f"Summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run segmentation and write only debug artifacts (no full inference).",
    )
    parser.add_argument("--image", type=Path, default=None, help="Single image path")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=_default_image_dir(),
        help="Directory containing page images",
    )
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "segmentation_debug",
    )

    parser.add_argument("--min-line-height", type=int, default=20)
    parser.add_argument("--max-line-height", type=int, default=200)
    parser.add_argument("--min-line-width", type=int, default=100)
    parser.add_argument("--margin-pixels", type=int, default=5)

    args = parser.parse_args()

    if args.image is not None and not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    if args.image is None:
        if args.image_dir is None or not args.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    images = _collect_images(args.image, args.image_dir, args.max_images)
    if not images:
        raise RuntimeError("No images found. Use --image or provide a valid --image-dir with image files.")

    run_debug(
        images=images,
        output_dir=args.output_dir,
        min_line_height=args.min_line_height,
        max_line_height=args.max_line_height,
        min_line_width=args.min_line_width,
        margin_pixels=args.margin_pixels,
    )


if __name__ == "__main__":
    main()
