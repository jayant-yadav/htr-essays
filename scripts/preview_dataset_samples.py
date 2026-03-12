"""Preview line crops produced by EssayLineDataset and save intermediate PIL images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from htr_essays.data.dataset import EssayLineDataset


def _default_paths() -> tuple[Path, Path, Path]:
    workspace_root = PROJECT_ROOT.parent
    annotations = workspace_root / "200essays" / "json_full.json"
    images_root = workspace_root / "200essays"
    split_file = PROJECT_ROOT / "data_splits.json"
    return annotations, images_root, split_file


def _bbox_pixels(sample_bbox: dict, width: int, height: int) -> tuple[int, int, int, int]:
    x = int(sample_bbox["x"] / 100 * width)
    y = int(sample_bbox["y"] / 100 * height)
    w = int(sample_bbox["width"] / 100 * width)
    h = int(sample_bbox["height"] / 100 * height)

    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))

    return x, y, x + w, y + h


def run_preview(
    annotations_file: Path,
    images_root: Path,
    split_file: Path | None,
    split_type: str,
    max_samples: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path_str = str(split_file) if split_file and split_file.exists() else None
    dataset = EssayLineDataset(
        annotations_file=str(annotations_file),
        images_root=str(images_root),
        processor=None,
        split_file=split_path_str,
        split_type=split_type,
        transform=None,
    )

    total = len(dataset.samples)
    num = min(max_samples, total)
    metadata = []

    for idx in range(num):
        sample = dataset.samples[idx]
        original = Image.open(sample["image_path"]).convert("RGB")
        cropped = dataset._crop_line(
            original,
            sample["bbox"],
            sample["original_width"],
            sample["original_height"],
        )

        boxed = original.copy()
        draw = ImageDraw.Draw(boxed)
        x0, y0, x1, y1 = _bbox_pixels(sample["bbox"], *original.size)
        draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=3)

        boxed_path = output_dir / f"{idx:03d}_boxed.png"
        crop_path = output_dir / f"{idx:03d}_crop.png"
        boxed.save(boxed_path)
        cropped.save(crop_path)

        metadata.append(
            {
                "index": idx,
                "source_image": sample["image_path"],
                "boxed_preview": str(boxed_path),
                "crop_preview": str(crop_path),
                "transcription": sample["transcription"],
                "bbox_percent": sample["bbox"],
                "bbox_pixels": [x0, y0, x1, y1],
                "crop_size": list(cropped.size),
            }
        )

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Split: {split_type}")
    print(f"Total samples in split: {total}")
    print(f"Saved previews for: {num} samples")
    print(f"Output dir: {output_dir}")
    print(f"Metadata: {metadata_path}")


def main() -> None:
    default_annotations, default_images_root, default_split = _default_paths()

    parser = argparse.ArgumentParser(description="Preview EssayLineDataset crops")
    parser.add_argument("--annotations-file", type=Path, default=default_annotations)
    parser.add_argument("--images-root", type=Path, default=default_images_root)
    parser.add_argument("--split-file", type=Path, default=default_split)
    parser.add_argument("--split-type", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "dataset_preview",
    )

    args = parser.parse_args()

    run_preview(
        annotations_file=args.annotations_file,
        images_root=args.images_root,
        split_file=args.split_file,
        split_type=args.split_type,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
