"""
Main inference script for HTR pipeline.

Usage:
    # Single image with YOLO segmentation
    python -m htr_essays.inference.infer --checkpoint outputs/final_model --image essay.jpg

    # Batch inference on directory
    python -m htr_essays.inference.infer --checkpoint outputs/final_model --image_dir essays/

    # Use ground truth bounding boxes from JSON
    python -m htr_essays.inference.infer --checkpoint outputs/final_model --annotations annotations.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from htr_essays.inference.predictor import HTRPredictor
from htr_essays.inference.segment import LineSegmenter


DEFAULT_YOLO_MODEL = (
    Path(__file__).resolve().parents[4]
    / "yolo-essays"
    / "outputs"
    / "yolo_line_seg"
    / "weights"
    / "best.pt"
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="HTR Inference Pipeline")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Path to single image")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to directory of images")
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Optional JSON with ground truth bounding boxes",
    )
    parser.add_argument("--output", type=str, default="predictions.json", help="Output JSON file")
    parser.add_argument("--visualize", action="store_true", help="Save visualization images")
    parser.add_argument("--viz_dir", type=str, default="visualizations", help="Directory for visualizations")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument(
        "--yolo_model",
        type=str,
        default=str(DEFAULT_YOLO_MODEL),
        help="Path to YOLO line segmentation model (.pt)",
    )
    parser.add_argument(
        "--yolo_conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold for line detections",
    )
    parser.add_argument(
        "--yolo_iou",
        type=float,
        default=0.45,
        help="YOLO IoU threshold for NMS",
    )
    return parser.parse_args()


def infer_single_image(
    image_path: str,
    predictor: HTRPredictor,
    segmenter: LineSegmenter,
    use_segmentation: bool = True,
) -> Dict:
    """
    Run inference on a single image.

    Args:
        image_path: Path to image
        predictor: HTR predictor
        segmenter: Line segmenter
        use_segmentation: Whether to use automated segmentation

    Returns:
        Dict with predictions
    """
    image = Image.open(image_path).convert("RGB")

    if use_segmentation:
        bboxes = segmenter.segment(image)
    else:
        img_width, img_height = image.size
        bboxes = [
            {
                "bbox": {"x": 0, "y": 0, "width": 100, "height": 100},
                "original_width": img_width,
                "original_height": img_height,
            }
        ]

    predictions = predictor.predict_from_bboxes(image, bboxes)

    return {
        "image_path": image_path,
        "num_lines": len(predictions),
        "lines": predictions,
        "full_text": "\n".join([p["text"] for p in predictions]),
    }


def main():
    """Main inference function."""
    args = parse_args()

    print("=" * 70)
    print("HTR Inference Pipeline")
    print("=" * 70)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"YOLO model: {args.yolo_model}")
    print(f"Output: {args.output}")
    print()

    print("Loading model...")
    predictor = HTRPredictor(
        model_path=args.checkpoint,
        device=args.device,
        num_beams=args.num_beams,
    )
    print()

    print("Initializing segmenter...")
    segmenter = LineSegmenter(
        model_path=args.yolo_model,
        conf_threshold=args.yolo_conf,
        iou_threshold=args.yolo_iou,
        device=args.device,
    )
    print()

    image_paths = []

    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            image_paths.extend(str(path) for path in image_dir.glob(ext))
        print(f"Found {len(image_paths)} images in {args.image_dir}")
    elif args.annotations:
        with open(args.annotations, "r", encoding="utf-8") as file_handle:
            _annotations = json.load(file_handle)
        print("Note: Using annotations requires adapting this script to your specific format")
        return
    else:
        print("Error: Must specify either --image, --image_dir, or --annotations")
        return

    if not image_paths:
        print("Error: No images found")
        return

    print(f"Processing {len(image_paths)} images...")
    print()

    if args.visualize:
        os.makedirs(args.viz_dir, exist_ok=True)

    all_results = []

    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            result = infer_single_image(
                image_path=image_path,
                predictor=predictor,
                segmenter=segmenter,
                use_segmentation=True,
            )
            all_results.append(result)

            if args.visualize:
                image = Image.open(image_path).convert("RGB")
                bboxes = [{"bbox": line["bbox"]} for line in result["lines"]]
                viz_path = os.path.join(args.viz_dir, Path(image_path).name)
                segmenter.visualize_segmentation(image, bboxes, output_path=viz_path)

        except Exception as error:
            print(f"\nError processing {image_path}: {error}")
            continue

    print(f"\nSaving results to {args.output}...")

    output_data = {
        "checkpoint": args.checkpoint,
        "num_images": len(all_results),
        "total_lines": sum(result["num_lines"] for result in all_results),
        "results": all_results,
    }

    with open(args.output, "w", encoding="utf-8") as file_handle:
        json.dump(output_data, file_handle, indent=2, ensure_ascii=False)

    print("\nInference complete!")
    print(f"Processed {len(all_results)} images")
    print(f"Total lines detected: {output_data['total_lines']}")
    print(f"Results saved to: {args.output}")

    if args.visualize:
        print(f"Visualizations saved to: {args.viz_dir}")

    print("\n" + "=" * 70)
    print("Sample Results (first 3 images):")
    print("=" * 70)

    for index, result in enumerate(all_results[:3]):
        print(f"\n{index + 1}. {Path(result['image_path']).name}")
        print(f"   Lines detected: {result['num_lines']}")
        print("   Full text:")

        lines = result["full_text"].split("\n")
        for line_index, line in enumerate(lines[:5], 1):
            print(f"     {line_index}. {line}")

        if len(lines) > 5:
            print(f"     ... ({len(lines) - 5} more lines)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
