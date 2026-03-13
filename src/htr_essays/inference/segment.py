"""
YOLO-based text line segmentation for handwritten essays.
"""

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def _default_yolo_model_path() -> Path:
    base_dir = Path(__file__).resolve().parents[4]
    return base_dir / "yolo-essays" / "outputs" / "yolo_line_seg" / "weights" / "best.pt"


class LineSegmenter:
    """
    Segment handwritten pages into text lines using a trained YOLO detector.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda",
    ):
        resolved_model_path = Path(model_path) if model_path else _default_yolo_model_path()
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at: {resolved_model_path}")

        self.model_path = resolved_model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = self._normalize_device(device)
        self.model = YOLO(str(self.model_path))

    @staticmethod
    def _normalize_device(device: str):
        if device == "cuda":
            return 0
        if device.startswith("cuda:"):
            _, gpu_id = device.split(":", 1)
            return int(gpu_id)
        return device

    @staticmethod
    def _clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int):
        x1 = max(0, min(int(round(x1)), width - 1))
        y1 = max(0, min(int(round(y1)), height - 1))
        x2 = max(x1 + 1, min(int(round(x2)), width))
        y2 = max(y1 + 1, min(int(round(y2)), height))
        return x1, y1, x2, y2

    def segment(
        self,
        image: Image.Image,
        debug_dir: Optional[str] = None,
    ) -> List[Dict]:
        """
        Segment an image into text lines using YOLO detections.

        Returns:
            List of dicts with keys: bbox (dict with x, y, width, height as %)
                                      original_width, original_height
        """
        img_array = np.array(image.convert("RGB"))
        img_height, img_width = img_array.shape[:2]

        predictions = self.model.predict(
            source=img_array,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        if not predictions:
            return []

        boxes = predictions[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else None

        detections = []
        for index, box in enumerate(xyxy):
            x1, y1, x2, y2 = self._clamp_box(box[0], box[1], box[2], box[3], img_width, img_height)
            score = float(confs[index]) if confs is not None else None
            detections.append((x1, y1, x2, y2, score))

        detections.sort(key=lambda d: ((d[1] + d[3]) / 2.0, d[0]))

        results = []
        for x1, y1, x2, y2, score in detections:
            box_width = x2 - x1
            box_height = y2 - y1
            result = {
                "bbox": {
                    "x": (x1 / img_width) * 100,
                    "y": (y1 / img_height) * 100,
                    "width": (box_width / img_width) * 100,
                    "height": (box_height / img_height) * 100,
                },
                "original_width": img_width,
                "original_height": img_height,
            }
            if score is not None:
                result["confidence"] = score
            results.append(result)

        if debug_dir:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            overlay = self.visualize_segmentation(image, results)
            overlay.save(debug_path / "00_yolo_bboxes.png")

        return results

    def visualize_segmentation(
        self,
        image: Image.Image,
        bboxes: List[Dict],
        output_path: Optional[str] = None,
    ) -> Image.Image:
        img_array = np.array(image.convert("RGB"))
        img_height, img_width = img_array.shape[:2]

        for bbox_data in bboxes:
            bbox = bbox_data["bbox"]
            x = int(bbox["x"] / 100 * img_width)
            y = int(bbox["y"] / 100 * img_height)
            w = int(bbox["width"] / 100 * img_width)
            h = int(bbox["height"] / 100 * img_height)
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

        result_img = Image.fromarray(img_array)
        if output_path:
            result_img.save(output_path)
        return result_img
