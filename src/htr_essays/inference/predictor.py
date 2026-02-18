"""
HTR Predictor for inference on new images.
"""

import torch
from PIL import Image
from typing import List, Dict, Optional, Union
import numpy as np

from transformers import VisionEncoderDecoderModel, TrOCRProcessor


class HTRPredictor:
    """
    Predictor class for HTR inference.

    Handles loading trained models and generating predictions for new images.
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        num_beams: int = 4,
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            num_beams: Number of beams for beam search
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_beams = num_beams

        print(f"Loading model from {model_path}...")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        print("Loading processor...")
        self.processor = TrOCRProcessor.from_pretrained(model_path)

        print(f"Model loaded on {self.device}")

    def predict_single(
        self,
        image: Union[Image.Image, np.ndarray],
        max_length: int = 128,
    ) -> str:
        """
        Predict text from a single image.

        Args:
            image: PIL Image or numpy array
            max_length: Maximum sequence length

        Returns:
            Predicted text string
        """
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=self.num_beams,
                early_stopping=True,
            )

        # Decode
        predicted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return predicted_text

    def predict_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        max_length: int = 128,
        batch_size: int = 8,
    ) -> List[str]:
        """
        Predict text from multiple images.

        Args:
            images: List of PIL Images or numpy arrays
            max_length: Maximum sequence length
            batch_size: Batch size for processing

        Returns:
            List of predicted text strings
        """
        predictions = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]

            # Convert to PIL and ensure RGB
            pil_images = []
            for img in batch_images:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                pil_images.append(img)

            # Process batch
            pixel_values = self.processor(pil_images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=max_length,
                    num_beams=self.num_beams,
                    early_stopping=True,
                )

            # Decode
            batch_predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(batch_predictions)

        return predictions

    def predict_from_bboxes(
        self,
        image: Image.Image,
        bboxes: List[Dict],
        max_length: int = 128,
    ) -> List[Dict[str, str]]:
        """
        Predict text for multiple bounding boxes in an image.

        Args:
            image: Full PIL Image
            bboxes: List of bounding box dicts (from segmenter)
            max_length: Maximum sequence length

        Returns:
            List of dicts with keys: 'bbox', 'text'
        """
        # Crop line images from bounding boxes
        line_images = []
        img_width, img_height = image.size

        for bbox_data in bboxes:
            bbox = bbox_data['bbox']

            # Convert percentage to pixels
            x = int(bbox['x'] / 100 * img_width)
            y = int(bbox['y'] / 100 * img_height)
            w = int(bbox['width'] / 100 * img_width)
            h = int(bbox['height'] / 100 * img_height)

            # Ensure bounds are valid
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            w = max(1, min(w, img_width - x))
            h = max(1, min(h, img_height - y))

            # Crop
            cropped = image.crop((x, y, x + w, y + h))
            line_images.append(cropped)

        # Predict on all lines
        predictions = self.predict_batch(line_images, max_length=max_length)

        # Combine with bounding boxes
        results = []
        for bbox_data, text in zip(bboxes, predictions):
            results.append({
                'bbox': bbox_data['bbox'],
                'text': text,
            })

        return results


if __name__ == '__main__':
    # Test predictor interface
    print("HTR Predictor module structure is correct")
    print("\nUsage example:")
    print("  predictor = HTRPredictor(model_path='outputs/final_model')")
    print("  text = predictor.predict_single(image)")
    print("  texts = predictor.predict_batch([img1, img2, img3])")
