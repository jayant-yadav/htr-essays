"""
Dataset module for loading and processing Swedish student essay images with HTR annotations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import TrOCRProcessor


class EssayLineDataset(Dataset):
    """
    Dataset for line-level handwriting recognition from Swedish student essays.

    Each sample is a single text line cropped from an essay image with its transcription.
    """

    def __init__(
        self,
        annotations_file: str,
        images_root: str,
        processor: TrOCRProcessor,
        split_file: Optional[str] = None,
        split_type: str = "train",
        transform=None,
    ):
        """
        Args:
            annotations_file: Path to json_full.json with annotations
            images_root: Root directory containing essay images (200essays/)
            processor: TrOCR processor for text and image processing
            split_file: Optional JSON file with train/val/test splits
            split_type: One of "train", "val", or "test"
            transform: Optional image transformations
        """
        self.images_root = Path(images_root)
        self.processor = processor
        self.transform = transform
        self.split_type = split_type

        # Load annotations
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        # Load or create splits
        if split_file and os.path.exists(split_file):
            with open(split_file, 'r') as f:
                splits = json.load(f)
            self.image_ids = set(splits[split_type])
        else:
            # Use all images if no split file provided
            self.image_ids = set(range(len(self.annotations)))

        # Extract all line-level samples
        self.samples = self._extract_line_samples()

        print(f"Loaded {len(self.samples)} line samples for {split_type} split")

    def _extract_line_samples(self) -> List[Dict]:
        """
        Extract individual text lines from annotations.

        Returns:
            List of dicts with keys: image_path, bbox, transcription
        """
        samples = []

        for idx, entry in enumerate(self.annotations):
            # Skip if not in this split
            if idx not in self.image_ids:
                continue

            # Get image path
            json_path = entry['data']['ocr']
            image_path = self._map_json_path_to_file(json_path)

            if image_path is None:
                print(f"Warning: Could not find image for {json_path}")
                continue

            # Extract annotations
            if not entry.get('annotations') or len(entry['annotations']) == 0:
                continue

            annotation = entry['annotations'][0]  # Take first annotation
            result = annotation.get('result', [])

            # Group results by ID to match bbox with transcription
            id_groups = {}
            for item in result:
                item_id = item.get('id')
                if item_id not in id_groups:
                    id_groups[item_id] = {}

                item_type = item.get('type')
                if item_type == 'rectangle':
                    id_groups[item_id]['bbox'] = item['value']
                    id_groups[item_id]['original_width'] = item['original_width']
                    id_groups[item_id]['original_height'] = item['original_height']
                elif item_type == 'textarea':
                    id_groups[item_id]['text'] = item['value'].get('text', [])

            # Create samples for each line
            for item_id, data in id_groups.items():
                if 'bbox' in data and 'text' in data:
                    text = data['text']
                    if isinstance(text, list) and len(text) > 0:
                        text = ' '.join(text)

                    samples.append({
                        'image_path': str(image_path),
                        'bbox': data['bbox'],
                        'original_width': data['original_width'],
                        'original_height': data['original_height'],
                        'transcription': text,
                    })

        return samples

    def _map_json_path_to_file(self, json_path: str) -> Optional[Path]:
        """
        Map JSON path like '/data/upload/1/30dba2af-06_33_1.png' to actual file.

        Returns:
            Path to actual image file, or None if not found
        """
        # Extract filename part after the hash
        basename = os.path.basename(json_path)
        parts = basename.split('-')

        if len(parts) < 2:
            return None

        # Get the actual filename (e.g., "06_33_1.png")
        actual_name = parts[1]

        # Determine year from filename prefix
        year_prefix = actual_name.split('_')[0]
        year_map = {
            '00': '2000',
            '06': '2006',
            '12': '2012',
            '18': '2018',
        }

        year = year_map.get(year_prefix)
        if year is None:
            return None

        # Try different formats
        for ext in ['jpg', 'png', 'pdf']:
            filename = actual_name.replace('.png', f'.{ext}')
            image_path = self.images_root / year / ext / filename
            if image_path.exists():
                return image_path

        return None

    def _crop_line(self, image: Image.Image, bbox: Dict, orig_width: int, orig_height: int) -> Image.Image:
        """
        Crop a line from the image using percentage-based bounding box.

        Args:
            image: PIL Image
            bbox: Dict with keys x, y, width, height (all in %)
            orig_width: Original image width from annotation
            orig_height: Original image height from annotation

        Returns:
            Cropped PIL Image
        """
        img_width, img_height = image.size

        # Convert percentage coordinates to pixels
        x_percent = bbox['x']
        y_percent = bbox['y']
        w_percent = bbox['width']
        h_percent = bbox['height']

        x = int(x_percent / 100 * img_width)
        y = int(y_percent / 100 * img_height)
        w = int(w_percent / 100 * img_width)
        h = int(h_percent / 100 * img_height)

        # Ensure bounds are within image
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = max(1, min(w, img_width - x))
        h = max(1, min(h, img_height - y))

        # Crop the image
        cropped = image.crop((x, y, x + w, y + h))

        return cropped

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single line sample.

        Returns:
            Dict with keys: pixel_values, labels
        """
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        # Crop the line
        cropped = self._crop_line(
            image,
            sample['bbox'],
            sample['original_width'],
            sample['original_height']
        )

        # Apply optional transforms
        if self.transform is not None:
            cropped = self.transform(cropped)

        # Process with TrOCR processor
        pixel_values = self.processor(cropped, return_tensors="pt").pixel_values.squeeze()

        # Encode text labels
        labels = self.processor.tokenizer(
            sample['transcription'],
            padding='max_length',
            max_length=128,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()

        # Replace padding token id with -100 for loss calculation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            'pixel_values': pixel_values,
            'labels': labels,
        }


def create_data_splits(
    annotations_file: str,
    output_file: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
):
    """
    Create train/validation/test splits and save to JSON file.

    Args:
        annotations_file: Path to json_full.json
        output_file: Where to save the splits JSON
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    # Load annotations
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    n_images = len(annotations)
    indices = list(range(n_images))

    # Shuffle with fixed seed
    random.seed(random_seed)
    random.shuffle(indices)

    # Calculate split sizes
    n_train = int(n_images * train_ratio)
    n_val = int(n_images * val_ratio)

    # Create splits
    splits = {
        'train': indices[:n_train],
        'val': indices[n_train:n_train + n_val],
        'test': indices[n_train + n_val:],
    }

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"Created splits:")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val: {len(splits['val'])} images")
    print(f"  Test: {len(splits['test'])} images")
    print(f"Saved to {output_file}")

    return splits


if __name__ == '__main__':
    # Example usage: create splits
    create_data_splits(
        annotations_file='../../../../200essays/json_full.json',
        output_file='../../../../data_splits.json',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
