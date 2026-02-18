"""
Preprocessing and augmentation utilities for handwritten text images.
"""

import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms as transforms


class HandwritingAugmentation:
    """
    Data augmentation for handwritten text images.

    Applies random transformations to simulate variations in handwriting,
    scanning quality, and image conditions.
    """

    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-2, 2),
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        blur_prob: float = 0.1,
        noise_prob: float = 0.1,
        elastic_prob: float = 0.0,  # Disabled by default (complex to implement)
    ):
        """
        Args:
            rotation_range: Min and max rotation in degrees
            brightness_range: Min and max brightness factor
            contrast_range: Min and max contrast factor
            blur_prob: Probability of applying blur
            noise_prob: Probability of adding noise
            elastic_prob: Probability of elastic deformation
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.elastic_prob = elastic_prob

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply augmentations to an image.

        Args:
            image: PIL Image

        Returns:
            Augmented PIL Image
        """
        # Random rotation
        if self.rotation_range != (0, 0):
            angle = random.uniform(*self.rotation_range)
            image = image.rotate(angle, fillcolor=(255, 255, 255), expand=False)

        # Random brightness
        if self.brightness_range != (1, 1):
            factor = random.uniform(*self.brightness_range)
            enhancer = ImageEnhance.Brightness(image)
            image =enhancer.enhance(factor)

        # Random contrast
        if self.contrast_range != (1, 1):
            factor = random.uniform(*self.contrast_range)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)

        # Random blur
        if random.random() < self.blur_prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))

        # Random noise
        if random.random() < self.noise_prob:
            image = self._add_noise(image)

        return image

    def _add_noise(self, image: Image.Image, noise_level: float = 0.02) -> Image.Image:
        """
        Add random noise to image.

        Args:
            image: PIL Image
            noise_level: Standard deviation of noise

        Returns:
            Noisy PIL Image
        """
        img_array = np.array(image).astype(np.float32) / 255.0
        noise = np.random.normal(0, noise_level, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((noisy * 255).astype(np.uint8))


def get_train_transform():
    """
    Get training data augmentation pipeline.

    Returns:
        Augmentation callable
    """
    return HandwritingAugmentation(
        rotation_range=(-2, 2),
        brightness_range=(0.85, 1.15),
        contrast_range=(0.85, 1.15),
        blur_prob=0.1,
        noise_prob=0.1,
    )


def get_val_transform():
    """
    Get validation/test transform (no augmentation).

    Returns:
        None (no augmentation for validation/test)
    """
    return None


def normalize_image(image: Image.Image) -> Image.Image:
    """
    Normalize image for HTR processing.

    Args:
        image: PIL Image

    Returns:
        Normalized PIL Image
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')

    # Convert back to RGB (TrOCR expects 3 channels)
    image = image.convert('RGB')

    return image


class AdaptiveResize:
    """
    Resize image while preserving aspect ratio.
    """

    def __init__(self, target_height: int = 384, max_width: int = 2048):
        """
        Args:
            target_height: Target height for resized image
            max_width: Maximum width to prevent very wide images
        """
        self.target_height = target_height
        self.max_width = max_width

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Resize image adaptively.

        Args:
            image: PIL Image

        Returns:
            Resized PIL Image
        """
        width, height = image.size

        # Calculate new width maintaining aspect ratio
        aspect_ratio = width / height
        new_width = int(self.target_height * aspect_ratio)

        # Limit width
        if new_width > self.max_width:
            new_width = self.max_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = self.target_height

        # Resize
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image


def binarize_image(image: Image.Image, threshold: int = 128) -> Image.Image:
    """
    Binarize image using simple thresholding.

    Args:
        image: PIL Image
        threshold: Threshold value (0-255)

    Returns:
        Binarized PIL Image
    """
    # Convert to grayscale
    gray = image.convert('L')

    # Apply threshold
    binary = gray.point(lambda x: 255 if x > threshold else 0)

    return binary


def adaptive_binarize(image: Image.Image, block_size: int = 11, c: int = 2) -> Image.Image:
    """
    Adaptive binarization using local thresholding.

    Note: This is a simple implementation. For better results, consider using
    cv2.adaptiveThreshold.

    Args:
        image: PIL Image
        block_size: Size of local neighborhood
        c: Constant subtracted from mean

    Returns:
        Binarized PIL Image
    """
    try:
        import cv2

        # Convert to OpenCV format
        img_array = np.array(image.convert('L'))

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            img_array,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c
        )

        return Image.fromarray(binary)
    except ImportError:
        # Fallback to simple binarization if cv2 not available
        return binarize_image(image)


def deskew_image(image: Image.Image) -> Image.Image:
    """
    Detect and correct skew in image.

    Args:
        image: PIL Image

    Returns:
        Deskewed PIL Image
    """
    try:
        import cv2

        # Convert to OpenCV format
        img_array = np.array(image.convert('L'))

        # Detect edges
        edges = cv2.Canny(img_array, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        if lines is None or len(lines) == 0:
            return image  # No lines detected, return original

        # Calculate median angle
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            angles.append(angle)

        median_angle = np.median(angles)

        # Only correct if angle is significant (> 0.5 degrees)
        if abs(median_angle) > 0.5:
            # Rotate image
            image = image.rotate(median_angle, fillcolor=(255, 255, 255), expand=False)

        return image
    except ImportError:
        # Return original if cv2 not available
        return image


if __name__ == '__main__':
    # Test augmentation
    from PIL import Image

    # Create a dummy image
    img = Image.new('RGB', (400, 100), color='white')

    # Apply augmentation
    aug = get_train_transform()
    aug_img = aug(img)

    print("Augmentation test passed")
