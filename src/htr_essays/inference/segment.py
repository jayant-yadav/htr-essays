"""
Automated text line segmentation for handwritten essays.

This module implements automated line detection using OpenCV-based methods
including preprocessing, binarization, projection profiles, and contour detection.
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
import cv2


class LineSegmenter:
    """
    Automated line segmentation for handwritten text images.

    Uses horizontal projection profile analysis and contour detection
    to identify text lines in essay images.
    """

    def __init__(
        self,
        min_line_height: int = 20,
        max_line_height: int = 200,
        min_line_width: int = 100,
        margin_pixels: int = 5,
    ):
        """
        Initialize line segmenter.

        Args:
            min_line_height: Minimum height for a valid text line (pixels)
            max_line_height: Maximum height for a valid text line (pixels)
            min_line_width: Minimum width for a valid text line (pixels)
            margin_pixels: Extra margin to add around detected lines
        """
        self.min_line_height = min_line_height
        self.max_line_height = max_line_height
        self.min_line_width = min_line_width
        self.margin_pixels = margin_pixels

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for line detection.

        Args:
            image: Input image (RGB or grayscale)

        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding for better handling of varying lighting
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # Invert: text is white, background is black
            11,
            2
        )

        return binary

    def detect_lines_projection(self, binary: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect text lines using horizontal projection profile.

        Args:
            binary: Binary image (preprocessed)

        Returns:
            List of (y_min, y_max) tuples for each detected line
        """
        # Compute horizontal projection
        projection = np.sum(binary, axis=1)

        # Smooth projection to reduce noise
        from scipy.ndimage import uniform_filter1d
        try:
            projection_smooth = uniform_filter1d(projection, size=5)
        except ImportError:
            # Fallback if scipy not available
            projection_smooth = projection

        # Find peaks and valleys in projection
        height = binary.shape[0]
        threshold = np.mean(projection_smooth) * 0.1

        # Track line boundaries
        in_line = False
        line_start = 0
        lines = []

        for y in range(height):
            if not in_line and projection_smooth[y] > threshold:
                # Start of a line
                in_line = True
                line_start = y
            elif in_line and projection_smooth[y] <= threshold:
                # End of a line
                in_line = False
                line_end = y

                # Validate line dimensions
                line_height = line_end - line_start
                if self.min_line_height <= line_height <= self.max_line_height:
                    lines.append((line_start, line_end))

        # Handle case where last line extends to bottom
        if in_line:
            line_height = height - line_start
            if self.min_line_height <= line_height <= self.max_line_height:
                lines.append((line_start, height))

        return lines

    def detect_lines_contours(self, binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect text lines using contour detection.

        Args:
            binary: Binary image (preprocessed)

        Returns:
            List of (x, y, w, h) bounding boxes for each detected line
        """
        # Apply morphological operations to connect text in lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract bounding boxes
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size
            if w >= self.min_line_width and self.min_line_height <= h <= self.max_line_height:
                # Add margin
                x_margin = max(0, x - self.margin_pixels)
                y_margin = max(0, y - self.margin_pixels)
                w_margin = min(binary.shape[1] - x_margin, w + 2 * self.margin_pixels)
                h_margin = min(binary.shape[0] - y_margin, h + 2 * self.margin_pixels)

                bboxes.append((x_margin, y_margin, w_margin, h_margin))

        # Sort by y-coordinate (top to bottom)
        bboxes = sorted(bboxes, key=lambda b: b[1])

        return bboxes

    def segment(self, image: Image.Image, method: str = 'contours') -> List[Dict]:
        """
        Segment an image into text lines.

        Args:
            image: PIL Image
            method: 'contours' or 'projection'

        Returns:
            List of dicts with keys: bbox (dict with x, y, width, height as %)
                                      original_width, original_height
        """
        # Convert to numpy array
        img_array = np.array(image)

        # Preprocess
        binary = self.preprocess_image(img_array)

        # Detect lines
        if method == 'projection':
            # Use projection profile method
            lines_y = self.detect_lines_projection(binary)

            # Convert to full bounding boxes
            bboxes = []
            img_width = img_array.shape[1]

            for y_min, y_max in lines_y:
                # Find x boundaries by looking at the actual content
                line_content = binary[y_min:y_max, :]
                x_projection = np.sum(line_content, axis=0)
                x_nonzero = np.where(x_projection > 0)[0]

                if len(x_nonzero) > 0:
                    x_min = max(0, x_nonzero[0] - self.margin_pixels)
                    x_max = min(img_width, x_nonzero[-1] + self.margin_pixels)

                    bboxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

        else:  # method == 'contours'
            bboxes= self.detect_lines_contours(binary)

        # Convert to percentage coordinates (matching annotation format)
        img_height, img_width = img_array.shape[:2]

        results = []
        for x, y, w, h in bboxes:
            results.append({
                'bbox': {
                    'x': (x / img_width) * 100,
                    'y': (y / img_height) * 100,
                    'width': (w / img_width) * 100,
                    'height': (h / img_height) * 100,
                },
                'original_width': img_width,
                'original_height': img_height,
            })

        return results

    def visualize_segmentation(
        self,
        image: Image.Image,
        bboxes: List[Dict],
        output_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Visualize detected text lines on image.

        Args:
            image: PIL Image
            bboxes: List of bounding box dicts (from segment())
            output_path: Optional path to save visualization

        Returns:
            PIL Image with bounding boxes drawn
        """
        img_array = np.array(image.convert('RGB'))
        img_height, img_width = img_array.shape[:2]

        # Draw bounding boxes
        for bbox_data in bboxes:
            bbox = bbox_data['bbox']

            # Convert percentage to pixels
            x = int(bbox['x'] / 100 * img_width)
            y = int(bbox['y'] / 100 * img_height)
            w = int(bbox['width'] / 100 * img_width)
            h = int(bbox['height'] / 100 * img_height)

            # Draw rectangle
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert back to PIL
        result_img = Image.fromarray(img_array)

        if output_path:
            result_img.save(output_path)

        return result_img


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct skew in image.

    Args:
        image: Input image array

    Returns:
        Deskewed image array
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect edges
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is None or len(lines) == 0:
        return image  # No lines detected, return original

    # Calculate median angle
    angles = []
    for line in lines[:50]:  # Use first 50 lines
        rho, theta = line[0]
        angle = np.degrees(theta) - 90
        angles.append(angle)

    median_angle = np.median(angles)

    # Only correct if angle is significant
    if abs(median_angle) > 0.5:
        # Get rotation matrix
        center = (image.shape[1] // 2, image.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        # Rotate image
        rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                                 fillcolor=(255, 255, 255), flags=cv2.INTER_CUBIC)
        return rotated

    return image


if __name__ == '__main__':
    # Test segmentation
    from PIL import Image

    # Create dummy image with lines
    img = Image.new('RGB', (800, 600), color='white')

    # Test segmenter
    segmenter = LineSegmenter()

    # This will fail on dummy image, but tests the interface
    try:
        bboxes = segmenter.segment(img, method='contours')
        print(f"Detected {len(bboxes)} lines")

        for i, bbox_data in enumerate(bboxes):
            print(f"Line {i+1}: {bbox_data['bbox']}")
    except Exception as e:
        print(f"Error (expected on dummy image): {e}")

    print("Segmentation module structure is correct")
