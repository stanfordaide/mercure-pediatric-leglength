import cv2
import torch
import pydicom
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def get_pixel_spacing(dicom_dataset):
    """
    Get pixel spacing from DICOM dataset, trying multiple fields.
    
    Args:
        dicom_dataset: PyDICOM dataset
        
    Returns:
        tuple: (pixel_spacing_x, pixel_spacing_y) or None if not found
    """
    # Try PixelSpacing first (most common)
    if hasattr(dicom_dataset, 'PixelSpacing') and dicom_dataset.PixelSpacing:
        return dicom_dataset.PixelSpacing
    
    # Try ImagerPixelSpacing as fallback
    if hasattr(dicom_dataset, 'ImagerPixelSpacing') and dicom_dataset.ImagerPixelSpacing:
        return dicom_dataset.ImagerPixelSpacing
    
    return None

"""Image processing helpers with logging.

This module is responsible for preparing DICOM images for model inference.
We add consistent debug logs at key steps to aid troubleshooting and
observability.
"""

class ImageProcessor:
    def __init__(self, target_size=512):
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.logger = logging.getLogger(__name__)
        self.original_height = None
        self.original_width = None
        self.pixel_spacing = None
        self.dicom = None

    def load_dicom(self, dicom_path: str) -> np.ndarray:
        """Load DICOM image and set metadata."""
        self.dicom = pydicom.dcmread(dicom_path)
        self.logger.debug(f"Loaded DICOM: {dicom_path}, shape={self.dicom.pixel_array.shape}")
        image = self.dicom.pixel_array
        self.original_height, self.original_width = image.shape
        self.pixel_spacing = get_pixel_spacing(self.dicom)
        if self.pixel_spacing is None:
            self.logger.warning(f"No PixelSpacing or ImagerPixelSpacing found in DICOM: {dicom_path}")
        self.logger.debug(f"Pixel spacing: {self.pixel_spacing}, original size: {self.original_height}x{self.original_width}")
        return image

    def preprocess_image(self, dicom_path: str) -> torch.Tensor:
        """Preprocess DICOM image for model input."""
        image = self.load_dicom(dicom_path)
        self.logger.debug("Starting image preprocessing")
        image, (pad_h, pad_w) = self.adaptive_pad(image)
        image = self.enhance_contrast(image)
        image = torch.from_numpy(image).float()
        image = self.clip_and_stretch(image)
        image = image.repeat(3, 1, 1)
        self.logger.debug("Finished image preprocessing")
        return image

    def adaptive_pad(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        h, w = image.shape
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        if max_dim != self.target_size:
            padded_image = cv2.resize(padded_image, (self.target_size, self.target_size))
        return padded_image, (pad_h, pad_w)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        image_float = image.astype(np.float32)
        image_norm = (image_float - image_float.min()) / (image_float.max() - image_float.min())
        image_clahe = self.clahe.apply((image_norm * 255).astype(np.uint8))
        image_enhanced = image_clahe.astype(np.float32) / 255.0
        return image_enhanced

    def clip_and_stretch(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.numpy()
        lower_quartile = np.quantile(x_np, 0.25)
        upper_quartile = np.quantile(x_np, 0.99)
        clipped = torch.clamp(x, min=float(lower_quartile), max=float(upper_quartile))
        stretched = (clipped - lower_quartile) / (upper_quartile - lower_quartile)
        return stretched

    def translate_boxes_to_original(self, boxes: torch.Tensor) -> torch.Tensor:
        # Store the original and target dimensions for debugging
        self.logger.debug(f"Original dimensions: {self.original_height}x{self.original_width}")
        self.logger.debug(f"Target size: {self.target_size}")
        
        # Calculate the scale factor based on how the image was resized during preprocessing
        # In adaptive_pad, we resize the max dimension to target_size
        scale_factor = self.target_size / max(self.original_height, self.original_width)
        self.logger.debug(f"Scale factor: {scale_factor}")
        
        # Remember the boxes might have been detected on a padded image
        # We need to account for padding before scaling back
        original_boxes = boxes.clone()
        
        # Scale back to original dimensions
        original_boxes = original_boxes / scale_factor
        
        # Clamp to ensure we're within image boundaries
        original_boxes[:, [0, 2]] = torch.clamp(original_boxes[:, [0, 2]], 0, self.original_width)
        original_boxes[:, [1, 3]] = torch.clamp(original_boxes[:, [1, 3]], 0, self.original_height)
        
        return original_boxes 