"""
Utility functions for image processing
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def normalize_image(image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Normalize image for consistent processing
    
    Args:
        image: Input image
        target_size: Target size for resizing (width, height)
        
    Returns:
        Normalized image
    """
    normalized = image.copy()
    
    # Convert to grayscale if needed
    if len(normalized.shape) == 3:
        normalized = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    
    # Resize if target size specified
    if target_size:
        normalized = cv2.resize(normalized, target_size)
    
    # Normalize pixel values to 0-255
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized

def calculate_image_statistics(image: np.ndarray) -> Dict:
    """
    Calculate comprehensive image statistics
    
    Args:
        image: Input image
        
    Returns:
        Dictionary of image statistics
    """
    stats = {}
    
    if len(image.shape) == 3:
        # Color