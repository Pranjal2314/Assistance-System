"""
Image Processing Module for Paper Quality Control System
Handles image preprocessing, enhancement, and basic transformations
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Process paper sheet images for quality analysis"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize image processor with configuration"""
        self.config = config or {}
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file path
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            logger.info(f"Successfully loaded image: {image_path}")
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for impurity detection
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to grayscale if configured
            if self.config.get('grayscale', True):
                processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                processed = image.copy()
            
            # Resize if dimensions specified
            resize_width = self.config.get('resize_width')
            resize_height = self.config.get('resize_height')
            if resize_width and resize_height:
                processed = cv2.resize(processed, (resize_width, resize_height))
            
            # Apply blur for noise reduction
            kernel_size = self.config.get('blur_kernel_size', 5)
            if kernel_size > 0:
                processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), 0)
            
            # Enhance contrast if configured
            if self.config.get('contrast_enhancement', True):
                processed = self.enhance_contrast(processed)
            
            # Apply histogram equalization if configured
            if self.config.get('histogram_equalization', True):
                processed = cv2.equalizeHist(processed)
            
            logger.debug("Image preprocessing completed")
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE
        
        Args:
            image: Input image
            
        Returns:
            Contrast enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in the image using Canny edge detection
        
        Args:
            image: Input image
            
        Returns:
            Edge detected image
        """
        sensitivity = self.config.get('edge_detection_sensitivity', 0.3)
        edges = cv2.Canny(image, 100 * sensitivity, 200 * sensitivity)
        return edges
    
    def calculate_brightness_stats(self, image: np.ndarray) -> Dict:
        """
        Calculate brightness statistics
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with brightness statistics
        """
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        stats = {
            'mean_brightness': np.mean(gray_image),
            'std_brightness': np.std(gray_image),
            'min_brightness': np.min(gray_image),
            'max_brightness': np.max(gray_image),
            'median_brightness': np.median(gray_image)
        }
        
        return stats
    
    def save_processed_image(self, image: np.ndarray, output_path: str):
        """
        Save processed image to file
        
        Args:
            image: Image to save
            output_path: Path to save the image
        """
        try:
            cv2.imwrite(output_path, image)
            logger.info(f"Saved processed image to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    processor = ImageProcessor()
    test_image = processor.load_image("sample_paper.jpg")
    processed = processor.preprocess_image(test_image)
    edges = processor.detect_edges(processed)
    stats = processor.calculate_brightness_stats(processed)
    print(f"Brightness Statistics: {stats}")