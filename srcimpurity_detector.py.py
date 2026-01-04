"""
Impurity Detection Module for Paper Quality Control System
Detects and analyzes impurities in paper sheets
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ImpurityType(Enum):
    """Types of impurities that can be detected"""
    SPOT = "spot"
    STREAK = "streak"
    DISCOLORATION = "discoloration"
    HOLE = "hole"
    FIBER_CLUMP = "fiber_clump"
    UNKNOWN = "unknown"

@dataclass
class Impurity:
    """Data class representing a detected impurity"""
    contour: np.ndarray
    area: float
    centroid: Tuple[float, float]
    bounding_box: Tuple[float, float, float, float]  # x, y, w, h
    impurity_type: ImpurityType
    intensity: float
    confidence: float

class PaperQualityInspector:
    """Main class for paper quality inspection and impurity detection"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the quality inspector"""
        self.config = self._load_config(config_path)
        self.impurities: List[Impurity] = []
        self.setup_logging()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file"""
        # In a real implementation, this would load from YAML/JSON
        default_config = {
            'min_impurity_size': 10,
            'max_impurity_size': 1000,
            'intensity_threshold': 30,
            'texture_analysis': True,
            'color_variation_threshold': 15
        }
        return default_config
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def analyze_paper(self, image_path: str) -> Dict:
        """
        Analyze paper sheet for impurities
        
        Args:
            image_path: Path to paper sheet image
            
        Returns:
            Dictionary with analysis results
        """
        try:
            logger.info(f"Starting analysis of paper sheet: {image_path}")
            
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Detect impurities
            self.impurities = self.detect_impurities(image)
            
            # Calculate quality metrics
            quality_score = self.calculate_quality_score(image)
            
            # Generate results
            results = {
                'image_path': image_path,
                'defect_count': len(self.impurities),
                'quality_score': quality_score,
                'is_passed': quality_score >= 95.0,  # 95% threshold
                'impurities': [
                    {
                        'type': impurity.impurity_type.value,
                        'area': impurity.area,
                        'confidence': impurity.confidence,
                        'position': impurity.centroid
                    }
                    for impurity in self.impurities
                ],
                'analysis_timestamp': cv2.getTickCount() / cv2.getTickFrequency()
            }
            
            logger.info(f"Analysis completed. Defects found: {len(self.impurities)}")
            logger.info(f"Quality score: {quality_score:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing paper sheet: {e}")
            raise
    
    def detect_impurities(self, image: np.ndarray) -> List[Impurity]:
        """
        Detect impurities in paper sheet
        
        Args:
            image: Input image
            
        Returns:
            List of detected impurities
        """
        impurities = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to identify potential impurities
            _, thresholded = cv2.threshold(
                gray, 
                self.config['intensity_threshold'], 
                255, 
                cv2.THRESH_BINARY_INV
            )
            
            # Find contours of potential impurities
            contours, _ = cv2.findContours(
                thresholded, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Analyze each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by size
                if (area >= self.config['min_impurity_size'] and 
                    area <= self.config['max_impurity_size']):
                    
                    # Calculate contour properties
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Determine impurity type
                        impurity_type = self.classify_impurity(contour, image)
                        
                        # Calculate intensity
                        intensity = self.calculate_impurity_intensity(contour, gray)
                        
                        # Create impurity object
                        impurity = Impurity(
                            contour=contour,
                            area=area,
                            centroid=(cx, cy),
                            bounding_box=(x, y, w, h),
                            impurity_type=impurity_type,
                            intensity=intensity,
                            confidence=self.calculate_confidence(contour, area, intensity)
                        )
                        
                        impurities.append(impurity)
            
            logger.info(f"Detected {len(impurities)} potential impurities")
            return impurities
            
        except Exception as e:
            logger.error(f"Error detecting impurities: {e}")
            raise
    
    def classify_impurity(self, contour: np.ndarray, image: np.ndarray) -> ImpurityType:
        """
        Classify the type of impurity
        
        Args:
            contour: Contour of impurity
            image: Original image
            
        Returns:
            ImpurityType classification
        """
        try:
            # Calculate contour properties for classification
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate circularity
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            # Get aspect ratio from bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Classify based on shape properties
            if circularity > 0.8:
                return ImpurityType.SPOT
            elif aspect_ratio > 3 or aspect_ratio < 0.33:
                return ImpurityType.STREAK
            elif area > 500:  # Large area
                return ImpurityType.DISCOLORATION
            else:
                return ImpurityType.UNKNOWN
                
        except Exception as e:
            logger.warning(f"Error classifying impurity: {e}")
            return ImpurityType.UNKNOWN
    
    def calculate_impurity_intensity(self, contour: np.ndarray, gray_image: np.ndarray) -> float:
        """
        Calculate intensity of impurity
        
        Args:
            contour: Contour of impurity
            gray_image: Grayscale image
            
        Returns:
            Average intensity within contour
        """
        # Create mask for the contour
        mask = np.zeros(gray_image.shape, np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        # Calculate mean intensity within contour
        mean_intensity = cv2.mean(gray_image, mask=mask)[0]
        
        return mean_intensity
    
    def calculate_confidence(self, contour: np.ndarray, area: float, intensity: float) -> float:
        """
        Calculate detection confidence
        
        Args:
            contour: Contour of impurity
            area: Area of impurity
            intensity: Intensity of impurity
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence on area (normalized)
        area_confidence = min(area / 100, 1.0)
        
        # Confidence based on intensity difference
        intensity_confidence = min(intensity / 100, 1.0)
        
        # Combine confidences
        confidence = (area_confidence * 0.6 + intensity_confidence * 0.4)
        
        return min(confidence, 1.0)
    
    def calculate_quality_score(self, image: np.ndarray) -> float:
        """
        Calculate overall quality score for paper sheet
        
        Args:
            image: Input image
            
        Returns:
            Quality score percentage (0-100)
        """
        try:
            base_score = 100.0
            
            # Deduct points for each impurity
            impurity_deduction = min(len(self.impurities) * 5, 30)
            
            # Deduct points for large impurities
            large_impurity_deduction = 0
            for impurity in self.impurities:
                if impurity.area > 100:  # Large impurity threshold
                    large_impurity_deduction += 2
            
            # Calculate uniformity score
            uniformity_score = self.calculate_uniformity_score(image)
            
            # Calculate final score
            final_score = (
                base_score 
                - impurity_deduction 
                - large_impurity_deduction 
                + uniformity_score * 0.3
            )
            
            return max(0, min(100, final_score))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def calculate_uniformity_score(self, image: np.ndarray) -> float:
        """
        Calculate uniformity score based on texture analysis
        
        Args:
            image: Input image
            
        Returns:
            Uniformity score (0-100)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate standard deviation (lower = more uniform)
            std_dev = np.std(gray)
            
            # Normalize to 0-100 scale (inverse relationship)
            # Lower std_dev should give higher score
            uniformity_score = max(0, 100 - std_dev / 2.55)  # Scale appropriately
            
            return uniformity_score
            
        except Exception:
            return 80.0  # Default score if calculation fails
    
    def visualize_results(self, image: np.ndarray, output_path: str):
        """
        Visualize detected impurities on the image
        
        Args:
            image: Original image
            output_path: Path to save visualization
        """
        try:
            visualization = image.copy()
            
            # Draw each impurity
            for impurity in self.impurities:
                # Draw contour
                cv2.drawContours(visualization, [impurity.contour], -1, (0, 0, 255), 2)
                
                # Draw bounding box
                x, y, w, h = impurity.bounding_box
                cv2.rectangle(visualization, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Draw centroid
                cv2.circle(visualization, 
                          (int(impurity.centroid[0]), int(impurity.centroid[1])), 
                          5, (0, 255, 0), -1)
                
                # Add label
                label = f"{impurity.impurity_type.value}: {impurity.area:.0f}px"
                cv2.putText(visualization, label, 
                           (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
            
            # Save visualization
            cv2.imwrite(output_path, visualization)
            logger.info(f"Saved visualization to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    inspector = PaperQualityInspector()
    results = inspector.analyze_paper("sample_paper.jpg")
    print(f"Analysis Results: {results}")