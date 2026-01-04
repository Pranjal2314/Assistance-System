"""
Quality Analyzer Module for Paper Quality Control System
Handles batch processing and quality reporting
"""

import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Process multiple paper sheet images in batch"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize batch processor"""
        self.config = self._load_config(config_path)
        self.results: List[Dict] = []
        self.setup_logging()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file"""
        default_config = {
            'batch_size': 10,
            'output_format': 'csv',
            'save_individual_reports': True,
            'generate_summary': True
        }
        return default_config
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def process_folder(self, folder_path: str) -> List[Dict]:
        """
        Process all paper sheet images in a folder
        
        Args:
            folder_path: Path to folder containing images
            
        Returns:
            List of analysis results for each image
        """
        try:
            logger.info(f"Processing folder: {folder_path}")
            
            # Get all image files in folder
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = []
            
            for file in os.listdir(folder_path):
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(os.path.join(folder_path, file))
            
            logger.info(f"Found {len(image_files)} image files")
            
            # Process each image
            for i, image_file in enumerate(image_files, 1):
                try:
                    logger.info(f"Processing image {i}/{len(image_files)}: {image_file}")
                    
                    # Analyze paper sheet
                    from src.impurity_detector import PaperQualityInspector
                    inspector = PaperQualityInspector()
                    result = inspector.analyze_paper(image_file)
                    
                    # Add batch processing metadata
                    result['batch_id'] = datetime.now().strftime('%Y%m%d_%H%M%S')
                    result['processing_timestamp'] = datetime.now().isoformat()
                    
                    self.results.append(result)
                    
                    logger.info(f"Completed: {image_file} - "
                              f"Score: {result['quality_score']:.2f}% - "
                              f"Defects: {result['defect_count']}")
                    
                except Exception as e:
                    logger.error(f"Error processing {image_file}: {e}")
                    continue
            
            logger.info(f"Batch processing completed. Processed {len(self.results)} images")
            return self.results
            
        except Exception as e:
            logger.error(f"Error processing folder {folder_path}: {e}")
            raise
    
    def generate_report(self, results: List[Dict], output_path: str):
        """
        Generate quality control report
        
        Args:
            results: List of analysis results
            output_path: Path to save the report
        """
        try:
            logger.info(f"Generating report to: {output_path}")
            
            # Convert results to DataFrame
            df = pd.DataFrame(results)
            
            # Extract impurity details
            impurity_data = []
            for result in results:
                for impurity in result.get('impurities', []):
                    impurity_data.append({
                        'image_path': result['image_path'],
                        'impurity_type': impurity['type'],
                        'area': impurity['area'],
                        'confidence': impurity['confidence'],
                        'position_x': impurity['position'][0],
                        'position_y': impurity['position'][1],
                        'quality_score': result['quality_score'],
                        'batch_id': result.get('batch_id', 'N/A')
                    })
            
            impurity_df = pd.DataFrame(impurity_data)
            
            # Generate summary statistics
            summary = {
                'total_images': len(results),
                'passed_images': sum(1 for r in results if r.get('is_passed', False)),
                'failed_images': sum(1 for r in results if not r.get('is_passed', False)),
                'average_quality_score': df['quality_score'].mean() if not df.empty else 0,
                'total_defects': df['defect_count'].sum() if not df.empty else 0,
                'defect_rate': df['defect_count'].mean() if not df.empty else 0,
                'pass_rate': (sum(1 for r in results if r.get('is_passed', False)) / 
                            len(results) * 100 if results else 0)
            }
            
            # Save reports
            if output_path.endswith('.csv'):
                df.to_csv(output_path, index=False)
                logger.info(f"Saved main report to: {output_path}")
                
                # Save impurity details to separate file
                impurity_output = output_path.replace('.csv', '_impurities.csv')
                impurity_df.to_csv(impurity_output, index=False)
                logger.info(f"Saved impurity details to: {impurity_output}")
                
                # Save summary to separate file
                summary_output = output_path.replace('.csv', '_summary.txt')
                with open(summary_output, 'w') as f:
                    f.write("PAPER QUALITY CONTROL REPORT SUMMARY\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total Images Processed: {summary['total_images']}\n")
                    f.write(f"Passed Images: {summary['passed_images']}\n")
                    f.write(f"Failed Images: {summary['failed_images']}\n")
                    f.write(f"Pass Rate: {summary['pass_rate']:.2f}%\n")
                    f.write(f"Average Quality Score: {summary['average_quality_score']:.2f}%\n")
                    f.write(f"Total Defects Found: {summary['total_defects']}\n")
                    f.write(f"Average Defects per Image: {summary['defect_rate']:.2f}\n")
                logger.info(f"Saved summary to: {summary_output}")
            
            elif output_path.endswith('.xlsx'):
                with pd.ExcelWriter(output_path) as writer:
                    df.to_excel(writer, sheet_name='Quality Results', index=False)
                    impurity_df.to_excel(writer, sheet_name='Impurity Details', index=False)
                    
                    # Create summary sheet
                    summary_df = pd.DataFrame([summary])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                logger.info(f"Saved Excel report to: {output_path}")
            
            else:
                raise ValueError(f"Unsupported output format: {output_path}")
            
            logger.info("Report generation completed successfully")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    def analyze_trends(self, results: List[Dict]) -> Dict:
        """
        Analyze trends in quality data
        
        Args:
            results: List of analysis results
            
        Returns:
            Dictionary with trend analysis
        """
        try:
            if not results:
                return {}
            
            df = pd.DataFrame(results)
            
            trends = {
                'quality_trend': {
                    'average': df['quality_score'].mean(),
                    'std_dev': df['quality_score'].std(),
                    'min': df['quality_score'].min(),
                    'max': df['quality_score'].max(),
                    'trend': 'improving' if len(results) > 1 and 
                            df['quality_score'].iloc[-1] > df['quality_score'].iloc[0] 
                            else 'declining' if len(results) > 1 else 'stable'
                },
                'defect_trend': {
                    'total_defects': df['defect_count'].sum(),
                    'average_defects': df['defect_count'].mean(),
                    'max_defects': df['defect_count'].max(),
                    'defect_types': {}
                },
                'pass_rate_trend': {
                    'current_rate': (df['is_passed'].sum() / len(df)) * 100,
                    'target_rate': 95.0  # Configurable target
                }
            }
            
            # Analyze defect types
            for result in results:
                for impurity in result.get('impurities', []):
                    defect_type = impurity['type']
                    trends['defect_trend']['defect_types'][defect_type] = \
                        trends['defect_trend']['defect_types'].get(defect_type, 0) + 1
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {}


class RealTimeMonitor:
    """Monitor paper quality in real-time"""
    
    def __init__(self, camera_index: int = 0):
        """Initialize real-time monitor"""
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def start_monitoring(self):
        """Start real-time quality monitoring"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open camera {self.camera_index}")
            
            self.running = True
            logger.info(f"Started real-time monitoring on camera {self.camera_index}")
            
            from src.impurity_detector import PaperQualityInspector
            inspector = PaperQualityInspector()
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    break
                
                # Analyze current frame
                # Note: In a real implementation, we would save frame and analyze
                # For demo, we'll just display
                
                cv2.imshow('Paper Quality Monitor', frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        except Exception as e:
            logger.error(f"Error in real-time monitoring: {e}")
        finally:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Real-time monitoring stopped")


if __name__ == "__main__":
    # Example usage
    processor = BatchProcessor()
    
    # Process a folder of images
    results = processor.process_folder("paper_samples/")
    
    # Generate report
    processor.generate_report(results, "quality_report.csv")
    
    # Analyze trends
    trends = processor.analyze_trends(results)
    print(f"Quality Trends: {trends}")