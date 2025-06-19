import os
import logging
from typing import Optional
import pandas as pd
from .detector import LegLengthDetector
from .processor import ImageProcessor
from .outputs import LegMeasurements
import torch

def run_inference(
    model_name: str,
    dicom_path: str,
    output_dir: str,
    confidence_threshold: float = 0.5,
    best_per_class: bool = True,
    logger: logging.Logger = None
) -> None:
    """
    Run inference on a DICOM image and save keypoint predictions.
    
    Args:
        model_name: Name of the model backbone to use (e.g., resnext101_32x8d, densenet201)
        dicom_path: Path to the DICOM image
        output_dir: Directory to save predictions
        confidence_threshold: Confidence threshold for predictions
        best_per_class: If True, returns only the highest confidence prediction for each class
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize detector and processor
        logger.info(f"Loading model: {model_name}")
        detector = LegLengthDetector.load_checkpoint(model_name)
        processor = ImageProcessor()
        
        # Preprocess image
        logger.info(f"Processing DICOM image: {dicom_path}")
        image_tensor = processor.preprocess_image(dicom_path)
        
        # Run inference
        logger.info("Running inference...")
        predictions = detector.predict(image_tensor, confidence_threshold, best_per_class)
        
        # Translate boxes back to original image space
        boxes = torch.tensor(predictions['boxes'])
        boxes = processor.translate_boxes_to_original(boxes)
        predictions['boxes'] = boxes.numpy()
        
        print(f"Best per class: {best_per_class}")
        print(predictions)
        
        # Get base name for output files
        base_name = os.path.splitext(os.path.basename(dicom_path))[0]
        
        # Calculate measurements
        logger.info("Calculating measurements...")
        measurements = LegMeasurements()
        results = measurements.calculate_distances(predictions, dicom_path)
        
        # Save JSON report
        json_path = os.path.join(output_dir, f"{base_name}_measurements.json")
        measurements.save_json_report(json_path)
        
        # Save DICOM SR
        sr_path = os.path.join(output_dir, f"{base_name}_measurements.dcm")
        measurements.create_structured_report(sr_path, dicom_path)
        
        # Create and save QA visualization
        qa_path = os.path.join(output_dir, f"{base_name}_qa.dcm")
        measurements.create_qa_dicom(predictions, dicom_path, qa_path, processor=processor)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise 