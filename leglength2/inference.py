import os
import logging
from typing import Optional
import pandas as pd
from .detector import LegLengthDetector
from .processor import ImageProcessor
from .measurements import LegMeasurements
import torch
from typing import Dict, List, Tuple, Optional
import time
import numpy as np



def calculate_weighted_centroid(positions: List[Tuple[float, float]], 
                               confidences: List[float]) -> Tuple[float, float, Dict]:
    """
    Calculate confidence-weighted centroid of multiple point predictions.
    
    Args:
        positions: List of (x, y) coordinates
        confidences: List of confidence scores for each position
        
    Returns:
        Tuple of (weighted_x, weighted_y, uncertainty_metrics)
    """
    if not positions or not confidences:
        return None, None, {}
    
    positions = np.array(positions)
    confidences = np.array(confidences)
    
    # Normalize confidences to sum to 1
    weights = confidences / np.sum(confidences)
    
    # Calculate weighted centroid
    weighted_x = np.sum(positions[:, 0] * weights)
    weighted_y = np.sum(positions[:, 1] * weights)
    
    # Calculate uncertainty metrics
    # Standard deviation of positions (spatial uncertainty)
    std_x = np.sqrt(np.sum(weights * (positions[:, 0] - weighted_x)**2))
    std_y = np.sqrt(np.sum(weights * (positions[:, 1] - weighted_y)**2))
    spatial_uncertainty = np.sqrt(std_x**2 + std_y**2)
    
    # Confidence uncertainty (spread in confidence values)
    confidence_mean = np.mean(confidences)
    confidence_std = np.std(confidences)
    confidence_uncertainty = confidence_std / confidence_mean if confidence_mean > 0 else 0
    
    # Overall uncertainty (combination of spatial and confidence uncertainty)
    overall_uncertainty = spatial_uncertainty + confidence_uncertainty * 10  # Scale confidence uncertainty
    
    uncertainty_metrics = {
        'spatial_uncertainty': float(spatial_uncertainty),
        'confidence_mean': float(confidence_mean),
        'confidence_std': float(confidence_std),
        'confidence_uncertainty': float(confidence_uncertainty),
        'overall_uncertainty': float(overall_uncertainty),
        'num_models': len(positions),
        'position_std_x': float(std_x),
        'position_std_y': float(std_y)
    }
    
    return float(weighted_x), float(weighted_y), uncertainty_metrics



def infer(
    model_name: str,
    dicom_path: str,
    output_dir: str,
    confidence_threshold: float = 0.5,
    best_per_class: bool = True,
    logger: logging.Logger = None,
) -> dict:
    """
    Run inference on a DICOM image and save keypoint predictions.
    
    Args:
        model_name: Name of the model backbone to use (e.g., resnext101_32x8d, densenet201)
        dicom_path: Path to the DICOM image
        output_dir: Directory to save predictions
        confidence_threshold: Confidence threshold for predictions
        best_per_class: If True, returns only the highest confidence prediction for each class
        
    Returns:
        Dict containing measurements, predictions, and metadata
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
        
        # Return data for ensemble analysis
        return {
            'measurements': results,
            'predictions': predictions,
            'pixel_spacing': measurements.pixel_spacing,
            'base_name': base_name
        }
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise
        
    
def run_inference(
    models: list[str],
    dicom_path: str,
    output_dir: str,
    confidence_threshold: float = 0.5,
    best_per_class: bool = True,
    logger: logging.Logger = None,
):
    logger = logging.getLogger(__name__)
    measurements = []
    start_time = time.time()
    logger.info(f"AAAAH Running inference for models: {models}")
    for model_name in models:
        measurements.append({model_name: infer(model_name, dicom_path, output_dir, confidence_threshold, best_per_class, logger)})
        
    fused_measurements = fuse_measurements(models, measurements, logger)
        

    
        
    
    end_time = time.time()
    logger.info(f"Inference time: {end_time - start_time} seconds")

    return fused_measurements



def fuse_measurements(
    models, model_measurements, logger
):
    logger = logging.getLogger(__name__)

    if len (models) == 1:
        logger.info(f"Only one model provided, returning measurements for {models[0]}")
        return model_measurements[0]
    
    else:
        logger.info(f"Fusing measurements for {models}")
        
        
        print(f"AAHH Model measurements: {model_measurements}")
        
        all_predictions = {}
        for model_name, results in model_measurements.items():
            if 'predictions' in results:
                predictions = results['predictions']
                boxes = predictions.get('boxes', [])
                scores = predictions.get('scores', [])
                labels = predictions.get('labels', [])
                
                # Convert to point-wise predictions
                for box, score, label in zip(boxes, scores, labels):
                    point_idx = int(label)
                    x_center = float((box[0] + box[2]) / 2)
                    y_center = float((box[1] + box[3]) / 2)
                    
                    if point_idx not in all_predictions:
                        all_predictions[point_idx] = {'positions': [], 'confidences': [], 'models': []}
                    
                    all_predictions[point_idx]['positions'].append((x_center, y_center))
                    all_predictions[point_idx]['confidences'].append(float(score))
                    all_predictions[point_idx]['models'].append(model_name)
            
        # Fuse predictions for each point
        fused_predictions = {
            'boxes': [],
            'scores': [],
            'labels': [],
            'uncertainties': {},
            'point_statistics': {},
            'problematic_points': []
        }
        
        for point_idx in range(1, 9):  # Points 1-8
            if point_idx in all_predictions:
                positions = all_predictions[point_idx]['positions']
                confidences = all_predictions[point_idx]['confidences']
                models = all_predictions[point_idx]['models']
                
                
                # Calculate weighted centroid
                fused_x, fused_y, uncertainty_metrics = calculate_weighted_centroid(positions, confidences)
                
                if fused_x is not None and fused_y is not None:
                    # Create fused bounding box (using small fixed size around centroid)
                    box_size = 10  # pixels
                    fused_box = [
                        float(fused_x - box_size/2),
                        float(fused_y - box_size/2), 
                        float(fused_x + box_size/2),
                        float(fused_y + box_size/2)
                    ]
                    
                    fused_predictions['boxes'].append(fused_box)
                    fused_predictions['scores'].append(float(uncertainty_metrics['confidence_mean']))
                    fused_predictions['labels'].append(int(point_idx))
                    fused_predictions['uncertainties'][point_idx] = uncertainty_metrics
                
                    # Point statistics
                    fused_predictions['point_statistics'][point_idx] = {
                        'num_models_detected': int(len(models)),
                        'detecting_models': models,
                        'position_range_x': float(max(pos[0] for pos in positions) - min(pos[0] for pos in positions)),
                        'position_range_y': float(max(pos[1] for pos in positions) - min(pos[1] for pos in positions)),
                        'confidence_range': float(max(confidences) - min(confidences))
                    }
                    
                    # Check if point is problematic
                    if uncertainty_metrics['overall_uncertainty'] > 15.0:  # Threshold for problematic points
                        fused_predictions['problematic_points'].append({
                            'point_id': int(point_idx),
                            'reason': 'high_uncertainty',
                            'uncertainty_score': float(uncertainty_metrics['overall_uncertainty']),
                            'models_detected': models
                        })
                        
                        
                    if len(models) < len(model_measurements):  # Not all models detected this point
                        fused_predictions['problematic_points'].append({
                            'point_id': int(point_idx),
                            'reason': 'detection_disagreement',
                            'models_detected': int(len(models)),
                            'total_models': int(len(model_measurements)),
                            'description': f"Point {point_idx} detected by only {len(models)}/{len(model_measurements)} models"
                        })
            else:
                # Point not detected by any model
                fused_predictions['problematic_points'].append({
                    'point_id': int(point_idx),
                    'reason': 'no_detection',
                    'description': f"Point {point_idx} not detected by any model"
                })
        
        logger.info(f"Fused predictions from {len(model_measurements)} models")
        logger.info(f"Successfully fused {len(fused_predictions['boxes'])} points")
        logger.info(f"Identified {len(fused_predictions['problematic_points'])} problematic points")
        
        return fused_predictions

                        
