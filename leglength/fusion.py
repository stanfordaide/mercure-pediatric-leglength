#!/usr/bin/env python3
"""
Prediction fusion module for leg length detection.
Combines multiple model predictions into unified results using confidence-weighted centroids.
"""
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

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

def fuse_ensemble_predictions(model_results: Dict[str, Dict]) -> Dict:
    """
    Fuse predictions from multiple models into a single unified prediction.
    
    Args:
        model_results: Dict mapping model_name -> inference_results
        
    Returns:
        Unified prediction dict with fused coordinates and uncertainty metrics
    """
    logger = logging.getLogger(__name__)
    
    # Extract predictions from each model
    all_predictions = {}
    for model_name, results in model_results.items():
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
                        'description': f"Point {point_idx} has high spatial/confidence uncertainty"
                    })
                
                if len(models) < len(model_results):  # Not all models detected this point
                    fused_predictions['problematic_points'].append({
                        'point_id': int(point_idx),
                        'reason': 'detection_disagreement',
                        'models_detected': int(len(models)),
                        'total_models': int(len(model_results)),
                        'description': f"Point {point_idx} detected by only {len(models)}/{len(model_results)} models"
                    })
        else:
            # Point not detected by any model
            fused_predictions['problematic_points'].append({
                'point_id': int(point_idx),
                'reason': 'no_detection',
                'description': f"Point {point_idx} not detected by any model"
            })
    
    logger.info(f"Fused predictions from {len(model_results)} models")
    logger.info(f"Successfully fused {len(fused_predictions['boxes'])} points")
    logger.info(f"Identified {len(fused_predictions['problematic_points'])} problematic points")
    
    return fused_predictions

def create_single_model_prediction(inference_results: Dict) -> Dict:
    """
    Convert single model results to the same format as fused predictions for consistency.
    
    Args:
        inference_results: Results from single model inference
        
    Returns:
        Prediction dict in unified format
    """
    if 'predictions' not in inference_results:
        return {
            'boxes': [],
            'scores': [],
            'labels': [],
            'uncertainties': {},
            'point_statistics': {},
            'problematic_points': []
        }
    
    predictions = inference_results['predictions']
    
    # Convert to unified format - ensure all arrays are native Python types
    unified_predictions = {
        'boxes': [box.tolist() if hasattr(box, 'tolist') else list(box) for box in predictions.get('boxes', [])],
        'scores': [float(score) for score in predictions.get('scores', [])],
        'labels': [int(label) for label in predictions.get('labels', [])],
        'uncertainties': {},
        'point_statistics': {},
        'problematic_points': []
    }
    
    # Add basic uncertainty metrics for single model (no fusion uncertainty)
    for i, (box, score, label) in enumerate(zip(
        predictions.get('boxes', []),
        predictions.get('scores', []), 
        predictions.get('labels', [])
    )):
        point_idx = int(label)
        
        # Single model uncertainty is just based on confidence
        uncertainty_metrics = {
            'spatial_uncertainty': 0.0,  # No spatial uncertainty for single model
            'confidence_mean': float(score),
            'confidence_std': 0.0,
            'confidence_uncertainty': 0.0,
            'overall_uncertainty': 1.0 - float(score),  # Inverse of confidence
            'num_models': 1,
            'position_std_x': 0.0,
            'position_std_y': 0.0
        }
        
        unified_predictions['uncertainties'][point_idx] = uncertainty_metrics
        
        # Point statistics
        unified_predictions['point_statistics'][point_idx] = {
            'num_models_detected': int(1),
            'detecting_models': ['single_model'],
            'position_range_x': float(0.0),
            'position_range_y': float(0.0),
            'confidence_range': float(0.0)
        }
        
        # Check if point is problematic (low confidence)
        if score < 0.5:
            unified_predictions['problematic_points'].append({
                'point_id': int(point_idx),
                'reason': 'low_confidence',
                'confidence_score': float(score),
                'description': f"Point {point_idx} has low confidence ({score:.2f})"
            })
    
    return unified_predictions 