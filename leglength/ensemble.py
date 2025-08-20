#!/usr/bin/env python3
"""
Ensemble inference module for leg length detection.
Provides multi-model inference and disagreement metrics calculation.
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List
from .inference import run_inference
from .fusion import fuse_ensemble_predictions
from .unified_outputs import UnifiedOutputGenerator
from .outputs import LegMeasurements

# Default ensemble models to use
DEFAULT_ENSEMBLE_MODELS = ['convnext_base', 'resnext101_32x8d', 'vit_l_16']

def extract_predictions_from_measurements(inference_results: Dict) -> Dict:
    """
    Extract predictions in the format expected by disagreement calculation.
    This converts from the inference results format to the expected format.
    
    Args:
        inference_results: Dict containing 'predictions', 'measurements', 'pixel_spacing'
    
    Returns:
        Dict with p1_x, p1_y, p1_conf, p2_x, p2_y, p2_conf, ... format
    """
    predictions_dict = {}
    
    if 'predictions' not in inference_results:
        return predictions_dict
    
    predictions = inference_results['predictions']
    boxes = predictions.get('boxes', [])
    scores = predictions.get('scores', [])
    labels = predictions.get('labels', [])
    
    # Extract keypoints from bounding boxes
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        # Calculate center point of bounding box
        x_center = float((box[0] + box[2]) / 2)
        y_center = float((box[1] + box[3]) / 2)
        
        # Map label to point index (assuming labels are 1-8 for points)
        point_idx = int(label)
        
        # Store in the expected format
        predictions_dict[f'p{point_idx}_x'] = float(x_center)
        predictions_dict[f'p{point_idx}_y'] = float(y_center)
        predictions_dict[f'p{point_idx}_conf'] = float(score)
    
    return predictions_dict

def calculate_disagreement_metrics_for_image(predictions_dict: Dict[str, Dict], 
                                           pixel_spacing: float = 1.0,
                                           confidence_threshold: float = 0.1,
                                           detection_weight: float = 0.5,
                                           outlier_weight: float = 0.35,
                                           localization_weight: float = 0.15) -> Dict:
    """
    Calculate disagreement metrics for a single image across all models.
    
    Args:
        predictions_dict: Dict mapping model_name -> {p1_x, p1_y, p1_conf, ...}
        pixel_spacing: Pixel spacing for mm conversion
        confidence_threshold: Minimum confidence to consider prediction valid
        
    Returns:
        Dict with disagreement metrics for this image
    """
    
    # Clinical thresholds
    SPATIAL_THRESHOLD_MM = 2.0
    BLUNDER_THRESHOLD_MM = 50.0
    spatial_threshold_pixels = SPATIAL_THRESHOLD_MM / pixel_spacing
    blunder_threshold_pixels = BLUNDER_THRESHOLD_MM / pixel_spacing
    
    # Extract valid predictions for each point
    valid_models = []
    model_points = {}
    
    for model_name, preds in predictions_dict.items():
        model_keypoints = {}
        
        # Extract predictions for each point (P1-P8)
        for point_idx in range(1, 9):
            x_key = f'p{point_idx}_x'
            y_key = f'p{point_idx}_y'
            conf_key = f'p{point_idx}_conf'
            
            try:
                x_val = preds.get(x_key, None)
                y_val = preds.get(y_key, None)
                conf_val = preds.get(conf_key, None)
                
                # Check if prediction is valid and meets confidence threshold
                if (x_val is not None and y_val is not None and conf_val is not None and
                    not pd.isna(x_val) and not pd.isna(y_val) and not pd.isna(conf_val) and
                    conf_val >= confidence_threshold):
                    
                    model_keypoints[point_idx] = {
                        'x': float(x_val),
                        'y': float(y_val),
                        'conf': float(conf_val)
                    }
            except Exception:
                continue
        
        if model_keypoints:
            model_points[model_name] = model_keypoints
            valid_models.append(model_name)
    
    if len(valid_models) < 2:
        # Need at least 2 models for disagreement calculation
        return {
            'detection_disagreement_score': float('nan'),
            'localization_disagreement_score': float('nan'),
            'outlier_risk_score': float('nan'),
            'overall_disagreement_score': float('nan'),
            'num_models_contributing': int(len(valid_models)),
            'pixel_spacing': float(pixel_spacing)
        }
    
    # Calculate disagreement metrics
    total_points = 8  # P1-P8
    points_detected_by_all = 0
    point_localization_disagreements = []
    point_outlier_risks = []
    
    for point_idx in range(1, 9):
        # Check which models detected this point
        models_detecting_point = []
        point_positions = []
        
        for model_name in valid_models:
            if point_idx in model_points[model_name]:
                models_detecting_point.append(model_name)
                point_positions.append((
                    model_points[model_name][point_idx]['x'], 
                    model_points[model_name][point_idx]['y']
                ))
        
        # Detection agreement: all models must detect the point
        if len(models_detecting_point) == len(valid_models):
            points_detected_by_all += 1
            
            # Calculate localization and outlier metrics for this point
            if len(point_positions) >= 2:
                # Localization disagreement for this point
                pairwise_distances = []
                for i in range(len(point_positions)):
                    for j in range(i + 1, len(point_positions)):
                        dx = point_positions[i][0] - point_positions[j][0]
                        dy = point_positions[i][1] - point_positions[j][1]
                        distance = np.sqrt(dx**2 + dy**2)
                        pairwise_distances.append(distance)
                
                if pairwise_distances:
                    # Count pairs within spatial threshold
                    pairs_within_threshold = sum(1 for d in pairwise_distances if d <= spatial_threshold_pixels)
                    total_pairs = len(pairwise_distances)
                    localization_agreement = pairs_within_threshold / total_pairs
                    point_localization_disagreement = 1.0 - localization_agreement
                    point_localization_disagreements.append(point_localization_disagreement)
                    
                    # Outlier risk for this point
                    max_distance = max(pairwise_distances)
                    max_distance_mm = max_distance * pixel_spacing
                    outlier_risk = min(max_distance_mm / BLUNDER_THRESHOLD_MM, 1.0)
                    point_outlier_risks.append(outlier_risk)
    
    # Calculate final scores
    detection_disagreement_score = 1.0 - (points_detected_by_all / total_points)
    
    localization_disagreement_score = (np.mean(point_localization_disagreements) 
                                     if point_localization_disagreements else 0.0)
    
    outlier_risk_score = (np.mean(point_outlier_risks) 
                        if point_outlier_risks else 0.0)
    
    # Overall disagreement score (weighted combination)
    # Validate weights sum to 1.0 for proper interpretation
    total_weight = detection_weight + outlier_weight + localization_weight
    if abs(total_weight - 1.0) > 0.001:
        # Normalize weights if they don't sum to 1.0
        detection_weight /= total_weight
        outlier_weight /= total_weight
        localization_weight /= total_weight
    
    overall_disagreement_score = (detection_weight * detection_disagreement_score + 
                                outlier_weight * outlier_risk_score +
                                localization_weight * localization_disagreement_score)
    
    return {
        'detection_disagreement_score': float(detection_disagreement_score),
        'localization_disagreement_score': float(localization_disagreement_score),
        'outlier_risk_score': float(outlier_risk_score),
        'overall_disagreement_score': float(overall_disagreement_score),
        'points_detected_by_all': int(points_detected_by_all),
        'num_models_contributing': int(len(valid_models)),
        'pixel_spacing': float(pixel_spacing)
    }

def run_ensemble_inference(
    models: List[str],
    dicom_path: str,
    output_dir: str,
    confidence_threshold: float = 0.0,
    best_per_class: bool = True,
    enable_disagreement: bool = True,
    detection_weight: float = 0.5,
    outlier_weight: float = 0.35,
    localization_weight: float = 0.15,
    logger: logging.Logger = None,
    discrepancy_threshold_cm: float = 2.0
) -> Dict:
    """
    Run ensemble inference with multiple models and generate unified outputs.
    
    Args:
        models: List of model names to use in ensemble
        dicom_path: Path to DICOM image
        output_dir: Directory to save predictions
        confidence_threshold: Confidence threshold for predictions
        best_per_class: Return only highest confidence prediction for each class
        enable_disagreement: Enable disagreement metrics calculation
        detection_weight: Weight for detection disagreement in overall score
        outlier_weight: Weight for outlier risk in overall score
        localization_weight: Weight for localization disagreement in overall score
        logger: Logger instance
        
    Returns:
        Dict containing ensemble results and disagreement metrics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Starting ensemble inference with {len(models)} models")
    logger.info(f"Models: {models}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Store predictions from each model in temporary directories
    model_inference_results = {}
    temp_dir = os.path.join(output_dir, 'temp_models')

    # Run inference with each model
    for model_name in models:
        logger.info(f"Running inference with model: {model_name}")
        
        # Create model-specific temporary directory
        model_output_dir = os.path.join(temp_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        try:
            # Run inference for this model
            inference_results = run_inference(
                model_name=model_name,
                dicom_path=dicom_path,
                output_dir=model_output_dir,
                confidence_threshold=confidence_threshold,
                best_per_class=best_per_class,
                logger=logger
            )
            
            model_inference_results[model_name] = inference_results
            logger.info(f"Successfully completed inference with {model_name}")
            
        except Exception as e:
            logger.error(f"Error running inference with {model_name}: {str(e)}")
            continue

    # Fuse ensemble predictions
    logger.info("Fusing ensemble predictions...")
    fused_predictions = fuse_ensemble_predictions(model_inference_results)
    
    # Calculate measurements using fused predictions
    logger.info("Calculating measurements from fused predictions...")
    measurements = LegMeasurements()
    measurements_data = measurements.calculate_distances(fused_predictions, dicom_path, discrepancy_threshold_cm)
    
    # Calculate disagreement metrics if enabled
    disagreement_metrics = None
    if enable_disagreement and len(model_inference_results) >= 2:
        logger.info("Calculating disagreement metrics...")
        
        try:
            # Convert measurements to format expected by disagreement calculation
            predictions_dict = {}
            pixel_spacing = 1.0  # Default
            
            for model_name, inference_results in model_inference_results.items():
                # Extract predictions in the expected format
                predictions_dict[model_name] = extract_predictions_from_measurements(inference_results)
                
                # Extract pixel spacing from the first model
                if pixel_spacing == 1.0 and inference_results.get('pixel_spacing'):
                    pixel_spacing = float(inference_results['pixel_spacing'][0])  # Use first value
            
            # Calculate disagreement metrics
            disagreement_metrics = calculate_disagreement_metrics_for_image(
                predictions_dict=predictions_dict,
                pixel_spacing=pixel_spacing,
                confidence_threshold=confidence_threshold,
                detection_weight=detection_weight,
                outlier_weight=outlier_weight,
                localization_weight=localization_weight
            )
            
            # Log disagreement summary
            if not np.isnan(disagreement_metrics['overall_disagreement_score']):
                logger.info(f"Overall disagreement score: {disagreement_metrics['overall_disagreement_score']:.3f}")
                logger.info(f"Detection disagreement: {disagreement_metrics['detection_disagreement_score']:.3f}")
                logger.info(f"Localization disagreement: {disagreement_metrics['localization_disagreement_score']:.3f}")
                logger.info(f"Outlier risk: {disagreement_metrics['outlier_risk_score']:.3f}")
                
                if disagreement_metrics['detection_disagreement_score'] > 0.0:
                    logger.warning("⚠️  Detection disagreement detected - expert review recommended")
                
                if disagreement_metrics['overall_disagreement_score'] > 0.5:
                    logger.warning("⚠️  High uncertainty detected - clinical caution advised")
            
        except Exception as e:
            logger.error(f"Error calculating disagreement metrics: {str(e)}")

    # Generate unified outputs
    logger.info("Generating unified outputs...")
    output_generator = UnifiedOutputGenerator()
    
    # Get base name for output files
    base_name = os.path.splitext(os.path.basename(dicom_path))[0]
    
    # Ensemble information
    ensemble_info = {
        'models': models,
        'num_models': len(models),
        'models_processed': list(model_inference_results.keys()),
        'fusion_method': 'confidence_weighted_centroid',
        'disagreement_metrics': disagreement_metrics if 'disagreement_metrics' in locals() else {}
    }
    
    # Generate the three output files
    qa_path = os.path.join(output_dir, f"{base_name}_qa_visualization.dcm")
    output_generator.create_enhanced_qa_visualization(
        unified_predictions=fused_predictions,
        dicom_path=dicom_path,
        output_path=qa_path,
        processor=None,
        ensemble_info=ensemble_info
    )
    
    sr_path = os.path.join(output_dir, f"{base_name}_measurements_report.dcm")
    output_generator.create_enhanced_secondary_capture(
        unified_predictions=fused_predictions,
        measurements_data=measurements_data,
        dicom_path=dicom_path,
        output_path=sr_path,
        ensemble_info=ensemble_info
    )
    
    json_path = os.path.join(output_dir, f"{base_name}_measurements_report.json")
    output_generator.create_enhanced_json_report(
        unified_predictions=fused_predictions,
        measurements_data=measurements_data,
        dicom_path=dicom_path,
        output_path=json_path,
        ensemble_info=ensemble_info,
        disagreement_metrics=disagreement_metrics
    )
    
    # Clean up temporary directories
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    logger.info("Ensemble inference completed successfully")
    logger.info(f"Generated unified outputs:")
    logger.info(f"  - QA Visualization: {qa_path}")
    logger.info(f"  - Secondary Capture: {sr_path}")
    logger.info(f"  - JSON Report: {json_path}")
    
    return {
        'fused_predictions': fused_predictions,
        'measurements_data': measurements_data,
        'disagreement_metrics': disagreement_metrics,
        'ensemble_info': ensemble_info,
        'output_files': {
            'qa_visualization': qa_path,
            'secondary_capture': sr_path,
            'json_report': json_path
        },
        'models_processed': list(model_inference_results.keys()),
        'output_dir': output_dir
    } 