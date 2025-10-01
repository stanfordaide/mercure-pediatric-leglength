import os
import logging
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
import pydicom
import torch
from .detector import LegLengthDetector
from .processor import ImageProcessor
from .measurements import LegMeasurements


def get_pixel_spacing(dicom_dataset):
    """
    Get pixel spacing from DICOM dataset, trying multiple fields.
    
    Args:
        dicom_dataset: PyDICOM dataset
        
    Returns:
        float: Pixel spacing value, or None if not found
    """
    # Try PixelSpacing first (most common)
    if hasattr(dicom_dataset, 'PixelSpacing') and dicom_dataset.PixelSpacing:
        return float(dicom_dataset.PixelSpacing[0])
    
    # Try ImagerPixelSpacing as fallback
    if hasattr(dicom_dataset, 'ImagerPixelSpacing') and dicom_dataset.ImagerPixelSpacing:
        return float(dicom_dataset.ImagerPixelSpacing[0])
    
    return None


def inference_handler(
    models: list[str],
    dicom_path: str,
    output_dir: str,
    config: dict,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Run inference using single or ensemble of models on a DICOM image.

    Args:
        models: List of model names to use for inference
        dicom_path: Path to input DICOM file
        output_dir: Directory to save results
        logger: Optional logger instance

    Returns:
        Dictionary containing predictions, measurements and metadata
    """

    # Read DICOM metadata
    dicom = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    pixel_spacing = get_pixel_spacing(dicom)
    
    if pixel_spacing is None:
        logger.warning(f"No PixelSpacing or ImagerPixelSpacing found in DICOM file: {dicom_path}")
        logger.info(f"Skipping image {dicom_path} - pixel spacing is required for distance calculations")
        # Return a result indicating the image was skipped
        return {
            'skipped': True,
            'reason': 'Missing pixel spacing (PixelSpacing or ImagerPixelSpacing)',
            'dicom_path': dicom_path,
            'dicom_metadata': {
                'pixel_spacing': None,
                'accession_number': getattr(dicom, 'AccessionNumber', None),
                'patient_id': getattr(dicom, 'PatientID', None),
                'study_instance_uid': getattr(dicom, 'StudyInstanceUID', None),
                'series_instance_uid': getattr(dicom, 'SeriesInstanceUID', None),
                'modality': getattr(dicom, 'Modality', None),
                'study_date': getattr(dicom, 'StudyDate', None),
                'series_description': getattr(dicom, 'SeriesDescription', None),
                'manufacturer': getattr(dicom, 'Manufacturer', None),
                'slice_thickness': getattr(dicom, 'SliceThickness', None),
            },
            'measurements': {},
            'boxes': [],
            'scores': [],
            'labels': [],
            'issues': ['Missing pixel spacing - cannot calculate measurements']
        }

    dicom_metadata = {
        'pixel_spacing': pixel_spacing,
        'accession_number': getattr(dicom, 'AccessionNumber', None),
        'patient_id': getattr(dicom, 'PatientID', None),
        'study_instance_uid': getattr(dicom, 'StudyInstanceUID', None),
        'series_instance_uid': getattr(dicom, 'SeriesInstanceUID', None),
        'modality': getattr(dicom, 'Modality', None),
        'study_date': getattr(dicom, 'StudyDate', None),
        'series_description': getattr(dicom, 'SeriesDescription', None),
        'manufacturer': getattr(dicom, 'Manufacturer', None),
        'slice_thickness': getattr(dicom, 'SliceThickness', None),
    }

    # Run ensemble inference if multiple models provided
    if len(models) > 1:
        # Get predictions from each model
        model_predictions = {}
        for model_name in models:
            model_predictions[model_name] = infer(
                model_name,
                dicom_path,
                output_dir, 
                config,
                logger=logger
            )

        # Fuse predictions from all models
        fused_results = fuse_predictions(
            models,
            model_predictions,
            dicom_path,
            pixel_spacing,
            config,
            logger=logger
        )

        # Format prediction object for measurements
        prediction_object = {
            'boxes': fused_results['boxes'],
            'scores': fused_results['scores'],
            'labels': fused_results['labels'],
        }

        # Calculate measurements
        measurement_calculator = LegMeasurements()
        measurements, issues = measurement_calculator.calculate_distances(
            prediction_object,
            dicom_path, 
            logger=logger
        )

        # Add measurements and metadata to results
        fused_results['measurements'] = measurements
        fused_results['issues'].extend(issues)
        fused_results['dicom_metadata'] = dicom_metadata

        return fused_results

    # Run single model inference
    else:
        single_model_results = infer(
            models[0],
            dicom_path,
            output_dir, 
            config,
            logger=logger   
        )

        # Calculate measurements
        measurement_calculator = LegMeasurements()
        measurement_results = measurement_calculator.calculate_distances(
            single_model_results['predictions'],
            dicom_path,
            logger=logger
        )

        # Format results
        results = {
            'measurements': measurement_results['measurements'],
            'issues': measurement_results['measurement_issues'],
            'uncertainties': {},
            'point_statistics': {},
            'dicom_metadata': dicom_metadata,
            'boxes': single_model_results['predictions']['boxes'],
            'scores': single_model_results['predictions']['scores'],
            'labels': single_model_results['predictions']['labels'],
        }

        return results


def infer(
    model_name: str,
    dicom_path: str,
    output_dir: str,
    config: dict,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Run inference on a DICOM image using a single model.

    Args:
        model_name: Name of model to use
        dicom_path: Path to DICOM file
        output_dir: Directory to save results
        logger: Optional logger instance

    Returns:
        Dictionary containing model predictions
    """

    try:
        os.makedirs(output_dir, exist_ok=True)

        # Initialize model and preprocessor
        logger.info(f"Loading model: {model_name}")
        detector = LegLengthDetector.load_checkpoint(model_name)
        preprocessor = ImageProcessor()

        # Preprocess image
        logger.info(f"Processing DICOM: {dicom_path}")
        image = preprocessor.preprocess_image(dicom_path)

        # Run inference
        logger.info("Running inference...")
        predictions = detector.predict(image, confidence_threshold=config['confidence_threshold'])

        # Convert boxes back to original image space
        boxes = torch.tensor(predictions['boxes'])
        boxes = preprocessor.translate_boxes_to_original(boxes)
        predictions['boxes'] = boxes.numpy()

        return {'predictions': predictions}

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise


def calculate_image_level_metrics(
    uncertainties: Dict,
    config: dict,
    total_models: int
) -> Dict:
    """
    Calculate image-level disagreement metrics.
    
    Args:
        uncertainties: Point-level uncertainty metrics
        config: Configuration containing thresholds and weights
        total_models: Total number of models in ensemble
        
    Returns:
        Dictionary containing image-level metrics
    """
    if not uncertainties:
        return {
            'image_dds': 0.0,
            'image_lds': 0.0, 
            'image_ors': 0.0,
            'image_cds': 0.0
        }
    
    # Get configuration parameters with defaults
    distance_threshold = config.get('image_metrics', {}).get('distance_threshold', 2.0)
    outlier_bounds = config.get('image_metrics', {}).get('threshold_bounds', [10.0, 50.0])
    weights = config.get('image_metrics', {}).get('composite_weights', [0.5, 0.35, 0.15])
    
    # Calculate image-level Detection Disagreement Score (DDS)
    detection_disagreements = [u.get('detection_disagreement', 0) for u in uncertainties.values()]
    image_dds = np.mean(detection_disagreements) if detection_disagreements else 0.0
    
    # Calculate image-level Localization Disagreement Score (LDS)
    # LDS should be based on pairwise distances between model predictions
    localization_disagreements = []
    for uncertainty in uncertainties.values():
        # Use the localization_disagreement from point-level calculation
        # This is already calculated as: sum(pairwise_distances > 2mm) / total_pairs
        point_lds = uncertainty.get('localization_disagreement', 0)
        localization_disagreements.append(point_lds)
    
    image_lds = np.mean(localization_disagreements) if localization_disagreements else 0.0
    
    # Calculate image-level Outlier Risk Score (ORS)
    outlier_risks = []
    for uncertainty in uncertainties.values():
        spatial_uncertainty = uncertainty.get('spatial_uncertainty_mm', 0)
        # Scale outlier risk based on threshold bounds
        min_bound, max_bound = outlier_bounds
        if spatial_uncertainty < min_bound:
            risk = 0.0
        elif spatial_uncertainty > max_bound:
            risk = 1.0
        else:
            risk = (spatial_uncertainty - min_bound) / (max_bound - min_bound)
        outlier_risks.append(risk)
    
    image_ors = np.mean(outlier_risks) if outlier_risks else 0.0
    
    # Calculate Composite Disagreement Score (CDS)
    w_dds, w_ors, w_lds = weights
    image_cds = w_dds * image_dds + w_ors * image_ors + w_lds * image_lds
    
    return {
        'image_dds': float(image_dds),
        'image_lds': float(image_lds),
        'image_ors': float(image_ors), 
        'image_cds': float(image_cds)
    }


def calculate_weighted_centroid(
    positions: List[Tuple[float, float]],
    confidences: List[float],
    pixel_spacing: float,
    total_models: int,
    distance_threshold: float = 2.0,
    outlier_bounds: List[float] = [10.0, 50.0],
    individual_model_boxes: List = None,
    model_names: List[str] = None
) -> Tuple[float, float, Dict]:
    """
    Calculate confidence-weighted centroid and uncertainty metrics for point predictions.

    Args:
        positions: List of (x,y) point coordinates
        confidences: Confidence scores for each point
        pixel_spacing: Pixel to mm conversion factor
        total_models: Total number of models in ensemble

    Returns:
        Tuple of (weighted_x, weighted_y, uncertainty_metrics)
    """
    if not positions or not confidences:
        return None, None, {}

    positions = np.array(positions)
    confidences = np.array(confidences)

    # Calculate weighted centroid
    weights = confidences / np.sum(confidences)
    weighted_x = np.sum(positions[:, 0] * weights)
    weighted_y = np.sum(positions[:, 1] * weights)

    # Convert to mm
    weighted_x_mm = weighted_x * pixel_spacing
    weighted_y_mm = weighted_y * pixel_spacing

    # Calculate disagreement scores
    detection_disagreement = (total_models - len(positions)) / total_models

    # Calculate pairwise distances using individual model bboxes if available
    pairwise_distances = []
    
    if individual_model_boxes and len(individual_model_boxes) > 1:
        # Use actual model bounding box centroids
        model_centroids = []
        for bbox in individual_model_boxes:
            if bbox is not None and len(bbox) >= 4:
                centroid_x = (bbox[0] + bbox[2]) / 2
                centroid_y = (bbox[1] + bbox[3]) / 2
                model_centroids.append((centroid_x, centroid_y))
        
        # Calculate pairwise distances between model centroids
        for i in range(len(model_centroids)):
            for j in range(i+1, len(model_centroids)):
                dist_pixels = np.sqrt((model_centroids[i][0] - model_centroids[j][0])**2 + 
                                    (model_centroids[i][1] - model_centroids[j][1])**2)
                dist_mm = dist_pixels * pixel_spacing
                pairwise_distances.append(dist_mm)
    else:
        # Fallback to using positions (converted to mm)
        positions_mm = np.array(positions) * pixel_spacing
        for i in range(len(positions_mm)):
            for j in range(i+1, len(positions_mm)):
                dist = np.linalg.norm(positions_mm[i] - positions_mm[j])
                pairwise_distances.append(dist)
    
    # Use the configurable distance threshold
    localization_disagreement = (
        np.sum(np.array(pairwise_distances) > distance_threshold) / len(pairwise_distances) 
        if pairwise_distances else 0.0
    )

    # Calculate outlier risk based on pairwise distances
    if pairwise_distances:
        outlier_threshold_min, outlier_threshold_max = outlier_bounds
        
        # Calculate risk based on how many pairs exceed thresholds
        outlier_pairs_count = np.sum(np.array(pairwise_distances) > outlier_threshold_min)
        total_pairs = len(pairwise_distances)
        
        if outlier_pairs_count == 0:
            outlier_risk = 0.0
        else:
            # Base risk from proportion of outlier pairs
            base_risk = outlier_pairs_count / total_pairs
            
            # Scale risk based on severity (how far beyond min threshold)
            outlier_distances = [d for d in pairwise_distances if d > outlier_threshold_min]
            max_outlier_distance = max(outlier_distances)
            
            if max_outlier_distance > outlier_threshold_max:
                severity_multiplier = 1.0  # Maximum severity
            else:
                # Linear scaling between min and max thresholds
                severity_multiplier = (max_outlier_distance - outlier_threshold_min) / (outlier_threshold_max - outlier_threshold_min)
            
            outlier_risk = min(base_risk * (1 + severity_multiplier), 1.0)
    else:
        outlier_risk = 0.0

    # Calculate uncertainty metrics
    std_x = np.sqrt(
        np.sum(weights * (positions[:, 0] - weighted_x)**2)
    ) * pixel_spacing
    std_y = np.sqrt(
        np.sum(weights * (positions[:, 1] - weighted_y)**2)
    ) * pixel_spacing
    spatial_uncertainty = np.sqrt(std_x**2 + std_y**2)

    confidence_mean = np.mean(confidences)
    confidence_std = np.std(confidences)
    confidence_uncertainty = (
        confidence_std / confidence_mean if confidence_mean > 0 else 0
    )

    uncertainty_metrics = {
        'weighted_x_mm': float(weighted_x_mm),
        'weighted_y_mm': float(weighted_y_mm),
        'detection_disagreement': float(detection_disagreement),
        'total_models': total_models,
        'localization_disagreement': float(localization_disagreement),
        'outlier_risk': float(outlier_risk),
        'spatial_uncertainty_mm': float(spatial_uncertainty),
        'confidence_mean': float(confidence_mean),
        'confidence_std': float(confidence_std),
        'confidence_uncertainty': float(confidence_uncertainty),
        'num_models': len(positions),
        'position_std_x_mm': float(std_x),
        'position_std_y_mm': float(std_y)
    }

    return float(weighted_x), float(weighted_y), uncertainty_metrics


def fuse_predictions(
    models: List[str],
    model_predictions: Dict,
    dicom_path: str,
    pixel_spacing: float,
    config: dict,
    logger: logging.Logger
) -> Dict:
    """
    Fuse predictions from multiple models into a single set of predictions.

    Args:
        models: List of model names
        model_predictions: Dictionary of predictions from each model
        dicom_path: Path to DICOM file
        pixel_spacing: Pixel to mm conversion factor
        logger: Logger instance

    Returns:
        Dictionary containing fused predictions and uncertainty metrics
    """

    if len(models) == 1:
        logger.info(f"Single model inference with {models[0]}")
        return model_predictions[0]

    logger.info(f"Fusing predictions from {len(models)} models")

    # Collect predictions per point from all models
    point_predictions = {}
    for model_name, results in model_predictions.items():
        if 'predictions' not in results:
            continue

        predictions = results['predictions']
        for box, score, label in zip(
            predictions.get('boxes', []),
            predictions.get('scores', []),
            predictions.get('labels', [])
        ):
            point_idx = int(label)
            x_center = float((box[0] + box[2]) / 2)
            y_center = float((box[1] + box[3]) / 2)

            if point_idx not in point_predictions:
                point_predictions[point_idx] = {
                    'positions': [],
                    'confidences': [],
                    'models': []
                }

            point_predictions[point_idx]['positions'].append((x_center, y_center))
            point_predictions[point_idx]['confidences'].append(float(score))
            point_predictions[point_idx]['models'].append(model_name)

    # Initialize fused predictions
    fused_results = {
        'boxes': [],
        'scores': [],
        'labels': [],
        'uncertainties': {},
        'point_statistics': {},
        'issues': [],
        'individual_model_predictions': model_predictions  # Store individual model predictions
    }

    # Fuse predictions for each point
    for point_idx in range(1, 9):
        if point_idx not in point_predictions:
            # Log missing point
            fused_results['issues'].append({
                'name': f"point_{point_idx}",
                'reason': 'no_detection',
                'total_models': len(model_predictions),
                'description': f"Point {point_idx} not detected by any model"
            })
            continue

        point_data = point_predictions[point_idx]

        # Calculate fused position and uncertainties
        distance_threshold = config.get('image_metrics', {}).get('distance_threshold', 2.0)
        outlier_bounds = config.get('image_metrics', {}).get('threshold_bounds', [10.0, 50.0])
        
        # Get individual model bboxes for this point
        individual_boxes = []
        model_names_for_point = []
        for model_name in point_data['models']:
            # Find the bbox for this model and point
            model_pred = model_predictions.get(model_name, {})
            boxes = model_pred.get('predictions', {}).get('boxes', [])
            labels = model_pred.get('predictions', {}).get('labels', [])
            
            # Find the box with matching label for this point
            target_label = point_idx  # Assuming point_idx corresponds to label
            for i, label in enumerate(labels):
                if label == target_label and i < len(boxes):
                    individual_boxes.append(boxes[i])
                    model_names_for_point.append(model_name)
                    break
        
        fused_x, fused_y, uncertainties = calculate_weighted_centroid(
            point_data['positions'],
            point_data['confidences'],
            pixel_spacing,
            len(model_predictions),
            distance_threshold,
            outlier_bounds,
            individual_boxes,
            model_names_for_point
        )

        if fused_x is None or fused_y is None:
            continue

        # Create bounding box from min/max coordinates
        x_coords = [pos[0] for pos in point_data['positions']]
        y_coords = [pos[1] for pos in point_data['positions']]
        fused_box = [
            float(min(x_coords)),
            float(min(y_coords)),
            float(max(x_coords)),
            float(max(y_coords))
        ]

        # Add fused prediction
        fused_results['boxes'].append(fused_box)
        fused_results['scores'].append(float(uncertainties['confidence_mean']))
        fused_results['labels'].append(int(point_idx))
        fused_results['uncertainties'][point_idx] = uncertainties

        # Add point statistics
        fused_results['point_statistics'][point_idx] = {
            'num_models_detected': int(len(point_data['models'])),
            'detecting_models': point_data['models'],
            'detection_disagreement': float(uncertainties['detection_disagreement']),
            'outlier_risk': float(uncertainties['outlier_risk']),
            'localization_disagreement': float(uncertainties['localization_disagreement']),
            'position_range_x': float(max(x_coords) - min(x_coords)),
            'position_range_y': float(max(y_coords) - min(y_coords)),
            'confidence_range': float(
                max(point_data['confidences']) - min(point_data['confidences'])
            ),
            'spatial_uncertainty_mm': float(uncertainties['spatial_uncertainty_mm']),
            'confidence_mean': float(uncertainties['confidence_mean']),
            'confidence_std': float(uncertainties['confidence_std'])
        }

        # Check for issues
        if uncertainties['detection_disagreement'] > 0.25:
            failed_models = [
                m for m in model_predictions.keys()
                if m not in point_data['models']
            ]

            fused_results['issues'].append({
                'name': f"point_{point_idx}",
                'reason': 'detection_disagreement',
                'disagreement_score': float(uncertainties['detection_disagreement']),
                'models_detected': int(len(point_data['models'])),
                'total_models': int(len(model_predictions)),
                'detecting_models': point_data['models'],
                'failed_models': failed_models,
                'description': (
                    f"Point {point_idx}: {len(failed_models)}/{len(model_predictions)} "
                    "models failed to detect"
                )
            })

        if uncertainties['localization_disagreement'] > 0.25:
            positions_mm = [
                np.array(pos) * pixel_spacing for pos in point_data['positions']
            ]
            pairwise_distances = [
                np.linalg.norm(p1 - p2)
                for i, p1 in enumerate(positions_mm)
                for p2 in positions_mm[i+1:]
            ]

            max_dist = max(pairwise_distances)
            mean_dist = np.mean(pairwise_distances)
            std_dist = np.std(pairwise_distances)

            # Calculate percentage of predictions more than 2mm apart
            dists_over_2mm = sum(1 for d in pairwise_distances if d > 2.0)
            total_pairs = len(pairwise_distances)
            pct_over_2mm = (dists_over_2mm / total_pairs) * 100 if total_pairs > 0 else 0

            fused_results['issues'].append({
                'name': f"point_{point_idx}",
                'reason': 'localization_disagreement', 
                'disagreement_score': float(uncertainties['localization_disagreement']),
                'max_spatial_disagreement_mm': float(max_dist),
                'mean_spatial_disagreement_mm': float(mean_dist),
                'std_spatial_disagreement_mm': float(std_dist),
                'spatial_uncertainty_mm': float(uncertainties['spatial_uncertainty_mm']),
                'pct_predictions_over_2mm': float(pct_over_2mm),
                'description': f"Point {point_idx}: Models disagree by up to {max_dist:.1f} mm, {pct_over_2mm:.1f}% prediction pairs >2 mm apart"
            })

        if uncertainties['outlier_risk'] > 0.5:
            distances = [
                np.sqrt(np.sum((np.array(pos) - np.array([fused_x, fused_y]))**2))
                for pos in point_data['positions']
            ]
            max_dist = max(distances) * pixel_spacing
            outlier_model = point_data['models'][np.argmax(distances)]

            fused_results['issues'].append({
                'name': f"point_{point_idx}",
                'reason': 'high_outlier_risk',
                'risk_score': float(uncertainties['outlier_risk']),
                'max_distance_mm': float(max_dist),
                'outlier_model': outlier_model,
                'fused_position_mm': [
                    float(fused_x * pixel_spacing),
                    float(fused_y * pixel_spacing)
                ],
                'description': (
                    f"Point {point_idx}: Potential outlier {max_dist:.1f} mm from consensus"
                )
            })

    # Calculate image-level metrics
    image_metrics = calculate_image_level_metrics(
        fused_results.get('uncertainties', {}),
        config,
        len(models)
    )
    fused_results['image_metrics'] = image_metrics
    fused_results['config_used'] = {
        'distance_threshold': config.get('image_metrics', {}).get('distance_threshold', 4.0),
        'threshold_bounds': config.get('image_metrics', {}).get('threshold_bounds', [10.0, 50.0])
    }
    
    logger.info(
        f"Fused {len(fused_results['boxes'])} points with "
        f"{len(fused_results['issues'])} issues"
    )
    logger.info(f"Image-level metrics: DDS={image_metrics['image_dds']:.3f}, "
                f"LDS={image_metrics['image_lds']:.3f}, ORS={image_metrics['image_ors']:.3f}, "
                f"CDS={image_metrics['image_cds']:.3f}")
    return fused_results
