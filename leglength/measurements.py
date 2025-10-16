import os
import json
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid
import logging
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path
import cv2
import torch

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


class LegMeasurements:
    """Class to handle leg length measurements and DICOM SR generation."""
    
    def __init__(self, config_path: str = None):
        """Initialize with measurement configurations.
        
        Args:
            config_path: Path to measurement configuration JSON file. If None, uses default config.
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'measurement_configs.json')
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize measurement tracking
        self.measurements = {}
        self.measurement_issues = []
        self.pixel_spacing = None
        
        # Define professional medical imaging colors (BGR format)
        self.colors = {
            'PLL_R_FEM': (255, 255, 0),  # Cyan for medical imaging
            'PLL_R_TIB': (255, 255, 0),  # Cyan for medical imaging
            'PLL_L_FEM': (255, 255, 0),  # Cyan for medical imaging
            'PLL_L_TIB': (255, 255, 0),  # Cyan for medical imaging
        }
        
    def calculate_distances(self, predictions: Dict, dicom_path: str, logger: logging.Logger) -> Dict[str, Dict[str, float]]:
        """Calculate distances between keypoints based on model predictions.
        
        Args:
            predictions: Dictionary containing model predictions with 'boxes' and 'labels'
            dicom_path: Path to DICOM file for pixel spacing information
            
        Returns:
            Tuple of:
                - Dictionary of measurements with distances in mm/cm and point coordinates
                - List of measurement issues encountered
        """
        # Get pixel spacing from DICOM
        dcm = pydicom.dcmread(dicom_path)
        self.pixel_spacing = get_pixel_spacing(dcm)
        if self.pixel_spacing is None:
            logger.error(f"No PixelSpacing or ImagerPixelSpacing found in DICOM: {dicom_path}")
            return {}, ['Missing pixel spacing - cannot calculate measurements']
        
        # Handle both 2-value and single-value (isotropic) pixel spacing
        if len(self.pixel_spacing) == 1:
            # If only one value, use it for both x and y (isotropic spacing)
            pixel_spacing_x = pixel_spacing_y = float(self.pixel_spacing[0])
            logger.info(f"Using isotropic pixel spacing: {pixel_spacing_x} mm")
        else:
            # Standard case: separate x and y spacing
            pixel_spacing_x, pixel_spacing_y = float(self.pixel_spacing[0]), float(self.pixel_spacing[1])
        
        # Extract keypoint coordinates from bounding box centers
        keypoints = {}
        for i, (box, label) in enumerate(zip(predictions['boxes'], predictions['labels'])):
            x_center = float((box[0] + box[2]) / 2)
            y_center = float((box[1] + box[3]) / 2)
            keypoints[label] = (x_center, y_center)
        
        # Calculate derived points (e.g. midpoints) if specified in config
        if 'derived_points' in self.config:
            for derived_point in self.config['derived_points']:
                name = derived_point['name']
                point_type = derived_point['type']
                source_points = derived_point['source_points']
                
                if point_type == 'midpoint':
                    if all(point in keypoints for point in source_points):
                        x_coords = [keypoints[point][0] for point in source_points]
                        y_coords = [keypoints[point][1] for point in source_points]
                        midpoint_x = sum(x_coords) / len(x_coords)
                        midpoint_y = sum(y_coords) / len(y_coords)
                        keypoints[name] = (midpoint_x, midpoint_y)
                    else:
                        logger.warning(f"Missing source points for derived point {name}")
        
        # Calculate measurements based on config specifications
        measurements = {}
        measurement_issues = []
        
        for measure in self.config['measurements']:
            name = measure['name']
            point1, point2 = measure['join_points']
            
            if point1 not in keypoints or point2 not in keypoints:
                logger.warning(f"Missing points for measurement {name}")
                continue
            
            # Get distance type (default to vertical)
            distance_type = measure.get('distance_type', 'vertical')
            
            p1 = keypoints[point1]
            p2 = keypoints[point2]
            
            # Calculate distance based on type
            if distance_type == 'vertical':
                # Vertical distance: y-coordinate difference only
                pixel_distance = abs(p2[1] - p1[1])
                # Convert pixel distance to physical units (using y-axis pixel spacing)
                mm_distance = pixel_distance * pixel_spacing_y
            elif distance_type == 'euclidean':
                # Euclidean distance: straight line between points
                pixel_distance = np.sqrt(
                    (p2[0] - p1[0])**2 + 
                    (p2[1] - p1[1])**2
                )
                # Convert pixel distance to physical units
                mm_distance = pixel_distance * np.sqrt(
                    (pixel_spacing_x**2 + pixel_spacing_y**2) / 2
                )
            else:
                logger.warning(f"Unknown distance_type '{distance_type}' for measurement {name}, using vertical")
                pixel_distance = abs(p2[1] - p1[1])
                mm_distance = pixel_distance * pixel_spacing_y
            
            cm_distance = mm_distance / 10.0
            
            # Store measurement data
            measurements[name] = {
                'millimeters': mm_distance,
                'centimeters': cm_distance,
                'distance_type': distance_type,
                'points': {
                    'start': {'x': float(p1[0]), 'y': float(p1[1])},
                    'end': {'x': float(p2[0]), 'y': float(p2[1])}
                }
            }
        
        # Track any missing measurements
        for measure in self.config['measurements']:
            if measure['name'] not in measurements:
                measurement_issues.append({
                    'name': measure['name'],
                    'reason': 'missing_points',
                    'description': f"Missing points: Could not compute {measure['name']} because of missing points"
                })
        
        self.measurements = measurements
        self.measurement_issues = measurement_issues
        
        # Calculate additional derived measurements
        self.downstream_calculations()
        return self.measurements, self.measurement_issues
            
    def downstream_calculations(self):
        """Calculate derived measurements like total leg lengths from component measurements."""
        
        # Calculate right leg length if components available
        if "PLL_R_TIB" in self.measurements and "PLL_R_FEM" in self.measurements:
            self.measurements["PLL_R_LGL"] = {
                'millimeters': self.measurements["PLL_R_TIB"]["millimeters"] + self.measurements["PLL_R_FEM"]["millimeters"],
                'centimeters': self.measurements["PLL_R_TIB"]["centimeters"] + self.measurements["PLL_R_FEM"]["centimeters"],
                'points': {
                    'start': {'x': float(self.measurements["PLL_R_FEM"]["points"]["start"]["x"]), 'y': float(self.measurements["PLL_R_FEM"]["points"]["start"]["y"])},
                    'end': {'x': float(self.measurements["PLL_R_TIB"]["points"]["end"]["x"]), 'y': float(self.measurements["PLL_R_TIB"]["points"]["end"]["y"])},
                }
            }
        else:
            self.measurement_issues.append({
                'name': "PLL_R_LGL",
                'reason': 'missing_measurements',
                'description': "Missing measurements: Could not compute PLL_R_LGL because of missing PLL_R_TIB or PLL_L_TIB measurements"
            })
            
        # Calculate left leg length if components available
        if "PLL_L_TIB" in self.measurements and "PLL_L_FEM" in self.measurements:
            self.measurements["PLL_L_LGL"] = {
                'millimeters': self.measurements["PLL_L_TIB"]["millimeters"] + self.measurements["PLL_L_FEM"]["millimeters"],
                'centimeters': self.measurements["PLL_L_TIB"]["centimeters"] + self.measurements["PLL_L_FEM"]["centimeters"],
                'points': {
                    'start': {'x': float(self.measurements["PLL_L_FEM"]["points"]["start"]["x"]), 'y': float(self.measurements["PLL_L_FEM"]["points"]["start"]["y"])},    
                    'end': {'x': float(self.measurements["PLL_L_TIB"]["points"]["end"]["x"]), 'y': float(self.measurements["PLL_L_TIB"]["points"]["end"]["y"])},
                }
            }
        else:
            self.measurement_issues.append({
                'name': "PLL_L_LGL", 
                'reason': 'missing_measurements',
                'description': "Missing measurements: Could not compute PLL_L_LGL because of missing PLL_L_TIB or PLL_L_FEM measurements"
            })
            
        return self.measurements, self.measurement_issues