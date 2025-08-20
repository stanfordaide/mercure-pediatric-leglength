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


class LegMeasurements:
    """Class to handle leg length measurements and DICOM SR generation."""
    
    def __init__(self, config_path: str = None):
        """Initialize with measurement configurations."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'measurement_configs.json')
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.measurements = {}
        self.pixel_spacing = None
        
        # Define colors for visualization (BGR format)
        self.colors = {
            'femur_r': (255,31,223),    # White
            'femur_l': (255,31,223),    # White 
            'tibia_r': (255,31,223),    # White
            'tibia_l': (255,31,223),    # White
        }
        
    def calculate_distances(self, predictions: Dict, dicom_path: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate distances between keypoints based on predictions.
        threshold_cm: optional threshold for reporting discrepancies in centimeters.
        """
        # Load DICOM for pixel spacing
        dcm = pydicom.dcmread(dicom_path)
        self.pixel_spacing = dcm.PixelSpacing
        pixel_spacing_x, pixel_spacing_y = float(self.pixel_spacing[0]), float(self.pixel_spacing[1])
        
        # Extract keypoints from predictions
        keypoints = {}
        for i, (box, label) in enumerate(zip(predictions['boxes'], predictions['labels'])):
            # Calculate center point of bounding box
            x_center = float((box[0] + box[2]) / 2)
            y_center = float((box[1] + box[3]) / 2)
            keypoints[label] = (x_center, y_center)
        
        # Calculate derived points if they exist in config
        if 'derived_points' in self.config:
            for derived_point in self.config['derived_points']:
                name = derived_point['name']
                point_type = derived_point['type']
                source_points = derived_point['source_points']
                
                if point_type == 'midpoint':
                    # Check if all source points are available
                    if all(point in keypoints for point in source_points):
                        # Calculate midpoint
                        x_coords = [keypoints[point][0] for point in source_points]
                        y_coords = [keypoints[point][1] for point in source_points]
                        midpoint_x = sum(x_coords) / len(x_coords)
                        midpoint_y = sum(y_coords) / len(y_coords)
                        keypoints[name] = (midpoint_x, midpoint_y)
                    else:
                        logger.warning(f"Missing source points for derived point {name}")
        
        # Calculate measurements based on config
        measurements = {}
        for measure in self.config['measurements']:
            name = measure['name']
            point1, point2 = measure['join_points']
            
            # Skip if either point is missing
            if point1 not in keypoints or point2 not in keypoints:
                logger.warning(f"Missing points for measurement {name}")
                continue
            
            # Calculate Euclidean distance in pixels
            p1 = keypoints[point1]
            p2 = keypoints[point2]
            pixel_distance = np.sqrt(
                (p2[0] - p1[0])**2 + 
                (p2[1] - p1[1])**2
            )
            
            # Convert to millimeters
            mm_distance = pixel_distance * np.sqrt(
                (pixel_spacing_x**2 + pixel_spacing_y**2) / 2
            )
            
            # Store both millimeters and centimeters
            cm_distance = mm_distance / 10.0
            measurements[name] = {
                'millimeters': mm_distance,
                'centimeters': cm_distance,
                'points': {
                    'start': {'x': float(p1[0]), 'y': float(p1[1])},
                    'end': {'x': float(p2[0]), 'y': float(p2[1])}
                }
            }
        
        self.measurements = measurements
        return measurements
    
    