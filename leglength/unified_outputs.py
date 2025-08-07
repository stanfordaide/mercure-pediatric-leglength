#!/usr/bin/env python3
"""
Unified output module for leg length detection.
Generates enhanced QA visualizations, secondary captures, and JSON reports with uncertainty information.
"""
import os
import cv2
import json
import pydicom
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid

from .outputs import LegMeasurements

class UnifiedOutputGenerator:
    """Generates unified outputs with uncertainty information and problematic point analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Enhanced color scheme for uncertainty visualization
        self.uncertainty_colors = {
            'low': (0, 255, 0),      # Green - low uncertainty
            'medium': (0, 255, 255), # Yellow - medium uncertainty  
            'high': (0, 165, 255),   # Orange - high uncertainty
            'critical': (0, 0, 255), # Red - critical uncertainty
            'problematic': (255, 0, 255), # Magenta - problematic points
        }
    
    def get_uncertainty_color(self, uncertainty_score: float) -> tuple:
        """Get color based on uncertainty level."""
        if uncertainty_score <= 5.0:
            return self.uncertainty_colors['low']
        elif uncertainty_score <= 10.0:
            return self.uncertainty_colors['medium']
        elif uncertainty_score <= 15.0:
            return self.uncertainty_colors['high']
        else:
            return self.uncertainty_colors['critical']
    
    def create_enhanced_qa_visualization(self, 
                                       unified_predictions: Dict,
                                       dicom_path: str,
                                       output_path: str,
                                       processor=None,
                                       ensemble_info: Optional[Dict] = None) -> None:
        """
        Create enhanced QA DICOM with uncertainty visualization.
        
        Args:
            unified_predictions: Fused predictions with uncertainty metrics
            dicom_path: Path to original DICOM file
            output_path: Path to save the QA DICOM
            processor: ImageProcessor instance
            ensemble_info: Information about ensemble models used
        """
        # Load original DICOM
        dcm = pydicom.dcmread(dicom_path)
        original_image = dcm.pixel_array.astype(float)
        
        # Normalize to 0-255 and convert to RGB
        normalized_image = ((original_image - original_image.min()) / 
                          (original_image.max() - original_image.min()) * 255).astype(np.uint8)
        visualization = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)
        
        # Extract keypoints from unified predictions
        keypoints = {}
        for i, (box, score, label) in enumerate(zip(
            unified_predictions.get('boxes', []),
            unified_predictions.get('scores', []),
            unified_predictions.get('labels', [])
        )):
            point_idx = int(label)
            x_center = int((box[0] + box[2]) / 2)
            y_center = int((box[1] + box[3]) / 2)
            keypoints[point_idx] = (x_center, y_center)
        
        # Calculate derived points if config available
        measurements = LegMeasurements()
        if 'derived_points' in measurements.config:
            for derived_point in measurements.config['derived_points']:
                name = derived_point['name']
                point_type = derived_point['type']
                source_points = derived_point['source_points']
                
                if point_type == 'midpoint':
                    if all(point in keypoints for point in source_points):
                        x_coords = [keypoints[point][0] for point in source_points]
                        y_coords = [keypoints[point][1] for point in source_points]
                        midpoint_x = int(sum(x_coords) / len(x_coords))
                        midpoint_y = int(sum(y_coords) / len(y_coords))
                        keypoints[name] = (midpoint_x, midpoint_y)
        
        # Draw measurements with uncertainty information
        for measure in measurements.config['measurements']:
            name = measure['name']
            point1, point2 = measure['join_points']
            
            if point1 in keypoints and point2 in keypoints:
                p1 = keypoints[point1]
                p2 = keypoints[point2]
                
                # Get uncertainty for both points (if available)
                uncertainty1 = unified_predictions.get('uncertainties', {}).get(point1, {}).get('overall_uncertainty', 0)
                uncertainty2 = unified_predictions.get('uncertainties', {}).get(point2, {}).get('overall_uncertainty', 0)
                avg_uncertainty = (uncertainty1 + uncertainty2) / 2
                
                # Choose color based on uncertainty
                color = self.get_uncertainty_color(avg_uncertainty)
                
                # Draw line between points with thickness based on uncertainty
                thickness = 2 if avg_uncertainty <= 10 else 3
                cv2.line(visualization, p1, p2, color, thickness)
                
                # Draw points with enhanced markers (circle outline with center dot)
                self._draw_enhanced_uncertainty_point(visualization, p1, uncertainty1)
                self._draw_enhanced_uncertainty_point(visualization, p2, uncertainty2)
                
                # Add measurement label with improved positioning and styling
                mid_point = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
                distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                
                # Convert to cm if pixel spacing available
                if hasattr(dcm, 'PixelSpacing'):
                    pixel_spacing = float(dcm.PixelSpacing[0])
                    distance_mm = distance_pixels * pixel_spacing
                    distance_cm = distance_mm / 10.0  # Convert mm to cm
                    label = f"{distance_cm:.1f}cm"
                else:
                    label = f"{distance_pixels:.1f}px"
                
                # Add uncertainty indicator to label
                if avg_uncertainty > 10:
                    label += f" ±{avg_uncertainty:.1f}"
                
                # Calculate text position with proper offset
                text_offset_y = -50
                
                # Calculate text width to properly offset
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                
                # Offset text to left for _r measurements, right for _l measurements
                if name.endswith('_r'):
                    text_offset_x = -text_width - 10  # Offset to the left
                elif name.endswith('_l'):
                    text_offset_x = 10   # Offset to the right
                else:
                    text_offset_x = -text_width // 2  # Center for other measurements
                
                text_pos = (mid_point[0] + text_offset_x, mid_point[1] + text_offset_y)
                
                # Add text with shadow/outline for better contrast
                self._draw_text_with_outline(visualization, label, text_pos, 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Add ensemble information and disagreement metrics
        if ensemble_info:
            # Main ensemble info
            info_text = f"Ensemble: {len(ensemble_info.get('models', []))} models"
            self._draw_text_with_outline(visualization, info_text, (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
        # Add disagreement metrics at bottom with larger font
        disagreement_data = unified_predictions.get('disagreement_metrics', {})
        if ensemble_info and 'disagreement_metrics' in ensemble_info:
            disagreement_data = ensemble_info['disagreement_metrics']
        
        if disagreement_data:
            # Position metrics at bottom of image
            metrics_y_start = visualization.shape[0] - 150
            metrics_x = 10
            font_scale = 1.0  # Larger font size
            
            # Detection disagreement
            detection_score = disagreement_data.get('detection_disagreement', 0.0)
            if not np.isnan(detection_score):
                detection_text = f"Detection Disagreement: {detection_score:.3f}"
                self._draw_text_with_outline(visualization, detection_text, 
                                           (metrics_x, metrics_y_start),
                                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)
            
            # Localization disagreement
            localization_score = disagreement_data.get('localization_disagreement', 0.0)
            if not np.isnan(localization_score):
                localization_text = f"Localization Disagreement: {localization_score:.3f}"
                self._draw_text_with_outline(visualization, localization_text, 
                                           (metrics_x, metrics_y_start + 35),
                                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)
            
            # Outlier risk
            outlier_score = disagreement_data.get('outlier_risk', 0.0)
            if not np.isnan(outlier_score):
                outlier_text = f"Outlier Risk: {outlier_score:.3f}"
                self._draw_text_with_outline(visualization, outlier_text, 
                                           (metrics_x, metrics_y_start + 70),
                                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)
            
            # Composite risk score (most important - make it stand out)
            composite_score = disagreement_data.get('overall_disagreement_score', 0.0)
            if not np.isnan(composite_score):
                composite_text = f"Overall Score: {composite_score:.3f}"
                self._draw_text_with_outline(visualization, composite_text, 
                                           (metrics_x, metrics_y_start + 105),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)  # Yellow for emphasis
        
        # Save as DICOM
        self._save_as_dicom(visualization, dcm, output_path, "Enhanced QA Visualization")
        self.logger.info(f"Enhanced QA visualization saved to {output_path}")
    
    def _draw_uncertainty_point(self, image, point, uncertainty, point_id):
        """Draw a point with uncertainty visualization."""
        x, y = point
        
        # Base circle
        color = self.get_uncertainty_color(uncertainty)
        cv2.circle(image, (x, y), 5, color, -1)
        
        # Uncertainty circle (larger for higher uncertainty)
        if uncertainty > 0:
            uncertainty_radius = int(5 + uncertainty)
            cv2.circle(image, (x, y), uncertainty_radius, color, 2)
        
        # Point label
        cv2.putText(image, str(point_id), (x + 8, y - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_enhanced_uncertainty_point(self, image, point, uncertainty):
        """Draw an enhanced point with circle outline and center dot."""
        x, y = point
        
        # Get color based on uncertainty
        color = self.get_uncertainty_color(uncertainty)
        
        # Draw circle outline
        cv2.circle(image, (x, y), 8, color, 2)
        
        # Draw center dot
        cv2.circle(image, (x, y), 2, color, -1)
        
        # Uncertainty circle (larger for higher uncertainty)
        if uncertainty > 0:
            uncertainty_radius = int(12 + uncertainty)
            cv2.circle(image, (x, y), uncertainty_radius, color, 1)
    
    def _draw_text_with_outline(self, image, text, position, font, font_scale, color, thickness):
        """Draw text with black outline for better contrast."""
        x, y = position
        
        # Draw black outline (shadow effect)
        cv2.putText(image, text, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(image, text, (x - 1, y - 1), font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(image, text, (x + 1, y - 1), font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(image, text, (x - 1, y + 1), font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(image, text, (x + 1, y + 1), font, font_scale, (0, 0, 0), thickness + 1)
        
        # Draw main text
        cv2.putText(image, text, (x, y), font, font_scale, color, thickness)
    

    
    def create_enhanced_secondary_capture(self,
                                        unified_predictions: Dict,
                                        measurements_data: Dict,
                                        dicom_path: str,
                                        output_path: str,
                                        ensemble_info: Optional[Dict] = None) -> None:
        """
        Create enhanced DICOM Structured Report with measurements, uncertainties, and coordinates.
        
        Args:
            unified_predictions: Fused predictions with uncertainty metrics
            measurements_data: Calculated measurements
            dicom_path: Path to original DICOM file
            output_path: Path to save the structured report
            ensemble_info: Information about ensemble models used
        """
        # Read source DICOM for patient information
        source_dcm = pydicom.dcmread(dicom_path)
        
        # Create DICOM dataset for Structured Report
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.88.11'  # SR Document
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        # Create the dataset
        ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Copy patient and study information
        attrs_to_copy = [
            'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
            'StudyInstanceUID', 'StudyID', 'StudyDate', 'ReferringPhysicianName',
            'AccessionNumber'
        ]
        
        for attr in attrs_to_copy:
            if hasattr(source_dcm, attr):
                setattr(ds, attr, getattr(source_dcm, attr))
        
        # Set required SR attributes
        dt = datetime.now()
        ds.SpecificCharacterSet = 'ISO_IR 100'
        ds.InstanceCreationDate = dt.strftime('%Y%m%d')
        ds.InstanceCreationTime = dt.strftime('%H%M%S')
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.88.11'  # SR Document
        ds.SOPInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesDescription = 'Enhanced Leg Length Analysis Report'
        ds.InstanceNumber = 1
        ds.StudyDate = dt.strftime('%Y%m%d')
        ds.ContentDate = dt.strftime('%Y%m%d')
        ds.ContentTime = dt.strftime('%H%M%S')
        ds.Modality = 'SR'
        ds.Manufacturer = 'STANFORD AIDE'
        ds.CompletionFlag = 'COMPLETE'
        ds.VerificationFlag = 'UNVERIFIED'
        
        # Create main container for measurements
        main_container = Dataset()
        main_container.RelationshipType = 'CONTAINS'
        main_container.ValueType = 'CONTAINER'
        main_container.ContinuityOfContent = 'SEPARATE'
        main_container.ConceptNameCodeSequence = [Dataset()]
        main_container.ConceptNameCodeSequence[0].CodeValue = '99LEG_ENHANCED'
        main_container.ConceptNameCodeSequence[0].CodingSchemeDesignator = '99LEG'
        main_container.ConceptNameCodeSequence[0].CodeMeaning = 'Enhanced Leg Length Analysis'
        
        # Create content sequence for all data
        content_items = []
        
        # 1. Add measurements container
        measurements_container = Dataset()
        measurements_container.RelationshipType = 'CONTAINS'
        measurements_container.ValueType = 'CONTAINER'
        measurements_container.ContinuityOfContent = 'SEPARATE'
        measurements_container.ConceptNameCodeSequence = [Dataset()]
        measurements_container.ConceptNameCodeSequence[0].CodeValue = 'MEASUREMENTS'
        measurements_container.ConceptNameCodeSequence[0].CodingSchemeDesignator = '99LEG'
        measurements_container.ConceptNameCodeSequence[0].CodeMeaning = 'Leg Length Measurements'
        
        measurement_items = []
        for name, data in measurements_data.items():
            # Add centimeter measurement only (removed pixel measurements)
            cm_item = Dataset()
            cm_item.RelationshipType = 'CONTAINS'
            cm_item.ValueType = 'NUM'
            cm_item.ConceptNameCodeSequence = [Dataset()]
            cm_item.ConceptNameCodeSequence[0].CodeValue = f"{name}_cm"
            cm_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = "99LEG"
            cm_item.ConceptNameCodeSequence[0].CodeMeaning = f"{name} length in centimeters"
            cm_item.MeasuredValueSequence = [Dataset()]
            cm_item.MeasuredValueSequence[0].NumericValue = str(round(data['millimeters'] / 10.0, 2))
            cm_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence = [Dataset()]
            cm_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeValue = "cm"
            cm_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodingSchemeDesignator = "UCUM"
            cm_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeMeaning = "centimeters"
            measurement_items.append(cm_item)
            
            # Add confidence if available
            if 'confidence' in data:
                conf_item = Dataset()
                conf_item.RelationshipType = 'CONTAINS'
                conf_item.ValueType = 'NUM'
                conf_item.ConceptNameCodeSequence = [Dataset()]
                conf_item.ConceptNameCodeSequence[0].CodeValue = f"{name}_conf"
                conf_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = "99LEG"
                conf_item.ConceptNameCodeSequence[0].CodeMeaning = f"{name} confidence score"
                conf_item.MeasuredValueSequence = [Dataset()]
                conf_item.MeasuredValueSequence[0].NumericValue = str(round(data['confidence'], 3))
                conf_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence = [Dataset()]
                conf_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeValue = "1"
                conf_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodingSchemeDesignator = "UCUM"
                conf_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeMeaning = "unity"
                measurement_items.append(conf_item)
        
        measurements_container.ContentSequence = measurement_items
        content_items.append(measurements_container)
        
        # 2. Add coordinates container
        coordinates_container = Dataset()
        coordinates_container.RelationshipType = 'CONTAINS'
        coordinates_container.ValueType = 'CONTAINER'
        coordinates_container.ContinuityOfContent = 'SEPARATE'
        coordinates_container.ConceptNameCodeSequence = [Dataset()]
        coordinates_container.ConceptNameCodeSequence[0].CodeValue = 'COORDINATES'
        coordinates_container.ConceptNameCodeSequence[0].CodingSchemeDesignator = '99LEG'
        coordinates_container.ConceptNameCodeSequence[0].CodeMeaning = 'Final Point Coordinates'
        
        coordinate_items = []
        
        # Extract coordinates from unified predictions in (X,Y) format
        for i, (box, label) in enumerate(zip(
            unified_predictions.get('boxes', []),
            unified_predictions.get('labels', [])
        )):
            point_id = int(label)
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            
            # Add coordinate in (X,Y) format
            coord_item = Dataset()
            coord_item.RelationshipType = 'CONTAINS'
            coord_item.ValueType = 'TEXT'
            coord_item.ConceptNameCodeSequence = [Dataset()]
            coord_item.ConceptNameCodeSequence[0].CodeValue = f"P{point_id}_COORD"
            coord_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = "99LEG"
            coord_item.ConceptNameCodeSequence[0].CodeMeaning = f"Point {point_id} coordinate"
            coord_item.TextValue = f"({x_center:.1f}, {y_center:.1f})"
            coordinate_items.append(coord_item)
        
        coordinates_container.ContentSequence = coordinate_items
        content_items.append(coordinates_container)
        
        # 3. Add image-level disagreement metrics container
        disagreement_container = Dataset()
        disagreement_container.RelationshipType = 'CONTAINS'
        disagreement_container.ValueType = 'CONTAINER'
        disagreement_container.ContinuityOfContent = 'SEPARATE'
        disagreement_container.ConceptNameCodeSequence = [Dataset()]
        disagreement_container.ConceptNameCodeSequence[0].CodeValue = 'DISAGREEMENT_METRICS'
        disagreement_container.ConceptNameCodeSequence[0].CodingSchemeDesignator = '99LEG'
        disagreement_container.ConceptNameCodeSequence[0].CodeMeaning = 'Image-Level Disagreement Analysis'
        
        disagreement_items = []
        
        # Get disagreement metrics from unified predictions or ensemble info
        disagreement_data = unified_predictions.get('disagreement_metrics', {})
        if ensemble_info and 'disagreement_metrics' in ensemble_info:
            disagreement_data = ensemble_info['disagreement_metrics']
        
        # Detection Disagreement
        detection_score = disagreement_data.get('detection_disagreement', 0.0)
        detection_item = Dataset()
        detection_item.RelationshipType = 'CONTAINS'
        detection_item.ValueType = 'TEXT'
        detection_item.ConceptNameCodeSequence = [Dataset()]
        detection_item.ConceptNameCodeSequence[0].CodeValue = "DETECTION_DISAGREEMENT"
        detection_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = "99LEG"
        detection_item.ConceptNameCodeSequence[0].CodeMeaning = "Detection Disagreement Analysis"
        
        # Interpret detection disagreement
        if detection_score < 0.3:
            detection_level = "Good: Minor disagreements on point existence"
        elif detection_score < 0.6:
            detection_level = "Moderate: Some disagreements on point existence"
        else:
            detection_level = "Concerning: Major disagreements on point existence"
        
        # Get detection details
        detection_details = disagreement_data.get('detection_details', {})
        missing_points = detection_details.get('missing_points', [])
        total_points = detection_details.get('total_points', 8)
        problem_points = detection_details.get('problem_points', [])
        
        detection_text = f"DETECTION DISAGREEMENT: {detection_score:.3f}\n"
        detection_text += f"→ {detection_level}\n"
        detection_text += f"Details: {len(missing_points)}/{total_points} points missing from some models\n"
        if problem_points:
            detection_text += f"Problem points: {', '.join([f'P{p}' for p in problem_points])}"
        
        detection_item.TextValue = detection_text
        disagreement_items.append(detection_item)
        
        # Localization Disagreement
        localization_score = disagreement_data.get('localization_disagreement', 0.0)
        localization_item = Dataset()
        localization_item.RelationshipType = 'CONTAINS'
        localization_item.ValueType = 'TEXT'
        localization_item.ConceptNameCodeSequence = [Dataset()]
        localization_item.ConceptNameCodeSequence[0].CodeValue = "LOCALIZATION_DISAGREEMENT"
        localization_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = "99LEG"
        localization_item.ConceptNameCodeSequence[0].CodeMeaning = "Localization Disagreement Analysis"
        
        # Interpret localization disagreement
        if localization_score < 0.3:
            localization_level = "Good: Low spatial disagreement between models"
        elif localization_score < 0.6:
            localization_level = "Moderate: Some spatial disagreement between models"
        else:
            localization_level = "Poor: High spatial disagreement between models"
        
        # Get localization details
        localization_details = disagreement_data.get('localization_details', {})
        threshold_exceeded = localization_details.get('points_exceeding_threshold', [])
        threshold_mm = localization_details.get('threshold_mm', 2.0)
        
        localization_text = f"LOCALIZATION DISAGREEMENT: {localization_score:.3f}\n"
        localization_text += f"→ {localization_level}\n"
        localization_text += f"Details: {len(threshold_exceeded)}/8 points exceed {threshold_mm}mm threshold\n"
        if threshold_exceeded:
            problem_list = []
            for point_data in threshold_exceeded:
                point_id = point_data.get('point_id', 'Unknown')
                distance = point_data.get('max_distance_mm', 0)
                problem_list.append(f"P{point_id}({distance:.1f}mm)")
            localization_text += f"Problem points: {', '.join(problem_list)}"
        
        localization_item.TextValue = localization_text
        disagreement_items.append(localization_item)
        
        # Outlier Risk
        outlier_score = disagreement_data.get('outlier_risk', 0.0)
        outlier_item = Dataset()
        outlier_item.RelationshipType = 'CONTAINS'
        outlier_item.ValueType = 'TEXT'
        outlier_item.ConceptNameCodeSequence = [Dataset()]
        outlier_item.ConceptNameCodeSequence[0].CodeValue = "OUTLIER_RISK"
        outlier_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = "99LEG"
        outlier_item.ConceptNameCodeSequence[0].CodeMeaning = "Outlier Risk Analysis"
        
        # Interpret outlier risk
        if outlier_score < 0.2:
            outlier_level = "Low risk: Minor distance disagreements detected"
        elif outlier_score < 0.5:
            outlier_level = "Medium risk: Moderate distance disagreements detected"
        else:
            outlier_level = "High risk: Major distance disagreements detected"
        
        # Get outlier details
        outlier_details = disagreement_data.get('outlier_details', {})
        outlier_threshold_mm = outlier_details.get('threshold_mm', 10.0)
        outlier_points = outlier_details.get('outlier_points', [])
        
        outlier_text = f"OUTLIER RISK: {outlier_score:.3f}\n"
        outlier_text += f"→ {outlier_level}\n"
        if outlier_points:
            outlier_text += f"Details: {len(outlier_points)} points exceed {outlier_threshold_mm}mm disagreement threshold\n"
            problem_list = []
            for point_data in outlier_points:
                point_id = point_data.get('point_id', 'Unknown')
                distance = point_data.get('max_distance_mm', 0)
                problem_list.append(f"P{point_id}({distance:.1f}mm)")
            outlier_text += f"Problem points: {', '.join(problem_list)}"
        else:
            outlier_text += f"Details: No points exceed {outlier_threshold_mm}mm disagreement threshold"
        
        outlier_item.TextValue = outlier_text
        disagreement_items.append(outlier_item)
        
        disagreement_container.ContentSequence = disagreement_items
        content_items.append(disagreement_container)
        
        # 4. Add problematic points as text
        if unified_predictions.get('problematic_points'):
            problems_item = Dataset()
            problems_item.RelationshipType = 'CONTAINS'
            problems_item.ValueType = 'TEXT'
            problems_item.ConceptNameCodeSequence = [Dataset()]
            problems_item.ConceptNameCodeSequence[0].CodeValue = "PROBLEMS"
            problems_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = "99LEG"
            problems_item.ConceptNameCodeSequence[0].CodeMeaning = "Problematic Points"
            
            problem_descriptions = []
            for problem in unified_predictions['problematic_points']:
                problem_descriptions.append(f"Point {problem['point_id']}: {problem['description']}")
            
            problems_item.TextValue = "; ".join(problem_descriptions)
            content_items.append(problems_item)
        
        # 5. Add ensemble information if available
        if ensemble_info:
            ensemble_item = Dataset()
            ensemble_item.RelationshipType = 'CONTAINS'
            ensemble_item.ValueType = 'TEXT'
            ensemble_item.ConceptNameCodeSequence = [Dataset()]
            ensemble_item.ConceptNameCodeSequence[0].CodeValue = "ENSEMBLE"
            ensemble_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = "99LEG"
            ensemble_item.ConceptNameCodeSequence[0].CodeMeaning = "Ensemble Information"
            ensemble_item.TextValue = f"Models used: {', '.join(ensemble_info.get('models', []))}"
            content_items.append(ensemble_item)
        
        # 6. Add pixel spacing information
        if hasattr(source_dcm, 'PixelSpacing'):
            spacing_item = Dataset()
            spacing_item.RelationshipType = 'CONTAINS'
            spacing_item.ValueType = 'TEXT'
            spacing_item.ConceptNameCodeSequence = [Dataset()]
            spacing_item.ConceptNameCodeSequence[0].CodeValue = "PIXEL_SPACING"
            spacing_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = "99LEG"
            spacing_item.ConceptNameCodeSequence[0].CodeMeaning = "Pixel Spacing"
            spacing_item.TextValue = f"Pixel spacing: {source_dcm.PixelSpacing[0]} x {source_dcm.PixelSpacing[1]} mm/pixel"
            content_items.append(spacing_item)
        
        # Set content sequence to main container
        main_container.ContentSequence = content_items
        
        # Add main container to dataset
        ds.ContentSequence = [main_container]
        
        # Save the structured report
        ds.save_as(output_path, write_like_original=False)
        self.logger.info(f"Enhanced structured report saved to {output_path}")
    
    def create_enhanced_json_report(self,
                                  unified_predictions: Dict,
                                  measurements_data: Dict,
                                  dicom_path: str,
                                  output_path: str,
                                  ensemble_info: Optional[Dict] = None,
                                  disagreement_metrics: Optional[Dict] = None) -> None:
        """
        Create enhanced JSON report with comprehensive statistics.
        
        Args:
            unified_predictions: Fused predictions with uncertainty metrics
            measurements_data: Calculated measurements
            dicom_path: Path to original DICOM file
            output_path: Path to save the JSON report
            ensemble_info: Information about ensemble models used
            disagreement_metrics: Disagreement analysis results
        """
        # Get pixel spacing
        dcm = pydicom.dcmread(dicom_path)
        pixel_spacing = None
        if hasattr(dcm, 'PixelSpacing'):
            pixel_spacing = [float(x) for x in dcm.PixelSpacing]
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Build comprehensive report
        report = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'dicom_file': str(Path(dicom_path).name),
                'pixel_spacing': pixel_spacing,
                'analysis_type': 'ensemble' if ensemble_info else 'single_model'
            },
            'measurements': convert_numpy_types(measurements_data),
            'point_uncertainties': convert_numpy_types(unified_predictions.get('uncertainties', {})),
            'point_statistics': convert_numpy_types(unified_predictions.get('point_statistics', {})),
            'problematic_points': convert_numpy_types(unified_predictions.get('problematic_points', [])),
            'quality_assessment': convert_numpy_types(self._assess_quality(unified_predictions, disagreement_metrics)),
        }
        
        # Add ensemble-specific information
        if ensemble_info:
            report['ensemble_info'] = convert_numpy_types(ensemble_info)
        
        # Add disagreement metrics
        if disagreement_metrics:
            report['disagreement_metrics'] = convert_numpy_types(disagreement_metrics)
        
        # Add clinical recommendations
        report['clinical_recommendations'] = convert_numpy_types(
            self._generate_clinical_recommendations(unified_predictions, disagreement_metrics)
        )
        
        # Save JSON report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        self.logger.info(f"Enhanced JSON report saved to {output_path}")
    
    def _assess_quality(self, unified_predictions: Dict, disagreement_metrics: Optional[Dict]) -> Dict:
        """Assess overall quality of the analysis."""
        quality = {
            'overall_score': 'good',  # good, moderate, poor
            'confidence_level': 'high',  # high, medium, low
            'reliability': 'reliable'  # reliable, uncertain, unreliable
        }
        
        # Count problematic points
        num_problematic = len(unified_predictions.get('problematic_points', []))
        total_points = 8
        
        # Assess based on problematic points
        if num_problematic == 0:
            quality['overall_score'] = 'excellent'
            quality['confidence_level'] = 'high'
        elif num_problematic <= 2:
            quality['overall_score'] = 'good'
            quality['confidence_level'] = 'medium'
        elif num_problematic <= 4:
            quality['overall_score'] = 'moderate'
            quality['confidence_level'] = 'medium'
            quality['reliability'] = 'uncertain'
        else:
            quality['overall_score'] = 'poor'
            quality['confidence_level'] = 'low'
            quality['reliability'] = 'unreliable'
        
        # Factor in disagreement metrics if available
        if disagreement_metrics:
            overall_disagreement = disagreement_metrics.get('overall_disagreement_score', 0)
            if not np.isnan(overall_disagreement):
                if overall_disagreement > 0.5:
                    quality['reliability'] = 'uncertain'
                    if quality['overall_score'] == 'excellent':
                        quality['overall_score'] = 'good'
        
        return quality
    
    def _generate_clinical_recommendations(self, unified_predictions: Dict, disagreement_metrics: Optional[Dict]) -> List[str]:
        """Generate clinical recommendations based on analysis results."""
        recommendations = []
        
        # Check for problematic points
        problematic_points = unified_predictions.get('problematic_points', [])
        if problematic_points:
            recommendations.append(f"Found {len(problematic_points)} problematic points requiring review")
            
            # Specific recommendations based on problem types
            problem_types = [p['reason'] for p in problematic_points]
            if 'no_detection' in problem_types:
                recommendations.append("Some anatomical landmarks not detected - consider manual verification")
            if 'high_uncertainty' in problem_types:
                recommendations.append("High spatial uncertainty detected - consider repeat imaging")
            if 'low_confidence' in problem_types:
                recommendations.append("Low confidence predictions detected - expert review recommended")
        
        # Check disagreement metrics
        if disagreement_metrics:
            overall_disagreement = disagreement_metrics.get('overall_disagreement_score', 0)
            if not np.isnan(overall_disagreement):
                if overall_disagreement > 0.5:
                    recommendations.append("High model disagreement - clinical caution advised")
                elif overall_disagreement > 0.2:
                    recommendations.append("Moderate model disagreement - consider additional validation")
        
        # Default recommendation if no issues
        if not recommendations:
            recommendations.append("Analysis completed successfully with high confidence")
        
        return recommendations
    
    def _save_as_dicom(self, image_array, original_dcm, output_path, description):
        """Save image array as DICOM file."""
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        # Create the dataset
        new_ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Copy necessary attributes from original DICOM
        attrs_to_copy = [
            'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
            'StudyInstanceUID', 'StudyID', 'StudyDate', 'SeriesNumber',
            'PixelSpacing', 'AccessionNumber', 'ReferringPhysicianName'
        ]
        
        for attr in attrs_to_copy:
            if hasattr(original_dcm, attr):
                setattr(new_ds, attr, getattr(original_dcm, attr))
        
        # Set required attributes
        new_ds.SeriesInstanceUID = generate_uid()
        new_ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        new_ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        new_ds.SeriesDescription = description
        new_ds.InstanceNumber = 1
        
        # Set image-specific attributes
        new_ds.SamplesPerPixel = 3
        new_ds.PhotometricInterpretation = 'RGB'
        new_ds.PlanarConfiguration = 0
        new_ds.BitsAllocated = 8
        new_ds.BitsStored = 8
        new_ds.HighBit = 7
        new_ds.PixelRepresentation = 0
        new_ds.Rows = image_array.shape[0]
        new_ds.Columns = image_array.shape[1]
        
        # Set additional required attributes
        new_ds.ImagePositionPatient = ['0', '0', '0']
        new_ds.ImageOrientationPatient = ['1', '0', '0', '0', '1', '0']
        new_ds.FrameOfReferenceUID = generate_uid()
        new_ds.Modality = 'SC'
        new_ds.ConversionType = 'WSD'
        
        dt = datetime.now()
        new_ds.ContentDate = dt.strftime('%Y%m%d')
        new_ds.ContentTime = dt.strftime('%H%M%S')
        
        # Set pixel data
        new_ds.PixelData = image_array.tobytes()
        
        # Save the DICOM
        new_ds.is_implicit_VR = False
        new_ds.is_little_endian = True
        new_ds.save_as(output_path, write_like_original=False) 