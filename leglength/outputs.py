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
    
    def create_qa_dicom(self, predictions: Dict, dicom_path: str, output_path: str, processor: 'ImageProcessor' = None) -> None:
        """
        Create a QA DICOM with visualizations of detected points and measurements.
        
        Args:
            predictions: Dictionary containing boxes, scores, and labels
            dicom_path: Path to original DICOM file
            output_path: Path to save the QA DICOM
            processor: ImageProcessor instance used for preprocessing
        """
        # Load original DICOM
        dcm = pydicom.dcmread(dicom_path)
        original_image = dcm.pixel_array.astype(float)
        
        # Normalize to 0-255 and convert to RGB
        normalized_image = ((original_image - original_image.min()) / 
                          (original_image.max() - original_image.min()) * 255).astype(np.uint8)
        visualization = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)
        
        # Extract keypoints from predictions (boxes are already in original space)
        keypoints = {}
        for i, (box, label) in enumerate(zip(predictions['boxes'], predictions['labels'])):
            x_center = int((box[0] + box[2]) / 2)
            y_center = int((box[1] + box[3]) / 2)
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
                        midpoint_x = int(sum(x_coords) / len(x_coords))
                        midpoint_y = int(sum(y_coords) / len(y_coords))
                        keypoints[name] = (midpoint_x, midpoint_y)
                    else:
                        logger.warning(f"Missing source points for derived point {name}")
        
        # Draw measurements
        for measure in self.config['measurements']:
            name = measure['name']
            point1, point2 = measure['join_points']
            
            # Skip if either point is missing
            if point1 not in keypoints or point2 not in keypoints:
                logger.warning(f"Missing points for measurement {name}")
                continue
            
            p1 = keypoints[point1]
            p2 = keypoints[point2]
            color = self.colors.get(name, (255,31,223))  # Blue as default
            
            # Draw line between points
            cv2.line(visualization, p1, p2, color, 2)
            
            # Draw points
            cv2.circle(visualization, p1, 5, color, -1)  # Filled circle
            cv2.circle(visualization, p2, 5, color, -1)  # Filled circle
            
            # Add label with distance
            if name in self.measurements:
                mid_point = (
                    int((p1[0] + p2[0]) / 2),
                    int((p1[1] + p2[1]) / 2)
                )
                distance_mm = self.measurements[name]['millimeters']
                label = f"{name}: {distance_mm:.1f}mm"
                label = f"{distance_mm:.1f}mm"
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
                
                # Add text with offset
                cv2.putText(
                    visualization,
                    label,
                    (mid_point[0] - text_width//2, mid_point[1] - 40),  # Offset text up by 40 pixels
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255,31,223),
                    2,
                    cv2.LINE_AA
                )
        # Create new DICOM dataset
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage
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
            if hasattr(dcm, attr):
                setattr(new_ds, attr, getattr(dcm, attr))
        
        # Set required attributes
        new_ds.SeriesInstanceUID = generate_uid()  # New series
        new_ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        new_ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        new_ds.SeriesDescription = 'QA Visualization'
        new_ds.InstanceNumber = 1
        
        # Set image-specific attributes
        new_ds.SamplesPerPixel = 3
        new_ds.PhotometricInterpretation = 'RGB'
        new_ds.PlanarConfiguration = 0
        new_ds.BitsAllocated = 8
        new_ds.BitsStored = 8
        new_ds.HighBit = 7
        new_ds.PixelRepresentation = 0
        new_ds.Rows = visualization.shape[0]
        new_ds.Columns = visualization.shape[1]
        
        # Set required image position and orientation attributes
        new_ds.ImagePositionPatient = ['0', '0', '0']
        new_ds.ImageOrientationPatient = ['1', '0', '0', '0', '1', '0']
        new_ds.FrameOfReferenceUID = generate_uid()
        
        # Additional required attributes
        new_ds.Modality = 'SC'  # Secondary Capture
        new_ds.ConversionType = 'WSD'  # Workstation
        dt = datetime.now()
        new_ds.ContentDate = dt.strftime('%Y%m%d')
        new_ds.ContentTime = dt.strftime('%H%M%S')
        new_ds.AcquisitionDate = dt.strftime('%Y%m%d')
        new_ds.AcquisitionTime = dt.strftime('%H%M%S')
        
        # Set pixel data
        new_ds.PixelData = visualization.tobytes()
        
        # Save the QA DICOM
        new_ds.is_implicit_VR = False
        new_ds.is_little_endian = True
        new_ds.save_as(output_path, write_like_original=False)
        logger.info(f"Saved QA visualization DICOM to {output_path}")

    
    def _classify_discrepancy(self, diff_cm: float, right_cm: float, left_cm: float,
                            threshold_cm: float, category: str) -> str:
        # category: 'femur', 'tibia', 'total'
        if threshold_cm is None:
            return ""

        if diff_cm <= threshold_cm:
            if category == 'femur':
                return "There is no femoral length discrepancy."
            elif category == 'tibia':
                return "There is no tibial length discrepancy."
            else:
                return "There is no leg length discrepancy."
        else:
            if category == 'femur':
                return "The right femur is longer." if right_cm > left_cm else "The left femur is longer."
            elif category == 'tibia':
                return "The right tibia is longer." if right_cm > left_cm else "The left tibia is longer."
            else:
                return "The right leg is longer." if right_cm > left_cm else "The left leg is longer."

    def calculate_distances(self, predictions: Dict, dicom_path: str, threshold_cm: float = None) -> Dict[str, Dict[str, float]]:
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
        
        # Derive right/left totals and differences (cm)
        # Expect keys: femur_r, tibia_r, femur_l, tibia_l
        def get_cm(k): return measurements.get(k, {}).get('centimeters', 0.0)
        femur_r_cm = get_cm('femur_r')
        tibia_r_cm = get_cm('tibia_r')
        femur_l_cm = get_cm('femur_l')
        tibia_l_cm = get_cm('tibia_l')

        total_right_cm = femur_r_cm + tibia_r_cm
        total_left_cm  = femur_l_cm + tibia_l_cm

        femur_diff_cm = abs(femur_r_cm - femur_l_cm)
        tibia_diff_cm = abs(tibia_r_cm - tibia_l_cm)
        total_diff_cm = abs(total_right_cm - total_left_cm)

        # Store derived metrics
        measurements['femur_r_cm'] = {'millimeters': femur_r_cm * 10, 'centimeters': femur_r_cm}
        measurements['tibia_r_cm'] = {'millimeters': tibia_r_cm * 10, 'centimeters': tibia_r_cm}
        measurements['femur_l_cm'] = {'millimeters': femur_l_cm * 10, 'centimeters': femur_l_cm}
        measurements['tibia_l_cm'] = {'millimeters': tibia_l_cm * 10, 'centimeters': tibia_l_cm}

        measurements['total_right_cm'] = {'millimeters': total_right_cm * 10, 'centimeters': total_right_cm}
        measurements['total_left_cm']  = {'millimeters': total_left_cm * 10, 'centimeters': total_left_cm}
        measurements['femur_diff_cm'] = {'value_cm': femur_diff_cm}
        measurements['tibia_diff_cm'] = {'value_cm': tibia_diff_cm}
        measurements['total_diff_cm'] = {'value_cm': total_diff_cm}

        # Discrepancy messages (threshold-driven)
        femur_disc  = self._classify_discrepancy(femur_diff_cm, femur_r_cm, femur_l_cm, threshold_cm, 'femur')
        tibia_disc  = self._classify_discrepancy(tibia_diff_cm, tibia_r_cm, tibia_l_cm, threshold_cm, 'tibia')
        total_disc  = self._classify_discrepancy(total_diff_cm, total_right_cm, total_left_cm, threshold_cm, 'total')

        measurements['femur_discrepancy'] = {'text': femur_disc}
        measurements['tibia_discrepancy'] = {'text': tibia_disc}
        measurements['total_discrepancy'] = {'text': total_disc}

        self.measurements = measurements
        return measurements
    
    def create_structured_report(self, output_path: str, source_dicom_path: str) -> None:
        """
        Create a DICOM Structured Report containing the measurements.
        
        Args:
            output_path: Path to save the structured report
            source_dicom_path: Path to source DICOM file for patient info
        """
        # Read source DICOM for patient information
        source_dcm = pydicom.dcmread(source_dicom_path)
        
        # Create DICOM dataset
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
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.88.11'
        ds.SOPInstanceUID = generate_uid()
        ds.StudyDate = dt.strftime('%Y%m%d')
        ds.ContentDate = dt.strftime('%Y%m%d')
        ds.ContentTime = dt.strftime('%H%M%S')
        ds.Modality = 'SR'
        ds.Manufacturer = 'STANFORD AIDE'
        ds.CompletionFlag = 'COMPLETE'
        ds.VerificationFlag = 'UNVERIFIED'
        
        # Create container for measurements
        container = Dataset()
        container.RelationshipType = 'CONTAINS'
        container.ValueType = 'CONTAINER'
        container.ContinuityOfContent = 'SEPARATE'
        container.ConceptNameCodeSequence = [Dataset()]
        container.ConceptNameCodeSequence[0].CodeValue = '99LEG_MEASURE'
        container.ConceptNameCodeSequence[0].CodingSchemeDesignator = '99LEG'
        container.ConceptNameCodeSequence[0].CodeMeaning = 'Leg Length'
        
        # Create content sequence for measurements
        content_seq = []
        
        # Add each measurement as a separate content item
        for name, data in self.measurements.items():
            # Add millimeter measurement
            mm_item = Dataset()
            mm_item.RelationshipType = 'CONTAINS'
            mm_item.ValueType = 'NUM'
            mm_item.ConceptNameCodeSequence = [Dataset()]
            mm_item.ConceptNameCodeSequence[0].CodeValue = f"{name.replace(' ', '_').lower()}_mm"
            mm_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = "99LEG"
            mm_item.ConceptNameCodeSequence[0].CodeMeaning = f"{name} length in millimeters"
            mm_item.MeasuredValueSequence = [Dataset()]
            mm_item.MeasuredValueSequence[0].NumericValue = str(round(data['millimeters'], 2))
            mm_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence = [Dataset()]
            mm_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeValue = "mm"
            mm_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodingSchemeDesignator = "UCUM"
            mm_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeMeaning = "millimeters"
            
            content_seq.extend([mm_item])
            
            # Add text version with "Properties:" prefix
            # text_item = Dataset()
            # text_item.RelationshipType = 'CONTAINS'
            # text_item.ValueType = 'TEXT'
            # text_item.ConceptNameCodeSequence = [Dataset()]
            # text_item.ConceptNameCodeSequence[0].CodeValue = f"{name.replace(' ', '_').lower()}_text"
            # text_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = "99LEG"
            # text_item.ConceptNameCodeSequence[0].CodeMeaning = name
            # text_item.TextValue = f"Properties: {name} = \"{data['millimeters']:.1f} mm\""
            
            # content_seq.extend([mm_item, text_item])
        
        # Add content sequence to container
        container.ContentSequence = content_seq
        
        # Add container to dataset
        ds.ContentSequence = [container]
        
        # Save the structured report
        ds.save_as(output_path, write_like_original=False)
        logger.info(f"Saved structured report to {output_path}")
    
    def save_json_report(self, output_path: str) -> None:
        """
        Save measurements as JSON file.
        
        Args:
            output_path: Path to save the JSON report
        """
        with open(output_path, 'w') as f:
            json.dump({
                'measurements': self.measurements,
                'discrepancies': {
                    'femur': self.measurements.get('femur_discrepancy', {}).get('text', None),
                    'tibia': self.measurements.get('tibia_discrepancy', {}).get('text', None),
                    'total': self.measurements.get('total_discrepancy', {}).get('text', None),
                },
                'pixel_spacing': [float(x) for x in self.pixel_spacing] if self.pixel_spacing else None
            }, f, indent=4)
        logger.info(f"Saved JSON report to {output_path}")
