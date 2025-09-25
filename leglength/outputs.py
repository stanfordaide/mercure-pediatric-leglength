import json
import os
import logging
import cv2
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

from typing import List

class StanfordAIDECodes:
    """Centralized Stanford AIDE codes and constants"""
    CODING_SCHEME = "STANFORD_AIDE"
    VERSION = "1.0"
    
    # Algorithm codes
    ALGORITHMS = {
        'pediatric_leg_length': 'PLL_001'
    }
    
    # Status codes
    STATUS = {
        'pending': 'PENDING',
        'processing': 'PROCESSING', 
        'complete': 'COMPLETE',
        'failed': 'FAILED'
    }
    
    # SOP Class UIDs
    SOP_CLASS_UIDS = {
        'secondary_capture': '1.2.840.10008.5.1.4.1.1.7',
        'structured_report': '1.2.840.10008.5.1.4.1.1.88.11'
    }
    
    # Measurement codes
    MEASUREMENTS = {
        'container': 'PLL_002',
        'length_mm': 'PLL_003',
        'length_cm': 'PLL_004'
    }
    
    # Institution info
    INSTITUTION = {
        'manufacturer': 'StanfordAIDE',
        'institution_name': 'SOM',
        'department': 'Radiology',
        'station_name': 'LPCH'
    }
    
    # Transfer syntax
    TRANSFER_SYNTAX = pydicom.uid.ExplicitVRLittleEndian
    
    # Color thresholds for uncertainty visualization
    UNCERTAINTY_THRESHOLDS = {
        'normal': 0.33,
        'high': 0.66
    }

class DicomProcessor:
    """DICOM processing class for Stanford AIDE pediatric leg length measurements"""
    
    def __init__(self, config_path: str = None, logger: logging.Logger = None):
        self.codes = StanfordAIDECodes()
        self.logger = logger or logging.getLogger(__name__)
        self._series_number_cache = {}  # Cache series numbers by StudyInstanceUID
        # Load config
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'measurement_configs.json')
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def _get_next_series_number(self, source_dcm: Dataset) -> str:
        """Get the next available series number for the study"""
        study_uid = getattr(source_dcm, 'StudyInstanceUID', None)
        if not study_uid:
            return "9999"  # Fallback if no study UID
        
        # Check cache first
        if study_uid in self._series_number_cache:
            self._series_number_cache[study_uid] += 1
            return str(self._series_number_cache[study_uid])
        
        # Find highest series number in source DICOM's study
        # This would require querying the PACS/Orthanc, but for now we'll use source + offset
        source_series_num = getattr(source_dcm, 'SeriesNumber', None)
        if source_series_num:
            try:
                base_num = int(source_series_num)
                # Start our series numbers at base + 1000 to avoid conflicts
                next_num = base_num + 1000
            except (ValueError, TypeError):
                next_num = 9000  # Fallback for non-numeric series numbers
        else:
            next_num = 9000
        
        self._series_number_cache[study_uid] = next_num
        return str(next_num)
    
    def _get_point_uncertainty(self, point_id, results: dict) -> float:
        """Get spatial uncertainty for a point (handle both direct points and derived points)"""
        if isinstance(point_id, int):
            # Direct point
            return results.get('uncertainties', {}).get(point_id, {}).get('localization_disagreement', 0.0)
        elif isinstance(point_id, str):
            # Derived point
            for derived in self.config.get('derived_points', []):
                if derived['name'] == point_id:
                    uncertainties = [self._get_point_uncertainty(sp, results) for sp in derived['source_points']]
                    return max(uncertainties) if uncertainties else 0.0
        return 0.0  # Default for missing data

    def _get_localization_uncertainty(self, measure: str, results: dict) -> float:
        """Calculate the maximum uncertainty for points involved in a measurement"""
        join_points = None
        
        # Find the measurement in config to get join_points
        for config_measure in self.config['measurements']:
            if config_measure['name'] == measure:
                join_points = config_measure['join_points']
                break
        
        if not join_points:
            return 0.0
        
        uncertainties = [self._get_point_uncertainty(point, results) for point in join_points]
        
        self.logger.info(f"All uncertainties for {measure}: {uncertainties}")
        return max(uncertainties) if uncertainties else 0.0

    def _get_uncertainty_color(self, uncertainty: float) -> tuple:
        """Return professional medical imaging color based on uncertainty level"""
        thresholds = self.codes.UNCERTAINTY_THRESHOLDS
        if uncertainty <= thresholds['normal']:
            return (255, 255, 0)    # Cyan - confident (medical imaging standard)
        elif uncertainty <= thresholds['high']:
            return (140, 255, 255)  # Orange - moderate uncertainty  
        else:
            return (255, 255, 255)   # White - high uncertainty (high contrast)

    def _add_stanford_aide_headers(self, ds: Dataset, algorithm_name: str = 'pediatric_leg_length', 
                                 status: str = 'complete', results: dict = None) -> Dataset:
        """Add Stanford AIDE specific headers to existing dataset"""
        # Stanford AIDE identification
        ds.Manufacturer = self.codes.INSTITUTION['manufacturer']
        ds.SoftwareVersions = f"{algorithm_name}_v{self.codes.VERSION}"
        ds.InstitutionName = self.codes.INSTITUTION['institution_name']
        ds.InstitutionalDepartmentName = self.codes.INSTITUTION['department']
        ds.StationName = self.codes.INSTITUTION['station_name']
        
        # Core Stanford AIDE private tags (group 0x7001)
        # Core Stanford AIDE private tags (group 0x7001) - USE CORRECT VR TYPES
        ds.add_new([0x7001, 0x0001], 'LO', self.codes.ALGORITHMS.get(algorithm_name, 'UNKNOWN'))  # Algorithm ID - LO instead of SH
        ds.add_new([0x7001, 0x0002], 'CS', self.codes.STATUS.get(status, 'UNKNOWN'))  # Processing Status - CS is fine
        ds.add_new([0x7001, 0x0003], 'DT', datetime.now().strftime('%Y%m%d%H%M%S'))  # Processing Timestamp - DT is fine
        
        # Optional additional metadata if results provided
        if results:
            # Overall confidence score
            confidence = results.get('overall_confidence', 0.0)
            ds.add_new([0x7001, 0x0004], 'DS', str(confidence))  # Confidence Score
            
            # Processing duration
            duration = results.get('processing_duration_seconds', 0.0)
            ds.add_new([0x7001, 0x0005], 'DS', str(duration))  # Processing Duration
            
            # Source image reference
            source_sop = results.get('source_sop_instance_uid', '')
            if source_sop:
                ds.add_new([0x7001, 0x0006], 'UI', source_sop)  # Source SOP Instance UID
        
        return ds

    def _copy_all_relevant_headers(self, source_ds: Dataset, new_ds: Dataset) -> Dataset:
        """Copy only relevant headers from source DICOM, excluding problematic ones."""
        
        # Headers to explicitly copy (add or remove as needed)
        headers_to_copy = [
            'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
            'StudyInstanceUID', 'StudyID', 'StudyDate', 'AccessionNumber',
            'ReferringPhysicianName', 'PatientAge', 'PatientSize', 'PatientWeight',
            'MedicalAlerts', 'Allergies', 'PregnancyStatus', 'StudyDescription'
        ]
        
        # Headers to explicitly exclude
        headers_to_exclude = [
            'PixelData', 'SOPInstanceUID', 'SeriesInstanceUID', 'InstanceNumber',
            'SeriesDescription', 'SeriesNumber', 'ContentDate', 'ContentTime',
            'AcquisitionDate', 'AcquisitionTime', 'PhotometricInterpretation',
            'Rows', 'Columns', 'SamplesPerPixel', 'BitsAllocated', 'BitsStored',
            'HighBit', 'PixelRepresentation', 'WindowCenter', 'WindowWidth',
            'RescaleIntercept', 'RescaleSlope', 'ImageType', 'SOPClassUID',
            'Modality', 'BodyPartExamined', 'PresentationLUTShape',
            'LossyImageCompression', 'LossyImageCompressionRatio',
            'LossyImageCompressionMethod'
        ]
        
        for elem in source_ds:
            if elem.keyword in headers_to_copy:
                new_ds.add(elem)
            elif elem.keyword not in headers_to_exclude:
                # For any header not explicitly included or excluded, log it
                self.logger.debug(f"Skipping header: {elem.keyword}")
        
        return new_ds

    def _add_headers(self, dcm: pydicom.Dataset, sop_class_uid: str, modality: str = None, 
                   series_description: str = None, algorithm_name: str = 'pediatric_leg_length',
                   status: str = 'complete', results: dict = None) -> Dataset:
        """
        Create new DICOM dataset copying all relevant headers and adding Stanford AIDE headers.
        """
        # Create new DICOM dataset
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = sop_class_uid
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = self.codes.TRANSFER_SYNTAX
        
        # Create the dataset
        new_ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Copy ALL relevant headers from source
        new_ds = self._copy_all_relevant_headers(dcm, new_ds)
        
        
        
        # Set new instance-specific attributes
        dt = datetime.now()
        new_ds.SeriesInstanceUID = generate_uid()  # New series
        new_ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        new_ds.SOPClassUID = sop_class_uid
        new_ds.InstanceNumber = 1
        new_ds.SeriesNumber = self._get_next_series_number(dcm)  # Bump series number
        new_ds.ContentDate = dt.strftime('%Y%m%d')
        new_ds.ContentTime = dt.strftime('%H%M%S')
        new_ds.AcquisitionDate = dt.strftime('%Y%m%d')
        new_ds.AcquisitionTime = dt.strftime('%H%M%S')
        
        # Set modality-specific attributes
        if modality == 'SC':  # Secondary Capture
            new_ds.Modality = 'SC'
            new_ds.ConversionType = 'WSD'  # Workstation
            new_ds.SeriesDescription = series_description or 'QA'
            new_ds.InstanceCreationDate = dt.strftime('%Y%m%d')
            new_ds.InstanceCreationTime = dt.strftime('%H%M%S')
            
            
            # Image-specific attributes for Secondary Capture
            new_ds.SamplesPerPixel = 3
            new_ds.PhotometricInterpretation = 'RGB'
            new_ds.PlanarConfiguration = 0
            new_ds.BitsAllocated = 8
            new_ds.BitsStored = 8
            new_ds.HighBit = 7
            new_ds.PixelRepresentation = 0
            new_ds.ImagePositionPatient = ['0', '0', '0']
            new_ds.ImageOrientationPatient = ['1', '0', '0', '0', '1', '0']
            new_ds.FrameOfReferenceUID = generate_uid()
            
        elif modality == 'SR':  # Structured Report
            new_ds.Modality = 'SR'
            new_ds.SeriesDescription = series_description or 'Measurenments'
            new_ds.SpecificCharacterSet = 'ISO_IR 100'
            new_ds.InstanceCreationDate = dt.strftime('%Y%m%d')
            new_ds.InstanceCreationTime = dt.strftime('%H%M%S')
            new_ds.CompletionFlag = 'COMPLETE'
            new_ds.VerificationFlag = 'UNVERIFIED'
        
        # Add Stanford AIDE specific headers
        new_ds = self._add_stanford_aide_headers(new_ds, algorithm_name, status, results)
        
        return new_ds

    def _create_visualization_image(self, results: dict, dicom_path: str) -> np.ndarray:
        """Create the visualization image with measurements and annotations."""
        self.logger.info("=== Starting visualization creation ===")
        
        # Load DICOM
        dcm = pydicom.dcmread(dicom_path)
        original_image = dcm.pixel_array.astype(float)
        
        
        # Check for problematic images
        if original_image.size == 0:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        
        if original_image.min() == original_image.max():
            # Create a test pattern
            test_img = np.random.randint(0, 255, original_image.shape, dtype=np.uint8)
            visualization = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
        else:
            # Apply medical imaging contrast enhancement
            # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
            img_range = original_image.max() - original_image.min()
            normalized_image = ((original_image - original_image.min()) / img_range * 255).astype(np.uint8)
            
            # Apply CLAHE for better bone visibility
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_image = clahe.apply(normalized_image)
            
            # Convert to RGB with slight blue tint (medical imaging standard)
            visualization = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
            # Add subtle blue tint for medical aesthetic
            visualization[:,:,0] = (visualization[:,:,0] * 0.95).astype(np.uint8)  # Reduce red
            visualization[:,:,1] = (visualization[:,:,1] * 0.98).astype(np.uint8)  # Slightly reduce green
        

        
        # Draw all 8 keypoints first
        self._draw_all_keypoints(visualization, results)
        
        # Add professional legend for keypoint symbols
        self._add_keypoint_legend(visualization)
        
        # Add medical imaging overlay with patient info if available
        self._add_medical_overlay(visualization, dcm)
        
        measurements_processed = 0
        
        # Draw measurements
        for measure in self.config['measurements']:
            name = measure['name']
            join_points = measure['join_points']
            

            if name not in results.get('measurements', {}):
                continue

            try:
                points = results['measurements'][name]['points']
                p1 = (int(points['start']['x']), int(points['start']['y']))
                p2 = (int(points['end']['x']), int(points['end']['y']))
                
                
                # Validate coordinates are within image bounds
                h, w = visualization.shape[:2]
                if not (0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h):
                    # Clamp to image bounds
                    p1 = (max(0, min(w-1, p1[0])), max(0, min(h-1, p1[1])))
                    p2 = (max(0, min(w-1, p2[0])), max(0, min(h-1, p2[1])))
                # Draw professional measurement line with gradient effect
                self._draw_measurement_line(visualization, p1, p2, name)

                # Draw professional measurement points with uncertainty-based styling
                for i, (point, point_id) in enumerate(zip([p1, p2], join_points)):
                    uncertainty = self._get_point_uncertainty(point_id, results)
                    self._draw_measurement_point(visualization, point, uncertainty, i == 0)

                # Add professional measurement label with callout
                mid_point = (
                    int((p1[0] + p2[0]) / 2),
                    int((p1[1] + p2[1]) / 2)
                )
                distance_cm = results['measurements'][name]['centimeters']
                
                # Create professional measurement label
                self._draw_measurement_label(visualization, mid_point, distance_cm, name, uncertainty)
                
                measurements_processed += 1
                
            except Exception as e:
                self.logger.error(f"Error processing measurement {name}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        self.logger.info(f"Processed {measurements_processed} measurements")
        
        # Save debug image
        debug_path = '/output/debug_qa_image.png'
        cv2.imwrite(debug_path, visualization)
        self.logger.info(f"Debug image saved to {debug_path}")
        
        # Final check
        self.logger.info(f"Final visualization - shape: {visualization.shape}, min/max: {visualization.min()}/{visualization.max()}")
        
        # Before returning, ensure the image is in the correct format
        if visualization.dtype != np.uint8:
            visualization = (visualization * 255).astype(np.uint8)
        
        if visualization.ndim == 2:
            visualization = cv2.cvtColor(visualization, cv2.COLOR_GRAY2RGB)
        elif visualization.shape[2] == 4:  # RGBA
            visualization = cv2.cvtColor(visualization, cv2.COLOR_RGBA2RGB)
        
        # Add professional watermark
        self._add_stanford_watermark(visualization)
        
        return visualization

    def _draw_measurement_line(self, visualization: np.ndarray, p1: tuple, p2: tuple, measurement_name: str):
        """Draw professional measurement line with gradient effect."""
        # Draw professional measurement line with medical imaging standards
        # Main measurement line in cyan for high contrast against X-ray backgrounds
        accent_color = (255, 255, 0)  # Cyan in BGR format
        cv2.line(visualization, p1, p2, accent_color, 3)  # Consistent 3px thickness
        
        # Add measurement arrows/caps at endpoints
        self._draw_measurement_caps(visualization, p1, p2)

    def _draw_measurement_caps(self, visualization: np.ndarray, p1: tuple, p2: tuple):
        """Draw professional measurement endpoint caps."""
        # Calculate perpendicular direction for caps
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            # Normalize and get perpendicular
            nx = -dy / length
            ny = dx / length
            
            cap_length = 12
            # Draw caps at both ends
            for point in [p1, p2]:
                cap_p1 = (int(point[0] + nx * cap_length), int(point[1] + ny * cap_length))
                cap_p2 = (int(point[0] - nx * cap_length), int(point[1] - ny * cap_length))
                cv2.line(visualization, cap_p1, cap_p2, (255, 255, 0), 2)  # Cyan caps with 2px thickness

    def _draw_measurement_point(self, visualization: np.ndarray, point: tuple, uncertainty: float, is_start: bool):
        """Draw professional measurement endpoint with uncertainty indication."""
        color = self._get_uncertainty_color(uncertainty)
        
        # Draw measurement endpoint with professional medical imaging style
        # Use larger cyan hollow circle for measurement endpoints
        cv2.circle(visualization, point, 8, (255, 255, 0), 2)  # Larger cyan hollow circle
        # Add crosshair for exact position
        cv2.line(visualization, (point[0]-4, point[1]), (point[0]+4, point[1]), (255, 255, 255), 1)
        cv2.line(visualization, (point[0], point[1]-4), (point[0], point[1]+4), (255, 255, 255), 1)

    def _draw_measurement_label(self, visualization: np.ndarray, mid_point: tuple, distance_cm: float, 
                               measurement_name: str, uncertainty: float):
        """Draw professional measurement label with callout."""
        h, w = visualization.shape[:2]
        
        # Create clean measurement label - only show distance value
        main_label = f"{distance_cm:.1f} cm"
        # Remove anatomical labels per medical imaging standards
        sub_label = ""  # No sub-label needed
        
        # Professional font settings
        main_font = cv2.FONT_HERSHEY_DUPLEX
        sub_font = cv2.FONT_HERSHEY_SIMPLEX
        main_scale = 1.2
        sub_scale = 0.6
        main_thickness = 2
        sub_thickness = 1
        
        # Get text dimensions - only for main label
        (main_w, main_h), _ = cv2.getTextSize(main_label, main_font, main_scale, main_thickness)
        
        # Calculate label box dimensions - simplified for single line
        box_width = main_w + 20
        box_height = main_h + 10
        
        # Position label box (offset from measurement line)
        label_x = max(10, min(w - box_width - 10, mid_point[0] - box_width//2))
        label_y = max(box_height + 10, min(h - 10, mid_point[1] - 60))
        
        # Draw professional label background
        overlay = visualization.copy()
        cv2.rectangle(overlay, 
                     (label_x - 10, label_y - box_height - 5), 
                     (label_x + box_width, label_y + 5), 
                     (0, 0, 0), -1)  # Black background
        
        # Apply transparency
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0, visualization)
        
        # Add border with uncertainty color
        border_color = self._get_uncertainty_color(uncertainty)
        cv2.rectangle(visualization, 
                     (label_x - 10, label_y - box_height - 5), 
                     (label_x + box_width, label_y + 5), 
                     border_color, 2)
        
        # Draw callout line to measurement in cyan
        cv2.line(visualization, (mid_point[0], mid_point[1]), 
                (label_x + box_width//2, label_y), (255, 255, 0), 2)  # Cyan callout line
        
        # Draw text - only measurement value in white
        text_x = label_x + 10
        cv2.putText(visualization, main_label, (text_x, label_y - 5), 
                   main_font, main_scale, (255, 255, 255), main_thickness, cv2.LINE_AA)

    def _draw_professional_label(self, visualization: np.ndarray, center_x: int, center_y: int, 
                                point_name: str, was_detected: bool):
        """Draw professional point label with callout."""
        h, w = visualization.shape[:2]
        
        # Professional label styling
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # Get text dimensions
        (text_width, text_height), _ = cv2.getTextSize(point_name, font, font_scale, font_thickness)
        
        # Smart label positioning to avoid overlap
        offset_distance = 25
        label_x = center_x + offset_distance
        label_y = center_y - offset_distance
        
        # Clamp to image bounds
        label_x = max(text_width//2, min(w - text_width//2, label_x))
        label_y = max(text_height + 5, min(h - 5, label_y))
        
        # Create label background
        padding = 4
        bg_x1 = label_x - text_width//2 - padding
        bg_y1 = label_y - text_height - padding
        bg_x2 = label_x + text_width//2 + padding  
        bg_y2 = label_y + padding
        
        # Draw label background with transparency
        overlay = visualization.copy()
        color = (0, 150, 0) if was_detected else (0, 80, 200)
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0, visualization)
        
        # Draw border
        cv2.rectangle(visualization, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 1)
        
        # Draw callout line
        cv2.line(visualization, (center_x, center_y), 
                (label_x, label_y - text_height//2), (255, 255, 255), 1)
        
        # Draw text
        cv2.putText(visualization, point_name, (label_x - text_width//2, label_y - 2), 
                   font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    def _add_medical_overlay(self, visualization: np.ndarray, dcm: pydicom.Dataset):
        """Add medical imaging overlay with patient info."""
        h, w = visualization.shape[:2]
        
        # Get patient info from DICOM
        patient_info = []
        if hasattr(dcm, 'PatientID'):
            patient_info.append(f"ID: {dcm.PatientID}")
        if hasattr(dcm, 'PatientAge'):
            patient_info.append(f"Age: {dcm.PatientAge}")
        if hasattr(dcm, 'StudyDate'):
            date_str = dcm.StudyDate
            if len(date_str) == 8:
                formatted_date = f"{date_str[4:6]}/{date_str[6:8]}/{date_str[:4]}"
                patient_info.append(f"Date: {formatted_date}")
        
        if patient_info:
            # Position in top-right corner
            overlay_x = w - 250
            overlay_y = h - 100
            
            # Create overlay background
            overlay = visualization.copy()
            cv2.rectangle(overlay, (overlay_x - 10, overlay_y - 60), 
                         (w - 10, overlay_y + 10), (0, 0, 0), -1)
            
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0, visualization)
            
            # Add border
            cv2.rectangle(visualization, (overlay_x - 10, overlay_y - 60), 
                         (w - 10, overlay_y + 10), (100, 100, 100), 1)
            
            # Add patient info
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            for i, info in enumerate(patient_info):
                cv2.putText(visualization, info, (overlay_x, overlay_y - 40 + i * 20), 
                           font, font_scale, (200, 200, 200), 1, cv2.LINE_AA)

    def _add_stanford_watermark(self, visualization: np.ndarray):
        """Add Stanford AIDE watermark."""
        h, w = visualization.shape[:2]
        
        # Position watermark in bottom-right
        watermark_text = "Stanford AIDE"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1
        
        (text_width, text_height), _ = cv2.getTextSize(watermark_text, font, font_scale, font_thickness)
        
        x = w - text_width - 20
        y = h - 20
        
        # Draw watermark with transparency
        overlay = visualization.copy()
        cv2.putText(overlay, watermark_text, (x, y), font, font_scale, (100, 100, 100), font_thickness, cv2.LINE_AA)
        
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0, visualization)
        
    def _create_measurements_container(self, content_seq: List[Dataset], results: dict, femur_threshold: float = 2.0, tibia_threshold: float = 2.0, total_threshold: float = 5.0) -> Dataset:
        """Create the measurements container for structured report.
        
        Args:
            results: Dictionary containing measurement results
            femur_threshold: Threshold in cm for femur length difference significance (default 0.5)
            tibia_threshold: Threshold in cm for tibia length difference significance (default 0.5) 
            total_threshold: Threshold in cm for total leg length difference significance (default 0.5)
        """
        # # Create container for measurements
        # container = Dataset()
        # container.RelationshipType = 'CONTAINS'
        # container.ValueType = 'CONTAINER'
        # container.ContinuityOfContent = 'SEPARATE'
        # container.ConceptNameCodeSequence = [Dataset()]
        # container.ConceptNameCodeSequence[0].CodeValue = self.codes.MEASUREMENTS['container']
        # container.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
        # container.ConceptNameCodeSequence[0].CodeMeaning = 'Pediatric Leg Length Measurements'
        
        # Create content sequence for measurements
        # content_seq = []
        
        # Create separate content items for each measurement
        for name, data in results['measurements'].items():
            text_item = Dataset()
            text_item.RelationshipType = 'HAS PROPERTIES'
            text_item.ValueType = 'TEXT'
            text_item.ConceptNameCodeSequence = [Dataset()]
            text_item.ConceptNameCodeSequence[0].CodeValue = f"99_{name}"
            text_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
            
            # Format text based on measurement name with 1 decimal place
            if name == 'PLL_R_FEM':
                text_item.ConceptNameCodeSequence[0].CodeMeaning = "Right femur length"
                text = f"{data['centimeters']:.1f} cm"
            elif name == 'PLL_R_TIB':
                text_item.ConceptNameCodeSequence[0].CodeMeaning = "Right tibia length"
                text = f"{data['centimeters']:.1f} cm"
            elif name == 'PLL_R_LGL':
                text_item.ConceptNameCodeSequence[0].CodeMeaning = "Total right lower extremity length"
                text = f"{data['centimeters']:.1f} cm"
            elif name == 'PLL_L_FEM':
                text_item.ConceptNameCodeSequence[0].CodeMeaning = "Left femur length"
                text = f"{data['centimeters']:.1f} cm"
            elif name == 'PLL_L_TIB':
                text_item.ConceptNameCodeSequence[0].CodeMeaning = "Left tibia length"
                text = f"{data['centimeters']:.1f} cm"
            elif name == 'PLL_L_LGL':
                text_item.ConceptNameCodeSequence[0].CodeMeaning = "Total left lower extremity length"
                text = f"{data['centimeters']:.1f} cm"
            else:
                text_item.ConceptNameCodeSequence[0].CodeMeaning = f"{name} length"
                text = f"{data['centimeters']:.1f} cm"

            text_item.TextValue = text
            content_seq.append(text_item)

        # Add difference measurements if both sides are available
        if all(k in results['measurements'] for k in ['PLL_R_FEM', 'PLL_L_FEM']):
            fem_diff = (results['measurements']['PLL_R_FEM']['centimeters'] - 
                         results['measurements']['PLL_L_FEM']['centimeters'])
            
            # Femur difference value
            diff_item = Dataset()
            diff_item.RelationshipType = 'HAS PROPERTIES'
            diff_item.ValueType = 'TEXT'
            diff_item.ConceptNameCodeSequence = [Dataset()]
            diff_item.ConceptNameCodeSequence[0].CodeValue = "99_FEM_DIFF_VAL"
            diff_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
            diff_item.ConceptNameCodeSequence[0].CodeMeaning = "Femur Length Difference"
            # Create descriptive text instead of negative values
            if abs(fem_diff) < 0.1:  # Less than 1mm difference
                diff_item.TextValue = "Femur lengths are equal"
            elif fem_diff > 0:
                diff_item.TextValue = f"Right femur is longer than left by {fem_diff:.1f} cm"
            else:
                diff_item.TextValue = f"Left femur is longer than right by {abs(fem_diff):.1f} cm"
            content_seq.append(diff_item)

            # # Femur discrepancy description
            # desc_item = Dataset()
            # desc_item.RelationshipType = 'HAS PROPERTIES'
            # desc_item.ValueType = 'TEXT'
            # desc_item.ConceptNameCodeSequence = [Dataset()]
            # desc_item.ConceptNameCodeSequence[0].CodeValue = "99_FEM_DIFF_DESC"
            # desc_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
            # desc_item.ConceptNameCodeSequence[0].CodeMeaning = "Femur Length Discrepancy"
            # if fem_diff > femur_threshold:
            #     if results['measurements']['PLL_R_FEM']['centimeters'] > results['measurements']['PLL_L_FEM']['centimeters']:
            #         desc_item.TextValue = "The right femur is longer"
            #     else:
            #         desc_item.TextValue = "The left femur is longer"
            # else:
            #     desc_item.TextValue = "There is no femoral length discrepancy"
            # content_seq.append(desc_item)

        if all(k in results['measurements'] for k in ['PLL_R_TIB', 'PLL_L_TIB']):
            tib_diff = (results['measurements']['PLL_R_TIB']['centimeters'] - 
                         results['measurements']['PLL_L_TIB']['centimeters'])
            
            # Tibia difference value
            diff_item = Dataset()
            diff_item.RelationshipType = 'HAS PROPERTIES'
            diff_item.ValueType = 'TEXT'
            diff_item.ConceptNameCodeSequence = [Dataset()]
            diff_item.ConceptNameCodeSequence[0].CodeValue = "99_TIB_DIFF_VAL"
            diff_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
            diff_item.ConceptNameCodeSequence[0].CodeMeaning = "Tibia Length Difference"
            # Create descriptive text instead of negative values
            if abs(tib_diff) < 0.1:  # Less than 1mm difference
                diff_item.TextValue = "Tibia lengths are equal"
            elif tib_diff > 0:
                diff_item.TextValue = f"Right tibia is longer than left by {tib_diff:.1f} cm"
            else:
                diff_item.TextValue = f"Left tibia is longer than right by {abs(tib_diff):.1f} cm"
            content_seq.append(diff_item)

            # # Tibia discrepancy description
            # desc_item = Dataset()
            # desc_item.RelationshipType = 'HAS PROPERTIES'
            # desc_item.ValueType = 'TEXT'
            # desc_item.ConceptNameCodeSequence = [Dataset()]
            # desc_item.ConceptNameCodeSequence[0].CodeValue = "99_TIB_DIFF_DESC"
            # desc_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
            # desc_item.ConceptNameCodeSequence[0].CodeMeaning = "Tibia Length Discrepancy"
            # if tib_diff > tibia_threshold:
            #     if results['measurements']['PLL_R_TIB']['centimeters'] > results['measurements']['PLL_L_TIB']['centimeters']:
            #         desc_item.TextValue = "The right tibia is longer"
            #     else:
            #         desc_item.TextValue = "The left tibia is longer"
            # else:
            #     desc_item.TextValue = "There is no tibial length discrepancy"
            # content_seq.append(desc_item)

        if all(k in results['measurements'] for k in ['PLL_R_LGL', 'PLL_L_LGL']):
            total_diff = (results['measurements']['PLL_R_LGL']['centimeters'] - 
                           results['measurements']['PLL_L_LGL']['centimeters'])
            
            # Total difference value
            diff_item = Dataset()
            diff_item.RelationshipType = 'HAS PROPERTIES'
            diff_item.ValueType = 'TEXT'
            diff_item.ConceptNameCodeSequence = [Dataset()]
            diff_item.ConceptNameCodeSequence[0].CodeValue = "99_TOT_DIFF_VAL"
            diff_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
            diff_item.ConceptNameCodeSequence[0].CodeMeaning = "Total Lower Extremity Length Difference"
            # Create descriptive text instead of negative values
            if abs(total_diff) < 0.1:  # Less than 1mm difference
                diff_item.TextValue = "Total leg lengths are equal"
            elif total_diff > 0:
                diff_item.TextValue = f"Right leg is longer than left by {total_diff:.1f} cm"
            else:
                diff_item.TextValue = f"Left leg is longer than right by {abs(total_diff):.1f} cm"
            content_seq.append(diff_item)

            # # Total discrepancy description
            # desc_item = Dataset()
            # desc_item.RelationshipType = 'HAS PROPERTIES'
            # desc_item.ValueType = 'TEXT'
            # desc_item.ConceptNameCodeSequence = [Dataset()]
            # desc_item.ConceptNameCodeSequence[0].CodeValue = "99_TOT_DIFF_DESC"
            # desc_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
            # desc_item.ConceptNameCodeSequence[0].CodeMeaning = "Total Lower Extremity Length Discrepancy"
            # if total_diff > total_threshold:
            #     if results['measurements']['PLL_R_LGL']['centimeters'] > results['measurements']['PLL_L_LGL']['centimeters']:
            #         desc_item.TextValue = "The right leg is longer"
            #     else:
            #         desc_item.TextValue = "The left leg is longer"
            # else:
            #     desc_item.TextValue = "There is no leg length discrepancy."
            # content_seq.append(desc_item)

        return content_seq

    def _create_issues_container(self, content_seq: List[Dataset], results: dict) -> Dataset:
        """Create a flat issues container for structured report."""
        # container = Dataset()
        # container.RelationshipType = 'CONTAINS'
        # container.ValueType = 'CONTAINER'
        # container.ContinuityOfContent = 'SEPARATE'
        # container.ConceptNameCodeSequence = [Dataset()]
        # container.ConceptNameCodeSequence[0].CodeValue = "PLL_ISSUES"
        # container.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
        # container.ConceptNameCodeSequence[0].CodeMeaning = 'Measurement Issues'
        
        # content_seq = []
        
        issues = results.get('issues', [])
        
        if not issues:
            no_issues_item = Dataset()
            no_issues_item.RelationshipType = 'HAS PROPERTIES'
            no_issues_item.ValueType = 'TEXT'
            no_issues_item.ConceptNameCodeSequence = [Dataset()]
            no_issues_item.ConceptNameCodeSequence[0].CodeValue = "99_ITEM_NO_ISSUES"
            no_issues_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
            no_issues_item.ConceptNameCodeSequence[0].CodeMeaning = "Analysis Status"
            no_issues_item.TextValue = "No measurement issues detected"
            content_seq.append(no_issues_item)
        else:
            for i, issue in enumerate(issues):
                issue_item = Dataset()
                issue_item.RelationshipType = 'HAS PROPERTIES'
                issue_item.ValueType = 'TEXT'
                issue_item.ConceptNameCodeSequence = [Dataset()]
                issue_item.ConceptNameCodeSequence[0].CodeValue = f"99_ITEM_ISSUE_{i+1}"
                issue_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
                issue_item.ConceptNameCodeSequence[0].CodeMeaning = f"Issue {i+1}"
                issue_item.TextValue = issue.get('description', f"Issue with {issue.get('name', 'unknown')}")
                content_seq.append(issue_item)

        # container.ContentSequence.extend(content_seq)
        return content_seq
    
    def get_qa_dicom(self, results: dict, dicom_path: str) -> Dataset:
        """Create a QA DICOM with visualizations of detected points and measurements."""
        # Load source DICOM
        dcm = pydicom.dcmread(dicom_path)
        
        # Create visualization image
        visualization = self._create_visualization_image(results, dicom_path)
        
        # Create DICOM dataset with all headers preserved + Stanford AIDE headers
        new_ds = self._add_headers(
            dcm=dcm,
            sop_class_uid=self.codes.SOP_CLASS_UIDS['secondary_capture'],
            modality='SC',
            series_description='QA Visualization',
            algorithm_name='pediatric_leg_length',
            status='complete',
            results=results
        )
        
        # Set pixel data and image dimensions
        new_ds.PixelData = visualization.tobytes()
        new_ds.Rows = visualization.shape[0]
        new_ds.Columns = visualization.shape[1]
        
        return new_ds

    def _draw_all_keypoints(self, visualization: np.ndarray, results: dict):
        """Draw all 8 keypoints on the visualization, marking derived points with X."""
        # Define keypoint names and labels with anatomical accuracy
        keypoint_names = {
            1: "R-FH",     # Right Femoral Head
            2: "R-LT",     # Right Lesser Trochanter  
            3: "R-MK",     # Right Medial Knee
            4: "R-LK",     # Right Lateral Knee
            5: "L-FH",     # Left Femoral Head
            6: "L-LT",     # Left Lesser Trochanter
            7: "L-MK",     # Left Medial Knee
            8: "L-LK"      # Left Lateral Knee
        }
        
        # Define professional medical imaging color scheme
        detected_color = (255, 255, 0)    # Cyan for detected points (BGR format)
        derived_color = (140, 255, 255)   # Orange for derived points (BGR format)
        text_bg_color = (0, 0, 0, 180)    # Semi-transparent black background
        
        # Get individual model predictions to see which points were actually detected
        individual_predictions = results.get('individual_model_predictions', {})
        detected_points = set()
        
        # Find which points were detected by at least one model
        for model_name, model_data in individual_predictions.items():
            predictions = model_data.get('predictions', {})
            labels = predictions.get('labels', [])
            boxes = predictions.get('boxes', [])
            
            for i, label in enumerate(labels):
                if i < len(boxes) and 1 <= label <= 8:
                    detected_points.add(label)
        
        # Get fused results to see final point positions
        fused_boxes = results.get('boxes', [])
        fused_labels = results.get('labels', [])
        
        # Draw all 8 keypoints
        for point_id in range(1, 9):
            point_name = keypoint_names.get(point_id, f"P{point_id}")
            
            # Check if this point exists in fused results
            fused_box = None
            for i, label in enumerate(fused_labels):
                if label == point_id and i < len(fused_boxes):
                    fused_box = fused_boxes[i]
                    break
            
            if fused_box is not None:
                # Calculate center of fused bounding box
                center_x = int((fused_box[0] + fused_box[2]) / 2)
                center_y = int((fused_box[1] + fused_box[3]) / 2)
                
                # Clamp to image bounds
                h, w = visualization.shape[:2]
                center_x = max(0, min(w-1, center_x))
                center_y = max(0, min(h-1, center_y))
                
                # Determine if this point was directly detected or derived
                was_detected = point_id in detected_points
                
                if was_detected:
                    # Draw single hollow circle for AI-predicted landmarks
                    cv2.circle(visualization, (center_x, center_y), 12, detected_color, 2)  # Hollow circle
                    # Add crosshair that ends at circle border
                    cv2.line(visualization, (center_x-12, center_y), (center_x+12, center_y), detected_color, 1)
                    cv2.line(visualization, (center_x, center_y-12), (center_x, center_y+12), detected_color, 1)
                else:
                    # Draw single hollow circle for manual/interpolated landmarks
                    cv2.circle(visualization, (center_x, center_y), 12, derived_color, 2)  # Hollow circle
                    # Add crosshair that ends at circle border
                    cv2.line(visualization, (center_x-12, center_y), (center_x+12, center_y), derived_color, 1)
                    cv2.line(visualization, (center_x, center_y-12), (center_x, center_y+12), derived_color, 1)
                
                # Remove anatomical joint labels per medical imaging standards
                # self._draw_professional_label(visualization, center_x, center_y, point_name, was_detected)

    def _add_keypoint_legend(self, visualization: np.ndarray):
        """Add professional legend explaining keypoint symbols."""
        h, w = visualization.shape[:2]
        
        # Professional legend positioning (top-left to avoid interference)
        legend_x = 20
        legend_y = 20
        legend_width = 220
        legend_height = 90
        
        # Create professional legend background with subtle gradient
        overlay = visualization.copy()
        cv2.rectangle(overlay, 
                     (legend_x - 10, legend_y - 5), 
                     (legend_x + legend_width, legend_y + legend_height), 
                     (40, 40, 40), -1)  # Dark gray background
        
        # Apply transparency
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0, visualization)
        
        # Add border
        cv2.rectangle(visualization, 
                     (legend_x - 10, legend_y - 5), 
                     (legend_x + legend_width, legend_y + legend_height), 
                     (200, 200, 200), 2)  # Light border
        
        # Professional typography
        title_font = cv2.FONT_HERSHEY_DUPLEX
        label_font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = 0.7
        label_scale = 0.5
        
        # Legend title
        cv2.putText(visualization, "ANATOMICAL LANDMARKS", (legend_x, legend_y + 20), 
                   title_font, title_scale, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Detected point symbol and label
        symbol_y = legend_y + 45
        detected_color = (255, 255, 0)  # Cyan
        cv2.circle(visualization, (legend_x + 15, symbol_y), 12, detected_color, 2)
        # Add crosshair that ends at circle border
        cv2.line(visualization, (legend_x + 3, symbol_y), (legend_x + 27, symbol_y), detected_color, 1)
        cv2.line(visualization, (legend_x + 15, symbol_y-12), (legend_x + 15, symbol_y+12), detected_color, 1)
        cv2.putText(visualization, "AI Detected", (legend_x + 35, symbol_y + 5), 
                   label_font, label_scale, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Derived point symbol and label  
        symbol_y = legend_y + 70
        derived_color = (140, 255, 255)  # Orange
        cv2.circle(visualization, (legend_x + 15, symbol_y), 12, derived_color, 2)
        # Add crosshair that ends at circle border
        cv2.line(visualization, (legend_x + 3, symbol_y), (legend_x + 27, symbol_y), derived_color, 1)
        cv2.line(visualization, (legend_x + 15, symbol_y-12), (legend_x + 15, symbol_y+12), derived_color, 1)
        cv2.putText(visualization, "Interpolated", (legend_x + 35, symbol_y + 5), 
                   label_font, label_scale, (255, 255, 255), 1, cv2.LINE_AA)

    def _create_model_coverage_table(self, results: dict, models: List[str]) -> np.ndarray:
        """Create Table 1: Model Coverage table showing which models produced output for each point."""
        # Create large, professional figure for medical reports
        plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='white', dpi=200)
        ax.axis('tight')
        ax.axis('off')
        fig.patch.set_facecolor('white')
        
        # Get individual model predictions from fused results
        model_predictions = results.get('individual_model_predictions', {})
        point_ids = set()
        
        # Debug: Log the structure of model_predictions
        self.logger.info(f"Model predictions keys: {list(model_predictions.keys())}")
        for model_name, predictions in model_predictions.items():
            self.logger.info(f"Model {model_name} predictions keys: {list(predictions.keys()) if isinstance(predictions, dict) else 'Not a dict'}")
            if isinstance(predictions, dict) and 'predictions' in predictions:
                pred_data = predictions['predictions']
                self.logger.info(f"Model {model_name} prediction data keys: {list(pred_data.keys()) if isinstance(pred_data, dict) else 'Not a dict'}")
                if isinstance(pred_data, dict) and 'boxes' in pred_data:
                    boxes = pred_data['boxes']
                    self.logger.info(f"Model {model_name} has {len(boxes)} boxes")
        
        # Collect all point IDs that were detected by any model
        # Use labels instead of box indices to get the actual anatomical points
        for model_name, predictions in model_predictions.items():
            if 'predictions' in predictions and 'labels' in predictions['predictions']:
                labels = predictions['predictions']['labels']
                for label in labels:
                    if 1 <= label <= 8:  # Valid anatomical point labels
                        point_ids.add(label - 1)  # Convert to 0-based for point_ids
        
        point_ids = sorted(point_ids)
        self.logger.info(f"Point IDs found: {[p+1 for p in point_ids]} (labels: {[p+1 for p in point_ids]})")
        
        # Create table data
        table_data = []
        headers = ['Point'] + [f'm{i+1}' for i in range(len(models))]
        table_data.append(headers)
        
        # Store model coverage for use in distance table
        self._model_coverage = {}
        
        for point_id in point_ids:
            row = [f'P{point_id + 1}']
            model_detections = []
            
            # Check each model for this point
            for i, model_name in enumerate(models):
                model_pred = model_predictions.get(model_name, {})
                predictions = model_pred.get('predictions', {})
                boxes = predictions.get('boxes', [])
                labels = predictions.get('labels', [])
                
                # Check if this model detected this specific anatomical point
                # Look for matching label (point_id corresponds to anatomical landmark)
                detected = False
                target_label = point_id + 1  # Labels are 1-based (1-8)
                
                # Debug: Show what labels this model has
                if point_id < 3:
                    self.logger.info(f"Model {model_name} labels: {labels}, looking for target_label: {target_label}")
                
                for j, label in enumerate(labels):
                    if label == target_label and j < len(boxes):
                        detected = True
                        break
                
                if detected:
                    row.append('✓')  # Green checkmark
                    model_detections.append(i)
                else:
                    row.append('✗')  # Red X
                    
                # Debug logging for first few points
                if point_id < 3:
                    self.logger.info(f"Point P{point_id+1} Model {model_name}: {len(boxes)} boxes, {len(labels)} labels, detected={detected}")
                    
            self._model_coverage[point_id] = model_detections
            table_data.append(row)
        
        # Handle case where no points were found
        if len(table_data) == 1:  # Only headers, no data rows
            # Add a "No data" row
            no_data_row = ['No Points'] + ['-'] * len(models)
            table_data.append(no_data_row)
            self._model_coverage = {}  # Empty coverage
        
        # Create large, readable medical table
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(16)  # Much larger font for readability
        table.scale(1.5, 3.5)   # Much larger table for visibility
        
        # Apply professional medical styling with high contrast
        for i in range(len(table_data)):
            for j in range(len(headers)):
                cell = table[(i, j)] if i > 0 else table[(0, j)]
                if i == 0:  # Header styling - medical blue
                    cell.set_facecolor('#1e3a5f')  # Dark medical blue
                    cell.set_text_props(weight='bold', color='white', fontsize=18)
                    cell.set_edgecolor('#ffffff')
                    cell.set_linewidth(2)
                else:
                    # High-contrast professional colors
                    if j > 0 and table_data[i][j] == '✓':
                        cell.set_facecolor('#c8e6c9')  # Medical green - detected
                        cell.set_text_props(weight='bold', color='#2e7d32', fontsize=16)
                    elif j > 0 and table_data[i][j] == '✗':
                        cell.set_facecolor('#ffcdd2')  # Medical red - not detected
                        cell.set_text_props(weight='bold', color='#c62828', fontsize=16)
                    else:
                        cell.set_facecolor('#f5f5f5')  # Light gray - labels
                        cell.set_text_props(weight='bold', color='#424242', fontsize=16)
                    
                    cell.set_edgecolor('#bdbdbd')
                    cell.set_linewidth(1.5)
        
        plt.title('AI Model Detection Coverage', fontsize=20, fontweight='bold', 
                 color='#1e3a5f', pad=30, family='serif')
        
        # Convert to image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        table_array = np.array(img)
        plt.close()
        
        return table_array

    def _create_spatial_distance_table(self, results: dict, models: List[str]) -> np.ndarray:
        """Create Table 3: Spatial distance matrix between model combinations."""
        # Create large, professional figure for medical reports
        plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white', dpi=200)
        ax.axis('tight')
        ax.axis('off')
        fig.patch.set_facecolor('white')
        
        # Get individual model predictions from fused results
        model_predictions = results.get('individual_model_predictions', {})
        
        # Create model combination headers
        model_combos = []
        model_pairs = []
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                model_combos.append(f'm{i+1}-m{j+1}')
                model_pairs.append((i, j))
        
        # Create table data
        table_data = []
        headers = ['Point'] + model_combos + ['Mean']
        table_data.append(headers)
        
        # Use the model coverage data from the previous table
        if not hasattr(self, '_model_coverage'):
            # Fallback if model coverage wasn't calculated
            point_ids = sorted(results.get('uncertainties', {}).keys())
            self._model_coverage = {pid: list(range(len(models))) for pid in point_ids}
        
        point_ids = sorted(self._model_coverage.keys())
        
        for point_id in point_ids:
            row = [f'P{point_id + 1}']
            model_detections = self._model_coverage[point_id]
            
            distances = []
            valid_distances = []
            
            # Calculate distances for each model pair
            for (i, j) in model_pairs:
                # Only calculate distance if both models detected this point
                if i in model_detections and j in model_detections:
                    # Get actual positions from model predictions
                    try:
                        model1_name = models[i]
                        model2_name = models[j]
                        
                        # Get positions from model predictions
                        pred1 = model_predictions.get(model1_name, {}).get('predictions', {})
                        pred2 = model_predictions.get(model2_name, {}).get('predictions', {})
                        
                        boxes1 = pred1.get('boxes', [])
                        boxes2 = pred2.get('boxes', [])
                        
                        # Find boxes for the same anatomical point (point_id + 1 since labels are 1-based)
                        target_label = point_id + 1
                        
                        labels1 = pred1.get('labels', [])
                        labels2 = pred2.get('labels', [])
                        
                        box1 = None
                        box2 = None
                        
                        # Find box for model 1
                        for idx, label in enumerate(labels1):
                            if label == target_label and idx < len(boxes1):
                                box1 = boxes1[idx]
                                break
                                
                        # Find box for model 2  
                        for idx, label in enumerate(labels2):
                            if label == target_label and idx < len(boxes2):
                                box2 = boxes2[idx]
                                break
                        
                        if box1 is not None and box2 is not None:
                            # Calculate center points of bounding boxes
                            center1 = [(box1[0] + box1[2])/2, (box1[1] + box1[3])/2]
                            center2 = [(box2[0] + box2[2])/2, (box2[1] + box2[3])/2]
                            
                            # Get pixel spacing from DICOM metadata - REQUIRED
                            pixel_spacing = results.get('dicom_metadata', {}).get('pixel_spacing')
                            
                            if pixel_spacing is None:
                                self.logger.error(f"No pixel spacing found in DICOM metadata - cannot calculate distances")
                                distances.append('-')
                                continue
                            
                            distance_pixels = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                            distance_mm = distance_pixels * pixel_spacing
                            
                            # Debug logging for first few calculations
                            if point_id < 3:  # Only log first few points to avoid spam
                                self.logger.info(f"Point {point_id+1} {model1_name}-{model2_name}: "
                                                f"target_label={target_label}, box1={box1 is not None}, box2={box2 is not None}, "
                                                f"centers=({center1[0]:.1f},{center1[1]:.1f}) vs ({center2[0]:.1f},{center2[1]:.1f}), "
                                                f"distance_pixels={distance_pixels:.1f}, pixel_spacing={pixel_spacing}, "
                                                f"distance_mm={distance_mm:.1f}")
                            
                            distances.append(f'{distance_mm:.1f}')
                            valid_distances.append(distance_mm)
                        else:
                            distances.append('-')  # One or both models didn't detect this point
                            if point_id < 3:  # Debug logging
                                self.logger.info(f"Point {point_id+1} {model1_name}-{model2_name}: Missing boxes (box1={box1 is not None}, box2={box2 is not None})")
                    except Exception:
                        distances.append('-')  # Error calculating distance
                else:
                    distances.append('-')  # One or both models didn't detect this point
            
            # Calculate mean only from valid distances
            if valid_distances:
                mean_dist = np.mean(valid_distances)
                row.extend(distances)
                row.append(f'{mean_dist:.1f}')
            else:
                row.extend(distances)
                row.append('-')  # No valid distances
                
            table_data.append(row)
        
        # Handle case where no points were found
        if len(table_data) == 1:  # Only headers, no data rows
            # Add a "No data" row
            no_data_row = ['No Points'] + ['-'] * len(model_combos) + ['-']
            table_data.append(no_data_row)
        
        # Create large, readable distance table
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(14)  # Larger font for readability
        table.scale(1.3, 3.0)   # Much larger table
        
        # Apply professional medical styling with clinical color coding
        distance_threshold = results.get('config_used', {}).get('distance_threshold', 4.0)
        
        for i in range(len(table_data)):
            for j in range(len(headers)):
                cell = table[(i, j)] if i > 0 else table[(0, j)]
                if i == 0:  # Header styling
                    cell.set_facecolor('#1e3a5f')  # Dark medical blue
                    cell.set_text_props(weight='bold', color='white', fontsize=16)
                    cell.set_edgecolor('#ffffff')
                    cell.set_linewidth(2)
                else:
                    if j > 0 and j < len(headers) - 1:  # Distance cells
                        cell_value = table_data[i][j]
                        if cell_value == '-':
                            # No data - clinical gray
                            cell.set_facecolor('#e0e0e0')
                            cell.set_text_props(weight='bold', color='#757575', fontsize=14)
                        else:
                            try:
                                val = float(cell_value)
                                if val < distance_threshold:
                                    # Good agreement - clinical green
                                    cell.set_facecolor('#c8e6c9')
                                    cell.set_text_props(weight='bold', color='#2e7d32', fontsize=14)
                                else:
                                    # Poor agreement - clinical amber
                                    cell.set_facecolor('#ffe0b2')
                                    cell.set_text_props(weight='bold', color='#f57c00', fontsize=14)
                            except:
                                cell.set_facecolor('#f5f5f5')
                                cell.set_text_props(color='#424242', fontsize=14)
                    else:
                        # Point labels and mean column
                        cell_value = table_data[i][j]
                        if j == 0:  # Point labels
                            cell.set_facecolor('#f5f5f5')
                            cell.set_text_props(weight='bold', color='#424242', fontsize=14)
                        else:  # Mean column
                            if cell_value == '-':
                                cell.set_facecolor('#e0e0e0')
                                cell.set_text_props(weight='bold', color='#757575', fontsize=14)
                            else:
                                cell.set_facecolor('#eceff1')  # Slightly darker for mean
                                cell.set_text_props(weight='bold', color='#37474f', fontsize=14)
                    
                    cell.set_edgecolor('#bdbdbd')
                    cell.set_linewidth(1.5)
        
        plt.title('Inter-Model Distance Analysis (mm)', fontsize=20, fontweight='bold', 
                 color='#1e3a5f', pad=30, family='serif')
        
        # Add clinical threshold legend
        legend_text = f"Clinical Thresholds: Green ≤{distance_threshold}mm (Acceptable) | Amber >{distance_threshold}mm (Review Required)"
        plt.figtext(0.5, 0.02, legend_text, ha='center', fontsize=12, 
                   style='italic', color='#424242', weight='bold')
        
        # Convert to image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        table_array = np.array(img)
        plt.close()
        
        return table_array

    def _create_combined_qa_table_image(self, results: dict, dicom_path: str, models: List[str]) -> np.ndarray:
        """Create combined QA image with original visualization on left and tables on right."""
        # Get original QA visualization
        qa_viz = self._create_visualization_image(results, dicom_path)
        
        # Create tables
        table1 = self._create_model_coverage_table(results, models)
        table3 = self._create_spatial_distance_table(results, models)
        
        # Debug: Log initial shapes
        self.logger.info(f"Initial shapes - QA: {qa_viz.shape}, Table1: {table1.shape}, Table3: {table3.shape}")
        
        # Convert to consistent format
        if qa_viz.dtype != np.uint8:
            qa_viz = qa_viz.astype(np.uint8)
        if table1.dtype != np.uint8:
            table1 = (table1 * 255).astype(np.uint8) if table1.max() <= 1 else table1.astype(np.uint8)
        if table3.dtype != np.uint8:
            table3 = (table3 * 255).astype(np.uint8) if table3.max() <= 1 else table3.astype(np.uint8)
        
        # Ensure all are 3-channel BGR
        if len(qa_viz.shape) == 2:
            qa_viz = cv2.cvtColor(qa_viz, cv2.COLOR_GRAY2BGR)
        elif len(qa_viz.shape) == 3 and qa_viz.shape[2] == 4:  # RGBA
            qa_viz = cv2.cvtColor(qa_viz, cv2.COLOR_RGBA2BGR)
            
        if len(table1.shape) == 4:  # Handle 4D array
            table1 = table1.squeeze()  # Remove extra dimension
        if len(table1.shape) == 3 and table1.shape[2] == 4:  # RGBA
            table1 = cv2.cvtColor(table1, cv2.COLOR_RGBA2BGR)
        elif len(table1.shape) == 2:
            table1 = cv2.cvtColor(table1, cv2.COLOR_GRAY2BGR)
        elif len(table1.shape) == 3 and table1.shape[2] == 3:  # Already BGR
            pass  # Keep as is
            
        if len(table3.shape) == 4:  # Handle 4D array
            table3 = table3.squeeze()  # Remove extra dimension
        if len(table3.shape) == 3 and table3.shape[2] == 4:  # RGBA
            table3 = cv2.cvtColor(table3, cv2.COLOR_RGBA2BGR)
        elif len(table3.shape) == 2:
            table3 = cv2.cvtColor(table3, cv2.COLOR_GRAY2BGR)
        elif len(table3.shape) == 3 and table3.shape[2] == 3:  # Already BGR
            pass  # Keep as is
        
        # Debug: Log final shapes before resizing
        self.logger.info(f"After channel conversion - QA: {qa_viz.shape}, Table1: {table1.shape}, Table3: {table3.shape}")
        
        # Professional medical report layout - much larger for readability
        target_height = 1600  # Taller for better table visibility
        target_width = 2400   # Wider for professional presentation
        
        # Create professional layout with proper margins
        margin = 40
        header_height = 120
        footer_height = 100
        content_height = target_height - header_height - footer_height
        
        # Optimal proportions: 45% image, 55% tables
        qa_width = int((target_width - 3 * margin) * 0.45)
        table_width = target_width - qa_width - 3 * margin
        
        # Resize QA visualization maintaining aspect ratio
        qa_aspect = qa_viz.shape[1] / qa_viz.shape[0]
        if qa_aspect > qa_width / content_height:
            qa_resized_width = qa_width
            qa_resized_height = int(qa_width / qa_aspect)
        else:
            qa_resized_height = content_height
            qa_resized_width = int(content_height * qa_aspect)
        
        qa_resized = cv2.resize(qa_viz, (qa_resized_width, qa_resized_height))
        
        # Resize tables with optimal proportions for readability
        table1_height = int(content_height * 0.35)  # 35% for coverage
        table3_height = int(content_height * 0.65)  # 65% for distances
        
        table1_resized = cv2.resize(table1, (table_width, table1_height))
        table3_resized = cv2.resize(table3, (table_width, table3_height))
        
        # Create professional layout with proper spacing and headers
        final_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255  # White background
        
        # Add professional header
        self._add_professional_header(final_image, results)
        
        # Position QA visualization (centered vertically in content area)
        qa_y_offset = header_height + (content_height - qa_resized_height) // 2
        qa_x_offset = margin
        final_image[qa_y_offset:qa_y_offset + qa_resized_height, 
                   qa_x_offset:qa_x_offset + qa_resized_width] = qa_resized
        
        # Position tables with proper spacing
        table_x_offset = qa_x_offset + qa_resized_width + margin
        table1_y_offset = header_height + margin
        table3_y_offset = table1_y_offset + table1_height + margin//2
        
        final_image[table1_y_offset:table1_y_offset + table1_height,
                   table_x_offset:table_x_offset + table_width] = table1_resized
        final_image[table3_y_offset:table3_y_offset + table3_height,
                   table_x_offset:table_x_offset + table_width] = table3_resized
        
        combined = final_image
        
        # Add image-level metrics with explainers at the bottom
        image_metrics = results.get('image_metrics', {})
        if image_metrics:
            # Generate concise explainers
            explainers = self._generate_metric_explainers(results)
            
            # Create metrics text with explainers
            metrics_lines = [
                f"Image Metrics - DDS: {image_metrics.get('image_dds', 0):.3f}, LDS: {image_metrics.get('image_lds', 0):.3f}, ORS: {image_metrics.get('image_ors', 0):.3f}, CDS: {image_metrics.get('image_cds', 0):.3f}"
            ]
            metrics_lines.extend(explainers)
            
            # Add white space at bottom for metrics (more space for explainers)
            metrics_height = 30 + (len(explainers) * 25)
            white_bar = np.ones((metrics_height, combined.shape[1], 3), dtype=np.uint8) * 255
            combined = np.vstack([combined, white_bar])
            
            # Add metrics text
            y_offset = combined.shape[0] - metrics_height + 20
            for i, line in enumerate(metrics_lines):
                font_size = 0.8 if i == 0 else 0.6  # Smaller font for explainers
                thickness = 2 if i == 0 else 1
                cv2.putText(combined, line, (20, y_offset + (i * 25)), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness, cv2.LINE_AA)
        
        # Add legend at the bottom
        legend_height = 40
        legend_bar = np.ones((legend_height, combined.shape[1], 3), dtype=np.uint8) * 240
        combined = np.vstack([combined, legend_bar])
        
        legend_text = "Legend: m1=resnet101, m2=vit_l_16, m3=resnext101_32x8d"
        cv2.putText(combined, legend_text, (20, combined.shape[0] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
        return combined

    def _generate_metric_explainers(self, results: dict) -> List[str]:
        """Generate concise explainers for image-level metrics."""
        explainers = []
        
        # DDS explainer - which points had detection failures
        point_stats = results.get('point_statistics', {})
        failed_points = []
        total_failures = 0
        
        for point_id, stats in point_stats.items():
            if stats.get('detection_disagreement', 0) > 0:
                models_detected = stats.get('num_models_detected', 0)
                total_models = 3  # Assuming 3 models
                failures = total_models - models_detected
                failed_points.append(f"P{point_id}")
                total_failures += failures
        
        if failed_points:
            explainers.append(f"DDS: {len(failed_points)} points with failures ({', '.join(failed_points[:3])}{'...' if len(failed_points) > 3 else ''})")
        
        # LDS explainer - which points have localization disagreement
        problem_points_lds = []
        for point_id, stats in point_stats.items():
            if stats.get('localization_disagreement', 0) > 0:
                problem_points_lds.append(f"P{point_id}")
        
        if problem_points_lds:
            explainers.append(f"LDS: {len(problem_points_lds)} points >threshold ({', '.join(problem_points_lds[:3])}{'...' if len(problem_points_lds) > 3 else ''})")
        
        # ORS explainer - which points have outlier risk
        problem_points_ors = []
        for point_id, stats in point_stats.items():
            if stats.get('outlier_risk', 0) > 0:
                problem_points_ors.append(f"P{point_id}")
        
        if problem_points_ors:
            explainers.append(f"ORS: {len(problem_points_ors)} points with outliers ({', '.join(problem_points_ors[:3])}{'...' if len(problem_points_ors) > 3 else ''})")
        elif not failed_points and not problem_points_lds:
            # Only show "no issues" if there are actually no issues at all
            explainers.append("No significant disagreement detected")
        
        return explainers

    def _add_professional_header(self, image: np.ndarray, results: dict):
        """Add professional medical report header."""
        h, w = image.shape[:2]
        header_height = 120
        
        # Create header background with gradient
        header_bg = np.ones((header_height, w, 3), dtype=np.uint8) * 248  # Very light gray
        
        # Add subtle gradient
        for i in range(header_height):
            alpha = 0.05 * (i / header_height)
            header_bg[i, :] = np.uint8(248 - alpha * 15)
        
        image[0:header_height, :] = header_bg
        
        # Add header content
        title_font = cv2.FONT_HERSHEY_DUPLEX
        subtitle_font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Main title - much larger and more prominent
        title = "PEDIATRIC LEG LENGTH ANALYSIS - AI QUALITY ASSURANCE REPORT"
        title_scale = 1.2
        (title_w, title_h), _ = cv2.getTextSize(title, title_font, title_scale, 3)
        title_x = (w - title_w) // 2
        cv2.putText(image, title, (title_x, 35), title_font, title_scale, 
                   (30, 60, 120), 3, cv2.LINE_AA)  # Dark medical blue
        
        # Subtitle with timestamp
        from datetime import datetime
        subtitle = f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | Stanford Medicine Radiology"
        subtitle_scale = 0.7
        (sub_w, sub_h), _ = cv2.getTextSize(subtitle, subtitle_font, subtitle_scale, 2)
        sub_x = (w - sub_w) // 2
        cv2.putText(image, subtitle, (sub_x, 65), subtitle_font, subtitle_scale, 
                   (80, 80, 80), 2, cv2.LINE_AA)
        
        # Stanford AIDE logo area
        logo_text = "Stanford AIDE"
        cv2.putText(image, logo_text, (w - 200, 45), cv2.FONT_HERSHEY_DUPLEX, 
                   0.8, (30, 60, 120), 2, cv2.LINE_AA)
        
        # Add separator line
        cv2.line(image, (40, header_height - 5), (w - 40, header_height - 5), 
                (30, 60, 120), 3)

    def _add_professional_footer(self, image: np.ndarray, results: dict):
        """Add professional footer with metrics and legend."""
        h, w = image.shape[:2]
        footer_height = 100
        
        # Create footer background
        footer_y = h - footer_height
        footer_bg = np.ones((footer_height, w, 3), dtype=np.uint8) * 248
        image[footer_y:, :] = footer_bg
        
        # Add separator line
        cv2.line(image, (40, footer_y + 5), (w - 40, footer_y + 5), 
                (30, 60, 120), 3)
        
        # Add metrics section with larger, more readable text
        image_metrics = results.get('image_metrics', {})
        if image_metrics:
            metrics_text = (f"UNCERTAINTY METRICS: "
                          f"DDS={image_metrics.get('image_dds', 0):.3f} | "
                          f"LDS={image_metrics.get('image_lds', 0):.3f} | "
                          f"ORS={image_metrics.get('image_ors', 0):.3f} | "
                          f"CDS={image_metrics.get('image_cds', 0):.3f}")
            
            cv2.putText(image, metrics_text, (40, footer_y + 30), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (30, 60, 120), 2, cv2.LINE_AA)
            
            # Add explainers with larger text
            explainers = self._generate_metric_explainers(results)
            for i, explainer in enumerate(explainers[:2]):  # Limit to 2 lines
                cv2.putText(image, explainer, (40, footer_y + 55 + i * 22), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2, cv2.LINE_AA)
        
        # Add model legend with larger text
        legend_text = "AI MODELS: m1=ResNet101 | m2=ViT-L/16 | m3=ResNeXt101-32x8d"
        cv2.putText(image, legend_text, (w - 650, footer_y + 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2, cv2.LINE_AA)

    def get_qa_table_dicom(self, results: dict, dicom_path: str, models: List[str]) -> Dataset:
        """Create a QA table DICOM with combined visualization and tables."""
        # Load source DICOM
        dcm = pydicom.dcmread(dicom_path)
        
        # Create combined visualization image
        visualization = self._create_combined_qa_table_image(results, dicom_path, models)
        
        # Create DICOM dataset with all headers preserved + Stanford AIDE headers
        new_ds = self._add_headers(
            dcm=dcm,
            sop_class_uid=self.codes.SOP_CLASS_UIDS['secondary_capture'],
            modality='SC',
            series_description='QA Table Visualization',
            algorithm_name='pediatric_leg_length',
            status='complete',
            results=results
        )
        
        # Set pixel data and image dimensions
        new_ds.PixelData = visualization.tobytes()
        new_ds.Rows = visualization.shape[0]
        new_ds.Columns = visualization.shape[1]
        
        return new_ds

    def get_sr_dicom(self, results: dict, dicom_path: str, config: dict) -> Dataset:
        """Create a DICOM Structured Report containing the measurements and issues."""
        # Load source DICOM
        dcm = pydicom.dcmread(dicom_path)
        
        # Create DICOM dataset with all headers preserved + Stanford AIDE headers
        new_ds = self._add_headers(
            dcm=dcm,
            sop_class_uid=self.codes.SOP_CLASS_UIDS['structured_report'],
            modality='SR',
            series_description='AI Measurements',
            algorithm_name='pediatric_leg_length',
            status='complete',
            results=results
        )
        
        # Create a single container for all content
        container = Dataset()
        container.RelationshipType = 'CONTAINS'
        container.ValueType = 'CONTAINER'
        container.ContinuityOfContent = 'SEPARATE'
        container.ConceptNameCodeSequence = [Dataset()]
        container.ConceptNameCodeSequence[0].CodeValue = "PLL_RESULTS"
        container.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
        container.ConceptNameCodeSequence[0].CodeMeaning = 'Pediatric Leg Length Results'
        
        # Initialize the ContentSequence
        content_seq = []
        
        # Add measurements
        content_seq = self._create_measurements_container(content_seq, results, femur_threshold=config['femur_threshold'], tibia_threshold=config['tibia_threshold'], total_threshold=config['total_threshold'])
        
        # # # Add a separator item
        # # separator = Dataset()
        # # separator.RelationshipType = 'CONTAINS'
        # # separator.ValueType = 'TEXT'
        # # separator.ConceptNameCodeSequence = [Dataset()]
        # # separator.ConceptNameCodeSequence[0].CodeValue = "99_SEPARATOR"
        # # separator.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
        # # separator.ConceptNameCodeSequence[0].CodeMeaning = "Measurement Issues"
        # # separator.TextValue = "Measurement Issues"
        # # content_seq.append(separator)
        
        # # Add issues
        # content_seq = self._create_issues_container(content_seq, results)
        
        # Set the content sequence
        container.ContentSequence = content_seq
        new_ds.ContentSequence = [container]
        
        return new_ds

    def create_processing_marker(self, dicom_path: str, algorithm_name: str = 'pediatric_leg_length', 
                               status: str = 'processing') -> Dataset:
        """Create a minimal processing status marker DICOM."""
        # Load source DICOM
        dcm = pydicom.dcmread(dicom_path)
        
        # Create minimal marker dataset
        marker_ds = self._add_headers(
            dcm=dcm,
            sop_class_uid=self.codes.SOP_CLASS_UIDS['secondary_capture'],
            modality='SC',
            series_description='AI Processing Status',
            algorithm_name=algorithm_name,
            status=status,
            results=None
        )
        
        # Minimal 1x1 pixel image
        marker_ds.SamplesPerPixel = 1
        marker_ds.PhotometricInterpretation = 'MONOCHROME2'
        marker_ds.Rows = 1
        marker_ds.Columns = 1
        marker_ds.BitsAllocated = 8
        marker_ds.BitsStored = 8
        marker_ds.HighBit = 7
        marker_ds.PixelRepresentation = 0
        marker_ds.PixelData = b'\x00'  # Single black pixel
        
        return marker_ds

# Example usage:
if __name__ == "__main__":
    # Initialize the processor
    processor = DicomProcessor(config_path='measurement_configs.json')
    
    # Example results data
    results = {
        'measurements': {
            'left_femur': {
                'centimeters': 25.4,
                'points': {
                    'start': {'x': 100, 'y': 200},
                    'end': {'x': 150, 'y': 300}
                }
            }
        },
        'uncertainties': {
            0: {'localization_disagreement': 0.2},
            1: {'localization_disagreement': 0.5}
        },
        'overall_confidence': 0.95,
        'processing_duration_seconds': 2.3
    }
    
    # Create QA DICOM
    qa_dicom = processor.get_qa_dicom(results, 'input.dcm')
    qa_dicom.save_as('qa_output.dcm')
    
    # Create SR DICOM
    sr_dicom = processor.get_sr_dicom(results, 'input.dcm')
    sr_dicom.save_as('sr_output.dcm')
    
    # Create processing marker
    marker = processor.create_processing_marker('input.dcm', status='processing')
    marker.save_as('processing_marker.dcm')