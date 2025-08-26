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
        """Return color based on uncertainty level"""
        thresholds = self.codes.UNCERTAINTY_THRESHOLDS
        if uncertainty <= thresholds['normal']:
            return (0, 255, 0)  # Green
        elif uncertainty <= thresholds['high']:
            return (255, 165, 0)  # Orange
        else:
            return (255, 0, 0)  # Red

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
            # Normalize to 0-255
            img_range = original_image.max() - original_image.min()
            normalized_image = ((original_image - original_image.min()) / img_range * 255).astype(np.uint8)
            
            
            # Convert to RGB
            visualization = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)
        

        
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
                # Draw line (keep white for visibility)
                cv2.line(visualization, p1, p2, (255, 255, 255), 3)  # Thicker line

                # Draw circles with point-specific uncertainty-based color
                for i, (point, point_id) in enumerate(zip([p1, p2], join_points)):
                    uncertainty = self._get_point_uncertainty(point_id, results)
                    color = self._get_uncertainty_color(uncertainty)
                    

                    cv2.circle(visualization, point, 3, (255,255,255), -1)  # Larger filled circle
                    cv2.circle(visualization, point, 15, color, 3)  # Larger circle outline

                # Add label with distance
                mid_point = (
                    int((p1[0] + p2[0]) / 2),
                    int((p1[1] + p2[1]) / 2)
                )
                distance_cm = results['measurements'][name]['centimeters']
                label = f"{distance_cm:.1f}cm"
                
                self.logger.info(f"Adding label '{label}' at {mid_point}")
                
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
                
                # Position text (make sure it's visible)
                text_x = max(0, min(w - text_width, mid_point[0] - text_width//2))
                text_y = max(text_height, min(h - 10, mid_point[1] - 40))
                
                # Draw black outline text
                cv2.putText(
                    visualization,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA
                )
                # Draw white text on top
                cv2.putText(
                    visualization,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
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
        
        return visualization
        
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
            
            # Format text based on measurement name
            if name == 'PLL_R_FEM':
                text_item.ConceptNameCodeSequence[0].CodeMeaning = "Right femur length"
                text = f"{str(round(data['centimeters'], 2))} cm"
            elif name == 'PLL_R_TIB':
                text_item.ConceptNameCodeSequence[0].CodeMeaning = "Right tibia length"
                text = f"{str(round(data['centimeters'], 2))} cm"
            elif name == 'PLL_R_LGL':
                text_item.ConceptNameCodeSequence[0].CodeMeaning = "Total right lower extremity length"
                text = f"{str(round(data['centimeters'], 2))} cm"
            elif name == 'PLL_L_FEM':
                text_item.ConceptNameCodeSequence[0].CodeMeaning = "Left femur length"
                text = f"{str(round(data['centimeters'], 2))} cm"
            elif name == 'PLL_L_TIB':
                text_item.ConceptNameCodeSequence[0].CodeMeaning = "Left tibia length"
                text = f"{str(round(data['centimeters'], 2))} cm"
            elif name == 'PLL_L_LGL':
                text_item.ConceptNameCodeSequence[0].CodeMeaning = "Total left lower extremity length"
                text = f"{str(round(data['centimeters'], 2))} cm"
            else:
                text_item.ConceptNameCodeSequence[0].CodeMeaning = f"{name} length"
                text = f"{str(round(data['centimeters'], 2))} cm"

            text_item.TextValue = text
            content_seq.append(text_item)

        # Add difference measurements if both sides are available
        if all(k in results['measurements'] for k in ['PLL_R_FEM', 'PLL_L_FEM']):
            fem_diff = abs(results['measurements']['PLL_R_FEM']['centimeters'] - 
                         results['measurements']['PLL_L_FEM']['centimeters'])
            
            # Femur difference value
            diff_item = Dataset()
            diff_item.RelationshipType = 'HAS PROPERTIES'
            diff_item.ValueType = 'TEXT'
            diff_item.ConceptNameCodeSequence = [Dataset()]
            diff_item.ConceptNameCodeSequence[0].CodeValue = "99_FEM_DIFF_VAL"
            diff_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
            diff_item.ConceptNameCodeSequence[0].CodeMeaning = "Femur Length Difference"
            diff_item.TextValue = f"{round(fem_diff,2)} cm"
            content_seq.append(diff_item)

            # Femur discrepancy description
            desc_item = Dataset()
            desc_item.RelationshipType = 'HAS PROPERTIES'
            desc_item.ValueType = 'TEXT'
            desc_item.ConceptNameCodeSequence = [Dataset()]
            desc_item.ConceptNameCodeSequence[0].CodeValue = "99_FEM_DIFF_DESC"
            desc_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
            desc_item.ConceptNameCodeSequence[0].CodeMeaning = "Femur Length Discrepancy"
            if fem_diff > femur_threshold:
                if results['measurements']['PLL_R_FEM']['centimeters'] > results['measurements']['PLL_L_FEM']['centimeters']:
                    desc_item.TextValue = "The right femur is longer"
                else:
                    desc_item.TextValue = "The left femur is longer"
            else:
                desc_item.TextValue = "There is no femoral length discrepancy"
            content_seq.append(desc_item)

        if all(k in results['measurements'] for k in ['PLL_R_TIB', 'PLL_L_TIB']):
            tib_diff = abs(results['measurements']['PLL_R_TIB']['centimeters'] - 
                         results['measurements']['PLL_L_TIB']['centimeters'])
            
            # Tibia difference value
            diff_item = Dataset()
            diff_item.RelationshipType = 'HAS PROPERTIES'
            diff_item.ValueType = 'TEXT'
            diff_item.ConceptNameCodeSequence = [Dataset()]
            diff_item.ConceptNameCodeSequence[0].CodeValue = "99_TIB_DIFF_VAL"
            diff_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
            diff_item.ConceptNameCodeSequence[0].CodeMeaning = "Tibia Length Difference"
            diff_item.TextValue = f"{round(tib_diff,2)} cm"
            content_seq.append(diff_item)

            # Tibia discrepancy description
            desc_item = Dataset()
            desc_item.RelationshipType = 'HAS PROPERTIES'
            desc_item.ValueType = 'TEXT'
            desc_item.ConceptNameCodeSequence = [Dataset()]
            desc_item.ConceptNameCodeSequence[0].CodeValue = "99_TIB_DIFF_DESC"
            desc_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
            desc_item.ConceptNameCodeSequence[0].CodeMeaning = "Tibia Length Discrepancy"
            if tib_diff > tibia_threshold:
                if results['measurements']['PLL_R_TIB']['centimeters'] > results['measurements']['PLL_L_TIB']['centimeters']:
                    desc_item.TextValue = "The right tibia is longer"
                else:
                    desc_item.TextValue = "The left tibia is longer"
            else:
                desc_item.TextValue = "There is no tibial length discrepancy"
            content_seq.append(desc_item)

        if all(k in results['measurements'] for k in ['PLL_R_LGL', 'PLL_L_LGL']):
            total_diff = abs(results['measurements']['PLL_R_LGL']['centimeters'] - 
                           results['measurements']['PLL_L_LGL']['centimeters'])
            
            # Total difference value
            diff_item = Dataset()
            diff_item.RelationshipType = 'HAS PROPERTIES'
            diff_item.ValueType = 'TEXT'
            diff_item.ConceptNameCodeSequence = [Dataset()]
            diff_item.ConceptNameCodeSequence[0].CodeValue = "99_TOT_DIFF_VAL"
            diff_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
            diff_item.ConceptNameCodeSequence[0].CodeMeaning = "Total Lower Extremity Length Difference"
            diff_item.TextValue = f"{round(total_diff,2)} cm"
            content_seq.append(diff_item)

            # Total discrepancy description
            desc_item = Dataset()
            desc_item.RelationshipType = 'HAS PROPERTIES'
            desc_item.ValueType = 'TEXT'
            desc_item.ConceptNameCodeSequence = [Dataset()]
            desc_item.ConceptNameCodeSequence[0].CodeValue = "99_TOT_DIFF_DESC"
            desc_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
            desc_item.ConceptNameCodeSequence[0].CodeMeaning = "Total Lower Extremity Length Discrepancy"
            if total_diff > total_threshold:
                if results['measurements']['PLL_R_LGL']['centimeters'] > results['measurements']['PLL_L_LGL']['centimeters']:
                    desc_item.TextValue = "The right leg is longer"
                else:
                    desc_item.TextValue = "The left leg is longer"
            else:
                desc_item.TextValue = "There is no leg length discrepancy."
            content_seq.append(desc_item)

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
        
        # # Add a separator item
        # separator = Dataset()
        # separator.RelationshipType = 'CONTAINS'
        # separator.ValueType = 'TEXT'
        # separator.ConceptNameCodeSequence = [Dataset()]
        # separator.ConceptNameCodeSequence[0].CodeValue = "99_SEPARATOR"
        # separator.ConceptNameCodeSequence[0].CodingSchemeDesignator = self.codes.CODING_SCHEME
        # separator.ConceptNameCodeSequence[0].CodeMeaning = "Measurement Issues"
        # separator.TextValue = "Measurement Issues"
        # content_seq.append(separator)
        
        # Add issues
        content_seq = self._create_issues_container(content_seq, results)
        
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