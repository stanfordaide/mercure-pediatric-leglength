"""
Metrics collector that formats data for both InfluxDB and Prometheus backends.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import psutil
import os
import calendar
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


class MetricsCollector:
    """Collects and formats metrics for both InfluxDB and Prometheus."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize metrics collector.
        
        Args:
            config: Metrics configuration dictionary
            logger: Logger instance
        """
        self.logger = logger
        self.config = config
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Configuration options
        self.include_system_metrics = config.get('include_system_metrics', True)
        self.include_model_metrics = config.get('include_model_metrics', True)
        self.collection_interval = config.get('collection_interval', 10)
        
        # System monitoring
        self.process = psutil.Process()
        
        self.logger.debug("Metrics collector initialized")
    
    def _extract_dicom_metadata(self, dicom_path: str) -> Dict[str, Any]:
        """Extract metadata from DICOM file for tagging."""
        metadata = {}
        
        if not PYDICOM_AVAILABLE:
            return metadata
            
        try:
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            
            # Patient information
            metadata['patient_id'] = getattr(ds, 'PatientID', 'unknown')
            metadata['patient_gender'] = getattr(ds, 'PatientSex', 'unknown').upper()
            
            # Calculate age group from patient age
            patient_age = getattr(ds, 'PatientAge', None)
            if patient_age:
                try:
                    # PatientAge format: "025Y" or "030M" etc.
                    age_value = int(patient_age[:3])
                    age_unit = patient_age[3]
                    
                    if age_unit == 'Y':  # Years
                        age_years = age_value
                    elif age_unit == 'M':  # Months
                        age_years = age_value / 12
                    else:
                        age_years = None
                        
                    if age_years is not None:
                        metadata['patient_age'] = int(age_years)
                        if age_years <= 2:
                            metadata['patient_age_group'] = '0-2'
                        elif age_years <= 8:
                            metadata['patient_age_group'] = '2-8'
                        elif age_years <= 18:
                            metadata['patient_age_group'] = '8-18'
                        else:
                            metadata['patient_age_group'] = '18+'
                    else:
                        metadata['patient_age_group'] = 'unknown'
                        
                except (ValueError, IndexError):
                    metadata['patient_age_group'] = 'unknown'
            else:
                metadata['patient_age_group'] = 'unknown'
            
            # Study and series information
            metadata['study_id'] = getattr(ds, 'StudyInstanceUID', 'unknown')
            metadata['series_id'] = getattr(ds, 'SeriesInstanceUID', 'unknown')
            metadata['accession_number'] = getattr(ds, 'AccessionNumber', 'unknown')
            
            # Scanner information
            metadata['scanner_manufacturer'] = getattr(ds, 'Manufacturer', 'unknown')
            metadata['pixel_spacing'] = float(getattr(ds, 'PixelSpacing', [0.0, 0.0])[0])
            
        except Exception as e:
            self.logger.debug(f"Failed to extract DICOM metadata: {e}")
            
        return metadata
    
    def _get_temporal_tags(self, timestamp: float) -> Dict[str, str]:
        """Generate temporal tags from timestamp."""
        dt = datetime.fromtimestamp(timestamp)
        
        # Time of day
        hour = dt.hour
        if 6 <= hour < 12:
            time_of_day = 'morning'
        elif 12 <= hour < 18:
            time_of_day = 'afternoon'
        elif 18 <= hour < 24:
            time_of_day = 'evening'
        else:
            time_of_day = 'night'
        
        # Day type
        day_type = 'weekend' if dt.weekday() >= 5 else 'weekday'
        
        # Week of month (1-based)
        week_of_month = f"week{((dt.day - 1) // 7) + 1}"
        
        return {
            'time_of_day': time_of_day,
            'day_of_week': dt.strftime('%A').lower(),
            'week_of_month': week_of_month,
            'month': dt.strftime('%B').lower(),
            'year': str(dt.year),
            'day_type': day_type
        }
    
    def _extract_table_level_features(self, individual_preds: Dict[str, Any], 
                                     models: List[str], 
                                     pixel_spacing: float) -> Dict[str, float]:
        """
        Extract table-level quality features from individual model predictions.
        
        These features capture model agreement and prediction quality across all points.
        Based on the feature extraction logic used in the decision tree analysis.
        
        Args:
            individual_preds: Dictionary of individual model predictions
            models: List of model names
            pixel_spacing: Pixel to mm conversion factor
            
        Returns:
            Dictionary of table-level quality features
        """
        features = {}
        num_models = len(models)
        
        if num_models < 2 or not individual_preds:
            # Return default values for single model or no predictions
            return {
                'mean_distance': 0.0,
                'max_distance': 0.0,
                'distance_std': 0.0,
                'distance_cv': 0.0,
                'overall_detection_rate': 0.0,
                'detection_consistency': 1.0,
                'missing_point_ratio': 1.0,
                'high_distance_ratio': 0.0,
                'extreme_distance_ratio': 0.0,
                'model_agreement_rate': 1.0
            }
        
        # Store all distances and detections
        all_distances = []
        point_distances = {}
        point_detections = {}
        
        # Extract features for each anatomical point (1-8)
        for point_id in range(8):
            target_label = point_id + 1  # Convert to 1-based
            point_distances[point_id] = []
            point_detections[point_id] = []
            
            # Get detection flags for each model
            for model_name in models:
                model_pred = individual_preds.get(model_name, {})
                predictions = model_pred.get('predictions', {})
                boxes = predictions.get('boxes', [])
                labels = predictions.get('labels', [])
                
                # Check if this point was detected
                detected = 0
                for j, label in enumerate(labels):
                    if label == target_label and j < len(boxes):
                        detected = 1
                        break
                
                point_detections[point_id].append(detected)
            
            # Calculate pairwise distances for detected points
            model_positions = {}
            for i, model_name in enumerate(models):
                if point_detections[point_id][i] == 1:  # Only if detected
                    model_pred = individual_preds.get(model_name, {})
                    predictions = model_pred.get('predictions', {})
                    boxes = predictions.get('boxes', [])
                    labels = predictions.get('labels', [])
                    
                    # Find the box for this point
                    for j, label in enumerate(labels):
                        if label == target_label and j < len(boxes):
                            box = boxes[j]
                            center = [(box[0] + box[2])/2, (box[1] + box[3])/2]
                            model_positions[i] = center
                            break
            
            # Calculate pairwise distances
            for i in range(num_models):
                for j in range(i+1, num_models):
                    if i in model_positions and j in model_positions:
                        pos1 = model_positions[i]
                        pos2 = model_positions[j]
                        distance_pixels = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                        distance_mm = distance_pixels * pixel_spacing
                        point_distances[point_id].append(distance_mm)
                        all_distances.append(distance_mm)
                    else:
                        # Missing distance - use sentinel value
                        point_distances[point_id].append(999.0)
                        all_distances.append(999.0)
        
        # Create feature set
        # 1. Global distance statistics
        valid_distances = [d for d in all_distances if d < 999.0]
        if valid_distances:
            features['mean_distance'] = float(np.mean(valid_distances))
            features['max_distance'] = float(np.max(valid_distances))
            features['distance_std'] = float(np.std(valid_distances))
            features['distance_cv'] = float(np.std(valid_distances) / (np.mean(valid_distances) + 1e-6))
        else:
            features['mean_distance'] = 999.0
            features['max_distance'] = 999.0
            features['distance_std'] = 0.0
            features['distance_cv'] = 0.0
        
        # 2. Detection quality metrics
        all_detection_rates = [np.mean(detections) for detections in point_detections.values()]
        features['overall_detection_rate'] = float(np.mean(all_detection_rates))
        features['detection_consistency'] = float(1.0 - np.std(all_detection_rates))
        
        # 3. Missing data severity
        total_points = len(point_detections)
        missing_points = sum(1 for detections in point_detections.values() if np.sum(detections) == 0)
        features['missing_point_ratio'] = float(missing_points / total_points)
        
        # 4. Distance distribution
        if valid_distances:
            features['high_distance_ratio'] = float(np.mean([d > 5.0 for d in valid_distances]))
            features['extreme_distance_ratio'] = float(np.mean([d > 10.0 for d in valid_distances]))
        else:
            features['high_distance_ratio'] = 1.0
            features['extreme_distance_ratio'] = 1.0
        
        # 5. Model agreement quality
        if valid_distances:
            agreement_threshold = 3.0  # mm
            features['model_agreement_rate'] = float(np.mean([d <= agreement_threshold for d in valid_distances]))
        else:
            features['model_agreement_rate'] = 0.0
        
        return features
    
    def start_session(self, session_id: str, config: Dict[str, Any]) -> None:
        """
        Start collecting metrics for a session.
        
        Args:
            session_id: Unique session identifier
            config: Session configuration
        """
        self.sessions[session_id] = {
            'session_id': session_id,
            'start_time': time.time(),
            'config': config,
            'timings': {},
            'metrics': {},
            'system_metrics': [],
            'model_metrics': {},
            'measurements': {},
            'status': 'started'
        }
        
        # Collect initial system metrics
        if self.include_system_metrics:
            self._collect_system_metrics(session_id)
        
        self.logger.debug(f"Started metrics collection for session {session_id}")
    
    def record_timing(self, session_id: str, stage: str, duration: float) -> None:
        """
        Record timing information for a processing stage.
        
        Args:
            session_id: Session identifier
            stage: Processing stage name
            duration: Duration in seconds
        """
        if session_id not in self.sessions:
            return
        
        self.sessions[session_id]['timings'][stage] = {
            'duration': duration,
            'timestamp': time.time()
        }
        
        self.logger.debug(f"Recorded timing {stage}: {duration:.2f}s")
    
    def record_model_metrics(self, session_id: str, model_name: str, 
                           metrics: Dict[str, Any]) -> None:
        """
        Record model performance metrics.
        
        Args:
            session_id: Session identifier
            model_name: Name of the model
            metrics: Dictionary of model metrics
        """
        if session_id not in self.sessions or not self.include_model_metrics:
            return
        
        self.sessions[session_id]['model_metrics'][model_name] = {
            'timestamp': time.time(),
            'metrics': metrics
        }
        
        self.logger.debug(f"Recorded model metrics for {model_name}")
    
    def record_measurements(self, session_id: str, measurements: Dict[str, Any], 
                           dicom_path: str = None) -> None:
        """
        Record measurement results with DICOM metadata.
        
        Args:
            session_id: Session identifier
            measurements: Dictionary of measurements
            dicom_path: Path to DICOM file for metadata extraction
        """
        if session_id not in self.sessions:
            return
        
        self.sessions[session_id]['measurements'] = measurements
        if dicom_path:
            self.sessions[session_id]['dicom_path'] = dicom_path
            
        self.logger.debug(f"Recorded {len(measurements)} measurements")
    
    def record_performance_data(self, session_id: str, performance_data: Dict[str, Any],
                               dicom_path: str = None) -> None:
        """
        Record AI performance data (uncertainties, point statistics).
        
        Args:
            session_id: Session identifier
            performance_data: Dictionary containing uncertainties and point statistics
            dicom_path: Path to DICOM file for metadata extraction
        """
        if session_id not in self.sessions:
            return
            
        self.sessions[session_id]['performance_data'] = performance_data
        if dicom_path:
            self.sessions[session_id]['dicom_path'] = dicom_path
            
        self.logger.debug(f"Recorded performance data with {len(performance_data.get('uncertainties', {}))} points")
    
    def record_custom_metric(self, session_id: str, name: str, value: Any, 
                           tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a custom metric.
        
        Args:
            session_id: Session identifier
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        if session_id not in self.sessions:
            return
        
        if 'custom_metrics' not in self.sessions[session_id]:
            self.sessions[session_id]['custom_metrics'] = []
        
        self.sessions[session_id]['custom_metrics'].append({
            'name': name,
            'value': value,
            'tags': tags or {},
            'timestamp': time.time()
        })
        
        self.logger.debug(f"Recorded custom metric {name}: {value}")
    
    def _collect_system_metrics(self, session_id: str) -> None:
        """Collect current system metrics."""
        if session_id not in self.sessions:
            return
        
        try:
            # CPU and memory metrics
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # System-wide metrics
            system_cpu = psutil.cpu_percent()
            system_memory = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')
            
            # GPU metrics if available
            gpu_metrics = self._get_gpu_metrics()
            
            system_data = {
                'timestamp': time.time(),
                'process_cpu_percent': cpu_percent,
                'process_memory_mb': memory_mb,
                'system_cpu_percent': system_cpu,
                'system_memory_percent': system_memory.percent,
                'system_memory_available_gb': system_memory.available / (1024**3),
                'disk_usage_percent': disk_usage.percent,
                'disk_free_gb': disk_usage.free / (1024**3)
            }
            
            if gpu_metrics:
                system_data.update(gpu_metrics)
            
            self.sessions[session_id]['system_metrics'].append(system_data)
            
        except Exception as e:
            self.logger.debug(f"Failed to collect system metrics: {e}")
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU metrics if available."""
        gpu_metrics = {}
        
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    # Memory usage
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**2)  # MB
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024**2)   # MB
                    
                    gpu_metrics[f'gpu_{i}_memory_allocated_mb'] = memory_allocated
                    gpu_metrics[f'gpu_{i}_memory_reserved_mb'] = memory_reserved
                    
                    # Temperature and utilization would require nvidia-ml-py
                    # Not including to keep dependencies minimal
                    
        except Exception as e:
            self.logger.debug(f"Failed to collect GPU metrics: {e}")
        
        return gpu_metrics
    
    def get_influx_data(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Format session data for InfluxDB using the new PLL schema.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of InfluxDB-formatted metric dictionaries
        """
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        influx_data = []
        
        try:
            config = session.get('config', {})
            dicom_path = session.get('dicom_path')
            completion_time = time.time()
            
            # Extract DICOM metadata
            dicom_metadata = {}
            if dicom_path:
                dicom_metadata = self._extract_dicom_metadata(dicom_path)
            
            # Get temporal tags
            temporal_tags = self._get_temporal_tags(completion_time)
            
            # AI model version (from config)
            ai_model_version = '_'.join(config.get('models', ['unknown']))
            
            # Base tags for all measurements
            base_tags = {
                'patient_id': dicom_metadata.get('patient_id', 'unknown'),
                'study_id': dicom_metadata.get('study_id', 'unknown'),
                'series_id': dicom_metadata.get('series_id', 'unknown'),
                'accession_number': dicom_metadata.get('accession_number', 'unknown'),
                'patient_gender': dicom_metadata.get('patient_gender', 'unknown'),
                'patient_age_group': dicom_metadata.get('patient_age_group', 'unknown'),
                'scanner_manufacturer': dicom_metadata.get('scanner_manufacturer', 'unknown'),
                'ai_model_version': ai_model_version,
                **temporal_tags
            }
            
            # Processing duration from timings
            processing_duration_ms = 0
            for timing in session.get('timings', {}).values():
                processing_duration_ms += timing['duration'] * 1000  # Convert to ms
            
            # PLL AI Measurements
            measurements = session.get('measurements', {})
            if measurements:
                for measurement_name, measurement_data in measurements.items():
                    if isinstance(measurement_data, dict) and 'millimeters' in measurement_data:
                        # Extract measurement confidence and uncertainty from performance data
                        performance_data = session.get('performance_data', {})
                        measurement_confidence = 0.0
                        measurement_uncertainty_mm = 0.0
                        
                        # Calculate average confidence from uncertainties if available
                        uncertainties = performance_data.get('uncertainties', {})
                        if uncertainties:
                            confidences = [u.get('confidence_mean', 0) for u in uncertainties.values()]
                            if confidences:
                                measurement_confidence = sum(confidences) / len(confidences)
                            
                            # Calculate average spatial uncertainty
                            spatial_uncertainties = [u.get('spatial_uncertainty_mm', 0) for u in uncertainties.values()]
                            if spatial_uncertainties:
                                measurement_uncertainty_mm = sum(spatial_uncertainties) / len(spatial_uncertainties)
                        
                        influx_data.append({
                            'measurement': 'pll_ai_measurements',
                            'tags': {
                                **base_tags,
                                'measurement_type': measurement_name
                            },
                            'fields': {
                                'patient_age': dicom_metadata.get('patient_age', 0),
                                'measurement_value_mm': float(measurement_data['millimeters']),
                                'processing_duration_ms': int(processing_duration_ms),
                                'measurement_confidence': float(measurement_confidence),
                                'measurement_uncertainty_mm': float(measurement_uncertainty_mm),
                                'pixel_spacing': float(dicom_metadata.get('pixel_spacing', 0.0))
                            },
                            'time': datetime.fromtimestamp(completion_time)
                        })
            
            # PLL AI Performance (per point)
            performance_data = session.get('performance_data', {})
            uncertainties = performance_data.get('uncertainties', {})
            
            for point_id, uncertainty_data in uncertainties.items():
                influx_data.append({
                    'measurement': 'pll_ai_performance',
                    'tags': {
                        **base_tags,
                        'point_id': str(point_id)
                    },
                    'fields': {
                        'weighted_x_mm': float(uncertainty_data.get('weighted_x_mm', 0)),
                        'weighted_y_mm': float(uncertainty_data.get('weighted_y_mm', 0)),
                        'detection_disagreement': float(uncertainty_data.get('detection_disagreement', 0)),
                        'total_models': int(uncertainty_data.get('total_models', 0)),
                        'localization_disagreement': float(uncertainty_data.get('localization_disagreement', 0)),
                        'outlier_risk': float(uncertainty_data.get('outlier_risk', 0)),
                        'spatial_uncertainty_mm': float(uncertainty_data.get('spatial_uncertainty_mm', 0)),
                        'confidence_mean': float(uncertainty_data.get('confidence_mean', 0)),
                        'confidence_std': float(uncertainty_data.get('confidence_std', 0)),
                        'confidence_uncertainty': float(uncertainty_data.get('confidence_uncertainty', 0)),
                        'num_models': int(uncertainty_data.get('num_models', 0)),
                        'position_std_x_mm': float(uncertainty_data.get('position_std_x_mm', 0)),
                        'position_std_y_mm': float(uncertainty_data.get('position_std_y_mm', 0))
                    },
                    'time': datetime.fromtimestamp(completion_time)
                })
            
            # PLL AI Image-Level Metrics (including table-level quality features)
            image_metrics = performance_data.get('image_metrics', {})
            if image_metrics:
                # Base image metrics fields
                fields = {
                    'image_dds': float(image_metrics.get('image_dds', 0.0)),
                    'image_lds': float(image_metrics.get('image_lds', 0.0)),
                    'image_ors': float(image_metrics.get('image_ors', 0.0)),
                    'image_cds': float(image_metrics.get('image_cds', 0.0)),
                    'processing_duration_ms': int(processing_duration_ms),
                    'total_landmarks': len(uncertainties)
                }
                
                # Extract table-level quality features from individual model predictions
                individual_preds = performance_data.get('individual_model_predictions', {})
                models = config.get('models', [])
                pixel_spacing = dicom_metadata.get('pixel_spacing', 1.0)
                
                if individual_preds and len(models) > 1:
                    try:
                        table_features = self._extract_table_level_features(
                            individual_preds, 
                            models, 
                            pixel_spacing
                        )
                        # Add table-level features to the fields
                        fields.update(table_features)
                        self.logger.debug(f"Added {len(table_features)} table-level quality features")
                    except Exception as e:
                        self.logger.warning(f"Failed to extract table-level features: {e}")
                        # Continue without table-level features
                
                influx_data.append({
                    'measurement': 'pll_ai_image_metrics',
                    'tags': base_tags,
                    'fields': fields,
                    'time': datetime.fromtimestamp(completion_time)
                })
            
        except Exception as e:
            self.logger.warning(f"Failed to format InfluxDB data: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        return influx_data
    
    def get_prometheus_data(self, session_id: str) -> Dict[str, Any]:
        """
        Format session data for Prometheus.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary of Prometheus-formatted metrics
        """
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        prometheus_data = {
            'session_id': session_id,
            'timings': session.get('timings', {}),
            'model_metrics': session.get('model_metrics', {}),
            'measurements': session.get('measurements', {}),
            'system_metrics': session.get('system_metrics', []),
            'custom_metrics': session.get('custom_metrics', [])
        }
        
        return prometheus_data
    
    def cleanup_session(self, session_id: str) -> None:
        """
        Clean up session data.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.logger.debug(f"Cleaned up session {session_id}")
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.sessions.keys())
