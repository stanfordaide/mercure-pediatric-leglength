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
