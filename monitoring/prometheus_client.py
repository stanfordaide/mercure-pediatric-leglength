"""
Prometheus client with pushgateway integration for medical imaging metrics.
"""

import logging
from typing import Dict, Any, List, Optional
import time
import requests
from urllib.parse import urljoin

try:
    from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway
    from prometheus_client.gateway import default_handler
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from .exceptions import ConnectionError


class PrometheusClient:
    """Prometheus client with pushgateway integration and error handling."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize Prometheus client.
        
        Args:
            config: Prometheus configuration dictionary
            logger: Logger instance
            
        Raises:
            ConnectionError: If Prometheus client cannot be initialized
        """
        if not PROMETHEUS_AVAILABLE:
            raise ConnectionError("prometheus-client package not available")
        
        self.logger = logger
        self.config = config
        self.connected = False
        
        # Connection parameters
        self.gateway_url = config['gateway_url']
        self.job_name = config['job_name']
        self.timeout = config.get('timeout', 5.0)
        self.instance = config.get('instance', 'pediatric-leglength')
        
        # Create registry for this client
        self.registry = CollectorRegistry()
        
        # Initialize metrics
        self._init_metrics()
        
        # Test connection
        self._test_connection()
    
    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        try:
            # Processing metrics
            self.processing_duration = Histogram(
                'processing_duration_seconds',
                'Time spent processing images',
                ['series_id', 'stage', 'model'],
                registry=self.registry
            )
            
            self.processing_total = Counter(
                'processing_total',
                'Total number of images processed',
                ['status', 'model'],
                registry=self.registry
            )
            
            # Model performance metrics
            self.landmarks_detected = Gauge(
                'landmarks_detected_count',
                'Number of landmarks detected',
                ['series_id', 'model'],
                registry=self.registry
            )
            
            self.confidence_score = Gauge(
                'model_confidence_score',
                'Average confidence score',
                ['series_id', 'model'],
                registry=self.registry
            )
            
            # System metrics
            self.memory_usage = Gauge(
                'memory_usage_bytes',
                'Memory usage in bytes',
                ['component'],
                registry=self.registry
            )
            
            self.cpu_usage = Gauge(
                'cpu_usage_percent',
                'CPU usage percentage',
                ['component'],
                registry=self.registry
            )
            
            # Measurement metrics
            self.measurements_count = Gauge(
                'measurements_count',
                'Number of measurements calculated',
                ['series_id', 'measurement_type'],
                registry=self.registry
            )
            
            self.measurement_value = Gauge(
                'measurement_value_mm',
                'Measurement value in millimeters',
                ['series_id', 'measurement_name'],
                registry=self.registry
            )
            
            # Image-level disagreement metrics
            self.image_dds = Gauge(
                'image_detection_disagreement_score',
                'Image-level Detection Disagreement Score',
                ['series_id', 'model'],
                registry=self.registry
            )
            
            self.image_lds = Gauge(
                'image_localization_disagreement_score',
                'Image-level Localization Disagreement Score',
                ['series_id', 'model'],
                registry=self.registry
            )
            
            self.image_ors = Gauge(
                'image_outlier_risk_score',
                'Image-level Outlier Risk Score',
                ['series_id', 'model'],
                registry=self.registry
            )
            
            self.image_cds = Gauge(
                'image_composite_disagreement_score',
                'Image-level Composite Disagreement Score',
                ['series_id', 'model'],
                registry=self.registry
            )
            
            self.logger.debug("Prometheus metrics initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Prometheus metrics: {e}")
            raise ConnectionError(f"Prometheus metrics initialization failed: {e}")
    
    def _test_connection(self) -> None:
        """Test connection to Prometheus pushgateway."""
        try:
            # Test with a simple HTTP request
            test_url = urljoin(self.gateway_url, '/metrics')
            response = requests.get(test_url, timeout=self.timeout)
            
            if response.status_code == 200:
                self.connected = True
                self.logger.info(f"Connected to Prometheus pushgateway at {self.gateway_url}")
            else:
                raise ConnectionError(f"Pushgateway returned status {response.status_code}")
                
        except requests.RequestException as e:
            self.logger.warning(f"Failed to connect to Prometheus pushgateway: {e}")
            self.connected = False
            raise ConnectionError(f"Prometheus pushgateway connection failed: {e}")
    
    def record_processing_duration(self, series_id: str, stage: str, duration: float, 
                                 model: str = "unknown") -> None:
        """Record processing duration metric."""
        if not self.connected:
            return
        
        try:
            self.processing_duration.labels(
                series_id=series_id,
                stage=stage,
                model=model
            ).observe(duration)
            
            self.logger.debug(f"Recorded processing duration: {stage}={duration:.2f}s")
            
        except Exception as e:
            self.logger.debug(f"Failed to record processing duration: {e}")
    
    def record_processing_count(self, status: str, model: str = "unknown") -> None:
        """Record processing count metric."""
        if not self.connected:
            return
        
        try:
            self.processing_total.labels(status=status, model=model).inc()
            self.logger.debug(f"Recorded processing count: status={status}, model={model}")
            
        except Exception as e:
            self.logger.debug(f"Failed to record processing count: {e}")
    
    def record_model_metrics(self, series_id: str, model: str, landmarks: int, 
                           confidence: float) -> None:
        """Record model performance metrics."""
        if not self.connected:
            return
        
        try:
            self.landmarks_detected.labels(series_id=series_id, model=model).set(landmarks)
            self.confidence_score.labels(series_id=series_id, model=model).set(confidence)
            
            self.logger.debug(f"Recorded model metrics: landmarks={landmarks}, confidence={confidence:.3f}")
            
        except Exception as e:
            self.logger.debug(f"Failed to record model metrics: {e}")
    
    def record_system_metrics(self, memory_mb: float, cpu_percent: float) -> None:
        """Record system resource metrics."""
        if not self.connected:
            return
        
        try:
            self.memory_usage.labels(component="main").set(memory_mb * 1024 * 1024)  # Convert to bytes
            self.cpu_usage.labels(component="main").set(cpu_percent)
            
            self.logger.debug(f"Recorded system metrics: memory={memory_mb:.1f}MB, cpu={cpu_percent:.1f}%")
            
        except Exception as e:
            self.logger.debug(f"Failed to record system metrics: {e}")
    
    def record_measurements(self, series_id: str, measurements: Dict[str, float]) -> None:
        """Record measurement metrics."""
        if not self.connected:
            return
        
        try:
            # Count total measurements
            self.measurements_count.labels(
                series_id=series_id, 
                measurement_type="total"
            ).set(len(measurements))
            
            # Record individual measurements
            for name, value in measurements.items():
                if isinstance(value, (int, float)):
                    self.measurement_value.labels(
                        series_id=series_id,
                        measurement_name=name
                    ).set(value)
            
            self.logger.debug(f"Recorded {len(measurements)} measurements")
            
        except Exception as e:
            self.logger.debug(f"Failed to record measurements: {e}")
    
    def record_image_metrics(self, series_id: str, image_metrics: Dict[str, float], 
                           model: str = "ensemble") -> None:
        """Record image-level disagreement metrics."""
        if not self.connected:
            return
        
        try:
            # Record each image-level metric
            if 'image_dds' in image_metrics:
                self.image_dds.labels(
                    series_id=series_id,
                    model=model
                ).set(image_metrics['image_dds'])
            
            if 'image_lds' in image_metrics:
                self.image_lds.labels(
                    series_id=series_id,
                    model=model
                ).set(image_metrics['image_lds'])
            
            if 'image_ors' in image_metrics:
                self.image_ors.labels(
                    series_id=series_id,
                    model=model
                ).set(image_metrics['image_ors'])
            
            if 'image_cds' in image_metrics:
                self.image_cds.labels(
                    series_id=series_id,
                    model=model
                ).set(image_metrics['image_cds'])
            
            self.logger.debug(f"Recorded image metrics for series {series_id}")
            
        except Exception as e:
            self.logger.debug(f"Failed to record image metrics: {e}")
    
    def push_metrics(self) -> bool:
        """
        Push all metrics to the Prometheus pushgateway.
        
        Returns:
            True if metrics were pushed successfully, False otherwise
        """
        if not self.connected:
            self.logger.debug("Prometheus not connected, skipping metrics push")
            return False
        
        try:
            # Add instance label and push to gateway
            push_to_gateway(
                gateway=self.gateway_url,
                job=self.job_name,
                registry=self.registry,
                grouping_key={'instance': self.instance},
                timeout=self.timeout,
                handler=default_handler
            )
            
            self.logger.debug("Successfully pushed metrics to Prometheus pushgateway")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to push metrics to Prometheus: {e}")
            return False
    
    def test_connection(self) -> bool:
        """
        Test Prometheus pushgateway connection.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            test_url = urljoin(self.gateway_url, '/metrics')
            response = requests.get(test_url, timeout=self.timeout)
            return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"Prometheus connection test failed: {e}")
            return False
    
    def clear_metrics(self) -> bool:
        """
        Clear metrics from the pushgateway for this job.
        
        Returns:
            True if metrics were cleared successfully, False otherwise
        """
        if not self.connected:
            return False
        
        try:
            # Send DELETE request to clear metrics
            delete_url = f"{self.gateway_url}/metrics/job/{self.job_name}/instance/{self.instance}"
            response = requests.delete(delete_url, timeout=self.timeout)
            
            if response.status_code in [200, 202]:
                self.logger.debug("Cleared metrics from Prometheus pushgateway")
                return True
            else:
                self.logger.debug(f"Failed to clear metrics: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.debug(f"Failed to clear Prometheus metrics: {e}")
            return False
