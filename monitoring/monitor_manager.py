"""
Main monitoring coordinator with graceful fallback capabilities.
"""

import logging
import time
from typing import Dict, Any, Optional
from .config_validator import validate_monitoring_config
from .exceptions import ConfigurationError, ConnectionError
from .influx_client import InfluxClient
from .prometheus_client import PrometheusClient
from .metrics_collector import MetricsCollector


class MonitorManager:
    """
    Main monitoring coordinator that handles optional InfluxDB and Prometheus integration.
    
    Features:
    - Graceful degradation when monitoring is unavailable
    - Configuration-driven initialization
    - Minimal performance impact on main application
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize monitoring manager.
        
        Args:
            config: Full application configuration (may include monitoring section)
            logger: Logger instance for monitoring messages
        """
        self.logger = logger
        self.enabled = False
        self.influx_client: Optional[InfluxClient] = None
        self.prometheus_client: Optional[PrometheusClient] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        
        # Initialize monitoring if configuration is present
        self._initialize_monitoring(config)
    
    def _initialize_monitoring(self, config: Dict[str, Any]) -> None:
        """
        Initialize monitoring components based on configuration.
        
        Args:
            config: Application configuration dictionary
        """
        monitoring_config = config.get('monitoring')
        
        # Check if monitoring is configured and enabled
        if not monitoring_config:
            self.logger.info("No monitoring configuration found - monitoring disabled")
            return
        
        if not monitoring_config.get('enabled', False):
            self.logger.info("Monitoring explicitly disabled in configuration")
            return
        
        self.logger.info("Monitoring is enabled in configuration, initializing components...")
        
        try:
            # Validate configuration
            validate_monitoring_config(monitoring_config)
            
            # Initialize metrics collector
            self.metrics_collector = MetricsCollector(
                monitoring_config.get('metrics', {}),
                self.logger
            )
            
            # Initialize InfluxDB client if configured
            if 'influxdb' in monitoring_config:
                self.logger.info("Initializing InfluxDB client...")
                try:
                    self.influx_client = InfluxClient(
                        monitoring_config['influxdb'],
                        self.logger
                    )
                    self.logger.info("InfluxDB client initialized successfully")
                except ConnectionError as e:
                    self.logger.warning(f"InfluxDB initialization failed: {e}")
                    self.influx_client = None
            
            # Initialize Prometheus client if configured
            if 'prometheus' in monitoring_config:
                self.logger.info("Initializing Prometheus client...")
                try:
                    self.prometheus_client = PrometheusClient(
                        monitoring_config['prometheus'],
                        self.logger
                    )
                    self.logger.info("Prometheus client initialized successfully")
                except ConnectionError as e:
                    self.logger.warning(f"Prometheus initialization failed: {e}")
                    self.prometheus_client = None
            
            # Enable monitoring if at least one backend is available
            if self.influx_client or self.prometheus_client:
                self.enabled = True
                backends = []
                if self.influx_client:
                    backends.append("InfluxDB")
                if self.prometheus_client:
                    backends.append("Prometheus")
                
                self.logger.info(f"Monitoring enabled with backends: {', '.join(backends)}")
            else:
                self.logger.warning("No monitoring backends available - monitoring disabled")
                self.enabled = False
            
        except ConfigurationError as e:
            self.logger.warning(f"Invalid monitoring configuration: {e}")
            self.logger.info("Continuing without monitoring")
            self.enabled = False
        except Exception as e:
            self.logger.warning(f"Failed to initialize monitoring: {e}")
            self.logger.info("Continuing without monitoring")
            self.enabled = False
    
    def start_session(self, series_id: str, config: Dict[str, Any]) -> str:
        """
        Start a monitoring session for image processing.
        
        Args:
            series_id: DICOM series identifier
            config: Processing configuration
            
        Returns:
            Session ID for tracking, empty string if monitoring disabled
        """
        if not self.enabled:
            return ""
        
        try:
            session_id = f"{series_id}_{int(time.time())}"
            
            # Initialize session in metrics collector
            if self.metrics_collector:
                self.metrics_collector.start_session(session_id, config)
            
            accession_number = config.get('accession_number')
            if accession_number:
                self.logger.info(f"Started monitoring session for accession {accession_number}: {session_id}")
            else:
                self.logger.debug(f"Started monitoring session: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.debug(f"Failed to start monitoring session: {e}")
            return ""
    
    def track_processing_time(self, session_id: str, stage: str, 
                            start_time: float, end_time: float) -> None:
        """
        Track processing time for a specific stage.
        
        Args:
            session_id: Session identifier
            stage: Processing stage name (e.g., 'inference', 'measurement')
            start_time: Stage start timestamp
            end_time: Stage end timestamp
        """
        if not self.enabled or not session_id:
            return
        
        try:
            duration = end_time - start_time
            
            # Record in metrics collector
            if self.metrics_collector:
                self.metrics_collector.record_timing(session_id, stage, duration)
            
            # Record in Prometheus
            if self.prometheus_client:
                self.prometheus_client.record_processing_duration(
                    session_id, stage, duration
                )
            
            self.logger.debug(f"Tracked {stage} timing: {duration:.2f}s")
            
        except Exception as e:
            self.logger.debug(f"Failed to track processing time: {e}")
    
    def record_metrics(self, session_id: str, metric_name: str, 
                      value: Any, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a custom metric.
        
        Args:
            session_id: Session identifier
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags for the metric
        """
        if not self.enabled or not session_id:
            return
        
        try:
            # Record in metrics collector
            if self.metrics_collector:
                self.metrics_collector.record_custom_metric(session_id, metric_name, value, tags)
            
            self.logger.debug(f"Recorded metric {metric_name}: {value}")
            
        except Exception as e:
            self.logger.debug(f"Failed to record metric: {e}")
    
    def record_model_performance(self, session_id: str, model_name: str, 
                               metrics: Dict[str, Any]) -> None:
        """
        Record model performance metrics.
        
        Args:
            session_id: Session identifier
            model_name: Name of the model
            metrics: Dictionary containing model metrics
        """
        if not self.enabled or not session_id:
            return
        
        try:
            # Record in metrics collector
            if self.metrics_collector:
                self.metrics_collector.record_model_metrics(session_id, model_name, metrics)
            
            # Record in Prometheus if specific metrics are available
            if self.prometheus_client:
                landmarks = metrics.get('landmarks_detected', 0)
                confidence_scores = metrics.get('confidence_scores', [])
                
                if confidence_scores:
                    avg_confidence = sum(confidence_scores) / len(confidence_scores)
                    self.prometheus_client.record_model_metrics(
                        session_id, model_name, landmarks, avg_confidence
                    )
            
            self.logger.debug(f"Recorded model performance for {model_name}")
            
        except Exception as e:
            self.logger.debug(f"Failed to record model performance: {e}")
    
    def record_measurements(self, session_id: str, measurements: Dict[str, Any], 
                           dicom_path: str = None) -> None:
        """
        Record measurement results.
        
        Args:
            session_id: Session identifier
            measurements: Dictionary of measurement results
            dicom_path: Optional path to DICOM file for metadata extraction
        """
        if not self.enabled or not session_id:
            return
        
        try:
            # Record in metrics collector
            if self.metrics_collector:
                # Try new method with dicom_path, fallback to old method
                try:
                    self.metrics_collector.record_measurements(session_id, measurements, dicom_path)
                except TypeError:
                    # Old method signature without dicom_path
                    self.metrics_collector.record_measurements(session_id, measurements)
            
            # Record in Prometheus
            if self.prometheus_client:
                # Convert measurements to numeric values only
                numeric_measurements = {
                    k: v for k, v in measurements.items() 
                    if isinstance(v, (int, float))
                }
                self.prometheus_client.record_measurements(session_id, numeric_measurements)
            
            self.logger.debug(f"Recorded {len(measurements)} measurements")
            
        except Exception as e:
            self.logger.debug(f"Failed to record measurements: {e}")
    
    def record_performance_data(self, session_id: str, performance_data: Dict[str, Any],
                               dicom_path: str = None) -> None:
        """
        Record AI performance data (uncertainties, point statistics).
        
        Args:
            session_id: Session identifier
            performance_data: Dictionary containing uncertainties and point statistics
            dicom_path: Optional path to DICOM file for metadata extraction
        """
        if not self.enabled or not session_id:
            return
            
        try:
            # Record in metrics collector if method exists
            if self.metrics_collector and hasattr(self.metrics_collector, 'record_performance_data'):
                self.metrics_collector.record_performance_data(session_id, performance_data, dicom_path)
            
            # Record image-level metrics in Prometheus if available
            if self.prometheus_client and 'image_metrics' in performance_data:
                image_metrics = performance_data['image_metrics']
                if image_metrics:
                    self.prometheus_client.record_image_metrics(session_id, image_metrics)
            
            self.logger.debug(f"Recorded performance data with {len(performance_data.get('uncertainties', {}))} points")
            
        except Exception as e:
            self.logger.debug(f"Failed to record performance data: {e}")
    
    def end_session(self, session_id: str, status: str = 'completed') -> None:
        """
        End a monitoring session and log summary.
        
        Args:
            session_id: Session identifier
            status: Final session status ('completed', 'failed', etc.)
        """
        if not self.enabled or not session_id:
            return
        
        try:
            # Record processing completion in Prometheus
            if self.prometheus_client:
                self.prometheus_client.record_processing_count(status)
            
            # Flush metrics to backends
            self._flush_metrics(session_id)
            
            # Get accession number before cleanup for logging
            accession_number = None
            if self.metrics_collector and session_id in self.metrics_collector.sessions:
                session_config = self.metrics_collector.sessions[session_id].get('config', {})
                accession_number = session_config.get('accession_number')
            
            # Cleanup session data
            if self.metrics_collector:
                self.metrics_collector.cleanup_session(session_id)
            
            if accession_number:
                self.logger.info(f"Session for accession {accession_number} {status}")
            else:
                self.logger.info(f"Session {session_id} {status}")
            
        except Exception as e:
            self.logger.debug(f"Failed to end monitoring session: {e}")
    
    def _flush_metrics(self, session_id: str) -> None:
        """
        Flush all collected metrics to monitoring backends.
        
        Args:
            session_id: Session identifier
        """
        if not self.metrics_collector:
            return
        
        try:
            # Send to InfluxDB
            if self.influx_client:
                influx_data = self.metrics_collector.get_influx_data(session_id)
                self.logger.info(f"Generated {len(influx_data)} InfluxDB data points for session {session_id}")
                if influx_data:
                    # Log first data point for debugging
                    self.logger.debug(f"Sample InfluxDB data: {influx_data[0] if influx_data else 'None'}")
                    success = self.influx_client.write_metrics(influx_data)
                    if success:
                        self.logger.info(f"Successfully flushed {len(influx_data)} metrics to InfluxDB")
                    else:
                        self.logger.warning("Failed to flush metrics to InfluxDB")
                else:
                    self.logger.warning("No InfluxDB data generated - check metrics collection")
            
            # Send to Prometheus
            if self.prometheus_client:
                success = self.prometheus_client.push_metrics()
                if success:
                    self.logger.debug("Flushed metrics to Prometheus")
                else:
                    self.logger.debug("Failed to flush metrics to Prometheus")
                    
        except Exception as e:
            self.logger.warning(f"Failed to flush metrics: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def is_enabled(self) -> bool:
        """
        Check if monitoring is enabled.
        
        Returns:
            True if monitoring is active, False otherwise
        """
        return self.enabled
