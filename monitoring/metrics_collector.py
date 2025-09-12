"""
Metrics collector that formats data for both InfluxDB and Prometheus backends.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import psutil
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
    
    def record_measurements(self, session_id: str, measurements: Dict[str, Any]) -> None:
        """
        Record measurement results.
        
        Args:
            session_id: Session identifier
            measurements: Dictionary of measurements
        """
        if session_id not in self.sessions:
            return
        
        self.sessions[session_id]['measurements'] = measurements
        self.logger.debug(f"Recorded {len(measurements)} measurements")
    
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
        Format session data for InfluxDB.
        
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
            # Base tags for all metrics
            base_tags = {
                'session_id': session_id,
                'series_id': session.get('config', {}).get('series_id', 'unknown'),
                'instance': 'pediatric-leglength'
            }
            
            # Processing timing metrics
            for stage, timing in session.get('timings', {}).items():
                influx_data.append({
                    'measurement': 'processing_duration',
                    'tags': {**base_tags, 'stage': stage},
                    'fields': {
                        'duration_seconds': timing['duration']
                    },
                    'time': datetime.fromtimestamp(timing['timestamp'])
                })
            
            # Model performance metrics
            for model_name, model_data in session.get('model_metrics', {}).items():
                metrics = model_data['metrics']
                
                # Confidence scores
                if 'confidence_scores' in metrics:
                    scores = metrics['confidence_scores']
                    if scores:
                        avg_confidence = sum(scores) / len(scores)
                        influx_data.append({
                            'measurement': 'model_performance',
                            'tags': {**base_tags, 'model': model_name},
                            'fields': {
                                'avg_confidence': avg_confidence,
                                'min_confidence': min(scores),
                                'max_confidence': max(scores),
                                'confidence_count': len(scores)
                            },
                            'time': datetime.fromtimestamp(model_data['timestamp'])
                        })
                
                # Landmark detection
                if 'landmarks_detected' in metrics:
                    influx_data.append({
                        'measurement': 'landmark_detection',
                        'tags': {**base_tags, 'model': model_name},
                        'fields': {
                            'landmarks_count': metrics['landmarks_detected']
                        },
                        'time': datetime.fromtimestamp(model_data['timestamp'])
                    })
            
            # Measurement metrics
            measurements = session.get('measurements', {})
            if measurements:
                for name, value in measurements.items():
                    if isinstance(value, (int, float)):
                        influx_data.append({
                            'measurement': 'leg_measurements',
                            'tags': {**base_tags, 'measurement_name': name},
                            'fields': {
                                'value_mm': value
                            },
                            'time': datetime.fromtimestamp(session['start_time'])
                        })
            
            # System metrics
            for sys_metric in session.get('system_metrics', []):
                influx_data.append({
                    'measurement': 'system_resources',
                    'tags': base_tags,
                    'fields': {
                        k: v for k, v in sys_metric.items() 
                        if k != 'timestamp' and isinstance(v, (int, float))
                    },
                    'time': datetime.fromtimestamp(sys_metric['timestamp'])
                })
            
            # Custom metrics
            for custom in session.get('custom_metrics', []):
                influx_data.append({
                    'measurement': 'custom_metrics',
                    'tags': {**base_tags, **custom.get('tags', {}), 'metric_name': custom['name']},
                    'fields': {
                        'value': custom['value']
                    },
                    'time': datetime.fromtimestamp(custom['timestamp'])
                })
            
        except Exception as e:
            self.logger.warning(f"Failed to format InfluxDB data: {e}")
        
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
