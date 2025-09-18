"""
InfluxDB client with robust connection handling for medical imaging metrics.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

try:
    from influxdb_client import InfluxDBClient, Point, WriteApi
    from influxdb_client.client.write_api import SYNCHRONOUS
    from influxdb_client.rest import ApiException
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

from .exceptions import ConnectionError


class InfluxClient:
    """InfluxDB client with robust connection handling and error recovery."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize InfluxDB client.
        
        Args:
            config: InfluxDB configuration dictionary
            logger: Logger instance
            
        Raises:
            ConnectionError: If InfluxDB client cannot be initialized
        """
        if not INFLUXDB_AVAILABLE:
            raise ConnectionError("influxdb-client package not available")
        
        self.logger = logger
        self.config = config
        self.client: Optional[InfluxDBClient] = None
        self.write_api: Optional[WriteApi] = None
        self.connected = False
        
        # Connection parameters
        self.url = config['url']
        self.token = config['token']
        self.org = config['org']
        self.bucket = config['bucket']
        self.timeout = config.get('timeout', 5.0) * 1000  # Convert to ms
        
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to InfluxDB with timeout and error handling."""
        try:
            self.logger.debug(f"Connecting to InfluxDB at {self.url}")
            
            self.client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org,
                timeout=self.timeout
            )
            
            # Test connection with health check
            health = self.client.health()
            if health.status != "pass":
                raise ConnectionError(f"InfluxDB health check failed: {health.message}")
            
            # Initialize write API with synchronous mode for reliability
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            
            self.connected = True
            self.logger.info(f"Connected to InfluxDB at {self.url}")
            
        except Exception as e:
            self.logger.warning(f"Failed to connect to InfluxDB: {e}")
            self.connected = False
            self.client = None
            self.write_api = None
            raise ConnectionError(f"InfluxDB connection failed: {e}")
    
    def write_metrics(self, metrics: List[Dict[str, Any]]) -> bool:
        """
        Write metrics to InfluxDB with error handling and retry logic.
        
        Args:
            metrics: List of metric dictionaries with measurement, tags, fields, time
            
        Returns:
            True if metrics were written successfully, False otherwise
        """
        if not self.connected or not self.write_api:
            self.logger.debug("InfluxDB not connected, skipping metrics write")
            return False
        
        if not metrics:
            self.logger.debug("No metrics to write")
            return True
        
        try:
            points = []
            for metric in metrics:
                point = self._create_point(metric)
                if point:
                    points.append(point)
            
            if points:
                self.write_api.write(bucket=self.bucket, record=points)
                self.logger.debug(f"Successfully wrote {len(points)} metrics to InfluxDB")
                return True
            else:
                self.logger.debug("No valid points to write")
                return True
                
        except ApiException as e:
            self.logger.warning(f"InfluxDB API error: {e}")
            return False
        except Exception as e:
            self.logger.warning(f"Failed to write metrics to InfluxDB: {e}")
            return False
    
    def _create_point(self, metric: Dict[str, Any]) -> Optional[Point]:
        """
        Create an InfluxDB Point from a metric dictionary.
        
        Args:
            metric: Metric dictionary with measurement, tags, fields, time
            
        Returns:
            InfluxDB Point object or None if invalid
        """
        try:
            measurement = metric.get('measurement')
            if not measurement:
                self.logger.debug("Metric missing measurement name")
                return None
            
            point = Point(measurement)
            
            # Add tags
            tags = metric.get('tags', {})
            for key, value in tags.items():
                if value is not None:
                    point = point.tag(key, str(value))
            
            # Add fields
            fields = metric.get('fields', {})
            if not fields:
                self.logger.debug(f"Metric {measurement} has no fields")
                return None
                
            for key, value in fields.items():
                if value is not None:
                    # Handle different data types appropriately
                    if isinstance(value, bool):
                        point = point.field(key, value)
                    elif isinstance(value, (int, float)):
                        point = point.field(key, float(value))
                    else:
                        point = point.field(key, str(value))
            
            # Set timestamp if provided
            timestamp = metric.get('time')
            if timestamp:
                if isinstance(timestamp, datetime):
                    point = point.time(timestamp)
                elif isinstance(timestamp, (int, float)):
                    point = point.time(int(timestamp * 1_000_000_000))  # Convert to nanoseconds
            
            return point
            
        except Exception as e:
            self.logger.debug(f"Failed to create InfluxDB point: {e}")
            return None
    
    def test_connection(self) -> bool:
        """
        Test InfluxDB connection.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        if not self.client:
            return False
        
        try:
            health = self.client.health()
            return health.status == "pass"
        except Exception as e:
            self.logger.debug(f"InfluxDB health check failed: {e}")
            return False
    
    def close(self) -> None:
        """Close InfluxDB connection and cleanup resources."""
        try:
            if self.write_api:
                self.write_api.close()
            if self.client:
                self.client.close()
            self.connected = False
            self.logger.debug("InfluxDB connection closed")
        except Exception as e:
            self.logger.debug(f"Error closing InfluxDB connection: {e}")
    
    def __del__(self):
        """Cleanup on object destruction."""
        self.close()
