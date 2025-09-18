"""
Configuration validation for monitoring module.
"""

from typing import Dict, Any
from .exceptions import ConfigurationError


def validate_monitoring_config(config: Dict[str, Any]) -> None:
    """
    Validate monitoring configuration and raise descriptive errors.
    
    Args:
        config: Monitoring configuration dictionary
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ConfigurationError("Monitoring configuration must be a dictionary")
    
    # Validate InfluxDB configuration if present
    if 'influxdb' in config:
        _validate_influxdb_config(config['influxdb'])
    
    # Validate Prometheus configuration if present
    if 'prometheus' in config:
        _validate_prometheus_config(config['prometheus'])
    
    # At least one monitoring backend should be configured
    if 'influxdb' not in config and 'prometheus' not in config:
        raise ConfigurationError("At least one monitoring backend (influxdb or prometheus) must be configured")


def _validate_influxdb_config(config: Dict[str, Any]) -> None:
    """Validate InfluxDB configuration."""
    required_fields = ['url', 'token', 'org', 'bucket']
    
    for field in required_fields:
        if field not in config:
            raise ConfigurationError(f"Missing required InfluxDB field: {field}")
        if not config[field]:
            raise ConfigurationError(f"InfluxDB field '{field}' cannot be empty")
    
    # Validate URL format
    url = config['url']
    if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
        raise ConfigurationError("InfluxDB URL must be a string starting with http:// or https://")


def _validate_prometheus_config(config: Dict[str, Any]) -> None:
    """Validate Prometheus configuration."""
    required_fields = ['gateway_url', 'job_name']
    
    for field in required_fields:
        if field not in config:
            raise ConfigurationError(f"Missing required Prometheus field: {field}")
        if not config[field]:
            raise ConfigurationError(f"Prometheus field '{field}' cannot be empty")
    
    # Validate gateway URL format
    url = config['gateway_url']
    if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
        raise ConfigurationError("Prometheus gateway_url must be a string starting with http:// or https://")
