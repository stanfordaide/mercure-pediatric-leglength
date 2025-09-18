"""
Custom exceptions for the monitoring module.
"""


class MonitoringError(Exception):
    """Base exception for monitoring-related errors."""
    pass


class ConfigurationError(MonitoringError):
    """Raised when monitoring configuration is invalid."""
    pass


class ConnectionError(MonitoringError):
    """Raised when connection to monitoring services fails."""
    pass
