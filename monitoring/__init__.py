"""
Monitoring module for pediatric leg length analysis.

Provides optional InfluxDB and Prometheus integration with graceful fallback
when monitoring services are not available.
"""

from .monitor_manager import MonitorManager

__version__ = "0.1.0"
__all__ = ["MonitorManager"]
