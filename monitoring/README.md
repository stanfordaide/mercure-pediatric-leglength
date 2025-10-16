# Monitoring Module - Stage 2

This is the full-featured monitoring module for the pediatric leg length analysis application. It provides production-ready InfluxDB and Prometheus integration with comprehensive metrics collection.

## Current Features (Stage 2)

- **Production-Ready Clients**: Full InfluxDB and Prometheus pushgateway integration
- **Comprehensive Metrics**: System resources, model performance, measurements, and custom metrics
- **Graceful Degradation**: Application runs normally even if monitoring is unavailable
- **Configuration-Driven**: All monitoring controlled through `task.json`
- **Session Management**: Track individual image processing sessions with full context
- **System Resource Monitoring**: CPU, memory, disk usage, and GPU metrics (if available)
- **Model Performance Tracking**: Confidence scores, landmark detection, inference timing
- **Medical Metrics**: Measurement values, quality scores, processing success rates
- **Error Handling**: Robust connection management with automatic retry and fallback

## Configuration

Add monitoring configuration to your `task.json` inside the `settings` section:

```json
{
    "process": {
        "settings": {
            "models": ["resnet101", "vit_l_16"],
            "series_offset": 1000,
            "monitoring": {
                "enabled": true,
                "influxdb": {
                    "url": "http://monitoring-host:8086",
                    "token": "your-token",
                    "org": "your-org",
                    "bucket": "pediatric-leglength"
                },
                "prometheus": {
                    "gateway_url": "http://monitoring-host:9091",
                    "job_name": "pediatric-leglength"
                }
            }
        }
    }
}
```

## Current Behavior

- **With monitoring config**: Collects comprehensive metrics and sends to InfluxDB/Prometheus
- **Without monitoring config**: Application runs normally with no monitoring overhead
- **Connection failures**: Application continues, monitoring gracefully disabled
- **Partial failures**: Works with only InfluxDB or only Prometheus available

## Metrics Collected

### Processing Metrics
- Total processing time per image
- Stage-wise timing (inference, measurement calculation, DICOM generation)
- Processing success/failure rates
- Queue depth and throughput

### Model Performance Metrics
- Confidence scores (average, min, max)
- Number of landmarks detected
- Model inference time
- Ensemble disagreement metrics (if applicable)

### Medical Metrics
- Measurement values (femur length, tibia length, etc.)
- Measurement quality scores
- Outlier detection
- Clinical validation results

### Image-Level Quality Metrics
- Detection Disagreement Score (DDS): Model disagreement on point detection
- Localization Disagreement Score (LDS): Spatial disagreement in mm
- Outlier Risk Score (ORS): Probability of outlier points
- Confidence Disagreement Score (CDS): Confidence variation across models

### Table-Level Quality Features (NEW)
Model agreement metrics calculated from individual model predictions:
- **Distance Statistics**: Mean, max, std, and CV of pairwise distances between models
- **Detection Quality**: Overall detection rate and consistency across points
- **Missing Data**: Ratio of completely missing anatomical points
- **Agreement Thresholds**: Ratios of high (>5mm), extreme (>10mm), and good (<3mm) agreement
- **Use Cases**: Automated QC, quality dashboards, performance monitoring, decision tree training

### System Resource Metrics
- CPU utilization (process and system-wide)
- Memory usage (RSS, available system memory)
- Disk usage and free space
- GPU memory usage (if CUDA available)

## Files

- `__init__.py` - Module initialization
- `monitor_manager.py` - Main monitoring coordinator
- `influx_client.py` - InfluxDB client with connection handling
- `prometheus_client.py` - Prometheus pushgateway client
- `metrics_collector.py` - Metrics collection and formatting
- `config_validator.py` - Configuration validation
- `exceptions.py` - Custom monitoring exceptions

## Future Stages

- Stage 3: Grafana dashboard templates and alerting rules
- Stage 4: Advanced analytics and anomaly detection

## Testing

The monitoring module is automatically initialized in `run.py`. Test with:

1. **No monitoring**: Use existing `task.json` files
2. **With monitoring**: Use `task_with_monitoring.json.example` as template

Check logs for monitoring status messages.

## Configuration Examples

### **Full Monitoring Enabled**
```json
{
    "process": {
        "settings": {
            "models": ["resnet101", "vit_l_16"],
            "series_offset": 1000,
            "monitoring": {
                "enabled": true,
                "influxdb": {
                    "url": "http://monitoring-stack:8086",
                    "token": "your-token",
                    "org": "medical",
                    "bucket": "leglength"
                },
                "prometheus": {
                    "gateway_url": "http://monitoring-stack:9091",
                    "job_name": "pediatric-leglength"
                }
            }
        }
    }
}
```

### **Monitoring Disabled (Default)**
```json
{
    "process": {
        "settings": {
            "models": ["resnet101", "vit_l_16"],
            "series_offset": 1000
        }
    }
}
```

### **InfluxDB Only**
```json
{
    "process": {
        "settings": {
            "models": ["resnet101", "vit_l_16"],
            "series_offset": 1000,
            "monitoring": {
                "enabled": true,
                "influxdb": {
                    "url": "http://monitoring-stack:8086",
                    "token": "your-token",
                    "org": "medical",
                    "bucket": "leglength"
                }
            }
        }
    }
}
```

### **Prometheus Only**
```json
{
    "process": {
        "settings": {
            "models": ["resnet101", "vit_l_16"],
            "series_offset": 1000,
            "monitoring": {
                "enabled": true,
                "prometheus": {
                    "gateway_url": "http://monitoring-stack:9091",
                    "job_name": "pediatric-leglength"
                }
            }
        }
    }
}
```
