# Example Metrics Output - Stage 2

This document shows examples of the actual metrics data sent to InfluxDB and Prometheus.

## InfluxDB Metrics Examples

### Processing Duration Metric
```
measurement: processing_duration
tags:
  session_id: "1.2.826.0.1_1704123456"
  series_id: "1.2.826.0.1"
  instance: "pediatric-leglength"
  stage: "total_processing"
fields:
  duration_seconds: 45.23
time: 2024-01-15T10:30:00Z
```

### Model Performance Metric
```
measurement: model_performance
tags:
  session_id: "1.2.826.0.1_1704123456"
  series_id: "1.2.826.0.1"
  instance: "pediatric-leglength"
  model: "resnext101_32x8d"
fields:
  avg_confidence: 0.87
  min_confidence: 0.65
  max_confidence: 0.95
  confidence_count: 8
time: 2024-01-15T10:30:00Z
```

### Landmark Detection Metric
```
measurement: landmark_detection
tags:
  session_id: "1.2.826.0.1_1704123456"
  series_id: "1.2.826.0.1"
  instance: "pediatric-leglength"
  model: "resnext101_32x8d"
fields:
  landmarks_count: 8
time: 2024-01-15T10:30:00Z
```

### Leg Measurements Metric
```
measurement: leg_measurements
tags:
  session_id: "1.2.826.0.1_1704123456"
  series_id: "1.2.826.0.1"
  instance: "pediatric-leglength"
  measurement_name: "PLL_R_FEM"
fields:
  value_mm: 245.7
time: 2024-01-15T10:30:00Z
```

### System Resources Metric
```
measurement: system_resources
tags:
  session_id: "1.2.826.0.1_1704123456"
  series_id: "1.2.826.0.1"
  instance: "pediatric-leglength"
fields:
  process_cpu_percent: 85.2
  process_memory_mb: 1024.5
  system_cpu_percent: 45.8
  system_memory_percent: 67.3
  system_memory_available_gb: 4.2
  disk_usage_percent: 78.9
  disk_free_gb: 15.6
  gpu_0_memory_allocated_mb: 2048.0
  gpu_0_memory_reserved_mb: 2560.0
time: 2024-01-15T10:30:00Z
```

### PLL AI Image-Level Metrics
This measurement includes both existing uncertainty metrics and new table-level quality features.

```
measurement: pll_ai_image_metrics
tags:
  patient_id: "12345"
  study_id: "1.2.826.0.1.12345"
  series_id: "1.2.826.0.1"
  accession_number: "ACC001"
  patient_gender: "M"
  patient_age_group: "2-8"
  scanner_manufacturer: "GE"
  ai_model_version: "resnet101_vit_l_16_resnext101_32x8d"
  time_of_day: "morning"
  day_of_week: "monday"
  week_of_month: "week2"
  month: "january"
  year: "2024"
  day_type: "weekday"
fields:
  # Existing uncertainty metrics
  image_dds: 0.15          # Detection Disagreement Score
  image_lds: 2.3           # Localization Disagreement Score (mm)
  image_ors: 0.08          # Outlier Risk Score
  image_cds: 0.12          # Confidence Disagreement Score
  processing_duration_ms: 4523
  total_landmarks: 8
  
  # Table-level quality features (model agreement metrics)
  mean_distance: 1.8       # Average pairwise distance between models (mm)
  max_distance: 4.2        # Maximum pairwise distance (mm)
  distance_std: 0.9        # Standard deviation of distances
  distance_cv: 0.5         # Coefficient of variation for distances
  overall_detection_rate: 0.95      # Average detection rate across all points
  detection_consistency: 0.88       # 1 - std of detection rates
  missing_point_ratio: 0.0          # Ratio of completely missing points
  high_distance_ratio: 0.12         # Ratio of distances > 5mm
  extreme_distance_ratio: 0.03      # Ratio of distances > 10mm
  model_agreement_rate: 0.85        # Ratio of distances <= 3mm
time: 2024-01-15T10:30:00Z
```

**Table-Level Quality Features Explained:**

- `mean_distance`: Average pairwise distance between model predictions across all landmarks (mm). Lower is better.
- `max_distance`: Maximum pairwise distance observed. Indicates worst-case disagreement.
- `distance_std`: Standard deviation of pairwise distances. Measures consistency.
- `distance_cv`: Coefficient of variation (std/mean). Normalized measure of variability.
- `overall_detection_rate`: What fraction of points were detected by models on average.
- `detection_consistency`: How consistent detection rates are across points (1 = perfect consistency).
- `missing_point_ratio`: Fraction of anatomical points not detected by any model.
- `high_distance_ratio`: Fraction of pairwise distances exceeding 5mm threshold.
- `extreme_distance_ratio`: Fraction of pairwise distances exceeding 10mm threshold.
- `model_agreement_rate`: Fraction of pairwise distances within 3mm agreement threshold.

These metrics enable advanced quality monitoring and can be used to:
- Flag images with poor model agreement for review
- Track model performance degradation over time
- Build quality control dashboards
- Train decision trees for automated QC

## Prometheus Metrics Examples

### Processing Duration Histogram
```
processing_duration_seconds_bucket{series_id="1.2.826.0.1", stage="total_processing", model="resnext101_32x8d", le="30"} 0
processing_duration_seconds_bucket{series_id="1.2.826.0.1", stage="total_processing", model="resnext101_32x8d", le="60"} 1
processing_duration_seconds_bucket{series_id="1.2.826.0.1", stage="total_processing", model="resnext101_32x8d", le="+Inf"} 1
processing_duration_seconds_count{series_id="1.2.826.0.1", stage="total_processing", model="resnext101_32x8d"} 1
processing_duration_seconds_sum{series_id="1.2.826.0.1", stage="total_processing", model="resnext101_32x8d"} 45.23
```

### Processing Count
```
processing_total{status="completed", model="resnext101_32x8d"} 1
```

### Model Performance Gauges
```
landmarks_detected_count{series_id="1.2.826.0.1", model="resnext101_32x8d"} 8
model_confidence_score{series_id="1.2.826.0.1", model="resnext101_32x8d"} 0.87
```

### System Resource Gauges
```
memory_usage_bytes{component="main"} 1073741824
cpu_usage_percent{component="main"} 85.2
```

### Measurement Gauges
```
measurements_count{series_id="1.2.826.0.1", measurement_type="total"} 4
measurement_value_mm{series_id="1.2.826.0.1", measurement_name="PLL_R_FEM"} 245.7
measurement_value_mm{series_id="1.2.826.0.1", measurement_name="PLL_L_FEM"} 243.2
measurement_value_mm{series_id="1.2.826.0.1", measurement_name="PLL_R_TIB"} 198.5
measurement_value_mm{series_id="1.2.826.0.1", measurement_name="PLL_L_TIB"} 197.8
```

## Query Examples

### InfluxDB Queries

**Average processing time by model:**
```flux
from(bucket: "pediatric-leglength")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "processing_duration")
  |> filter(fn: (r) => r.stage == "total_processing")
  |> group(columns: ["model"])
  |> mean(column: "_value")
```

**System resource utilization over time:**
```flux
from(bucket: "pediatric-leglength")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "system_resources")
  |> filter(fn: (r) => r._field == "system_cpu_percent" or r._field == "system_memory_percent")
  |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
```

**Measurement distribution:**
```flux
from(bucket: "pediatric-leglength")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "leg_measurements")
  |> filter(fn: (r) => r.measurement_name == "PLL_R_FEM")
  |> histogram(bins: [200.0, 220.0, 240.0, 260.0, 280.0, 300.0])
```

### Prometheus Queries

**Processing rate (images per hour):**
```promql
rate(processing_total[1h]) * 3600
```

**95th percentile processing time:**
```promql
histogram_quantile(0.95, processing_duration_seconds_bucket)
```

**Average confidence score by model:**
```promql
avg by (model) (model_confidence_score)
```

**System memory usage percentage:**
```promql
(memory_usage_bytes / (1024^3)) / 16 * 100  # Assuming 16GB total RAM
```

## Alerting Examples

### Prometheus Alerting Rules

**High processing time:**
```yaml
- alert: HighProcessingTime
  expr: histogram_quantile(0.95, processing_duration_seconds_bucket) > 120
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Processing time is high"
    description: "95th percentile processing time is {{ $value }}s"
```

**Low confidence scores:**
```yaml
- alert: LowModelConfidence
  expr: avg by (model) (model_confidence_score) < 0.7
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Model confidence is low"
    description: "Average confidence for {{ $labels.model }} is {{ $value }}"
```

**High memory usage:**
```yaml
- alert: HighMemoryUsage
  expr: memory_usage_bytes > 6 * 1024^3  # 6GB
  for: 1m
  labels:
    severity: warning
  annotations:
    summary: "High memory usage detected"
    description: "Memory usage is {{ $value | humanizeBytes }}"
```

This comprehensive metrics collection enables detailed monitoring of the medical imaging pipeline performance, quality, and system health.
