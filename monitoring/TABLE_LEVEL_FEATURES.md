# Table-Level Quality Features

## Overview

A new set of **10 table-level quality features** have been added to the InfluxDB metrics export. These features are calculated from individual model predictions and provide comprehensive quality metrics for automated QC and performance monitoring.

## What Was Added

### Code Changes

1. **`monitoring/metrics_collector.py`**
   - Added `numpy` import for statistical calculations
   - Added `_extract_table_level_features()` method to calculate quality metrics from individual model predictions
   - Updated `get_influx_data()` to include table-level features in the `pll_ai_image_metrics` measurement

2. **`run.py`**
   - Updated performance data recording to include `individual_model_predictions`
   - Ensures model predictions are passed to the monitoring system

3. **Documentation Updates**
   - `monitoring/README.md`: Added section describing table-level quality features
   - `monitoring/example_metrics.md`: Added comprehensive examples with field explanations
   - `monitoring/TABLE_LEVEL_FEATURES.md`: This file!

## The 10 New Metrics

All metrics are added as fields in the existing `pll_ai_image_metrics` InfluxDB measurement:

### Distance-Based Metrics (mm)
1. **`mean_distance`**: Average pairwise distance between model predictions across all landmarks
   - Lower values indicate better model agreement
   - Typical good value: < 3mm

2. **`max_distance`**: Maximum pairwise distance observed
   - Indicates worst-case disagreement
   - Use to flag potential outliers

3. **`distance_std`**: Standard deviation of pairwise distances
   - Measures consistency of disagreement
   - Lower is more consistent

4. **`distance_cv`**: Coefficient of variation (std/mean)
   - Normalized measure of variability
   - Accounts for scale differences

### Detection Quality Metrics
5. **`overall_detection_rate`**: Average detection rate across all 8 anatomical points
   - Range: 0.0 to 1.0
   - 1.0 = all models detected all points

6. **`detection_consistency`**: 1 - std(detection rates)
   - How consistent detection is across different points
   - 1.0 = perfect consistency

7. **`missing_point_ratio`**: Fraction of points not detected by ANY model
   - Range: 0.0 to 1.0
   - 0.0 = no missing points (ideal)

### Threshold-Based Metrics
8. **`high_distance_ratio`**: Fraction of pairwise distances > 5mm
   - Indicates moderate disagreement
   - Useful for flagging cases needing review

9. **`extreme_distance_ratio`**: Fraction of pairwise distances > 10mm
   - Indicates severe disagreement
   - Strong signal for QC failure

10. **`model_agreement_rate`**: Fraction of pairwise distances ≤ 3mm
    - Direct measure of good agreement
    - Higher is better (closer to 1.0)

## How It Works

### Calculation Process

For each processed image with multiple models:

1. **Extract Individual Predictions**: Get bounding boxes and labels from each model
2. **Calculate Pairwise Distances**: For each anatomical point (1-8):
   - Find box centers for each model
   - Calculate all pairwise distances
   - Convert from pixels to millimeters using DICOM pixel spacing
3. **Aggregate Statistics**: Compute global statistics across all points and model pairs
4. **Export to InfluxDB**: Add as additional fields in `pll_ai_image_metrics`

### Example Calculation

With 3 models and 8 points:
- 3 models = 3 pairs per point (C(3,2) = 3)
- 8 points × 3 pairs = 24 pairwise distances
- Statistics calculated over these 24 distances

## Usage Examples

### Grafana Query: Flag Poor Quality Images

```flux
from(bucket: "ai-inference-spc")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "pll_ai_image_metrics")
  |> filter(fn: (r) => r._field == "mean_distance" or 
                        r._field == "extreme_distance_ratio" or
                        r._field == "missing_point_ratio")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> filter(fn: (r) => 
      r.mean_distance > 3.0 or 
      r.extreme_distance_ratio > 0.1 or 
      r.missing_point_ratio > 0.0
  )
```

### Grafana Query: Model Performance Trending

```flux
from(bucket: "ai-inference-spc")
  |> range(start: -30d)
  |> filter(fn: (r) => r._measurement == "pll_ai_image_metrics")
  |> filter(fn: (r) => r._field == "model_agreement_rate")
  |> aggregateWindow(every: 1d, fn: mean, createEmpty: false)
  |> yield(name: "daily_agreement_rate")
```

### Grafana Query: Quality by Scanner Manufacturer

```flux
from(bucket: "ai-inference-spc")
  |> range(start: -30d)
  |> filter(fn: (r) => r._measurement == "pll_ai_image_metrics")
  |> filter(fn: (r) => r._field == "mean_distance")
  |> group(columns: ["scanner_manufacturer"])
  |> mean()
  |> yield(name: "mean_distance_by_scanner")
```

### Grafana Alert: High Disagreement

Create an alert when model agreement drops:

```flux
from(bucket: "ai-inference-spc")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "pll_ai_image_metrics")
  |> filter(fn: (r) => r._field == "model_agreement_rate")
  |> mean()
  |> map(fn: (r) => ({ r with _level:
      if r._value < 0.7 then "critical"
      else if r._value < 0.85 then "warning"
      else "ok"
  }))
```

## Decision Tree Training

These features are designed to match the features used in the decision tree analysis. To train a QC model:

1. **Export Training Data**:
```python
from influxdb_client import InfluxDBClient

client = InfluxDBClient(url="http://localhost:9051", token="your-token", org="mercure-ai")
query_api = client.query_api()

query = '''
from(bucket: "ai-inference-spc")
  |> range(start: -90d)
  |> filter(fn: (r) => r._measurement == "pll_ai_image_metrics")
  |> pivot(rowKey: ["_time", "series_id"], columnKey: ["_field"], valueColumn: "_value")
'''

df = query_api.query_data_frame(query)
```

2. **Train Decision Tree**:
```python
from sklearn.tree import DecisionTreeClassifier

feature_cols = [
    'mean_distance', 'max_distance', 'distance_std', 'distance_cv',
    'overall_detection_rate', 'detection_consistency', 'missing_point_ratio',
    'high_distance_ratio', 'extreme_distance_ratio', 'model_agreement_rate'
]

X = df[feature_cols]
y = df['quality_label']  # Ground truth from manual review

clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X, y)
```

## Benefits

### Automated Quality Control
- Automatically flag images needing manual review
- Reduce radiologist workload by pre-filtering good quality cases

### Performance Monitoring
- Track model drift over time
- Detect degradation in model agreement
- Monitor by demographic groups, scanner types, time periods

### Research & Analysis
- Understand model behavior patterns
- Identify challenging cases
- Correlate quality metrics with clinical outcomes

### Integration with Existing Metrics
- Complements existing `image_dds`, `image_lds`, `image_ors`, `image_cds` metrics
- Provides richer feature set for advanced analytics
- All stored in same InfluxDB measurement for easy querying

## Configuration

No configuration changes needed! The features are automatically calculated when:
- Monitoring is enabled in `task.json`
- Multiple models are configured (ensemble mode)
- Individual model predictions are available

### Example task.json

```json
{
    "process": {
        "settings": {
            "models": ["resnet101", "vit_l_16", "resnext101_32x8d"],
            "monitoring": {
                "enabled": true,
                "influxdb": {
                    "url": "http://localhost:9051",
                    "token": "ai-inference-token-12345",
                    "org": "mercure-ai",
                    "bucket": "ai-inference-spc"
                }
            }
        }
    }
}
```

## Troubleshooting

### Features Not Appearing

**Single Model Mode**: Table-level features are only calculated with 2+ models
- Check `task.json` has multiple models configured
- Default values (0.0) are exported for single model

**Missing Individual Predictions**: Ensure results include `individual_model_predictions`
- Check logs for "Added X table-level quality features"
- Verify InfluxDB write succeeded

### Unexpected Values

**999.0 distances**: Sentinel value for missing pairwise distances
- Occurs when models disagree on detection
- Excluded from statistics calculation

**Low detection rates**: Some points may not be detected by all models
- Check `missing_point_ratio` and `overall_detection_rate`
- Review original images if consistently low

## Version Info

- **Added**: October 2025
- **InfluxDB Schema Version**: 2.0
- **Requires**: numpy, ensemble mode (2+ models)
- **Backward Compatible**: Yes (graceful degradation to single model)



