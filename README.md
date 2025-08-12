# Leg Length Analysis Module

A Docker-based application for automated leg length measurements from DICOM images using deep learning models.

## Features

- 🔍 Detects 8 anatomical landmarks with confidence scoring
- 📐 Calculates leg length measurements
- 🎯 Supports single model and ensemble inference
- 📊 Generates DICOM-compliant reports and visualizations

## Prerequisites

- Docker installed
- DICOM files (.dcm format)
- ~10GB disk space for models

## Quick Start

1. **Build the Docker image:**
```bash
docker build -t leglength-test .
```

2. **Prepare your data:**
```bash
mkdir -p input output
cp your-image.dcm input/
```

3. **File Naming Requirements:**

The module supports two file naming patterns:
```
Simple:   <SeriesID>.dcm
Optional: <SeriesID>#<description>.dcm
```

Examples:
- Simple naming:
  - `1.2.826.0.1.dcm`
  - `1.2.826.0.2.dcm`
- With optional descriptions:
  - `1.2.826.0.1#AP_standing.dcm`
  - `1.2.826.0.2#lateral.dcm`

Files from the same series should share the same `SeriesID`. The `#` and description are optional and can be used to add human-readable labels to your files.

4. **Configure Analysis:**

Create a `tasks.json` file to configure the analysis:
```json
{
    "mode": "single",
    "model": "resnext101_32x8d",
    "ensemble_models": ["resnet101", "efficientnet_v2_m", "mobilenet_v3_large"],
    "conf_threshold": 0.1,
    "enable_disagreement": false,
    "detection_weight": 0.5,
    "outlier_weight": 0.35,
    "localization_weight": 0.15,
    "series_offset": 1000
}
```

5. **Run analysis:**

Using environment variables:
```bash
docker run -v $(pwd)/input:/input -v $(pwd)/output:/output \
    -v $(pwd)/tasks.json:/app/v0/tasks.json \
    -e MERCURE_IN_DIR=/input \
    -e MERCURE_OUT_DIR=/output \
    leglength-test
```

Or using command line arguments:
```bash
docker run -v $(pwd)/input:/input -v $(pwd)/output:/output \
    -v $(pwd)/tasks.json:/app/v0/tasks.json \
    leglength-test \
    /input /output
```

## Configuration Options

All configuration is done through `tasks.json`:

| Option | Default | Description |
|----------|---------|-------------|
| `mode` | `single` | Inference mode: `single` or `ensemble` |
| `model` | `resnext101_32x8d` | Model for single mode |
| `ensemble_models` | `[...]` | Models for ensemble mode |
| `conf_threshold` | `0.1` | Minimum confidence threshold |
| `enable_disagreement` | `false` | Enable disagreement analysis |
| `detection_weight` | `0.5` | Weight for detection disagreement |
| `outlier_weight` | `0.35` | Weight for outlier risk |
| `localization_weight` | `0.15` | Weight for localization disagreement |
| `series_offset` | `1000` | Offset for DICOM series number |

## Available Models

### Single Model Options
- `resnext101_32x8d` (default)
- `resnet101`
- `densenet201`
- `vit_l_16`
- `efficientnet_v2_m`
- `mobilenet_v3_large`
- `swin_v2_b`
- `convnext_base`

### Default Ensemble
- `resnet101`
- `efficientnet_v2_m`
- `mobilenet_v3_large`

## Output Files

For each processed image, three files are generated:

1. **QA Visualization** (`*_qa_visualization.dcm`)
   - Enhanced DICOM with uncertainty indicators
   - Measurement overlays and annotations

2. **Measurements Report** (`*_measurements_report.dcm`)
   - Structured clinical report in DICOM format
   - Quality assessment scores

3. **JSON Report** (`*_measurements_report.json`)
   - Comprehensive analysis data
   - Uncertainty metrics and measurements

## Example Configurations

### Single Model Analysis
```json
{
    "mode": "single",
    "model": "resnext101_32x8d",
    "conf_threshold": 0.1
}
```

### Ensemble Analysis
```json
{
  "mode": "ensemble",
  "ensemble_models": ["resnet101", "efficientnet_v2_m", "mobilenet_v3_large"],
  "enable_disagreement": true,
  "detection_weight": 0.5,
  "outlier_weight": 0.35,
  "localization_weight": 0.15
}
```

## Troubleshooting

1. **Memory Issues**
   - Reduce number of ensemble models
   - Use single model mode

2. **Model Download Failures**
   - Check internet connection
   - Rebuild with `docker build --no-cache`

3. **Permission Errors**
   - Ensure proper permissions on input/output directories
   - Run `chmod 755 ./input ./output`

## Performance

- Single Model: ~30-60 seconds per image
- Ensemble Mode: ~2-4 minutes per image
- Memory Usage: 4-8GB RAM (varies with ensemble size)
- Disk Space: ~10GB for model checkpoints

---

**Note**: This module provides automated measurements for research and clinical decision support. All results should be reviewed by qualified medical professionals before clinical use.