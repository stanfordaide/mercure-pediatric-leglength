# Pediatric Leg Length Analysis Module

A Docker-based application for automated pediatric leg length measurements from DICOM images using deep learning models.

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
docker build -t stanfordaide/pediatric-leglength:latest .
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

Create a `task.json` file to configure the analysis. Here's the complete specification:

```json
{
    "mode": "ensemble",
    "model": "resnext101_32x8d",
    "ensemble_models": ["resnet101", "resnext101_32x8d", "vit_l_16"],
    "conf_threshold": 0.1,
    "enable_disagreement": true,
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
    -v $(pwd)/task.json.example:/input/task.json \
    -e MERCURE_IN_DIR=/input \
    -e MERCURE_OUT_DIR=/output \
    stanfordaide/pediatric-leglength
```

Or using command line arguments:
```bash
docker run -v $(pwd)/input:/input -v $(pwd)/output:/output \
    -v $(pwd)/task.json.example:/input/task.json \
    stanfordaide/pediatric-leglength \
    /input /output
```

**Note on Permissions**: The container automatically handles output directory permissions. If you encounter permission issues, the container will run as root temporarily to fix permissions and then switch to a non-root user for security.

### Advanced Usage

For more control over user permissions, you can:

1. **Run with specific user ID:**
```bash
docker run -v $(pwd)/input:/input -v $(pwd)/output:/output \
    -e DOCKER_UID=$(id -u) -e DOCKER_GID=$(id -g) \
    stanfordaide/pediatric-leglength
```

2. **Run with host user mapping:**
```bash
docker run -v $(pwd)/input:/input -v $(pwd)/output:/output \
    --user $(id -u):$(id -g) \
    stanfordaide/pediatric-leglength
```

## Configuration Options

### **Analysis Mode**
- **`mode`** (string, required): Analysis mode selection
  - `"single"`: Use a single model for inference
  - `"ensemble"`: Use multiple models with consensus fusion

### **Model Selection**
- **`model`** (string, required for single mode): Primary model for single-mode inference
  - Available models: `"resnet101"`, `"resnext101_32x8d"`, `"vit_l_16"`, `"densenet201"`, `"efficientnet_v2_m"`, `"mobilenet_v3_large"`, `"swin_v2_b"`, `"convnext_base"`

- **`ensemble_models`** (array, required for ensemble mode): List of models to use in ensemble
  - Must contain 2 or more model names from the available list above
  - Example: `["resnet101", "resnext101_32x8d", "vit_l_16"]`

### **Detection Parameters**
- **`conf_threshold`** (float, optional): Confidence threshold for landmark detection
  - Range: 0.0 to 1.0
  - Default: 0.1
  - Lower values detect more landmarks but may include false positives
  - Higher values are more selective but may miss valid landmarks

### **Ensemble Analysis**
- **`enable_disagreement`** (boolean, optional): Enable disagreement metrics calculation
  - Default: true
  - Only applies to ensemble mode
  - Calculates uncertainty quantification between models

### **Disagreement Weights** (ensemble mode only)
- **`detection_weight`** (float, optional): Weight for detection disagreement in overall uncertainty score
  - Range: 0.0 to 1.0
  - Default: 0.5
  - Higher values emphasize whether all models detect the same landmarks

- **`outlier_weight`** (float, optional): Weight for outlier risk in overall uncertainty score
  - Range: 0.0 to 1.0
  - Default: 0.35
  - Higher values emphasize spatial consistency between models

- **`localization_weight`** (float, optional): Weight for localization disagreement in overall uncertainty score
  - Range: 0.0 to 1.0
  - Default: 0.15
  - Higher values emphasize precise landmark positioning

**Note**: The three weights should sum to 1.0 for proper interpretation. If they don't sum to 1.0, the system will automatically normalize them.

### **Output Configuration**
- **`series_offset`** (integer, optional): Offset for DICOM series numbering
  - Default: 1000
  - Used to avoid conflicts with original series numbers
  - Output series will be numbered as: original_series + series_offset

## Example Configurations

### **Single Model Analysis**
```json
{
    "mode": "single",
    "model": "resnext101_32x8d",
    "conf_threshold": 0.2,
    "series_offset": 1000
}
```

### **Ensemble Analysis with High Confidence**
```json
{
    "mode": "ensemble",
    "ensemble_models": ["resnet101", "resnext101_32x8d", "vit_l_16"],
    "conf_threshold": 0.3,
    "enable_disagreement": true,
    "detection_weight": 0.6,
    "outlier_weight": 0.3,
    "localization_weight": 0.1,
    "series_offset": 1000
}
```

### **Ensemble Analysis with Balanced Weights**
```json
{
    "mode": "ensemble",
    "ensemble_models": ["resnet101", "efficientnet_v2_m", "mobilenet_v3_large"],
    "conf_threshold": 0.1,
    "enable_disagreement": true,
    "detection_weight": 0.4,
    "outlier_weight": 0.4,
    "localization_weight": 0.2,
    "series_offset": 1000
}
```

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