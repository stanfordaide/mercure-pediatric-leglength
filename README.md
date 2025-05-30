# Leg Length Analysis - Flywheel Gear

A Flywheel gear for automated leg length measurements from DICOM images using deep learning models.

## Overview

This Docker-based application analyzes pediatric leg radiographs to automatically detect anatomical landmarks and calculate leg length measurements. It supports multiple deep learning model backbones and generates structured DICOM reports with QA visualizations.

## Features

- 🔍 **Automated Keypoint Detection**: Detects anatomical landmarks on leg radiographs
- 📏 **Precision Measurements**: Calculates leg length measurements with confidence scores
- 📊 **Multiple Output Formats**: JSON reports, DICOM Structured Reports, and QA visualizations
- 🤖 **Multiple Models**: Support for 5 different deep learning backbones
- 🐳 **Docker Ready**: Containerized for easy deployment and reproducibility

## Supported Models

- `resnext101_32x8d` (default)
- `densenet201`
- `resnet101`
- `efficientnet_b0`
- `mobilenet_v2`

## Prerequisites

- Docker installed on your system
- DICOM files (.dcm format) for analysis

## Building the Docker Image

```bash
# Clone the repository
git clone <repository-url>
cd fw-leglength

# Build the Docker image
docker build -t fw-leglength:latest .
```

**Note**: The build process will download ~8GB of model checkpoints, so ensure you have sufficient disk space and a stable internet connection.

## Running the Docker Container

### Basic Usage with Volume Mounting

```bash
# Create directories for input and output
mkdir -p ./input ./output

# Copy your DICOM file to the input directory
cp /path/to/your/sample.dcm ./input/

# Run the container with volume mounts
docker run --rm \
  -v $(pwd)/input:/flywheel/v0/input \
  -v $(pwd)/output:/flywheel/v0/output \
  fw-leglength:latest \
  --dicom_path /flywheel/v0/input/sample.dcm \
  --output_dir /flywheel/v0/output
```

### Advanced Usage with Custom Parameters

```bash
# Run with specific model and confidence threshold
docker run --rm \
  -v $(pwd)/input:/flywheel/v0/input \
  -v $(pwd)/output:/flywheel/v0/output \
  fw-leglength:latest \
  --model densenet201 \
  --dicom_path /flywheel/v0/input/sample.dcm \
  --output_dir /flywheel/v0/output \
  --confidence_threshold 0.7 \
  --best_per_class true
```

### Flywheel Gear Usage

When deployed as a Flywheel gear, the container expects inputs according to the Flywheel specification:

```bash
# Flywheel gear execution
docker run --rm \
  -v /flywheel/v0/input:/flywheel/v0/input \
  -v /flywheel/v0/output:/flywheel/v0/output \
  fw-leglength:latest
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `resnext101_32x8d` | Model backbone to use |
| `--dicom_path` | `/flywheel/v0/data/2000-1597.dcm` | Path to input DICOM file |
| `--output_dir` | `/flywheel/v0/outputs` | Directory to save results |
| `--confidence_threshold` | `0.0` | Minimum confidence for detections |
| `--best_per_class` | `true` | Return only highest confidence per class |

## Output Files

The analysis generates three types of output files:

### 1. JSON Measurements Report
```
{filename}_measurements.json
```
- Detailed measurements in JSON format
- Confidence scores for each detection
- Coordinate information for all keypoints

### 2. DICOM Structured Report
```
{filename}_measurements.dcm
```
- Standardized DICOM SR format
- Compatible with PACS systems
- Contains measurement data and metadata

### 3. QA Visualization
```
{filename}_qa.dcm
```
- DICOM file with overlaid detections
- Visual quality assurance
- Shows detected keypoints and measurements

## Example Usage Scenarios

### Scenario 1: Single File Analysis
```bash
# Analyze a single DICOM file
mkdir -p input output
cp my_xray.dcm input/
docker run --rm \
  -v $(pwd)/input:/flywheel/v0/input \
  -v $(pwd)/output:/flywheel/v0/output \
  fw-leglength:latest \
  --dicom_path /flywheel/v0/input/my_xray.dcm \
  --output_dir /flywheel/v0/output
```

### Scenario 2: Batch Processing Script
```bash
#!/bin/bash
INPUT_DIR="/path/to/dicom/files"
OUTPUT_DIR="/path/to/results"

for dicom_file in "$INPUT_DIR"/*.dcm; do
    filename=$(basename "$dicom_file" .dcm)
    echo "Processing $filename..."
    
    docker run --rm \
      -v "$INPUT_DIR":/flywheel/v0/input \
      -v "$OUTPUT_DIR":/flywheel/v0/output \
      fw-leglength:latest \
      --dicom_path "/flywheel/v0/input/$(basename "$dicom_file")" \
      --output_dir /flywheel/v0/output \
      --model resnext101_32x8d
done
```

### Scenario 3: High Confidence Analysis
```bash
# Only return high-confidence detections
docker run --rm \
  -v $(pwd)/input:/flywheel/v0/input \
  -v $(pwd)/output:/flywheel/v0/output \
  fw-leglength:latest \
  --dicom_path /flywheel/v0/input/sample.dcm \
  --output_dir /flywheel/v0/output \
  --confidence_threshold 0.8 \
  --model densenet201
```

## Container Health Check

The container includes a health check that verifies Python can start correctly:

```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"
```

## Troubleshooting

### Common Issues

1. **Permission Denied Errors**
   ```bash
   # Ensure proper permissions on mounted directories
   chmod 755 ./input ./output
   ```

2. **Model Download Failures**
   ```bash
   # Rebuild with clean cache if models fail to download
   docker build --no-cache -t fw-leglength:latest .
   ```

3. **Memory Issues**
   ```bash
   # Increase Docker memory limit for large models
   docker run --memory=8g --rm \
     -v $(pwd)/input:/flywheel/v0/input \
     -v $(pwd)/output:/flywheel/v0/output \
     fw-leglength:latest
   ```

4. **DICOM File Issues**
   - Ensure DICOM files are valid and readable
   - Check file permissions in mounted volumes
   - Verify file format is supported (.dcm extension)

## Development

### Local Development Setup
```bash
# Create virtual environment
python -m venv fw_leglength_env
source fw_leglength_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run locally
python run.py --dicom_path data/sample.dcm --output_dir outputs/
```

### Testing the Docker Build
```bash
# Quick test with sample data
docker run --rm fw-leglength:latest --help
```

## System Requirements

- **Docker**: Version 20.10+
- **Memory**: Minimum 8GB RAM recommended
- **Storage**: ~10GB for Docker image + models
- **Network**: Required for initial model download during build

## License

[Add your license information here]

## Support

For issues and questions:
- Create an issue in the repository
- Contact the development team
- Check the troubleshooting section above

## Version History

- **v1.0**: Initial release with 5 model backbones
- Multi-format output support
- Docker containerization
- Flywheel gear compatibility 