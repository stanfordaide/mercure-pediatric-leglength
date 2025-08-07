# Leg Length Analysis - Flywheel Gear

A comprehensive Flywheel gear for automated leg length measurements from DICOM images using state-of-the-art deep learning models with ensemble support and advanced uncertainty quantification.

![Unified Workflow](docs/images/unified_workflow.svg)

## 🚀 What's New

### **🎯 Unified Output System** 
Both single and ensemble modes now generate **identical three-file outputs** with enhanced clinical information:
- **QA Visualization** with uncertainty color coding and error indicators
- **Secondary Capture** with structured clinical data and quality metrics
- **JSON Report** with comprehensive statistics and clinical recommendations

### **🔗 Derived Points Support**
Advanced measurement configuration supporting:
- **Midpoint calculations** between anatomical landmarks
- **Mixed point references** (integers + derived point names)
- **Flexible measurement definitions** with clinical anatomical names

### **🎭 Ensemble Inference**
Multi-model consensus with:
- **Confidence-weighted centroids** for robust predictions
- **Disagreement metrics** for uncertainty quantification
- **Problematic point detection** for quality control
- **Clinical recommendations** based on model agreement

## Overview

This Docker-based application analyzes pediatric leg radiographs to automatically detect anatomical landmarks and calculate leg length measurements. It features advanced ensemble capabilities, uncertainty visualization, and comprehensive clinical reporting.

## 🌟 Key Features

- 🔍 **Smart Keypoint Detection**: Detects 8 anatomical landmarks with confidence scoring
- 📐 **Derived Point Calculations**: Automatic midpoint and custom point derivations
- 🎯 **Ensemble Inference**: Multi-model consensus with uncertainty quantification
- 📊 **Unified Output Format**: Consistent outputs across single and ensemble modes
- 🎨 **Uncertainty Visualization**: Color-coded uncertainty indicators and error bounds
- 📋 **Clinical Integration**: DICOM-compliant reports with quality assessments
- 🤖 **Multiple Models**: Support for ConvNeXt, ResNeXt, ViT, and more
- 🐳 **Docker Ready**: Containerized for easy deployment and reproducibility

## 🤖 Supported Models

### Single Model Options
- `resnext101_32x8d` (default)
- `resnet101`
- `densenet201`
- `vit_l_16`
- `efficientnet_v2_m`
- `mobilenet_v3_large`
- `swin_v2_b`
- `convnext_base`

### Ensemble Configurations
- **Default Ensemble**: `resnet101`, `efficientnet_v2_m`, `mobilenet_v3_large`
- **Custom Ensembles**: Any combination of supported models
- **Disagreement Analysis**: Automatic quality assessment with clinical thresholds

## 📏 Measurement Configuration

The system supports flexible measurement definitions with derived points:

```json
{
  "derived_points": [
    {
      "name": "A",
      "type": "midpoint",
      "source_points": [3, 5]
    },
    {
      "name": "B", 
      "type": "midpoint",
      "source_points": [4, 6]
    }
  ],
  "measurements": [
    {
      "name": "femur_r",
      "join_points": [1, "A"]
    },
    {
      "name": "femur_l",
      "join_points": [2, "B"]
    },
    {
      "name": "tibia_r",
      "join_points": ["A", 7]
    },
    {
      "name": "tibia_l", 
      "join_points": ["B", 8]
    }
  ]
}
```

## 🛠️ Prerequisites

- Docker installed on your system
- DICOM files (.dcm format) for analysis
- ~10GB disk space for model checkpoints (ensemble mode)

## 🏗️ Building the Docker Image

```bash
# Clone the repository
git clone <repository-url>
cd fw-leglength

# Build the Docker image
docker build -t stanfordaide/aide-leglength:0.1.3 .
```

**Note**: The build process downloads multiple model checkpoints (~8-10GB total), so ensure sufficient disk space and stable internet connection.

## 🚀 Running the Container

### Single Model Mode

```bash
# Create directories
mkdir -p ./input ./output

# Copy your DICOM file
cp /path/to/your/sample.dcm ./input/

# Run single model analysis
docker run --rm \
  -v $(pwd)/input:/flywheel/v0/input \
  -v $(pwd)/output:/flywheel/v0/output \
  stanfordaide/aide-leglength:0.1.3 \
  --mode single \
  --model resnext101_32x8d \
  --dicom_path /flywheel/v0/input/sample.dcm \
  --output_dir /flywheel/v0/output
```

### Ensemble Mode

```bash
# Run ensemble analysis with default models
docker run --rm \
  -v $(pwd)/input:/flywheel/v0/input \
  -v $(pwd)/output:/flywheel/v0/output \
  stanfordaide/aide-leglength:0.1.3 \
  --mode ensemble \
  --dicom_path /flywheel/v0/input/sample.dcm \
  --output_dir /flywheel/v0/output \
  --enable_disagreement

# Custom ensemble with specific models
docker run --rm \
  -v $(pwd)/input:/flywheel/v0/input \
  -v $(pwd)/output:/flywheel/v0/output \
  stanfordaide/aide-leglength:0.1.3 \
  --mode ensemble \
  --ensemble_models efficientnet_v2_m mobilenet_v3_large resnet101 \
  --dicom_path /flywheel/v0/input/sample.dcm \
  --output_dir /flywheel/v0/output
```

## ⚙️ Command Line Arguments

### Core Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `single` | Inference mode: `single` or `ensemble` |
| `--dicom_path` | Auto-detected | Path to input DICOM file |
| `--output_dir` | `/flywheel/v0/outputs` | Directory to save results |
| `--confidence_threshold` | `0.0` | Minimum confidence for detections |

### Single Model Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `resnext101_32x8d` | Model backbone to use |

### Ensemble Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--ensemble_models` | `[convnext_base, resnext101_32x8d, vit_l_16]` | Models for ensemble |
| `--enable_disagreement` | `true` | Enable disagreement metrics |
| `--detection_weight` | `0.5` | Weight for detection disagreement |
| `--outlier_weight` | `0.35` | Weight for outlier risk |
| `--localization_weight` | `0.15` | Weight for localization disagreement |

## 📄 Output Files

Both single and ensemble modes generate **three identical output files**:

### 1. 🎨 QA Visualization (`*_qa_visualization.dcm`)
Enhanced DICOM visualization featuring:
- **Uncertainty color coding**: Green (low) → Yellow (medium) → Orange (high) → Red (critical)
- **Error indicators**: Uncertainty circles with variable radius
- **Measurement overlays**: Distance labels with uncertainty bounds
- **Interactive legend**: Visual guide for uncertainty interpretation
- **Ensemble metadata**: Model information and fusion details

### 2. 📋 Secondary Capture (`*_measurements_report.dcm`)
DICOM Secondary Capture with clinical data:
- **Structured measurements** in DICOM format
- **Problematic points analysis** with clinical descriptions
- **Quality assessment scores** (excellent/good/moderate/poor)
- **Model agreement statistics** (ensemble mode)
- **Clinical metadata** and patient information

### 3. 📊 JSON Report (`*_measurements_report.json`)
Comprehensive analysis report:
```json
{
  "metadata": {
    "analysis_timestamp": "2024-01-15T10:30:00",
    "analysis_type": "ensemble",
    "pixel_spacing": [0.143, 0.143]
  },
  "measurements": {
    "femur_r": {"millimeters": 245.7, "pixels": 1718, "confidence": 0.92},
    "femur_l": {"millimeters": 243.1, "pixels": 1700, "confidence": 0.89}
  },
  "point_uncertainties": {
    "1": {"overall_uncertainty": 2.3, "spatial_uncertainty": 1.8},
    "2": {"overall_uncertainty": 3.1, "spatial_uncertainty": 2.4}
  },
  "problematic_points": [
    {
      "point_id": 3,
      "reason": "high_uncertainty", 
      "description": "Point 3 has high spatial/confidence uncertainty"
    }
  ],
  "quality_assessment": {
    "overall_score": "good",
    "confidence_level": "high",
    "reliability": "reliable"
  },
  "clinical_recommendations": [
    "Analysis completed successfully with high confidence"
  ],
  "disagreement_metrics": {
    "overall_disagreement_score": 0.12,
    "detection_disagreement_score": 0.0,
    "localization_disagreement_score": 0.15
  }
}
```

## 🎯 Usage Examples

### High-Confidence Ensemble Analysis
```bash
docker run --rm \
  -v $(pwd)/input:/flywheel/v0/input \
  -v $(pwd)/output:/flywheel/v0/output \
  stanfordaide/aide-leglength:0.1.3 \
  --mode ensemble \
  --confidence_threshold 0.7 \
  --enable_disagreement \
  --dicom_path /flywheel/v0/input/sample.dcm
```

### Custom Ensemble with Specific Models
```bash
docker run --rm \
  -v $(pwd)/input:/flywheel/v0/input \
  -v $(pwd)/output:/flywheel/v0/output \
  stanfordaide/aide-leglength:0.1.3 \
  --mode ensemble \
  --ensemble_models resnet101 vit_l_16 densenet201 \
  --detection_weight 0.6 \
  --outlier_weight 0.3 \
  --localization_weight 0.1
```

### Batch Processing with Ensemble
```bash
#!/bin/bash
INPUT_DIR="/path/to/dicom/files"
OUTPUT_DIR="/path/to/results"

for dicom_file in "$INPUT_DIR"/*.dcm; do
    filename=$(basename "$dicom_file" .dcm)
    echo "Processing $filename with ensemble..."
    
    docker run --rm \
      -v "$INPUT_DIR":/flywheel/v0/input \
      -v "$OUTPUT_DIR/$filename":/flywheel/v0/output \
      stanfordaide/aide-leglength:0.1.3 \
      --mode ensemble \
      --dicom_path "/flywheel/v0/input/$(basename "$dicom_file")" \
      --output_dir /flywheel/v0/output \
      --enable_disagreement
done
```

## 🔬 Clinical Features

### Uncertainty Quantification
- **Spatial Uncertainty**: Variability in landmark localization
- **Confidence Uncertainty**: Spread in model confidence scores
- **Overall Uncertainty**: Combined uncertainty metric for clinical decision-making

### Quality Assessment
- **Excellent**: No problematic points, high confidence across all models
- **Good**: Minor issues, generally reliable measurements
- **Moderate**: Some problematic points, manual review recommended
- **Poor**: Significant issues, expert interpretation required

### Problematic Point Detection
- **High Uncertainty**: Points with significant spatial or confidence uncertainty
- **Detection Disagreement**: Points not detected by all models (ensemble)
- **Low Confidence**: Points with confidence below clinical thresholds
- **No Detection**: Anatomical landmarks not found by any model

### Clinical Recommendations
Automated suggestions based on analysis results:
- High confidence analyses: "Analysis completed successfully"
- Moderate uncertainty: "Consider additional validation"
- High disagreement: "Clinical caution advised"
- Missing landmarks: "Manual verification recommended"

## 🧠 Technical Architecture

### Fusion Algorithm
Ensemble predictions are fused using confidence-weighted centroids:
```python
weighted_position = Σ(confidence_i × position_i) / Σ(confidence_i)
uncertainty = √(spatial_variance + confidence_variance)
```

### Disagreement Metrics
- **Detection Disagreement**: Fraction of points not detected by all models
- **Localization Disagreement**: Spatial disagreement between model predictions  
- **Outlier Risk**: Risk of blunder errors based on inter-model distances

### Derived Points
- **Midpoint Calculation**: Automatic calculation of anatomical midpoints
- **Flexible References**: Support for both integer IDs and string names
- **Clinical Mapping**: Intuitive anatomical measurement names

## 🔧 Configuration

### Gear Configuration (Flywheel)
```json
{
  "mode": "ensemble",
  "ensemble_models": ["resnet101", "efficientnet_v2_m", "mobilenet_v3_large"],
  "enable_disagreement": true,
  "confidence_threshold": 0.1,
  "detection_weight": 0.5,
  "outlier_weight": 0.35,
  "localization_weight": 0.15
}
```

### Measurement Configuration
Located in `leglength/measurement_configs.json`:
- Define derived points (midpoints, custom calculations)
- Specify measurement connections between points
- Configure clinical measurement names

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues (Ensemble Mode)**
   ```bash
   # Reduce number of ensemble models
   --ensemble_models resnet101 efficientnet_v2_m
   ```

2. **High Uncertainty Warnings**
   ```bash
   # Lower confidence threshold for more detections
   --confidence_threshold 0.1
   ```

3. **Model Download Failures**
   ```bash
   # Rebuild with clean cache
   docker build --no-cache -t stanfordaide/aide-leglength:0.1.3 .
   ```

4. **Permission Denied Errors**
   ```bash
   # Ensure proper permissions
   chmod 755 ./input ./output
   ```

### Performance Optimization

- **Single Model**: ~30-60 seconds per image
- **Ensemble (3 models)**: ~2-4 minutes per image  
- **Memory Usage**: ~4-8GB RAM (depending on ensemble size)
- **Disk Space**: ~10GB for all model checkpoints

## 📚 References

- [Flywheel Gear Specification](https://gear-toolkit.readthedocs.io/)
- [DICOM Secondary Capture Standard](https://dicom.nema.org/)
- [PyTorch Model Hub](https://pytorch.org/hub/)

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

---

**Note**: This gear provides automated measurements for research and clinical decision support. All results should be reviewed by qualified medical professionals before clinical use. 