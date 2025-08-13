#!/usr/bin/env python
from leglength.inference import run_unified_single_inference
from leglength.ensemble import run_ensemble_inference, DEFAULT_ENSEMBLE_MODELS
import os
import json
import argparse
import logging
import sys
from pathlib import Path
import shutil
from pydicom.uid import generate_uid
import pydicom
import tempfile

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run leg length detection inference')
    
    # Only required arguments remain in CLI
    parser.add_argument('input_dir', type=Path,
                    help='Input directory containing DICOM images')
    parser.add_argument('output_dir', type=Path,
                    help='Output directory for results')
    
    return parser.parse_args()

def load_config(input_dir: Path, log: logging.Logger) -> dict:
    """Load configuration from task.json in input directory, or use defaults."""
    # Default configuration
    defaults = {
        'mode': 'ensemble',
        'model': 'resnext101_32x8d',
        'ensemble_models': ['resnet101', 'efficientnet_v2_m', 'mobilenet_v3_large'],
        'conf_threshold': 0.1,
        'enable_disagreement': False,
        'detection_weight': 0.5,
        'outlier_weight': 0.35,
        'localization_weight': 0.15,
        'series_offset': 1000
    }
    
    task_file = input_dir / "task.json"
    
    try:
        if task_file.exists():
            with open(task_file, "r") as f:
                task = json.load(f)
            
            # Extract settings from task file (if present)
            log.info(f"Task file content: {json.dumps(task, indent=2)}")
            if "process" in task and task["process"]:
                config = task["process"].get("settings", {})
                log.info(f"Loaded configuration from {task_file}")
                log.info(f"Extracted config: {json.dumps(config, indent=2)}")
                # Ensure we have the actual settings, not nested structure
                if isinstance(config, dict) and "settings" in config:
                    config = config["settings"]
                    log.info(f"Final config after extraction: {json.dumps(config, indent=2)}")
            else:
                config = {}
                log.warning("No 'process.settings' found in task.json")
        else:
            config = {}
            log.warning(f"No task.json found in {input_dir}, using defaults")
        
        # Merge with defaults
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
                log.info(f"Using default value for {key}: {value}")
        
        return config
        
    except Exception as e:
        log.error(f"Error loading configuration: {e}")
        log.info("Using default configuration")
        return defaults

def validate_config(config: dict, log: logging.Logger) -> list:
    """Validate configuration values and provide helpful error messages."""
    errors = []
    
    # Validate mode
    mode = config.get('mode', 'single')
    if mode not in ['single', 'ensemble']:
        errors.append(f"Invalid mode '{mode}'. Must be 'single' or 'ensemble'.")
    
    # Load valid models from registry
    valid_models = [
        'resnet101', 'resnext101_32x8d', 'densenet201', 'vit_l_16',
        'efficientnet_v2_m', 'mobilenet_v3_large', 'swin_v2_b', 'convnext_base'
    ]
    
    # Validate model (for single mode)
    if mode == 'single':
        model = config.get('model', 'resnext101_32x8d')
        if model not in valid_models:
            errors.append(f"Invalid model '{model}'. Must be one of: {valid_models}")
    
    # Validate ensemble_models (for ensemble mode)
    if mode == 'ensemble':
        ensemble_models = config.get('ensemble_models', DEFAULT_ENSEMBLE_MODELS)
        if not isinstance(ensemble_models, list):
            errors.append("ensemble_models must be a list of model names")
        elif len(ensemble_models) < 2:
            errors.append("ensemble_models must contain at least 2 models")
        elif len(ensemble_models) > 5:
            errors.append("ensemble_models must contain at most 5 models")
        else:
            for em in ensemble_models:
                if em not in valid_models:
                    errors.append(f"Invalid ensemble model '{em}'. Must be one of: {valid_models}")
            if len(set(ensemble_models)) != len(ensemble_models):
                errors.append("ensemble_models must contain unique model names")
    
    # Validate confidence threshold
    conf_threshold = config.get('conf_threshold', 0.1)
    if not isinstance(conf_threshold, (int, float)) or not (0.0 <= conf_threshold <= 1.0):
        errors.append("conf_threshold must be a number between 0.0 and 1.0")
    
    # Validate disagreement weights
    weights = {
        'detection_weight': 0.5,
        'outlier_weight': 0.35,
        'localization_weight': 0.15
    }
    weight_values = []
    
    for weight_name, default_value in weights.items():
        value = config.get(weight_name, default_value)
        if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
            errors.append(f"{weight_name} must be a number between 0.0 and 1.0")
        weight_values.append(value)
    
    # Check that weights sum to approximately 1.0
    if abs(sum(weight_values) - 1.0) > 0.01:
        log.warning(f"Disagreement weights sum to {sum(weight_values):.3f}, not 1.0. Consider adjusting for proper interpretation.")
    
    return errors

def process_image(dicom_path: Path, output_dir: Path, config: dict, log: logging.Logger) -> None:
    """Process a single DICOM image with either single or ensemble mode."""
    if config['mode'] == 'single':
        results = run_unified_single_inference(
            model_name=config['model'],
            dicom_path=str(dicom_path),
            output_dir=str(output_dir),
            confidence_threshold=config['conf_threshold'],
            best_per_class=True,  # Always use best per class
            logger=log
        )
    else:
        results = run_ensemble_inference(
            models=config['ensemble_models'],
            dicom_path=str(dicom_path),
            output_dir=str(output_dir),
            confidence_threshold=config['conf_threshold'],
            best_per_class=True,  # Always use best per class
            enable_disagreement=config['enable_disagreement'],
            detection_weight=config['detection_weight'],
            outlier_weight=config['outlier_weight'],
            localization_weight=config['localization_weight'],
            logger=log
        )
    
    # Log output files
    for output_type, path in results['output_files'].items():
        log.info(f"  📁 {output_type}: {Path(path).name}")
    
    # Log disagreement metrics for ensemble mode
    if config['mode'] == 'ensemble' and config['enable_disagreement']:
        if dm := results.get('disagreement_metrics'):
            if score := dm.get('overall_disagreement_score'):
                log.info(f"Overall disagreement score: {score:.3f}")
                if score > 0.5:
                    log.warning("⚠️  High disagreement detected - consider manual review")
                elif score > 0.2:
                    log.info("ℹ️  Moderate disagreement detected")
                else:
                    log.info("✅ Low disagreement - high confidence results")

def ready_outputs(tmp_dir: Path, output_dir: Path, qa_series_uid: str, sr_series_uid: str, config: dict, log: logging.Logger) -> None:
    """Move and update output files from temporary directory to final output directory."""
    log.info("Moving output files to final destination...")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Move and process all files
    for file in os.listdir(tmp_dir):
        src = tmp_dir / file
        dst = output_dir / file
        
        try:
            if file.endswith('.dcm'):
                # Update DICOM headers
                ds = pydicom.dcmread(src)
                if 'qa_visualization' in file:
                    ds.SeriesInstanceUID = qa_series_uid
                elif 'measurements_report' in file:
                    ds.SeriesInstanceUID = sr_series_uid
                else:
                    ds.SeriesInstanceUID = qa_series_uid  # Default to QA series
                ds.SOPInstanceUID = generate_uid()
                
                # Safely handle SeriesNumber with sequential offset for multiple output series
                try:
                    current_series = int(getattr(ds, "SeriesNumber", 0) or 0)
                    if 'qa_visualization' in file:
                        # First output series: original + series_offset
                        ds.SeriesNumber = current_series + config["series_offset"]
                    elif 'measurements_report' in file:
                        # Second output series: original + (2 * series_offset)
                        ds.SeriesNumber = current_series + (2 * config["series_offset"])
                    else:
                        # Other files: original + series_offset
                        ds.SeriesNumber = current_series + config["series_offset"]
                except (ValueError, TypeError):
                    if 'qa_visualization' in file:
                        ds.SeriesNumber = config["series_offset"]
                    elif 'measurements_report' in file:
                        ds.SeriesNumber = 2 * config["series_offset"]
                    else:
                        ds.SeriesNumber = config["series_offset"]
                
                # Update descriptions based on file type
                if 'qa_visualization' in file:
                    # Preserve original series description, just add SC prefix
                    original_series_desc = getattr(ds, 'SeriesDescription', '')
                    ds.SeriesDescription = f"SC({original_series_desc})" if original_series_desc else "SC"
                    
                    # Preserve original study description, just add AIDEOUT(LL()) prefix
                    original_study_desc = getattr(ds, 'StudyDescription', '')
                    if original_study_desc:
                        ds.StudyDescription = f"AIDEOUT(SC({original_study_desc}))"
                    else:
                        ds.StudyDescription = "AIDEOUT(SC)"
                        
                elif 'measurements_report' in file:
                    # Preserve original series description, just add SR prefix
                    original_series_desc = getattr(ds, 'SeriesDescription', '')
                    ds.SeriesDescription = f"SR({original_series_desc})" if original_series_desc else "SR"
                    
                    # Preserve original study description, just add AIDEOUT(SR()) prefix
                    original_study_desc = getattr(ds, 'StudyDescription', '')
                    if original_study_desc:
                        ds.StudyDescription = f"AIDEOUT(SR({original_study_desc}))"
                    else:
                        ds.StudyDescription = "AIDEOUT(SR)"
                        
                else:
                    # For any other DICOM files, preserve original descriptions
                    original_study_desc = getattr(ds, 'StudyDescription', '')
                    if original_study_desc:
                        ds.StudyDescription = f"AIDEOUT({original_study_desc})"
                    else:
                        ds.StudyDescription = "AIDEOUT"
                
                ds.save_as(dst)
            else:
                # Move non-DICOM files (e.g., JSON) directly
                shutil.move(src, dst)
                
        except Exception as e:
            log.error(f"Error processing {file}: {e}")
    
    # Clean up temporary directory
    shutil.rmtree(tmp_dir, ignore_errors=True)
    log.debug("Cleaned up temporary directory")

def main():
    """Main execution function."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    log = logging.getLogger(__name__)
    
    # Check if the input and output folders are provided as arguments
    if len(sys.argv) < 3:
        print("Error: Missing arguments!")
        print("Usage: run.py [input-folder] [output-folder]")
        sys.exit(1)
    
    try:
        log.info("=" * 60)
        log.info("LPCH Pediatric Leg Length Analysis Module v0.2.0")
        log.info("=" * 60)
        
        # Parse arguments and load configuration
        args = parse_args()
        
        # Check if the input and output folders actually exist
        if not args.input_dir.exists() or not args.output_dir.exists():
            print("IN/OUT paths do not exist")
            sys.exit(1)
        
        config = load_config(args.input_dir, log)
        
        # Validate configuration
        validation_errors = validate_config(config, log)
        if validation_errors:
            log.error("Configuration validation failed:")
            for error in validation_errors:
                log.error(f"  - {error}")
            sys.exit(1)
        
        # Log configuration
        log.info("Configuration:")
        for key, value in config.items():
            log.info(f"  {key}: {value}")
        
        # Create temporary directory
        tmp_dir = Path(tempfile.mkdtemp(prefix="leglength_"))
        
        # Process input directory
        series = {}
        for entry in os.scandir(args.input_dir):
            if entry.name.endswith(".dcm") and not entry.is_dir():
                # Group files by series
                series_id = entry.name.split("#", 1)[0]
                if series_id not in series:
                    series[series_id] = []
                series[series_id].append(entry.path)
        
        # Process each series
        for series_id, dicom_files in series.items():
            # For now, process first file in series (could be enhanced to handle multi-slice)
            if dicom_files:
                dicom_path = Path(dicom_files[0])
                log.info(f"Processing series {series_id}: {dicom_path.name}")
                process_image(dicom_path, tmp_dir, config, log)
        
        # Move outputs to final destination with separate series UIDs for each output type
        qa_series_uid = generate_uid()
        sr_series_uid = generate_uid()
        ready_outputs(tmp_dir, args.output_dir, qa_series_uid, sr_series_uid, config, log)
        
        # Log completion
        log.info("=" * 60)
        log.info("Analysis completed successfully!")
        log.info("Output file descriptions:")
        log.info("  🎨 QA Visualization: Enhanced DICOM with uncertainty indicators")
        log.info("  📋 Measurements Report (DICOM): Structured clinical report")
        log.info("  📊 Measurements Report (JSON): Comprehensive analysis data")
        log.info("=" * 60)
        log.info("✅ Module execution completed successfully")
        
    except Exception as e:
        log.error(f"❌ Module execution failed: {e}")
        log.error("Please check the configuration and input files")
        sys.exit(1)

if __name__ == "__main__":
    main()