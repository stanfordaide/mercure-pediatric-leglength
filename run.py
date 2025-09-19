#!/usr/bin/env python
from leglength.outputs import DicomProcessor
from leglength.inference import inference_handler
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
import time


# Default ensemble models to use
DEFAULT_ENSEMBLE_MODELS = ['convnext_base', 'resnext101_32x8d', 'vit_l_16']

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
        'models': ['resnet101', 'efficientnet_v2_m', 'mobilenet_v3_large'],
        'series_offset': 1000,
        'femur_threshold': 2.0,
        'tibia_threshold': 2.0,
        'total_threshold': 5.0,
        'confidence_threshold': 0.0
    }
    
    task_file = input_dir / "task.json"
    
    try:
        if task_file.exists():
            with open(task_file, "r") as f:
                task = json.load(f)
            
            # Extract settings from task file (if present)
            # log.info(f"Task file content: {json.dumps(task, indent=2)}")
            if "process" in task and task["process"]:
                config = task["process"].get("settings", {})
                log.info(f"Loaded configuration from {task_file}")
                # log.info(f"Extracted config: {json.dumps(config, indent=2)}")
                # Ensure we have the actual settings, not nested structure
                if isinstance(config, dict) and "settings" in config:
                    config = config["settings"]
                    log.info(f"Final config after extraction: {json.dumps(config, indent=2)}")
                
                # Monitoring configuration is now part of settings - no special handling needed
                # It will be included automatically with the settings
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
    
    # Load valid models from registry
    valid_models = [
        'resnet101', 'resnext101_32x8d', 'densenet201', 'vit_l_16',
        'efficientnet_v2_m', 'mobilenet_v3_large', 'swin_v2_b', 'convnext_base'
    ]

    # Validate models
    models = config.get('models', DEFAULT_ENSEMBLE_MODELS)
    if not isinstance(models, list):
        errors.append("models must be passed as a list of model names")
    else:
        for em in models:
            if em not in valid_models:
                errors.append(f"Invalid ensemble model '{em}'. Must be one of: {valid_models}")
        if len(set(models)) != len(models):
            errors.append("models must contain unique model names")

    # Validate numeric thresholds
    for threshold in ['femur_threshold', 'tibia_threshold', 'total_threshold']:
        value = config.get(threshold)
        if not isinstance(value, (int, float)):
            errors.append(f"{threshold} must be a number")
        elif value < 0:
            errors.append(f"{threshold} must be positive")

    # Validate series offset
    series_offset = config.get('series_offset')
    if not isinstance(series_offset, int):
        errors.append("series_offset must be an integer")
    elif series_offset < 0:
        errors.append("series_offset must be positive")

    # Validate confidence threshold
    confidence_threshold = config.get('confidence_threshold')
    if not isinstance(confidence_threshold, (int, float)):
        errors.append("confidence_threshold must be a number")
    elif confidence_threshold < 0 or confidence_threshold > 1:
        errors.append("conf_threshold must be between 0 and 1")

    return errors


def process_image(dicom_path: Path, output_dir: Path, config: dict, logger: logging.Logger) -> None:
    
    
    
    results = inference_handler(
        models=config['models'],
        dicom_path=str(dicom_path),
        output_dir=str(output_dir),
        config=config,
        logger=logger
    )
    
    
    logger.info(f"Output directory: {output_dir}")
    
    output_processor = DicomProcessor()
    
    qa_dicom = output_processor.get_qa_dicom(results, str(dicom_path))
    
    qa_dicom.is_implicit_VR = False
    qa_dicom.is_little_endian = True
    qa_dicom.save_as(str(output_dir / 'qa_output.dcm'), write_like_original=False)
    
    
    
    # Create SR DICOM
    sr_dicom = output_processor.get_sr_dicom(results, str(dicom_path), config)
    sr_dicom.save_as(str(output_dir / 'sr_output.dcm'), write_like_original=False)
    

    return results
    

# def process_image(dicom_path: Path, output_dir: Path, config: dict, log: logging.Logger) -> None:
#     """Process a single DICOM image with either single or ensemble mode."""
#     if config['mode'] == 'single':
#         results = run_unified_single_inference(
#             model_name=config['model'],
#             dicom_path=str(dicom_path),
#             output_dir=str(output_dir),
#             confidence_threshold=config['conf_threshold'],
#             best_per_class=True,  # Always use best per class
#             discrepancy_threshold_cm=config['discrepancy_threshold_cm'],
#             logger=log
#         )
#     else:
#         results = run_ensemble_inference(
#             models=config['ensemble_models'],
#             dicom_path=str(dicom_path),
#             output_dir=str(output_dir),
#             confidence_threshold=config['conf_threshold'],
#             best_per_class=True,  # Always use best per class
#             enable_disagreement=config['enable_disagreement'],
#             detection_weight=config['detection_weight'],
#             outlier_weight=config['outlier_weight'],
#             localization_weight=config['localization_weight'],
#             discrepancy_threshold_cm=config['discrepancy_threshold_cm'],
#             logger=log
#         )
    
#     # Log output files
#     for output_type, path in results['output_files'].items():
#         log.info(f"  📁 {output_type}: {Path(path).name}")
    
#     # Log disagreement metrics for ensemble mode
#     if config['mode'] == 'ensemble' and config['enable_disagreement']:
#         if dm := results.get('disagreement_metrics'):
#             if score := dm.get('overall_disagreement_score'):
#                 log.info(f"Overall disagreement score: {score:.3f}")
#                 if score > 0.5:
#                     log.warning("⚠️  High disagreement detected - consider manual review")
#                 elif score > 0.2:
#                     log.info("ℹ️  Moderate disagreement detected")
#                 else:
#                     log.info("✅ Low disagreement - high confidence results")

# def ready_outputs(tmp_dir: Path, output_dir: Path, qa_series_uid: str, sr_series_uid: str, config: dict, log: logging.Logger) -> None:
#     """Move and update output files from temporary directory to final output directory."""
#     log.info("Moving output files to final destination...")
    
#     # Create output directory if it doesn't exist
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Move and process all files
#     for file in os.listdir(tmp_dir):
#         src = tmp_dir / file
#         dst = output_dir / file
        
#         try:
#             if file.endswith('.dcm'):
#                 # Update DICOM headers
#                 ds = pydicom.dcmread(src)
#                 if 'qa_visualization' in file:
#                     ds.SeriesInstanceUID = qa_series_uid
#                 elif 'measurements_report' in file:
#                     ds.SeriesInstanceUID = sr_series_uid
#                 else:
#                     ds.SeriesInstanceUID = qa_series_uid  # Default to QA series
#                 ds.SOPInstanceUID = generate_uid()
                
#                 # Safely handle SeriesNumber with sequential offset for multiple output series
#                 try:
#                     current_series = int(getattr(ds, "SeriesNumber", 0) or 0)
#                     if 'qa_visualization' in file:
#                         # First output series: original + series_offset
#                         ds.SeriesNumber = current_series + config["series_offset"]
#                     elif 'measurements_report' in file:
#                         # Second output series: original + (2 * series_offset)
#                         ds.SeriesNumber = current_series + (2 * config["series_offset"])
#                     else:
#                         # Other files: original + series_offset
#                         ds.SeriesNumber = current_series + config["series_offset"]
#                 except (ValueError, TypeError):
#                     if 'qa_visualization' in file:
#                         ds.SeriesNumber = config["series_offset"]
#                     elif 'measurements_report' in file:
#                         ds.SeriesNumber = 2 * config["series_offset"]
#                     else:
#                         ds.SeriesNumber = config["series_offset"]
                
#                 # Update descriptions based on file type
#                 if 'qa_visualization' in file:
#                     # Preserve original series description, just add SC prefix
#                     original_series_desc = getattr(ds, 'SeriesDescription', '')
#                     ds.SeriesDescription = f"SC({original_series_desc})" if original_series_desc else "SC"
                    
#                     # Preserve original study description, just add AIDEOUT(LL()) prefix
#                     original_study_desc = getattr(ds, 'StudyDescription', '')
#                     if original_study_desc:
#                         ds.StudyDescription = f"AIDEOUT(SC({original_study_desc}))"
#                     else:
#                         ds.StudyDescription = "AIDEOUT(SC)"
                        
#                 elif 'measurements_report' in file:
#                     # Preserve original series description, just add SR prefix
#                     original_series_desc = getattr(ds, 'SeriesDescription', '')
#                     ds.SeriesDescription = f"SR({original_series_desc})" if original_series_desc else "SR"
                    
#                     # Preserve original study description, just add AIDEOUT(SR()) prefix
#                     original_study_desc = getattr(ds, 'StudyDescription', '')
#                     if original_study_desc:
#                         ds.StudyDescription = f"AIDEOUT(SR({original_study_desc}))"
#                     else:
#                         ds.StudyDescription = "AIDEOUT(SR)"
                        
#                 else:
#                     # For any other DICOM files, preserve original descriptions
#                     original_study_desc = getattr(ds, 'StudyDescription', '')
#                     if original_study_desc:
#                         ds.StudyDescription = f"AIDEOUT({original_study_desc})"
#                     else:
#                         ds.StudyDescription = "AIDEOUT"
                
#                 ds.save_as(dst)
#             else:
#                 # Move non-DICOM files (e.g., JSON) directly
#                 shutil.move(src, dst)
                
#         except Exception as e:
#             log.error(f"Error processing {file}: {e}")
    
#     # Clean up temporary directory
#     shutil.rmtree(tmp_dir, ignore_errors=True)
#     log.debug("Cleaned up temporary directory")

def main():
    """Main execution function."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)
    
    # Check if the input and output folders are provided as arguments
    if len(sys.argv) < 3:
        print("Error: Missing arguments!")
        print("Usage: run.py [input-folder] [output-folder]")
        sys.exit(1)
    
    # try:
    logger.info("=" * 60)
    logger.info("LPCH Pediatric Leg Length Analysis Module v0.2.0")
    logger.info("=" * 60)
    
    # Parse arguments and load configuration
    args = parse_args()
    
    # Check if the input and output folders actually exist
    if not args.input_dir.exists() or not args.output_dir.exists():
        print("IN/OUT paths do not exist")
        sys.exit(1)
    
    config = load_config(args.input_dir, logger)
    
    # Initialize monitoring (optional)
    try:
        logger.info("Attempting to initialize monitoring...")
        from monitoring import MonitorManager
        monitor = MonitorManager(config, logger)
        logger.info(f"Monitoring initialization completed. Enabled: {monitor.is_enabled()}")
    except ImportError:
        logger.warning("Monitoring module not available - monitoring disabled")
        monitor = None
    except Exception as e:
        logger.warning(f"Failed to initialize monitoring: {e}")
        logger.debug("Full monitoring error:", exc_info=True)
        monitor = None
    
    # Validate configuration
    validation_errors = validate_config(config, logger)
    if validation_errors:
        logger.error("Configuration validation failed:")
        for error in validation_errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    # Log configuration
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
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
    
    
    logger.info(f"Series: {series}")
    
    
    # Process each series
    for series_id, dicom_files in series.items():
        
        dicom_headers = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
        accession_number = getattr(dicom_headers, "AccessionNumber", None)
        
        # Start monitoring session using accession number as primary identifier
        monitoring_id = accession_number if accession_number else series_id
        
        # Add accession number and series info to monitoring config
        monitoring_config = {**config}
        if accession_number:
            monitoring_config['accession_number'] = accession_number
        monitoring_config['series_id'] = series_id
        
        session_id = monitor.start_session(monitoring_id, monitoring_config) if monitor else ""
        
        try:
            # Select the file with the highest InstanceNumber in the series
            if dicom_files:
                best_path = None
                best_inst = -1
                for f in dicom_files:
                    try:
                        ds = pydicom.dcmread(f, stop_before_pixels=True)
                        inst = int(getattr(ds, "InstanceNumber", 0) or 0)
                        if inst > best_inst:
                            best_inst = inst
                            best_path = Path(f)
                    except Exception as e:
                        logger.debug(f"Could not read InstanceNumber from {f}: {e}")
                if best_path is None:
                    dicom_path = Path(dicom_files[0])
                    logger.warning(f"No valid InstanceNumber found; falling back to first file in series: {dicom_path.name}")
                else:
                    dicom_path = best_path
                    if accession_number:
                        logger.info(f"Processing accession {accession_number} (series {series_id}): {dicom_path.name} (highest InstanceNumber={best_inst})")
                    else:
                        logger.info(f"Processing series {series_id}: {dicom_path.name} (highest InstanceNumber={best_inst})")
                
                # Track processing time
                start_time = time.time()
                results = process_image(dicom_path, args.output_dir, config, logger)
                end_time = time.time()
                
                # Record metrics
                if monitor:
                    monitor.track_processing_time(session_id, "total_processing", start_time, end_time)
                    
                    # Record model performance metrics
                    model_metrics = {
                        'landmarks_detected': len(results.get('boxes', [])),
                        'confidence_scores': results.get('scores', []),
                        'measurements': results.get('measurements', {})
                    }
                    monitor.record_model_performance(session_id, config.get('models', ['unknown'])[0], model_metrics)
                    
                    # Record measurements with DICOM path
                    if results.get('measurements'):
                        monitor.record_measurements(
                            session_id, 
                            results['measurements'], 
                            str(dicom_path)
                        )
                    
                    # Record performance data (uncertainties and point statistics)
                    performance_data = {
                        'uncertainties': results.get('uncertainties', {}),
                        'point_statistics': results.get('point_statistics', {}),
                        'issues': results.get('issues', [])
                    }
                    monitor.record_performance_data(
                        session_id, 
                        performance_data, 
                        str(dicom_path)
                    )
                
                logger.info(f"Results: {json.dumps(results, indent=2)}")
                
                # Mark session as completed
                if monitor:
                    monitor.end_session(session_id, "completed")
                    
        except Exception as e:
            logger.error(f"Error processing series {series_id}: {e}")
            if monitor:
                monitor.end_session(session_id, "failed")
            raise
            
            
        
    
        # # Move outputs to final destination with separate series UIDs for each output type
        # qa_series_uid = generate_uid()
        # sr_series_uid = generate_uid()
        # ready_outputs(tmp_dir, args.output_dir, qa_series_uid, sr_series_uid, config, logger)
        
        # # Log completion
        # logger.info("=" * 60)
        # logger.info("Analysis completed successfully!")
        # logger.info("Output file descriptions:")
        # logger.info("  🎨 QA Visualization: Enhanced DICOM with uncertainty indicators")
        # logger.info("  📋 Measurements Report (DICOM): Structured clinical report")
        # logger.info("  📊 Measurements Report (JSON): Comprehensive analysis data")
        # logger.info("=" * 60)
        # logger.info("✅ Module execution completed successfully")
        
    # except Exception as e:
    #     logger.error(f"❌ Module execution failed: {e}")
    #     logger.error("Please check the configuration and input files")
    #     sys.exit(1)

if __name__ == "__main__":
    
    main()