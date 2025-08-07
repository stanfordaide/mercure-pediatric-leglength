#!/usr/bin/env python
from leglength.inference import run_unified_single_inference
from leglength.ensemble import run_ensemble_inference, DEFAULT_ENSEMBLE_MODELS
import os
import argparse
import logging
import sys
from pathlib import Path

from flywheel_gear_toolkit import GearToolkitContext

def parse_args():
    """Parse command line arguments for development/testing purposes only."""
    parser = argparse.ArgumentParser(description='Run leg length detection inference')
    parser.add_argument('--dev-mode', action='store_true', 
                      help='Enable development mode with command line overrides')
    parser.add_argument('--dicom_path', type=str, default=None,
                      help='Path to DICOM image')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Directory to save predictions')
    
    # Inference configuration arguments
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'ensemble'],
                      help='Inference mode: single or ensemble')
    parser.add_argument('--model', type=str, default='resnext101_32x8d',
                      help='Model backbone for single mode')
    parser.add_argument('--ensemble_models', nargs='+', default=['resnet101', 'efficientnet_v2_m', 'mobilenet_v3_large'],
                      help='Models for ensemble mode')
    parser.add_argument('--confidence_threshold', type=float, default=0.1,
                      help='Minimum confidence threshold for detections')
    parser.add_argument('--best_per_class', action='store_true', default=True,
                      help='Use best per class filtering')
    parser.add_argument('--enable_disagreement', action='store_true', default=True,
                      help='Enable disagreement analysis for ensemble mode')
    parser.add_argument('--detection_weight', type=float, default=0.5,
                      help='Weight for detection disagreement')
    parser.add_argument('--outlier_weight', type=float, default=0.35,
                      help='Weight for outlier risk')
    parser.add_argument('--localization_weight', type=float, default=0.15,
                      help='Weight for localization disagreement')
    
    return parser.parse_args()

def validate_config(config, log):
    """Validate configuration values and provide helpful error messages."""
    errors = []
    
    # Validate mode
    mode = config.get('mode', 'single')
    if mode not in ['single', 'ensemble']:
        errors.append(f"Invalid mode '{mode}'. Must be 'single' or 'ensemble'.")
    
    # Validate model (for single mode) - models from registry_full.json
    valid_models = [
        'resnet101', 'resnext101_32x8d', 'densenet201', 'vit_l_16',
        'efficientnet_v2_m', 'mobilenet_v3_large', 'swin_v2_b', 'convnext_base'
    ]
    model = config.get('model', 'resnext101_32x8d')
    if model not in valid_models:
        errors.append(f"Invalid model '{model}'. Must be one of: {valid_models}")
    
    # Validate ensemble_models (for ensemble mode)
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
    
    # Validate confidence_threshold
    confidence_threshold = config.get('confidence_threshold', 0.1)
    if not isinstance(confidence_threshold, (int, float)) or not (0.0 <= confidence_threshold <= 1.0):
        errors.append("confidence_threshold must be a number between 0.0 and 1.0")
    
    # Validate disagreement weights
    weights = ['detection_weight', 'outlier_weight', 'localization_weight']
    weight_values = []
    for weight in weights:
        value = config.get(weight, {'detection_weight': 0.5, 'outlier_weight': 0.35, 'localization_weight': 0.15}[weight])
        if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
            errors.append(f"{weight} must be a number between 0.0 and 1.0")
        weight_values.append(value)
    
    # Check that weights sum to approximately 1.0
    if abs(sum(weight_values) - 1.0) > 0.01:
        log.warning(f"Disagreement weights sum to {sum(weight_values):.3f}, not 1.0. Consider adjusting for proper interpretation.")
    
    return errors

def get_gear_config(context, args, log):
    """Extract and validate configuration from gear context with command-line fallbacks."""
    config = context.config.copy() if hasattr(context, 'config') and context.config else {}
    
    # Check if we have a gear config or need to use command-line arguments
    is_gear_mode = bool(config)
    
    if not is_gear_mode:
        log.info("No Flywheel gear config found - using command line arguments")
        # Use command-line arguments as the primary source
        config = {
            'mode': args.mode,
            'model': args.model,
            'ensemble_models': args.ensemble_models,
            'confidence_threshold': args.confidence_threshold,
            'best_per_class': args.best_per_class,
            'enable_disagreement': args.enable_disagreement,
            'detection_weight': args.detection_weight,
            'outlier_weight': args.outlier_weight,
            'localization_weight': args.localization_weight
        }
    elif args.dev_mode:
        log.info("Development mode enabled - command line arguments will override gear config")
        # Override gear config with command line arguments if specified
        for key in ['mode', 'model', 'ensemble_models', 'confidence_threshold', 
                   'best_per_class', 'enable_disagreement', 'detection_weight', 
                   'outlier_weight', 'localization_weight']:
            if hasattr(args, key):
                arg_value = getattr(args, key)
                if arg_value is not None:
                    # Only override if the argument was explicitly provided
                    config[key] = arg_value
                    log.info(f"Development override: {key} = {arg_value}")
    
    # Set defaults for any missing values
    defaults = {
        'mode': 'single',
        'model': 'resnext101_32x8d',
        'ensemble_models': ['resnet101', 'efficientnet_v2_m', 'mobilenet_v3_large'],  # Updated to use registry models
        'confidence_threshold': 0.1,
        'best_per_class': True,
        'enable_disagreement': True,
        'detection_weight': 0.5,
        'outlier_weight': 0.35,
        'localization_weight': 0.15
    }
    
    # Apply defaults for missing values
    for key, default_value in defaults.items():
        if key not in config or config[key] is None:
            config[key] = default_value
            log.info(f"Using default value for {key}: {default_value}")
    
    # Validate configuration
    validation_errors = validate_config(config, log)
    if validation_errors:
        log.error("Configuration validation failed:")
        for error in validation_errors:
            log.error(f"  - {error}")
        raise ValueError(f"Invalid configuration: {'; '.join(validation_errors)}")
    
    # Log configuration summary
    log.info("Configuration summary:")
    log.info(f"  Mode: {config['mode']}")
    if config['mode'] == 'single':
        log.info(f"  Model: {config['model']}")
    else:
        log.info(f"  Ensemble models: {config['ensemble_models']}")
        log.info(f"  Disagreement analysis: {'enabled' if config['enable_disagreement'] else 'disabled'}")
        if config['enable_disagreement']:
            log.info(f"  Disagreement weights: detection={config['detection_weight']}, outlier={config['outlier_weight']}, localization={config['localization_weight']}")
    log.info(f"  Confidence threshold: {config['confidence_threshold']}")
    log.info(f"  Best per class: {config['best_per_class']}")
    
    return config

def get_paths(context, args, log):
    """Get input and output paths from context with command-line fallbacks."""
    try:
        input_file = None
        output_dir = None
        
        # Try to get paths from gear context if available
        if hasattr(context, 'get_input_path') and hasattr(context, 'output_dir'):
            try:
                input_file = Path(context.get_input_path('dicom_in'))
                output_dir = Path(context.output_dir)
                log.info("Using paths from Flywheel gear context")
            except Exception as e:
                log.info(f"Could not get paths from gear context: {e}")
        
        # Use command-line arguments as fallbacks or primary source
        if args.dicom_path:
            input_file = Path(args.dicom_path)
            log.info(f"Using DICOM path from command line: {input_file}")
        
        if args.output_dir:
            output_dir = Path(args.output_dir)
            log.info(f"Using output directory from command line: {output_dir}")
        
        # Set defaults if still not available
        if input_file is None:
            # Try to find a DICOM file in the current directory
            dcm_files = list(Path('.').glob('*.dcm'))
            if dcm_files:
                input_file = dcm_files[0]
                log.info(f"Auto-detected DICOM file: {input_file}")
            else:
                raise FileNotFoundError("No DICOM input file specified or found")
        
        if output_dir is None:
            output_dir = Path('./output')
            log.info(f"Using default output directory: {output_dir}")
        
        # Validate paths
        if not input_file.exists():
            raise FileNotFoundError(f"Input DICOM file not found: {input_file}")
        
        if not input_file.suffix.lower() == '.dcm':
            log.warning(f"Input file does not have .dcm extension: {input_file}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Input DICOM: {input_file}")
        log.info(f"Output directory: {output_dir}")
        
        return str(input_file), str(output_dir)
        
    except Exception as e:
        log.error(f"Error setting up paths: {e}")
        raise

def main(context=None):
    """Main execution function for the gear."""
    if context and hasattr(context, 'init_logging'):
        context.init_logging()
    else:
        # Set up basic logging for standalone mode
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    log = logging.getLogger(__name__)
    
    # Parse command line arguments
    args = parse_args()
    
    try:
        log.info("=" * 60)
        log.info("LPCH Pediatric Leg Length Analysis Gear v0.2.0")
        log.info("=" * 60)
        
        # Get and validate configuration
        config = get_gear_config(context, args, log)
        
        # Get input/output paths
        dicom_path, output_dir = get_paths(context, args, log)
        
        # Execute analysis based on mode
        if config['mode'] == 'ensemble':
            log.info("Starting ensemble analysis...")
            
            # Run ensemble inference
            results = run_ensemble_inference(
                models=config['ensemble_models'],
                dicom_path=dicom_path,
                output_dir=output_dir,
                confidence_threshold=config['confidence_threshold'],
                best_per_class=config['best_per_class'],
                enable_disagreement=config['enable_disagreement'],
                detection_weight=config['detection_weight'],
                outlier_weight=config['outlier_weight'],
                localization_weight=config['localization_weight'],
                logger=log
            )
            
            log.info(f"Ensemble analysis completed with {len(results['models_processed'])} models")
            
            # Log disagreement summary
            if config['enable_disagreement'] and results.get('disagreement_metrics'):
                dm = results['disagreement_metrics']
                if 'overall_disagreement_score' in dm and dm['overall_disagreement_score'] is not None:
                    log.info(f"Overall disagreement score: {dm['overall_disagreement_score']:.3f}")
                    if dm['overall_disagreement_score'] > 0.5:
                        log.warning("⚠️  High disagreement detected - consider manual review")
                    elif dm['overall_disagreement_score'] > 0.2:
                        log.info("ℹ️  Moderate disagreement detected")
                    else:
                        log.info("✅ Low disagreement - high confidence results")
            
        else:
            log.info("Starting single model analysis...")
            
            # Run single model inference
            results = run_unified_single_inference(
                model_name=config['model'],
                dicom_path=dicom_path,
                output_dir=output_dir,
                confidence_threshold=config['confidence_threshold'],
                best_per_class=config['best_per_class'],
                logger=log
            )
            
            log.info("Single model analysis completed")
        
        # Log output summary
        log.info("=" * 60)
        log.info("Analysis completed successfully!")
        log.info("Generated outputs:")
        for output_type, path in results['output_files'].items():
            log.info(f"  📁 {output_type}: {Path(path).name}")
        
        log.info("=" * 60)
        log.info("Output file descriptions:")
        log.info("  🎨 QA Visualization: Enhanced DICOM with uncertainty indicators")
        log.info("  📋 Measurements Report (DICOM): Structured clinical report")
        log.info("  📊 Measurements Report (JSON): Comprehensive analysis data")
        log.info("=" * 60)
        
        # Check for problematic points
        if hasattr(results, 'fused_predictions') and results['fused_predictions'].get('problematic_points'):
            num_problems = len(results['fused_predictions']['problematic_points'])
            if num_problems > 0:
                log.warning(f"⚠️  {num_problems} problematic points detected - see reports for details")
        
        log.info("✅ Gear execution completed successfully")
        
    except Exception as e:
        log.error(f"❌ Gear execution failed: {e}")
        log.error("Please check the configuration and input files")
        raise

if __name__ == "__main__":
    try:
        # Try to run with Flywheel gear context
        with GearToolkitContext() as context:
            main(context)
    except Exception as e:
        # Handle case where gear context is not available - run in standalone mode
        print(f"Flywheel gear context not available: {e}")
        print("Running in standalone mode with command-line arguments...")
        try:
            main(context=None)
        except Exception as standalone_error:
            print(f"Error in standalone mode: {standalone_error}")
            print("Please ensure proper command-line arguments are provided")
            sys.exit(1) 