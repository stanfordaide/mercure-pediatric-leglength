#!/usr/bin/env python
from leglength.inference import run_inference
import os
import argparse
import logging
from pathlib import Path

from flywheel_gear_toolkit import GearToolkitContext

def parse_args():
    parser = argparse.ArgumentParser(description='Run leg length detection inference')
    parser.add_argument('--model', type=str, default='resnext101_32x8d',
                      help='Model backbone to use (e.g., resnet101, densenet201, efficientnet_b0, mobilenet_v2, resnext101_32x8d)')
    parser.add_argument('--dicom_path', type=str, default=None,
                      help='Path to DICOM image. If not provided, will use default path.')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Directory to save predictions. If not provided, will use default path.')
    parser.add_argument('--confidence_threshold', type=float, default=0.0,
                      help='Confidence threshold for predictions')
    parser.add_argument('--best_per_class', type=bool, default=True,
                      help='Return only the highest confidence prediction for each class')
    return parser.parse_args()

def get_default_paths(context):
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default paths relative to the project directory
    input_file = Path(context.get_input_path('dicom_in'))
    return {
        'dicom_path': input_file,
        'output_dir': os.path.join(current_dir, "output")
    }

def main(context):
    context.init_logging()
    log = logging.getLogger(__name__)
    log.info("Starting gear")
    
    # From the gear context, get the config settings  
    config = context.config   

    # Get the model name from the config   
    model_name = config.get('model_name')


    default_paths = get_default_paths(context)

    args = parse_args()

    # Use provided paths or defaults
    dicom_path = args.dicom_path or default_paths['dicom_path']
    output_dir = args.output_dir or default_paths['output_dir']


    log.info(f"Dicom path: {dicom_path}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Model name: {model_name}")
    log.info(f"Model name is currently overridden to resnet101")
    model_name = 'resnet101'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run inference
    run_inference(
        model_name=model_name,
        dicom_path=dicom_path,
        output_dir=output_dir,
        confidence_threshold=args.confidence_threshold,
        best_per_class=args.best_per_class,
        logger=log
    )

if __name__ == "__main__":
    with GearToolkitContext() as context:
        main(context) 