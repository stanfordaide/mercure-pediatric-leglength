from leglength.inference import run_inference
import os
import argparse

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

def get_default_paths():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default paths relative to the project directory
    return {
        'dicom_path': os.path.join(current_dir, "data", "sample.dcm"),
        'output_dir': os.path.join(current_dir, "outputs")
    }

if __name__ == "__main__":
    args = parse_args()
    default_paths = get_default_paths()
    
    # Use provided paths or defaults
    dicom_path = args.dicom_path or default_paths['dicom_path']
    output_dir = args.output_dir or default_paths['output_dir']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference
    run_inference(
        model_name=args.model,
        dicom_path=dicom_path,
        output_dir=output_dir,
        confidence_threshold=args.confidence_threshold,
        best_per_class=args.best_per_class
    ) 