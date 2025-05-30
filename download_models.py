import json
import os
import wget
from pathlib import Path

def download_models():
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load registry from leglength package
    registry_path = os.path.join(current_dir, 'leglength', 'registry.json')
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    # Create models directory in project root
    models_dir = Path(os.path.join(current_dir, 'models'))
    models_dir.mkdir(exist_ok=True)
    
    # Download each model
    for model_name, url in registry.items():
        output_path = models_dir / f"{model_name}.pth"
        if not output_path.exists():
            print(f"Downloading {model_name} model...")
            wget.download(url, str(output_path))
            print(f"\nDownloaded {model_name} model to {output_path}")
        else:
            print(f"Model {model_name} already exists at {output_path}")

if __name__ == "__main__":
    download_models() 