import json
import os
import time
import requests
import torch
import torchvision.models
from pathlib import Path
from tqdm import tqdm

def download_backbone_weights(model_name):
    """Pre-download torchvision backbone weights for a specific model."""
    try:
        print(f"🏗️  Pre-downloading backbone weights for: {model_name}")
        
        # Map model names to torchvision functions
        backbone_map = {
            # Models from registry_full.json
            "resnet101": lambda: torchvision.models.resnet101(weights='DEFAULT'),
            "resnext101_32x8d": lambda: torchvision.models.resnext101_32x8d(weights='DEFAULT'),
            "densenet201": lambda: torchvision.models.densenet201(weights='DEFAULT'),
            "vit_l_16": lambda: torchvision.models.vit_l_16(weights='DEFAULT'),
            "efficientnet_v2_m": lambda: torchvision.models.efficientnet_v2_m(weights='DEFAULT'),
            "mobilenet_v3_large": lambda: torchvision.models.mobilenet_v3_large(weights='DEFAULT'),
            "swin_v2_b": lambda: torchvision.models.swin_v2_b(weights='DEFAULT'),
            "convnext_base": lambda: torchvision.models.convnext_base(weights='DEFAULT'),
            
            # Keep some useful extras for development
            "resnet50": lambda: torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT'),
            "mobilenet_v3_large_320": lambda: torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights='DEFAULT'),
            "resnet50_fpn_v2": lambda: torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
        }
        
        if model_name in backbone_map:
            # This will trigger the download and cache the weights
            model = backbone_map[model_name]()
            print(f"✅ Cached backbone weights for: {model_name}")
            return True
        else:
            print(f"⚠️  Unknown backbone: {model_name}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to cache backbone for {model_name}: {e}")
        return False

def download_with_progress(url, output_path, max_retries=3, delay=5):
    """Download file with progress bar and retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1}/{max_retries}: Downloading from {url}")
            
            # Get file size first
            response = requests.head(url, allow_redirects=True)
            total_size = int(response.headers.get('content-length', 0))
            
            if total_size > 0:
                print(f"File size: {total_size / (1024*1024*1024):.2f} GB")
            
            # Download with progress bar
            response = requests.get(url, stream=True, allow_redirects=True)
            response.raise_for_status()
            
            # Create progress bar
            progress_bar = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {output_path.name}"
            )
            
            with open(output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            
            progress_bar.close()
            print(f"✅ Download completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Download failed (attempt {attempt + 1}): {e}")
            
            # Remove partial file if it exists
            if output_path.exists():
                output_path.unlink()
                print(f"Removed partial file: {output_path}")
            
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"💥 Failed to download after {max_retries} attempts")
                return False
    return False

def download_models():
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load registry from project root (same directory as download_models.py)
    registry_path = os.path.join(current_dir, 'registry.json')
    
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    except FileNotFoundError:
        print(f"❌ Registry file not found: {registry_path}")
        print("Please ensure registry.json exists in the project root with model URLs")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in registry file: {e}")
        return
    
    # Create models directory in project root
    models_dir = Path(os.path.join(current_dir, 'models'))
    models_dir.mkdir(exist_ok=True)
    print(f"📁 Models directory: {models_dir}")
    
    # Set torch cache directory to be persistent
    cache_dir = Path(os.path.join(current_dir, '.cache', 'torch'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['TORCH_HOME'] = str(cache_dir.parent)
    print(f"🏗️  Torch cache directory: {cache_dir}")
    
    # Step 1: Pre-download backbone weights for models in registry
    print(f"\n{'='*60}")
    print(f"🏗️  STEP 1: PRE-DOWNLOADING BACKBONE WEIGHTS")
    print(f"{'='*60}")
    
    backbone_success = 0
    backbone_total = 0
    
    for model_name in registry.keys():
        if 'example.com' not in registry[model_name]:  # Skip placeholder URLs
            backbone_total += 1
            if download_backbone_weights(model_name):
                backbone_success += 1
    
    print(f"\n🏗️  Backbone weights cached: {backbone_success}/{backbone_total}")
    
    # Step 2: Download custom model checkpoints
    print(f"\n{'='*60}")
    print(f"📦 STEP 2: DOWNLOADING CUSTOM MODEL CHECKPOINTS")
    print(f"{'='*60}")
    
    # Check for placeholder URLs
    placeholder_urls = [url for url in registry.values() if 'example.com' in url]
    if placeholder_urls:
        print(f"⚠️  Found {len(placeholder_urls)} placeholder URLs in registry.json")
        print("Please replace example.com URLs with actual model download links")
        print("Skipping download of placeholder models...")
    
    # Download each model
    total_models = len(registry)
    successful_downloads = 0
    failed_downloads = 0
    
    print(f"\n🚀 Starting download of {total_models} custom models...")
    
    for i, (model_name, url) in enumerate(registry.items(), 1):
        print(f"\n{'='*60}")
        print(f"📦 Model {i}/{total_models}: {model_name}")
        print(f"{'='*60}")
        
        output_path = models_dir / f"{model_name}.pth"
        
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024*1024)
            print(f"✅ Model already exists: {output_path} ({file_size:.1f} MB)")
            successful_downloads += 1
            continue
        
        # Skip placeholder URLs
        if 'example.com' in url:
            print(f"⏭️  Skipping placeholder URL: {url}")
            continue
        
        success = download_with_progress(url, output_path)
        if success:
            file_size = output_path.stat().st_size / (1024*1024)
            print(f"✅ Successfully downloaded: {output_path} ({file_size:.1f} MB)")
            successful_downloads += 1
        else:
            print(f"❌ Failed to download: {model_name}")
            failed_downloads += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"🏗️  Backbone weights cached: {backbone_success}/{backbone_total}")
    print(f"✅ Custom models downloaded: {successful_downloads}")
    print(f"❌ Failed downloads: {failed_downloads}")
    print(f"📁 Models directory: {models_dir}")
    print(f"🏗️  Cache directory: {cache_dir}")
    
    if failed_downloads > 0:
        print(f"\n⚠️  Some downloads failed. You can re-run this script to retry failed downloads.")

if __name__ == "__main__":
    download_models() 