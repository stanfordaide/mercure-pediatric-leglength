#!/usr/bin/env python
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
import logging
import os
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch.nn as nn

logger = logging.getLogger(__name__)

class LegLengthDetector:
    """Detector for leg length measurements using Faster R-CNN."""
    
    def __init__(self, backbone_name='resnext101_32x8d', num_classes=9, weights='DEFAULT'):
        """
        Initialize the leg length detector.
        
        Args:
            backbone_name: Name of the backbone model to use
            num_classes: Number of classes to detect (default: 9 for 8 landmarks + background)
            weights: Pre-trained weights to use ('DEFAULT', 'IMAGENET1K_V1', or None)
        """
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.weights = weights
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self._create_model()
        self.model.to(self.device)
    
    def _create_model(self):
        """Create Faster R-CNN model with specified backbone."""
        
        # Common anchor generator and ROI pooler for all models
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # ResNet and ResNeXt models
        if self.backbone_name in ["resnet101", "resnext101_32x8d"]:
            backbone_fn = getattr(torchvision.models, self.backbone_name)
            backbone_model = backbone_fn(weights=self.weights)
            backbone_layers = list(backbone_model.children())[:-2]
            backbone = torch.nn.Sequential(*backbone_layers)
            backbone.out_channels = 2048
            
            model = FasterRCNN(
                backbone=backbone,
                num_classes=self.num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler
            )
        
        # DenseNet models
        elif self.backbone_name == "densenet201":
            backbone_model = torchvision.models.densenet201(weights=self.weights)
            backbone = torch.nn.Sequential(*list(backbone_model.children())[:-1])
            backbone.out_channels = 1920
            
            model = FasterRCNN(
                backbone=backbone,
                num_classes=self.num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler
            )
        
        # Vision Transformer
        elif self.backbone_name == "vit_l_16":
            backbone_model = torchvision.models.vit_l_16(weights=self.weights)
            
            class ViTBackbone(nn.Module):
                def __init__(self, vit_model, out_channels=1024, patch_size=16):
                    super().__init__()
                    self.conv_proj = vit_model.conv_proj
                    self.encoder = vit_model.encoder
                    self.out_channels = out_channels
                    self.patch_size = patch_size
                    
                    # Store original positional embedding
                    self.original_pos_embed = vit_model.encoder.pos_embedding
                    num_patches = self.original_pos_embed.shape[1] - 1
                    self.original_grid_size = int(num_patches ** 0.5)
                
                def interpolate_pos_encoding(self, pos_embed, H, W):
                    N = pos_embed.shape[1] - 1
                    if N == H * W:
                        return pos_embed
                    
                    class_pos_embed = pos_embed[:, :1]
                    patch_pos_embed = pos_embed[:, 1:]
                    
                    patch_pos_embed = patch_pos_embed.reshape(1, self.original_grid_size, self.original_grid_size, -1)
                    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
                    
                    patch_pos_embed = torch.nn.functional.interpolate(
                        patch_pos_embed, size=(H, W), mode='bicubic', align_corners=False
                    )
                    
                    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, H * W, -1)
                    return torch.cat([class_pos_embed, patch_pos_embed], dim=1)
                
                def forward(self, x):
                    x = self.conv_proj(x)
                    B, C, H, W = x.shape
                    
                    x = x.flatten(2).transpose(1, 2)
                    class_token = self.original_pos_embed[:, :1].expand(B, -1, -1)
                    x = torch.cat([class_token, x], dim=1)
                    
                    pos_embed = self.interpolate_pos_encoding(self.original_pos_embed, H, W)
                    x = x + pos_embed
                    x = self.encoder.dropout(x)
                    
                    for layer in self.encoder.layers:
                        x = layer(x)
                    
                    x = x[:, 1:]
                    x = x.transpose(1, 2).view(B, C, H, W)
                    return {'0': x}
            
            backbone = ViTBackbone(backbone_model)
            
            model = FasterRCNN(
                backbone=backbone,
                num_classes=self.num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler
            )
        
        # EfficientNet V2
        elif self.backbone_name == "efficientnet_v2_m":
            backbone_model = torchvision.models.efficientnet_v2_m(weights=self.weights)
            backbone = backbone_model.features
            backbone.out_channels = 1280
            
            model = FasterRCNN(
                backbone=backbone,
                num_classes=self.num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler
            )
        
        # MobileNet V3
        elif self.backbone_name == "mobilenet_v3_large":
            backbone = torchvision.models.mobilenet_v3_large(weights=self.weights).features
            backbone.out_channels = 960
            
            model = FasterRCNN(
                backbone=backbone,
                num_classes=self.num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler
            )
        
        # Swin Transformer V2
        elif self.backbone_name == "swin_v2_b":
            backbone_model = torchvision.models.swin_v2_b(weights=self.weights)
            
            class SwinBackbone(nn.Module):
                def __init__(self, swin_model, out_channels=1024):
                    super().__init__()
                    self.features = swin_model.features
                    self.norm = swin_model.norm
                    self.permute = swin_model.permute
                    self.out_channels = out_channels
                
                def forward(self, x):
                    x = self.features(x)
                    x = self.norm(x)
                    x = self.permute(x)
                    return {'0': x}
            
            backbone = SwinBackbone(backbone_model)
            
            model = FasterRCNN(
                backbone=backbone,
                num_classes=self.num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler
            )
        
        # ConvNeXt
        elif self.backbone_name == "convnext_base":
            backbone_model = torchvision.models.convnext_base(weights=self.weights)
            backbone = torch.nn.Sequential(*list(backbone_model.children())[:-2])
            backbone.out_channels = 1024
            
            model = FasterRCNN(
                backbone=backbone,
                num_classes=self.num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler
            )
        
        else:
            raise ValueError(f"Backbone '{self.backbone_name}' not found in registry. Available backbones: {list(self._get_registry().keys())}")
        
        return model
    
    @staticmethod
    def _get_registry() -> dict:
        """Get the model registry from registry.json."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Look for registry.json in the project root (parent directory of leglength module)
        registry_path = os.path.join(os.path.dirname(current_dir), 'registry.json')
        with open(registry_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def get_model_path(backbone_name: str) -> str:
        """Get the path to the model checkpoint for a given backbone."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load registry from project root
        registry_path = os.path.join(os.path.dirname(current_dir), 'registry.json')
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        if backbone_name not in registry:
            raise ValueError(f"Backbone '{backbone_name}' not found in registry. Available backbones: {list(registry.keys())}")
        
        model_path = os.path.join(os.path.dirname(current_dir), 'models', f"{backbone_name}.pth")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Please ensure the model has been downloaded.")
        
        return model_path
    
    @staticmethod
    def load_checkpoint(backbone_name: str, device: Optional[torch.device] = None) -> 'LegLengthDetector':
        """Load model from checkpoint."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get model path from registry
        model_path = LegLengthDetector.get_model_path(backbone_name)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract metadata from checkpoint
        if 'metadata' in checkpoint:
            metadata = checkpoint['metadata']
            num_classes = metadata.get('num_classes', 9)
            logger.info(f"Using metadata from checkpoint - num_classes: {num_classes}")
        else:
            num_classes = 9
            logger.info(f"No metadata found in checkpoint, using default num_classes: {num_classes}")
        
        # Create model with specified backbone
        model = LegLengthDetector(backbone_name=backbone_name, num_classes=num_classes, weights='DEFAULT')
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.model.to(device)
        
        logger.info(f"Loaded checkpoint from {model_path}")
        return model
    
    @torch.no_grad()
    def predict(self, image: torch.Tensor, confidence_threshold: float = 0.0, best_per_class: bool = True) -> Dict:
        """Run inference on a single image."""
        self.model.eval()
        image = image.to(self.device)
        
        # Get predictions
        predictions = self.model([image])[0]
        
        # Filter predictions by confidence
        keep = predictions['scores'] > confidence_threshold
        filtered_preds = {
            'boxes': predictions['boxes'][keep].cpu().numpy(),
            'scores': predictions['scores'][keep].cpu().numpy(),
            'labels': predictions['labels'][keep].cpu().numpy()
        }
        
        if best_per_class:
            logger.info(f"Best per class set to TRUE, getting best predictions")
            unique_labels = np.unique(filtered_preds['labels'])
            best_indices = []
            
            for label in unique_labels:
                class_mask = filtered_preds['labels'] == label
                if np.any(class_mask):
                    class_scores = filtered_preds['scores'][class_mask]
                    best_idx = np.argmax(class_scores)
                    original_indices = np.where(class_mask)[0]
                    best_indices.append(original_indices[best_idx])
            
            filtered_preds = {
                'boxes': filtered_preds['boxes'][best_indices],
                'scores': filtered_preds['scores'][best_indices],
                'labels': filtered_preds['labels'][best_indices]
            }
        
        return filtered_preds