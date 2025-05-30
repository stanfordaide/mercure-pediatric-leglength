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
import pandas as pd

logger = logging.getLogger(__name__)

class LegLengthDetector:
    """Detector for leg length measurements using Faster R-CNN."""
    
    def __init__(self, backbone_name='resnext101_32x8d', num_classes=9, pretrained=True):
        """
        Initialize the leg length detector.
        
        Args:
            backbone_name (str): Name of the backbone model to use
            num_classes (int): Number of classes (including background)
            pretrained (bool): Whether to use pretrained weights
        """
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = self._create_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def _create_model(self):
        """Create Faster R-CNN model with specified backbone."""
        def _modify_classifier(model, num_classes):
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            return model
        
        # Available backbones directly from torchvision.models.detection
        if self.backbone_name == "resnet50":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=self.pretrained,
                pretrained_backbone=True
            )
            model = _modify_classifier(model, self.num_classes)
        elif self.backbone_name == "mobilenet_v3_large":
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                pretrained=self.pretrained,
                pretrained_backbone=True
            )
            model = _modify_classifier(model, self.num_classes)
        elif self.backbone_name == "mobilenet_v3_large_320":
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
                pretrained=self.pretrained,
                pretrained_backbone=True
            )
            model = _modify_classifier(model, self.num_classes)
        elif self.backbone_name == "resnet50_fpn_v2":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                pretrained=self.pretrained,
                pretrained_backbone=True
            )
            model = _modify_classifier(model, self.num_classes)
        
        # Custom backbones requiring manual integration
        elif self.backbone_name in ["resnet18", "resnet34", "resnet101", "resnet152"]:
            backbone_fn = getattr(torchvision.models, self.backbone_name)
            backbone_model = backbone_fn(pretrained=self.pretrained)
            backbone_layers = list(backbone_model.children())[:-2]
            backbone = torch.nn.Sequential(*backbone_layers)
            backbone.out_channels = 512 if self.backbone_name in ["resnet18", "resnet34"] else 2048
            
            model = self._create_custom_fasterrcnn(backbone)
            
        elif self.backbone_name == "mobilenet_v2":
            backbone = torchvision.models.mobilenet_v2(pretrained=self.pretrained).features
            backbone.out_channels = 1280
            model = self._create_custom_fasterrcnn(backbone)
            
        elif self.backbone_name == "efficientnet_b0":
            backbone = torchvision.models.efficientnet_b0(pretrained=self.pretrained).features
            backbone.out_channels = 1280
            model = self._create_custom_fasterrcnn(backbone)
            
        # ConvNeXt models
        elif self.backbone_name in ["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"]:
            backbone_fn = getattr(torchvision.models, self.backbone_name)
            backbone_model = backbone_fn(pretrained=self.pretrained)
            
            out_channels = 768  # default for tiny and small
            if self.backbone_name == "convnext_base":
                out_channels = 1024
            elif self.backbone_name == "convnext_large":
                out_channels = 1536
            
            backbone = torch.nn.Sequential(*list(backbone_model.children())[:-1])
            backbone.out_channels = out_channels
            model = self._create_custom_fasterrcnn(backbone)
            
        # ResNeXt models
        elif self.backbone_name in ["resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d"]:
            backbone_fn = getattr(torchvision.models, self.backbone_name)
            backbone_model = backbone_fn(pretrained=self.pretrained)
            backbone_layers = list(backbone_model.children())[:-2]
            backbone = torch.nn.Sequential(*backbone_layers)
            backbone.out_channels = 2048
            model = self._create_custom_fasterrcnn(backbone)
            
        # DenseNet models
        elif self.backbone_name in ["densenet121", "densenet169", "densenet201"]:
            backbone_fn = getattr(torchvision.models, self.backbone_name)
            backbone_model = backbone_fn(pretrained=self.pretrained)
            
            out_channels = 1024  # default for densenet121
            if self.backbone_name == "densenet169":
                out_channels = 1664
            elif self.backbone_name == "densenet201":
                out_channels = 1920
            
            backbone = torch.nn.Sequential(*list(backbone_model.children())[:-1])
            backbone.out_channels = out_channels
            model = self._create_custom_fasterrcnn(backbone)
            
        else:
            raise ValueError(
                f"Backbone '{self.backbone_name}' not supported. Choose from: "
                "resnet50, mobilenet_v3_large, mobilenet_v3_large_320, resnet50_fpn_v2, "
                "resnet18, resnet34, resnet101, resnet152, mobilenet_v2, efficientnet_b0, "
                "convnext_tiny, convnext_small, convnext_base, convnext_large, "
                "resnext50_32x4d, resnext101_32x8d, resnext101_64x4d, "
                "densenet121, densenet169, densenet201"
            )
        
        return model
    
    def _create_custom_fasterrcnn(self, backbone):
        """Helper method to create FasterRCNN with custom backbone."""
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        model = FasterRCNN(
            backbone=backbone,
            num_classes=self.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
        
        return model
    
    @staticmethod
    def get_model_path(backbone_name: str) -> str:
        """Get the path to the model checkpoint for a given backbone."""
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load registry
        registry_path = os.path.join(current_dir, 'registry.json')
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        if backbone_name not in registry:
            raise ValueError(f"Backbone '{backbone_name}' not found in registry. Available backbones: {list(registry.keys())}")
        
        # Get the model path
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
        model = LegLengthDetector(backbone_name=backbone_name, num_classes=num_classes, pretrained=True)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.model.to(device)
        
        logger.info(f"Loaded checkpoint from {model_path}")
        return model
    
    @torch.no_grad()
    def predict(self, image: torch.Tensor, confidence_threshold: float = 0.5, best_per_class: bool = True) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image: Input image tensor
            confidence_threshold: Minimum confidence threshold for predictions
            best_per_class: If True, returns only the highest confidence prediction for each class
            
        Returns:
            Dictionary containing predictions with keys:
            - 'boxes': numpy array of bounding boxes [x1, y1, x2, y2]
            - 'scores': numpy array of confidence scores
            - 'labels': numpy array of class labels
        """
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
            # Get unique classes
            unique_labels = np.unique(filtered_preds['labels'])
            best_indices = []
            
            # For each class, find the prediction with highest confidence
            for label in unique_labels:
                class_mask = filtered_preds['labels'] == label
                if np.any(class_mask):
                    class_scores = filtered_preds['scores'][class_mask]
                    best_idx = np.argmax(class_scores)
                    # Get the original index in the filtered predictions
                    original_indices = np.where(class_mask)[0]
                    best_indices.append(original_indices[best_idx])
            
            # Keep only the best predictions
            filtered_preds = {
                'boxes': filtered_preds['boxes'][best_indices],
                'scores': filtered_preds['scores'][best_indices],
                'labels': filtered_preds['labels'][best_indices]
            }
        
        return filtered_preds 