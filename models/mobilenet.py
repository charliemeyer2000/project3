"""MobileNet-based student models for knowledge distillation."""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large


class MobileNetV2Student(nn.Module):
    """MobileNetV2 student model.
    
    Args:
        num_classes: Number of output classes
        width_mult: Width multiplier for channels
        pretrained: Use ImageNet weights
        dropout: Dropout before classifier
    """
    
    def __init__(self,
                 num_classes: int = 10,
                 width_mult: float = 1.0,
                 pretrained: bool = False,
                 dropout: float = 0.2):
        super().__init__()
        
        self.num_classes = num_classes
        self.width_mult = width_mult
        
        # Create base model
        self.backbone = mobilenet_v2(pretrained=pretrained, width_mult=width_mult)
        
        # Get feature dimension (last_channel)
        self.feature_dim = self.backbone.last_channel
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input images [B, 3, 224, 224]
            
        Returns:
            Logits [B, num_classes]
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification."""
        x = self.backbone.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x


class MobileNetV3Student(nn.Module):
    """MobileNetV3 student model.
    
    Args:
        num_classes: Number of output classes
        variant: 'small' or 'large'
        pretrained: Use ImageNet weights
        dropout: Dropout before classifier
    """
    
    def __init__(self,
                 num_classes: int = 10,
                 variant: str = 'small',
                 pretrained: bool = False,
                 dropout: float = 0.2):
        super().__init__()
        
        self.num_classes = num_classes
        self.variant = variant
        
        # Create base model
        if variant == 'small':
            self.backbone = mobilenet_v3_small(pretrained=pretrained)
            self.feature_dim = 576
        else:  # large
            self.backbone = mobilenet_v3_large(pretrained=pretrained)
            self.feature_dim = 960
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True) if dropout > 0 else nn.Identity(),
            nn.Linear(1280, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def get_mobilenet_student(variant: str = 'v2',
                          num_classes: int = 10,
                          width_mult: float = 1.0,
                          pretrained: bool = False,
                          **kwargs) -> nn.Module:
    """Factory function for MobileNet students.
    
    Args:
        variant: 'v2', 'v3_small', or 'v3_large'
        num_classes: Number of classes
        width_mult: Width multiplier (MobileNetV2 only)
        pretrained: Use ImageNet weights
        **kwargs: Additional arguments
        
    Returns:
        Student model
    """
    if variant == 'v2':
        return MobileNetV2Student(num_classes, width_mult, pretrained, **kwargs)
    elif variant == 'v3_small':
        return MobileNetV3Student(num_classes, 'small', pretrained, **kwargs)
    elif variant == 'v3_large':
        return MobileNetV3Student(num_classes, 'large', pretrained, **kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")


