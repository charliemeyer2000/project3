"""EfficientNet-based student models for knowledge distillation."""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetB0Student(nn.Module):
    """EfficientNet-B0 student model.
    
    EfficientNet-B0 has ~5.3M parameters (~21MB), just under the 25MB limit.
    Provides better accuracy than ShuffleNet/MobileNet at the cost of size.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet weights
        dropout: Dropout before classifier
    """
    
    def __init__(self,
                 num_classes: int = 10,
                 pretrained: bool = False,
                 dropout: float = 0.2):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Create base model
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        
        # Get feature dimension (1280 for EfficientNet-B0)
        self.feature_dim = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
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
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def get_efficientnet_student(variant: str = 'b0',
                             num_classes: int = 10,
                             pretrained: bool = False,
                             **kwargs) -> nn.Module:
    """Factory function for EfficientNet students.
    
    Args:
        variant: 'b0' (only b0 fits under 25MB)
        num_classes: Number of classes
        pretrained: Use ImageNet weights
        **kwargs: Additional arguments
        
    Returns:
        Student model
    """
    if variant == 'b0':
        return EfficientNetB0Student(num_classes, pretrained, **kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}. Only 'b0' is supported (size constraint).")

