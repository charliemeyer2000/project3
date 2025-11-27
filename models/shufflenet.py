"""ShuffleNetV2-based student models for knowledge distillation.

ShuffleNetV2 is recommended in the starter code (~5 MB), leaving room for experimentation.
"""

import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5


class ShuffleNetStudent(nn.Module):
    """ShuffleNetV2 student model for skin disease classification.
    
    Args:
        num_classes: Number of output classes (default: 10)
        width_mult: Width multiplier for channels (0.5, 1.0, 1.5, 2.0)
        pretrained: Whether to use ImageNet pretrained weights
        dropout: Dropout rate before final classifier
    """
    
    def __init__(self, 
                 num_classes: int = 10,
                 width_mult: float = 0.5,
                 pretrained: bool = False,
                 dropout: float = 0.2):
        super().__init__()
        
        self.num_classes = num_classes
        self.width_mult = width_mult
        
        # Select base model based on width multiplier
        if width_mult <= 0.5:
            self.backbone = shufflenet_v2_x0_5(pretrained=pretrained)
            self.feature_dim = 1024
        elif width_mult <= 1.0:
            self.backbone = shufflenet_v2_x1_0(pretrained=pretrained)
            self.feature_dim = 1024
        else:  # 1.5
            self.backbone = shufflenet_v2_x1_5(pretrained=pretrained)
            self.feature_dim = 1024
        
        # Replace final classifier
        self.backbone.fc = nn.Sequential(
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
        """Extract features before final classification layer.
        
        Useful for feature-based distillation.
        
        Args:
            x: Input images [B, 3, 224, 224]
            
        Returns:
            Features [B, feature_dim]
        """
        # ShuffleNetV2 structure: conv1 -> maxpool -> stage2 -> stage3 -> stage4 -> conv5
        x = self.backbone.conv1(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.stage2(x)
        x = self.backbone.stage3(x)
        x = self.backbone.stage4(x)
        x = self.backbone.conv5(x)
        x = x.mean([2, 3])  # Global average pooling
        return x


class CustomShuffleNetStudent(nn.Module):
    """Custom ShuffleNetV2 student with intermediate feature extraction.
    
    Supports multi-level feature distillation by exposing intermediate layers.
    
    Args:
        num_classes: Number of output classes (default: 10)
        width_mult: Width multiplier (0.5, 1.0, 1.5)
        pretrained: Whether to use ImageNet weights
        dropout: Dropout before classifier
        use_attention: Whether to add attention modules
    """
    
    def __init__(self,
                 num_classes: int = 10,
                 width_mult: float = 0.5,
                 pretrained: bool = False,
                 dropout: float = 0.2,
                 use_attention: bool = False):
        super().__init__()
        
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.use_attention = use_attention
        
        # Base model
        if width_mult <= 0.5:
            base = shufflenet_v2_x0_5(pretrained=pretrained)
        elif width_mult <= 1.0:
            base = shufflenet_v2_x1_0(pretrained=pretrained)
        else:
            base = shufflenet_v2_x1_5(pretrained=pretrained)
        
        # Extract backbone layers
        self.conv1 = base.conv1
        self.maxpool = base.maxpool
        self.stage2 = base.stage2
        self.stage3 = base.stage3
        self.stage4 = base.stage4
        self.conv5 = base.conv5
        
        self.feature_dim = 1024
        
        # Optional attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 4),
                nn.ReLU(inplace=False),
                nn.Linear(self.feature_dim // 4, self.feature_dim),
                nn.Sigmoid()
            )
        
        # Classifier
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.feature_dim, num_classes)
    
    def forward(self, x: torch.Tensor, return_features: bool = False):
        """Forward pass with optional feature extraction.
        
        Args:
            x: Input images [B, 3, 224, 224]
            return_features: Whether to return intermediate features
            
        Returns:
            If return_features=False: logits [B, num_classes]
            If return_features=True: (logits, features_dict)
        """
        features = {}
        
        # Stage 1
        x = self.conv1(x)
        x = self.maxpool(x)
        if return_features:
            features['stage1'] = x
        
        # Stage 2-4
        x = self.stage2(x)
        if return_features:
            features['stage2'] = x
        
        x = self.stage3(x)
        if return_features:
            features['stage3'] = x
        
        x = self.stage4(x)
        if return_features:
            features['stage4'] = x
        
        # Final conv and pooling
        x = self.conv5(x)
        x = x.mean([2, 3])  # Global average pooling
        
        if return_features:
            features['final'] = x
        
        # Optional attention
        if self.use_attention:
            attention_weights = self.attention(x)
            x = x * attention_weights
        
        # Classification
        x = self.dropout(x)
        logits = self.fc(x)
        
        if return_features:
            return logits, features
        return logits


def get_shufflenet_student(variant: str = 'base',
                           num_classes: int = 10,
                           width_mult: float = 0.5,
                           pretrained: bool = False,
                           **kwargs) -> nn.Module:
    """Factory function to create ShuffleNet student models.
    
    Args:
        variant: Model variant ('base', 'custom')
        num_classes: Number of classes
        width_mult: Width multiplier
        pretrained: Use ImageNet weights
        **kwargs: Additional model-specific arguments
        
    Returns:
        Student model
    """
    if variant == 'base':
        return ShuffleNetStudent(num_classes, width_mult, pretrained, **kwargs)
    elif variant == 'custom':
        return CustomShuffleNetStudent(num_classes, width_mult, pretrained, **kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")




