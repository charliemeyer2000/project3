"""Test-Time Augmentation (TTA) wrapper for student models.

TTA improves predictions by averaging results over augmented versions of the input.
This wrapper is designed to be TorchScript-compatible for server submission.

Reference: Cino et al., J. Imaging 2025 - "Skin Lesion Classification with TTA"
showed that TTA (flips/rotations) improved balanced accuracy on ISIC 2019.
"""

import torch
import torch.nn as nn
from typing import List


class TTAWrapper(nn.Module):
    """Wraps a model to apply test-time augmentation.
    
    Applies horizontal flip and averages predictions.
    This gives a small but consistent boost to accuracy.
    
    Args:
        model: The student model to wrap
        use_hflip: Apply horizontal flip (default: True)
        use_vflip: Apply vertical flip (default: False)
    
    Example:
        >>> student = get_student_model('efficientnet_b0', 10)
        >>> tta_model = TTAWrapper(student, use_hflip=True)
        >>> # Export for submission
        >>> scripted = torch.jit.script(tta_model)
    """
    
    def __init__(self, 
                 model: nn.Module,
                 use_hflip: bool = True,
                 use_vflip: bool = False):
        super().__init__()
        self.model = model
        self.use_hflip = use_hflip
        self.use_vflip = use_vflip
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with TTA.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Averaged logits [B, num_classes]
        """
        # Original prediction
        logits = self.model(x)
        count = 1.0
        
        # Horizontal flip
        if self.use_hflip:
            x_hflip = torch.flip(x, dims=[3])  # Flip width dimension
            logits = logits + self.model(x_hflip)
            count += 1.0
        
        # Vertical flip (less common for skin lesions, but can help)
        if self.use_vflip:
            x_vflip = torch.flip(x, dims=[2])  # Flip height dimension
            logits = logits + self.model(x_vflip)
            count += 1.0
            
            # Also do both flips if both are enabled
            if self.use_hflip:
                x_both = torch.flip(x, dims=[2, 3])
                logits = logits + self.model(x_both)
                count += 1.0
        
        # Average
        return logits / count


class MultiHeadTTAWrapper(nn.Module):
    """TTA wrapper with multiple classifier heads for ensemble effect.
    
    Combines TTA with multi-head ensemble for maximum benefit.
    Each head is trained with different dropout seeds/initialization.
    
    Args:
        model: Base model (without classifier)
        heads: List of classifier heads
        use_hflip: Apply horizontal flip
    """
    
    def __init__(self,
                 model: nn.Module,
                 heads: nn.ModuleList,
                 use_hflip: bool = True):
        super().__init__()
        self.model = model
        self.heads = heads
        self.use_hflip = use_hflip
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with multi-head ensemble + TTA.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Averaged logits [B, num_classes]
        """
        def get_ensemble_logits(features: torch.Tensor) -> torch.Tensor:
            """Average predictions from all heads."""
            logits_sum = self.heads[0](features)
            for head in self.heads[1:]:
                logits_sum = logits_sum + head(features)
            return logits_sum / float(len(self.heads))
        
        # Get features (assuming model.get_features exists)
        if hasattr(self.model, 'get_features'):
            features = self.model.get_features(x)
        else:
            features = self.model(x)
        
        logits = get_ensemble_logits(features)
        count = 1.0
        
        # Horizontal flip
        if self.use_hflip:
            x_hflip = torch.flip(x, dims=[3])
            if hasattr(self.model, 'get_features'):
                features_hflip = self.model.get_features(x_hflip)
            else:
                features_hflip = self.model(x_hflip)
            logits = logits + get_ensemble_logits(features_hflip)
            count += 1.0
        
        return logits / count


def wrap_model_with_tta(model: nn.Module, 
                        use_hflip: bool = True,
                        use_vflip: bool = False) -> nn.Module:
    """Factory function to wrap a model with TTA.
    
    Args:
        model: Student model to wrap
        use_hflip: Use horizontal flip (recommended)
        use_vflip: Use vertical flip (optional)
        
    Returns:
        TTA-wrapped model
    """
    return TTAWrapper(model, use_hflip=use_hflip, use_vflip=use_vflip)



