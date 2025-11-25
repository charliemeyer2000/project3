"""Student model architectures for knowledge distillation."""

import torch.nn as nn
from typing import Dict, Any

from .shufflenet import get_shufflenet_student, ShuffleNetStudent, CustomShuffleNetStudent
from .mobilenet import get_mobilenet_student, MobileNetV2Student, MobileNetV3Student
from .efficientnet import get_efficientnet_student, EfficientNetB0Student


def get_student_model(architecture: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """Factory function to create student models.
    
    Args:
        architecture: Model architecture name
            - 'shufflenet': ShuffleNetV2 (recommended, ~1.5MB)
            - 'shufflenet_custom': ShuffleNetV2 with feature extraction
            - 'mobilenet_v2': MobileNetV2 (~14MB)
            - 'mobilenet_v3_small': MobileNetV3-Small (~10MB)
            - 'mobilenet_v3_large': MobileNetV3-Large
            - 'efficientnet_b0': EfficientNet-B0 (~21MB, close to limit)
        num_classes: Number of output classes (default: 10)
        **kwargs: Architecture-specific arguments
        
    Returns:
        Student model instance
        
    Examples:
        >>> # ShuffleNetV2 with width multiplier 0.5 (recommended)
        >>> model = get_student_model('shufflenet', num_classes=10, width_mult=0.5)
        
        >>> # MobileNetV2 with width multiplier 0.75
        >>> model = get_student_model('mobilenet_v2', num_classes=10, width_mult=0.75)
        
        >>> # EfficientNet-B0 (highest capacity under 25MB)
        >>> model = get_student_model('efficientnet_b0', num_classes=10)
    """
    # ShuffleNet variants
    if architecture == 'shufflenet':
        return get_shufflenet_student('base', num_classes, **kwargs)
    elif architecture == 'shufflenet_custom':
        return get_shufflenet_student('custom', num_classes, **kwargs)
    
    # MobileNet variants
    elif architecture == 'mobilenet_v2':
        return get_mobilenet_student('v2', num_classes, **kwargs)
    elif architecture == 'mobilenet_v3_small':
        return get_mobilenet_student('v3_small', num_classes, **kwargs)
    elif architecture == 'mobilenet_v3_large':
        return get_mobilenet_student('v3_large', num_classes, **kwargs)
    
    # EfficientNet variants
    elif architecture == 'efficientnet_b0':
        return get_efficientnet_student('b0', num_classes, **kwargs)
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_size_mb(model: nn.Module) -> float:
    """Estimate model size in MB (assumes float32)."""
    return count_parameters(model) * 4 / (1024 ** 2)


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Get comprehensive model information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model info (params, size, architecture)
    """
    num_params = count_parameters(model)
    size_mb = estimate_model_size_mb(model)
    
    return {
        'architecture': model.__class__.__name__,
        'num_parameters': num_params,
        'estimated_size_mb': size_mb,
        'size_ok': size_mb < 25.0  # HW3 requirement
    }


__all__ = [
    'get_student_model',
    'get_shufflenet_student',
    'get_mobilenet_student',
    'get_efficientnet_student',
    'ShuffleNetStudent',
    'CustomShuffleNetStudent',
    'MobileNetV2Student',
    'MobileNetV3Student',
    'EfficientNetB0Student',
    'count_parameters',
    'estimate_model_size_mb',
    'get_model_info',
]

