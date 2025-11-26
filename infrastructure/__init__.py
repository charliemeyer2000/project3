"""Infrastructure for knowledge distillation project."""

from .data import (
    SkinDiseaseDataset,
    get_train_transform,
    get_val_transform,
    create_dataloaders
)

from .distillation import (
    DistillationLoss,
    FeatureDistillationLoss,
    HybridDistillationLoss,
    get_distillation_loss
)

from .training import (
    Trainer,
    train_with_distillation,
    save_torchscript
)

from .database import ExperimentDatabase

from .server import ServerAPI

__all__ = [
    # Data
    'SkinDiseaseDataset',
    'get_train_transform',
    'get_val_transform',
    'create_dataloaders',
    
    # Distillation
    'DistillationLoss',
    'FeatureDistillationLoss',
    'HybridDistillationLoss',
    'get_distillation_loss',
    
    # Training
    'Trainer',
    'train_with_distillation',
    'save_torchscript',
    
    # Database
    'ExperimentDatabase',
    
    # Server
    'ServerAPI',
]


