"""Data loading and preprocessing for skin disease classification."""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# Try to import kornia for GPU augmentation
try:
    import kornia.augmentation as K
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False


class GPUAugmentation(nn.Module):
    """GPU-accelerated augmentation using Kornia.
    
    Apply this to batched tensors already on GPU for faster augmentation.
    Works best with AMP (float16) for even better performance.
    
    Args:
        strength: 'none', 'light', 'medium', or 'strong'
    """
    
    def __init__(self, strength: str = 'light'):
        super().__init__()
        
        if not KORNIA_AVAILABLE:
            raise ImportError("Kornia not available. Install with: pip install kornia")
        
        self.strength = strength
        
        # ImageNet normalization
        self.normalize = K.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )
        
        if strength == 'none':
            self.augment = nn.Identity()
            
        elif strength == 'light':
            self.augment = K.AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.8),
                data_keys=["input"],
            )
            
        elif strength == 'medium':
            self.augment = K.AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomRotation(degrees=15, p=0.5),
                K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
                K.RandomAffine(degrees=0, translate=(0.1, 0.1), p=0.3),
                data_keys=["input"],
            )
            
        elif strength == 'strong':
            self.augment = K.AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomRotation(degrees=20, p=0.5),
                K.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
                K.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), p=0.5),
                K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.2),
                data_keys=["input"],
            )
        else:
            raise ValueError(f"Unknown augmentation strength: {strength}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to batch.
        
        Args:
            x: Batch of images [B, C, H, W], already normalized or not
            
        Returns:
            Augmented batch
        """
        if self.strength == 'none':
            return x
        return self.augment(x)


def get_gpu_augmentation(strength: str = 'light') -> Optional[nn.Module]:
    """Get GPU augmentation module if kornia is available.
    
    Args:
        strength: Augmentation strength
        
    Returns:
        GPUAugmentation module or None if kornia unavailable
    """
    if KORNIA_AVAILABLE:
        return GPUAugmentation(strength)
    return None


class SkinDiseaseDataset(Dataset):
    """Skin disease classification dataset.
    
    Args:
        root_dir: Path to dataset root (contains class folders)
        transform: Image transformations
        class_to_idx: Optional class name to index mapping
    """
    
    def __init__(self, root_dir: str, transform=None, class_to_idx: Optional[Dict[str, int]] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Get classes
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(self.root_dir / d)])
        
        # Create class mapping
        if class_to_idx is None:
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []
        
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif')
        
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(valid_exts):
                    self.image_paths.append(cls_dir / fname)
                    self.labels.append(self.class_to_idx[cls_name])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get count of samples per class."""
        counts = {}
        for cls_name in self.classes:
            counts[cls_name] = sum(1 for label in self.labels 
                                  if label == self.class_to_idx[cls_name])
        return counts
    
    def compute_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced data (inverse frequency)."""
        class_counts = np.bincount(self.labels)
        total = len(self.labels)
        n_classes = len(self.classes)
        
        # Inverse frequency weighting
        weights = total / (n_classes * class_counts)
        return torch.FloatTensor(weights)


def get_train_transform(img_size: int = 224, 
                        augmentation_strength: str = 'light') -> transforms.Compose:
    """Get training data augmentation transforms.
    
    Args:
        img_size: Target image size
        augmentation_strength: 'none', 'light', 'medium', or 'strong'
        
    Returns:
        Composed transforms
    """
    if augmentation_strength == 'none':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    elif augmentation_strength == 'light':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    elif augmentation_strength == 'medium':
        return transforms.Compose([
            transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                  saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    elif augmentation_strength == 'strong':
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                                  saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    else:
        raise ValueError(f"Unknown augmentation strength: {augmentation_strength}")


def get_val_transform(img_size: int = 224) -> transforms.Compose:
    """Get validation data transforms (no augmentation).
    
    Args:
        img_size: Target image size
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    train_split: float = 0.9,
    num_workers: int = 4,
    pin_memory: bool = True,
    augmentation_strength: str = 'light',
    use_class_weights: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create train and validation dataloaders with stratified split.
    
    Args:
        data_root: Path to dataset root
        batch_size: Batch size
        train_split: Fraction of data for training
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for faster GPU transfer
        augmentation_strength: Augmentation strength for training
        use_class_weights: Whether to use weighted sampling for training
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, info_dict)
    """
    # Create full dataset with validation transform to get labels
    full_dataset = SkinDiseaseDataset(
        data_root,
        transform=get_val_transform()
    )
    
    # Get labels and class info
    labels = full_dataset.labels
    class_counts = full_dataset.get_class_counts()
    
    # Stratified train/val split
    indices = np.arange(len(full_dataset))
    train_indices, val_indices = train_test_split(
        indices,
        train_size=train_split,
        stratify=labels,
        random_state=seed
    )
    
    # Create train dataset with augmentation
    train_dataset = SkinDiseaseDataset(
        data_root,
        transform=get_train_transform(augmentation_strength=augmentation_strength),
        class_to_idx=full_dataset.class_to_idx
    )
    
    # Create val dataset
    val_dataset = SkinDiseaseDataset(
        data_root,
        transform=get_val_transform(),
        class_to_idx=full_dataset.class_to_idx
    )
    
    # Create subset samplers
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Optional: Weighted sampling for class balance
    train_sampler = None
    if use_class_weights:
        # Compute sample weights
        train_labels = [labels[i] for i in train_indices]
        class_weights = full_dataset.compute_class_weights()
        sample_weights = [class_weights[label] for label in train_labels]
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    # Create dataloaders with performance optimizations
    # persistent_workers: Keep workers alive between epochs (faster)
    # prefetch_factor: How many batches to prefetch per worker
    use_persistent = num_workers > 0
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Shuffle if not using sampler
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # For stable batch norm
        persistent_workers=use_persistent,  # Keep workers alive between epochs
        prefetch_factor=4 if num_workers > 0 else None,  # Aggressive prefetching
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    
    # Info dict
    info = {
        'num_classes': len(full_dataset.classes),
        'classes': full_dataset.classes,
        'class_to_idx': full_dataset.class_to_idx,
        'class_counts': class_counts,
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'class_weights': full_dataset.compute_class_weights()
    }
    
    return train_loader, val_loader, info

