"""Train a linear classifier head on top of frozen MedSigLIP features.

This script trains a linear probe on the MedSigLIP teacher's features,
which gives us meaningful soft targets for knowledge distillation.
The randomly initialized head was producing noise - this fixes that.

Usage:
    modal run train_teacher_head.py
    
After running, the trained head will be saved to Modal volume at:
    /data/teacher_head/medsiglip_classifier.pth

Reference: SimKD (Chen et al., CVPR 2022) - reusing the trained teacher classifier
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import modal

# Create Modal app
app = modal.App("project3-train-teacher-head")

# Use same volume as main training
volume = modal.Volume.from_name("project3-kd-data", create_if_missing=True)

# Docker image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "huggingface-hub>=0.16.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
    )
)


@app.function(
    gpu="H100",
    image=image,
    volumes={"/data": volume},
    timeout=7200,  # 2 hours max
    memory=65536,
    cpu=16,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_teacher_classifier(
    epochs: int = 100,  # More epochs for better convergence
    batch_size: int = 64,
    learning_rate: float = 0.001,  # Lower LR for stability
    weight_decay: float = 0.0001,
    use_class_weights: bool = True,
    train_split: float = 0.9,
    use_mlp: bool = True,  # Use MLP head instead of linear
):
    """Train a linear classifier on MedSigLIP features.
    
    The process:
    1. Load frozen MedSigLIP vision encoder
    2. Extract features for all training images
    3. Train a linear classifier (logistic regression) on those features
    4. Save the classifier weights
    
    This gives us a teacher that actually knows the task, instead of random noise.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
    from torchvision import transforms
    from PIL import Image
    from tqdm import tqdm
    from sklearn.metrics import f1_score, accuracy_score
    from huggingface_hub import login
    from transformers import AutoModel, AutoProcessor
    import numpy as np
    from collections import Counter
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"Training Teacher Classifier Head (Linear Probe)")
    print(f"{'='*80}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Enable optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # Login to HuggingFace
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    print("Logging in to HuggingFace...")
    login(token=HF_TOKEN)
    print("✓ HuggingFace login successful\n")
    
    # Load teacher model AND processor (correct preprocessing!)
    print("Loading MedSigLIP teacher model and processor...")
    teacher = AutoModel.from_pretrained(
        "google/medsiglip-448",
        trust_remote_code=True,
        token=HF_TOKEN
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        "google/medsiglip-448",
        trust_remote_code=True,
        token=HF_TOKEN
    )
    teacher.eval()
    
    # Freeze all parameters
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Get feature dimension
    with torch.no_grad():
        sample = torch.randn(1, 3, 448, 448).to(device)
        features = teacher.vision_model(sample).pooler_output
        feature_dim = features.shape[1]
    
    print(f"✓ Teacher loaded with AutoProcessor:")
    print(f"  Model: google/medsiglip-448")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Using HuggingFace processor for correct normalization!")
    print()
    
    # Simple dataset class using HuggingFace processor for correct preprocessing
    class SkinDataset(Dataset):
        def __init__(self, root_dir, processor):
            self.root_dir = Path(root_dir)
            self.processor = processor
            self.samples = []
            self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
            
            for class_name in self.class_names:
                class_dir = self.root_dir / class_name
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            # Use the HuggingFace processor for correct preprocessing!
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)  # Remove batch dim
            return pixel_values, label
    
    data_root = "/data/data/training_dataset/train_dataset"
    full_dataset = SkinDataset(data_root, processor)
    
    # Split into train/val
    n_total = len(full_dataset)
    n_train = int(n_total * train_split)
    n_val = n_total - n_train
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"✓ Dataset loaded:")
    print(f"  Total: {n_total} images")
    print(f"  Train: {n_train} images")
    print(f"  Val: {n_val} images")
    print(f"  Classes: {len(full_dataset.class_names)}")
    print()
    
    # Compute class weights if needed
    class_weights = None
    sampler = None
    if use_class_weights:
        # Get labels for training subset
        train_labels = [full_dataset.samples[i][1] for i in train_dataset.indices]
        class_counts = Counter(train_labels)
        total = sum(class_counts.values())
        
        # Inverse frequency weights
        weights = torch.tensor([total / (len(class_counts) * class_counts[i]) 
                               for i in range(len(full_dataset.class_names))]).to(device)
        class_weights = weights
        
        # Weighted sampler for balanced batches
        sample_weights = [1.0 / class_counts[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
        print(f"✓ Class weights computed:")
        for i, (name, weight) in enumerate(zip(full_dataset.class_names, weights)):
            print(f"  Class {i} ({name}): {weight:.3f}")
        print()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # Create classifier head (MLP for better expressiveness)
    if use_mlp:
        classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        ).to(device)
        print(f"✓ MLP Classifier created: {feature_dim} -> 512 -> 256 -> 10")
    else:
        classifier = nn.Linear(feature_dim, 10).to(device)
        print(f"✓ Linear Classifier created: {feature_dim} -> 10")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Training loop
    print(f"\n{'='*80}")
    print(f"Starting training for {epochs} epochs...")
    print(f"{'='*80}\n")
    
    best_f1 = 0.0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        # Train
        classifier.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Extract features (no grad, frozen teacher)
            with torch.no_grad():
                features = teacher.vision_model(images).pooler_output
            
            # Forward through classifier
            optimizer.zero_grad()
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Evaluate
        classifier.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Evaluating"):
                images = images.to(device, non_blocking=True)
                features = teacher.vision_model(images).pooler_output
                logits = classifier(features)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / n_batches
        
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best
        if f1 > best_f1:
            best_f1 = f1
            best_state = classifier.state_dict().copy()
            print(f"  ✅ New best F1: {best_f1:.4f}")
        
        scheduler.step()
    
    # Save the trained classifier
    output_dir = Path("/data/teacher_head")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / "medsiglip_classifier.pth"
    torch.save({
        'state_dict': best_state,
        'feature_dim': feature_dim,
        'num_classes': 10,
        'best_f1': best_f1,
        'class_names': full_dataset.class_names,
        'use_mlp': use_mlp,
        'trained_at': datetime.now().isoformat()
    }, save_path)
    
    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"{'='*80}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Saved to: {save_path}")
    print()
    print("Now you can use --use-trained-teacher-head in modal_train.py")
    print(f"{'='*80}\n")
    
    # Commit volume
    volume.commit()
    
    return {
        'best_f1': best_f1,
        'feature_dim': feature_dim,
        'save_path': str(save_path)
    }


@app.local_entrypoint()
def main(
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    use_class_weights: bool = True,
    train_split: float = 0.9,
    use_mlp: bool = True,
):
    """Train the teacher's classifier head."""
    print(f"\n{'='*80}")
    print("Training Teacher Classifier Head (with correct preprocessing!)")
    print(f"{'='*80}\n")
    
    result = train_teacher_classifier.remote(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_class_weights=use_class_weights,
        train_split=train_split,
        use_mlp=use_mlp,
    )
    
    print(f"\n✓ Teacher head trained!")
    print(f"  Best F1: {result['best_f1']:.4f}")
    print(f"  Saved to: {result['save_path']}")


