"""Modal training script for knowledge distillation on H100 GPUs.

Usage:
    modal run modal_train.py --architecture shufflenet --epochs 50
    modal run modal_train.py --architecture mobilenet_v2 --temperature 6.0 --alpha 0.1
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import modal

# Create Modal app
app = modal.App("project3-knowledge-distillation")

# Create Modal volume for data and models
volume = modal.Volume.from_name("project3-kd-data", create_if_missing=True)

# HuggingFace token - use Modal secret or environment variable
# Set via: modal secret create huggingface-secret HF_TOKEN=hf_xxx
HF_TOKEN = os.environ.get("HF_TOKEN", "")

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
        "beautifulsoup4>=4.12.0",
        "kornia>=0.7.0",  # GPU-accelerated augmentation
    )
    .add_local_dir("models", "/root/code/models")
    .add_local_dir("infrastructure", "/root/code/infrastructure")
)


@app.function(
    gpu="H100",
    image=image,
    volumes={"/data": volume},
    timeout=43200,  # 12 hours
    memory=65536,  # H100 has more memory
    cpu=16,
    secrets=[modal.Secret.from_name("huggingface-secret")],  # HF_TOKEN env var
)
def train_on_h100(
    architecture: str = "shufflenet",
    width_mult: float = 0.5,
    epochs: int = 50,
    batch_size: int = 128,  # Larger batch size for H100
    learning_rate: float = 0.0003,
    weight_decay: float = 0.0001,
    temperature: float = 4.0,
    alpha: float = 0.3,
    augmentation_strength: str = "light",
    use_class_weights: bool = True,
    train_split: float = 0.9,
    run_name: Optional[str] = None,
    early_stopping_patience: int = 10,
    grad_clip: float = 1.0,
):
    """Train knowledge distillation model on H100 GPU with optimizations."""
    import torch
    import torch.nn as nn
    from datetime import datetime
    from huggingface_hub import login
    from transformers import AutoModel
    
    # Import our modules
    import sys
    sys.path.insert(0, "/root/code")
    
    from models import get_student_model, get_model_info
    from infrastructure.data import create_dataloaders, get_gpu_augmentation
    from infrastructure.distillation import get_distillation_loss
    from infrastructure.training import train_with_distillation, save_torchscript
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"{'='*80}\n")
    
    # Enable optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Enable flash attention for transformer models (teacher)
        torch.backends.cuda.enable_flash_sdp(True)
        
        # Disable debug APIs for speed
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)
        
        print("✓ Enabled TF32, cuDNN, flash attention, and disabled debug APIs")
    
    # Generate run name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{architecture}_T{temperature}_A{alpha}_{timestamp}"
    
    print(f"Run name: {run_name}\n")
    
    # Configuration
    config = {
        'run_name': run_name,
        'architecture': architecture,
        'width_mult': width_mult,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'temperature': temperature,
        'alpha': alpha,
        'augmentation_strength': augmentation_strength,
        'use_class_weights': use_class_weights,
        'train_split': train_split,
        'early_stopping_patience': early_stopping_patience,
        'grad_clip': grad_clip,
        'timestamp': datetime.now().isoformat(),
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Login to HuggingFace
    print("Logging in to HuggingFace...")
    login(token=HF_TOKEN)
    print("✓ HuggingFace login successful\n")
    
    # Create dataloaders
    print("Creating dataloaders...")
    data_root = "/data/data/training_dataset/train_dataset"
    
    train_loader, val_loader, info = create_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        train_split=train_split,
        num_workers=16,
        pin_memory=True,
        augmentation_strength=augmentation_strength,
        use_class_weights=use_class_weights,
        seed=42
    )
    
    print(f"✓ Data loaded:")
    print(f"  Train: {info['train_size']} images")
    print(f"  Val: {info['val_size']} images")
    print(f"  Classes: {info['num_classes']}")
    print(f"  Batches per epoch: {len(train_loader)}\n")
    
    # Create student model
    print(f"Creating student model ({architecture})...")
    student_model = get_student_model(
        architecture=architecture,
        num_classes=10,
        width_mult=width_mult,
        pretrained=False,
        dropout=0.2
    ).to(device)
    
    model_info = get_model_info(student_model)
    print(f"✓ Student model created:")
    print(f"  Architecture: {model_info['architecture']}")
    print(f"  Parameters: {model_info['num_parameters']:,}")
    print(f"  Estimated size: {model_info['estimated_size_mb']:.2f} MB")
    print(f"  Within limit: {model_info['size_ok']}")
    
    # Compile student model for faster training (H100 optimized)
    # Note: We'll use the compiled model for training, but create a fresh
    # model for TorchScript export (torch.compile not compatible with jit.script)
    try:
        compiled_student = torch.compile(student_model, mode="reduce-overhead")
        print(f"✓ torch.compile enabled (reduce-overhead mode)\n")
        use_compiled = True
    except Exception as e:
        print(f"⚠ torch.compile unavailable: {e}")
        compiled_student = student_model
        use_compiled = False
    
    # Load teacher model (MedSigLIP)
    print("Loading teacher model (MedSigLIP)...")
    teacher_model = AutoModel.from_pretrained(
        "google/medsiglip-448",
        trust_remote_code=True,
        token=HF_TOKEN
    ).to(device)
    teacher_model.eval()
    
    # Freeze teacher
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # Add classifier head to teacher (fixed random projection for soft targets)
    with torch.no_grad():
        sample_images = torch.randn(1, 3, 224, 224).to(device)
        # Resize to 448x448 for MedSigLIP
        sample_images_448 = torch.nn.functional.interpolate(
            sample_images, size=(448, 448), mode='bilinear', align_corners=False
        )
        teacher_features = teacher_model.vision_model(sample_images_448).pooler_output
        hidden_dim = teacher_features.shape[1]
    
    teacher_model.classifier_head = nn.Linear(hidden_dim, 10).to(device)
    print(f"✓ Teacher model loaded:")
    print(f"  Model: google/medsiglip-448")
    print(f"  Feature dim: {hidden_dim}")
    print(f"  Classifier: {hidden_dim} -> 10 classes\n")
    
    # Wrap teacher to handle image resizing
    class TeacherWrapper(nn.Module):
        def __init__(self, teacher):
            super().__init__()
            self.teacher = teacher
        
        def forward(self, x):
            # Resize from 224x224 to 448x448 for MedSigLIP
            x_448 = torch.nn.functional.interpolate(
                x, size=(448, 448), mode='bilinear', align_corners=False
            )
            features = self.teacher.vision_model(x_448).pooler_output
            logits = self.teacher.classifier_head(features)
            return logits
    
    teacher_wrapped = TeacherWrapper(teacher_model).to(device)
    teacher_wrapped.eval()
    
    # Create distillation loss
    distillation_loss = get_distillation_loss(
        method='logit',
        temperature=temperature,
        alpha=alpha,
        use_hard_loss=True
    )
    print(f"✓ Distillation loss created:")
    print(f"  Method: logit")
    print(f"  Temperature: {temperature}")
    print(f"  Alpha: {alpha}\n")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        student_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )
    
    # Create GPU augmentation (faster than CPU augmentation)
    gpu_aug = get_gpu_augmentation(augmentation_strength)
    if gpu_aug is not None:
        print(f"✓ GPU augmentation (Kornia) enabled: {augmentation_strength}\n")
    
    # Train model
    print(f"{'='*80}")
    print("Starting training...")
    print(f"{'='*80}\n")
    
    result = train_with_distillation(
        student_model=compiled_student,  # Use compiled model for training
        teacher_model=teacher_wrapped,
        train_loader=train_loader,
        val_loader=val_loader,
        distillation_loss=distillation_loss,
        optimizer=optimizer,
        device=device,
        num_epochs=epochs,
        scheduler=scheduler,
        early_stopping_patience=early_stopping_patience,
        checkpoint_dir='/data/checkpoints',
        run_name=run_name,
        grad_clip=grad_clip,
        save_best_only=True,
        gpu_augmentation=gpu_aug
    )
    
    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"{'='*80}")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Best F1: {result['best_f1']:.4f}")
    print(f"Final val F1: {result['final_val_f1']:.4f}")
    print(f"Final val accuracy: {result['final_val_accuracy']:.4f}")
    print(f"Training time: {result['training_time_seconds']:.2f}s\n")
    
    # Save TorchScript model - need fresh model for compatibility
    output_dir = f"/data/models/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create fresh model for TorchScript (compiled models can't be scripted)
    if use_compiled:
        print("Creating fresh model for TorchScript export...")
        export_model = get_student_model(
            architecture=architecture,
            num_classes=10,
            width_mult=width_mult,
            pretrained=False,
            dropout=0.2
        )
        # Load best weights from checkpoint
        best_checkpoint = torch.load(f'/data/checkpoints/{run_name}/best_model.pth')
        # torch.compile adds '_orig_mod.' prefix to state dict keys - strip it
        state_dict = best_checkpoint['model_state_dict']
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            # Remove '_orig_mod.' prefix if present
            new_key = k.replace('_orig_mod.', '')
            cleaned_state_dict[new_key] = v
        export_model.load_state_dict(cleaned_state_dict)
    else:
        export_model = student_model
    
    torchscript_path = f"{output_dir}/model.pt"
    file_size = save_torchscript(export_model, torchscript_path)
    size_mb = file_size / (1024 ** 2)
    
    print(f"✓ TorchScript model saved:")
    print(f"  Path: {torchscript_path}")
    print(f"  Size: {size_mb:.2f} MB\n")
    
    # Save results
    results = {
        'config': config,
        'history': result['history'],
        'best_epoch': result['best_epoch'],
        'best_f1': result['best_f1'],
        'final_val_f1': result['final_val_f1'],
        'final_val_accuracy': result['final_val_accuracy'],
        'final_val_loss': result['final_val_loss'],
        'training_time_seconds': result['training_time_seconds'],
        'num_parameters': model_info['num_parameters'],
        'model_size_mb': size_mb,
        'torchscript_path': torchscript_path
    }
    
    results_path = f"{output_dir}/results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {results_path}\n")
    
    # Instructions for download
    print(f"{'='*80}")
    print("Next steps:")
    print(f"{'='*80}")
    print(f"1. Download model:")
    print(f"   modal volume get project3-kd-data {torchscript_path} ./models/{run_name}.pt")
    print(f"")
    print(f"2. Submit to server:")
    print(f"   uv run python server_cli.py submit models/{run_name}.pt --run-name {run_name}")
    print(f"")
    print(f"3. Wait and sync:")
    print(f"   uv run python server_cli.py wait-and-sync --run-name {run_name}")
    print(f"{'='*80}\n")
    
    # Commit volume changes
    volume.commit()
    
    return results


@app.local_entrypoint()
def main(
    architecture: str = "shufflenet",
    width_mult: float = 0.5,
    epochs: int = 50,
    batch_size: int = 128,  # Larger batch size for H100
    learning_rate: float = 0.0003,
    weight_decay: float = 0.0001,
    temperature: float = 4.0,
    alpha: float = 0.3,
    augmentation_strength: str = "light",
    use_class_weights: bool = True,
    train_split: float = 0.9,
    run_name: Optional[str] = None,
    early_stopping_patience: int = 10,
    grad_clip: float = 1.0,
):
    """Main entry point for Modal training."""
    print(f"\n{'='*80}")
    print(f"Starting training on Modal H100...")
    print(f"{'='*80}\n")
    
    # Train on Modal
    results = train_on_h100.remote(
        architecture=architecture,
        width_mult=width_mult,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        temperature=temperature,
        alpha=alpha,
        augmentation_strength=augmentation_strength,
        use_class_weights=use_class_weights,
        train_split=train_split,
        run_name=run_name,
        early_stopping_patience=early_stopping_patience,
        grad_clip=grad_clip,
    )
    
    print("\n" + "="*80)
    print("Training job completed!")
    print("="*80)
    print(f"\nRun: {results['config']['run_name']}")
    print(f"Best F1: {results['best_f1']:.4f} (epoch {results['best_epoch']})")
    print(f"Final val F1: {results['final_val_f1']:.4f}")
    print(f"Model size: {results['model_size_mb']:.2f} MB")
    print(f"Parameters: {results['num_parameters']:,}")
    print("="*80 + "\n")

