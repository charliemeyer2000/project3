"""Training infrastructure for knowledge distillation."""

import time
import threading
import queue
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report

logger = logging.getLogger(__name__)


def create_scheduler(optimizer: torch.optim.Optimizer,
                     scheduler_type: str = 'plateau',
                     num_epochs: int = 50,
                     warmup_epochs: int = 0,
                     **kwargs) -> Tuple[Optional[Any], Optional[Callable]]:
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler ('plateau', 'cosine', 'cosine_warmup')
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs (linear warmup)
        **kwargs: Additional scheduler arguments
        
    Returns:
        Tuple of (scheduler, warmup_fn) where warmup_fn adjusts LR during warmup
    """
    # Create warmup function if needed
    warmup_fn = None
    if warmup_epochs > 0:
        base_lr = optimizer.param_groups[0]['lr']
        def warmup_fn(epoch: int) -> float:
            """Linear warmup."""
            if epoch < warmup_epochs:
                return base_lr * (epoch + 1) / warmup_epochs
            return None  # Return None when warmup is done
    
    # Create main scheduler
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 5),
            min_lr=kwargs.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=kwargs.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'cosine_warmup':
        # Cosine annealing with warm restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', 10),
            T_mult=kwargs.get('T_mult', 2),
            eta_min=kwargs.get('min_lr', 1e-6)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler, warmup_fn


class AsyncCheckpointer:
    """Async checkpoint saver to avoid blocking training."""
    
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            state_dict, path = item
            torch.save(state_dict, path)
            self.queue.task_done()
    
    def save(self, checkpoint_dict: Dict, path: str):
        """Save checkpoint asynchronously."""
        # Deep copy state dicts to avoid race conditions
        copied = {}
        for k, v in checkpoint_dict.items():
            if isinstance(v, dict):
                # State dicts need deep copy
                copied[k] = {sk: sv.cpu().clone() if isinstance(sv, torch.Tensor) else sv 
                            for sk, sv in v.items()}
            else:
                copied[k] = v
        self.queue.put((copied, path))
    
    def wait(self):
        """Wait for all pending saves to complete."""
        self.queue.join()
    
    def shutdown(self):
        """Shutdown the worker thread."""
        self.queue.put(None)
        self.thread.join()


class Trainer:
    """Trainer for knowledge distillation with AMP and GPU augmentation support.
    
    Args:
        student_model: Student model to train
        teacher_model: Teacher model (frozen)
        distillation_loss: Distillation loss function
        optimizer: Optimizer for student
        device: Device to train on
        scheduler: Optional learning rate scheduler
        grad_clip: Optional gradient clipping value
        use_amp: Whether to use automatic mixed precision (default: True for CUDA)
        gpu_augmentation: Optional GPU augmentation module (from data.py)
        mixup_cutmix: Optional Mixup/CutMix augmentation module
        warmup_fn: Optional warmup function that takes epoch and returns LR (or None if warmup done)
    """
    
    def __init__(self,
                 student_model: nn.Module,
                 teacher_model: nn.Module,
                 distillation_loss: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 scheduler: Optional[Any] = None,
                 grad_clip: Optional[float] = None,
                 use_amp: bool = True,
                 gpu_augmentation: Optional[nn.Module] = None,
                 mixup_cutmix: Optional[Any] = None,
                 warmup_fn: Optional[Callable] = None):
        
        self.student = student_model
        self.teacher = teacher_model
        self.distillation_loss = distillation_loss
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.mixup_cutmix = mixup_cutmix
        self.warmup_fn = warmup_fn
        
        # AMP setup - only use on CUDA
        self.use_amp = use_amp and device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler('cuda')
            logger.info("✓ Mixed precision training (AMP) enabled")
        else:
            self.scaler = None
        
        # GPU augmentation
        self.gpu_augmentation = gpu_augmentation
        if gpu_augmentation is not None:
            self.gpu_augmentation = gpu_augmentation.to(device)
            logger.info("✓ GPU augmentation (Kornia) enabled")
        
        # Mixup/CutMix
        if mixup_cutmix is not None:
            logger.info("✓ Mixup/CutMix augmentation enabled")
        
        # Ensure teacher is in eval mode and frozen
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with AMP support.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.student.train()
        
        # Apply warmup if in warmup phase
        if self.warmup_fn is not None:
            warmup_lr = self.warmup_fn(epoch - 1)  # epoch is 1-indexed
            if warmup_lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                logger.info(f"Warmup: LR set to {warmup_lr:.6f}")
        
        total_loss = 0.0
        total_hard_loss = 0.0
        total_soft_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for images, labels in pbar:
            # Non-blocking transfer to GPU
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Apply GPU augmentation if available (faster than CPU augmentation)
            if self.gpu_augmentation is not None:
                images = self.gpu_augmentation(images)
            
            # Apply Mixup/CutMix if enabled (note: this returns soft labels)
            # For now, we use original labels for distillation loss
            # TODO: Support soft label training when Mixup is enabled
            use_mixup = self.mixup_cutmix is not None
            if use_mixup:
                images, _ = self.mixup_cutmix(images, labels)
                # Note: We still use hard labels for distillation loss
                # The mixed images help with regularization
            
            # Get teacher predictions (no gradients, with AMP)
            with torch.no_grad():
                if self.use_amp:
                    with autocast('cuda', dtype=torch.float16):
                        teacher_logits = self.teacher(images)
                        if isinstance(teacher_logits, tuple):
                            teacher_logits = teacher_logits[0]
                else:
                    teacher_logits = self.teacher(images)
                    if isinstance(teacher_logits, tuple):
                        teacher_logits = teacher_logits[0]
            
            # Forward pass with AMP
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            if self.use_amp:
                with autocast('cuda', dtype=torch.float16):
                    # Get student predictions
                    student_logits = self.student(images)
                    if isinstance(student_logits, tuple):
                        student_logits = student_logits[0]
                    
                    # Compute distillation loss
                    loss, loss_dict = self.distillation_loss(student_logits, teacher_logits, labels)
                
                # Scaled backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping with scaler
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
                
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard FP32 training
                student_logits = self.student(images)
                if isinstance(student_logits, tuple):
                    student_logits = student_logits[0]
                
                loss, loss_dict = self.distillation_loss(student_logits, teacher_logits, labels)
                loss.backward()
                
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
                
                self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_hard_loss += loss_dict['hard_loss'].item()
            total_soft_loss += loss_dict['soft_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'hard': f'{loss_dict["hard_loss"].item():.4f}',
                'soft': f'{loss_dict["soft_loss"].item():.4f}'
            })
        
        return {
            'train_loss': total_loss / num_batches,
            'train_hard_loss': total_hard_loss / num_batches,
            'train_soft_loss': total_soft_loss / num_batches
        }
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate student model with AMP support.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.student.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        for images, labels in tqdm(val_loader, desc='Evaluating'):
            # Non-blocking transfer
            images = images.to(self.device, non_blocking=True)
            labels_dev = labels.to(self.device, non_blocking=True)
            
            if self.use_amp:
                with autocast('cuda', dtype=torch.float16):
                    # Get student predictions
                    student_logits = self.student(images)
                    if isinstance(student_logits, tuple):
                        student_logits = student_logits[0]
                    
                    # Get teacher predictions for loss computation
                    teacher_logits = self.teacher(images)
                    if isinstance(teacher_logits, tuple):
                        teacher_logits = teacher_logits[0]
                    
                    # Compute loss
                    loss, _ = self.distillation_loss(student_logits, teacher_logits, labels_dev)
            else:
                student_logits = self.student(images)
                if isinstance(student_logits, tuple):
                    student_logits = student_logits[0]
                
                teacher_logits = self.teacher(images)
                if isinstance(teacher_logits, tuple):
                    teacher_logits = teacher_logits[0]
                
                loss, _ = self.distillation_loss(student_logits, teacher_logits, labels_dev)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions
            preds = torch.argmax(student_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': accuracy,
            'val_f1_macro': f1_macro,
            'val_f1_weighted': f1_weighted,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def step_scheduler(self, metric: Optional[float] = None):
        """Step the learning rate scheduler.
        
        Args:
            metric: Metric for ReduceLROnPlateau scheduler
        """
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if metric is not None:
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()


def train_with_distillation(
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    distillation_loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 50,
    scheduler: Optional[Any] = None,
    early_stopping_patience: int = 10,
    checkpoint_dir: str = 'checkpoints',
    run_name: str = 'experiment',
    grad_clip: Optional[float] = 1.0,
    save_best_only: bool = True,
    use_amp: bool = True,
    use_async_checkpoint: bool = True,
    gpu_augmentation: Optional[nn.Module] = None,
    mixup_cutmix: Optional[Any] = None,
    warmup_fn: Optional[Callable] = None
) -> Dict[str, Any]:
    """Train student model with knowledge distillation.
    
    Args:
        student_model: Student model to train
        teacher_model: Teacher model (frozen)
        train_loader: Training data loader
        val_loader: Validation data loader
        distillation_loss: Distillation loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs
        scheduler: Optional LR scheduler
        early_stopping_patience: Patience for early stopping
        checkpoint_dir: Directory to save checkpoints
        run_name: Name of this run
        grad_clip: Gradient clipping value
        save_best_only: Whether to save only the best model
        use_amp: Whether to use automatic mixed precision
        use_async_checkpoint: Whether to save checkpoints asynchronously
        gpu_augmentation: Optional GPU augmentation module (Kornia)
        mixup_cutmix: Optional Mixup/CutMix augmentation module
        warmup_fn: Optional warmup function
        
    Returns:
        Dictionary with training history and best model info
    """
    # Setup
    checkpoint_path = Path(checkpoint_dir) / run_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Async checkpointer
    checkpointer = AsyncCheckpointer() if use_async_checkpoint else None
    if checkpointer:
        logger.info("✓ Async checkpointing enabled")
    
    trainer = Trainer(
        student_model=student_model,
        teacher_model=teacher_model,
        distillation_loss=distillation_loss,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        grad_clip=grad_clip,
        use_amp=use_amp,
        gpu_augmentation=gpu_augmentation,
        mixup_cutmix=mixup_cutmix,
        warmup_fn=warmup_fn
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1_macro': [],
        'val_f1_weighted': [],
        'learning_rate': []
    }
    
    # Early stopping
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Checkpoint directory: {checkpoint_path}")
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader)
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_metrics['train_loss'])
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_accuracy'].append(val_metrics['val_accuracy'])
        history['val_f1_macro'].append(val_metrics['val_f1_macro'])
        history['val_f1_weighted'].append(val_metrics['val_f1_weighted'])
        history['learning_rate'].append(current_lr)
        
        # Log metrics
        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
            f"Val F1: {val_metrics['val_f1_macro']:.4f}, "
            f"LR: {current_lr:.6f}"
        )
        
        # Check for improvement
        current_f1 = val_metrics['val_f1_macro']
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model (async or sync)
            best_model_path = checkpoint_path / 'best_model.pth'
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_f1,
                'val_accuracy': val_metrics['val_accuracy'],
                'val_loss': val_metrics['val_loss']
            }
            
            if checkpointer:
                checkpointer.save(checkpoint_dict, str(best_model_path))
            else:
                torch.save(checkpoint_dict, best_model_path)
            
            logger.info(f"✅ New best F1: {best_f1:.4f} (epoch {epoch})")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs")
        
        # Save checkpoint (if not save_best_only)
        if not save_best_only:
            checkpoint_file = checkpoint_path / f'checkpoint_epoch_{epoch}.pth'
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': current_f1,
                'val_accuracy': val_metrics['val_accuracy'],
                'val_loss': val_metrics['val_loss']
            }
            
            if checkpointer:
                checkpointer.save(checkpoint_dict, str(checkpoint_file))
            else:
                torch.save(checkpoint_dict, checkpoint_file)
        
        # Learning rate scheduling
        trainer.step_scheduler(val_metrics['val_loss'])
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    training_time = time.time() - start_time
    logger.info(f"Training complete! Best F1: {best_f1:.4f} at epoch {best_epoch}")
    logger.info(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Wait for pending async saves before loading best model
    if checkpointer:
        logger.info("Waiting for async checkpoints to complete...")
        checkpointer.wait()
        checkpointer.shutdown()
    
    # Load best model
    best_checkpoint = torch.load(checkpoint_path / 'best_model.pth')
    student_model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Final evaluation
    final_metrics = trainer.evaluate(val_loader)
    
    return {
        'history': history,
        'best_epoch': best_epoch,
        'best_f1': best_f1,
        'final_val_f1': final_metrics['val_f1_macro'],
        'final_val_accuracy': final_metrics['val_accuracy'],
        'final_val_loss': final_metrics['val_loss'],
        'training_time_seconds': training_time,
        'model_path': str(checkpoint_path / 'best_model.pth'),
        'student_model': student_model
    }


def save_torchscript(model: nn.Module, output_path: str) -> int:
    """Save model as TorchScript.
    
    Args:
        model: PyTorch model
        output_path: Path to save TorchScript file
        
    Returns:
        File size in bytes
    """
    model.eval()
    model.cpu()
    
    # Script the model
    scripted_model = torch.jit.script(model)
    scripted_model.save(output_path)
    
    # Get file size
    file_size = Path(output_path).stat().st_size
    logger.info(f"✅ TorchScript model saved to {output_path}")
    logger.info(f"   File size: {file_size / (1024**2):.2f} MB")
    
    return file_size

