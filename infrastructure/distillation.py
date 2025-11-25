"""Knowledge distillation losses and utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class DistillationLoss(nn.Module):
    """Knowledge distillation loss following Hinton et al. (2015).
    
    Combines:
    - Hard loss: Cross-entropy with ground truth labels
    - Soft loss: KL divergence between teacher and student soft predictions with temperature scaling
    
    Args:
        temperature: Temperature for softening distributions (default: 4.0)
        alpha: Weight for hard loss vs soft loss (default: 0.3)
            Total loss = alpha * hard_loss + (1 - alpha) * soft_loss
        use_hard_loss: Whether to include hard loss component
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.3, use_hard_loss: bool = True):
        super().__init__()
        
        self.temperature = temperature
        self.alpha = alpha
        self.use_hard_loss = use_hard_loss
        
        # Hard loss criterion
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, 
                student_logits: torch.Tensor, 
                teacher_logits: torch.Tensor, 
                labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute distillation loss.
        
        Args:
            student_logits: Student model logits [B, num_classes]
            teacher_logits: Teacher model logits [B, num_classes]
            labels: Ground truth labels [B]
            
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual components
        """
        # Hard loss: standard cross-entropy with ground truth
        if self.use_hard_loss:
            hard_loss = self.ce_loss(student_logits, labels)
        else:
            hard_loss = torch.tensor(0.0, device=student_logits.device)
        
        # Soft loss: KL divergence between teacher and student soft predictions
        # Temperature scaling softens the distributions for better knowledge transfer
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # KL divergence scaled by T^2 (compensates for gradient scaling)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)
        
        # Combine losses
        if self.use_hard_loss:
            total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        else:
            total_loss = soft_loss
        
        # Return loss components for logging
        loss_dict = {
            'hard_loss': hard_loss,
            'soft_loss': soft_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict


class FeatureDistillationLoss(nn.Module):
    """Feature-based distillation loss.
    
    Matches intermediate feature representations between teacher and student.
    Useful when teacher and student have similar architectures.
    
    Args:
        loss_type: Type of feature loss ('mse', 'cosine', 'l1')
        normalize_features: Whether to L2-normalize features before computing loss
    """
    
    def __init__(self, loss_type: str = 'mse', normalize_features: bool = True):
        super().__init__()
        
        self.loss_type = loss_type
        self.normalize_features = normalize_features
    
    def forward(self, 
                student_features: torch.Tensor, 
                teacher_features: torch.Tensor) -> torch.Tensor:
        """Compute feature distillation loss.
        
        Args:
            student_features: Student feature maps [B, C, H, W] or [B, D]
            teacher_features: Teacher feature maps [B, C, H, W] or [B, D]
            
        Returns:
            Feature distillation loss
        """
        # Optionally normalize
        if self.normalize_features:
            student_features = F.normalize(student_features, p=2, dim=1)
            teacher_features = F.normalize(teacher_features, p=2, dim=1)
        
        # Compute loss based on type
        if self.loss_type == 'mse':
            loss = F.mse_loss(student_features, teacher_features)
        elif self.loss_type == 'cosine':
            # Cosine similarity loss (1 - cosine similarity)
            cosine_sim = F.cosine_similarity(student_features, teacher_features, dim=1)
            loss = (1 - cosine_sim).mean()
        elif self.loss_type == 'l1':
            loss = F.l1_loss(student_features, teacher_features)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class HybridDistillationLoss(nn.Module):
    """Hybrid distillation combining logit and feature matching.
    
    Args:
        temperature: Temperature for logit distillation
        alpha: Weight for hard loss
        beta: Weight for feature loss
            Total = alpha * hard + (1-alpha) * (beta * feature + (1-beta) * soft_logit)
        feature_loss_type: Type of feature loss
    """
    
    def __init__(self, 
                 temperature: float = 4.0,
                 alpha: float = 0.3,
                 beta: float = 0.5,
                 feature_loss_type: str = 'mse'):
        super().__init__()
        
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.feature_loss = FeatureDistillationLoss(loss_type=feature_loss_type)
    
    def forward(self,
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                student_features: Optional[torch.Tensor],
                teacher_features: Optional[torch.Tensor],
                labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute hybrid distillation loss.
        
        Args:
            student_logits: Student logits [B, num_classes]
            teacher_logits: Teacher logits [B, num_classes]
            student_features: Student features (optional)
            teacher_features: Teacher features (optional)
            labels: Ground truth labels [B]
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Hard loss
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Soft logit loss
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)
        
        # Feature loss (if features provided)
        if student_features is not None and teacher_features is not None:
            feat_loss = self.feature_loss(student_features, teacher_features)
            distill_loss = self.beta * feat_loss + (1 - self.beta) * soft_loss
        else:
            feat_loss = torch.tensor(0.0, device=student_logits.device)
            distill_loss = soft_loss
        
        # Total loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * distill_loss
        
        loss_dict = {
            'hard_loss': hard_loss,
            'soft_loss': soft_loss,
            'feature_loss': feat_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict


def get_distillation_loss(method: str = 'logit', **kwargs) -> nn.Module:
    """Factory function for distillation losses.
    
    Args:
        method: Distillation method ('logit', 'feature', 'hybrid')
        **kwargs: Method-specific arguments
        
    Returns:
        Distillation loss module
    """
    if method == 'logit':
        return DistillationLoss(**kwargs)
    elif method == 'feature':
        return FeatureDistillationLoss(**kwargs)
    elif method == 'hybrid':
        return HybridDistillationLoss(**kwargs)
    else:
        raise ValueError(f"Unknown distillation method: {method}")

