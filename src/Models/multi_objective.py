"""
Multi-Objective Optimization Methods for Multi-Task Learning

This module provides various strategies for combining multiple loss functions
during training. Each method has different trade-offs between simplicity,
computational cost, and adaptability.

Supported Methods:
  - weighted_sum: Static weighted combination
  - grad_norm: Gradient magnitude balancing
  - uncertainty: Homoscedastic uncertainty weighting (Kendall et al., 2018)
  - dwa: Dynamic Weight Average (Liu et al., 2019)
  - gradnorm: GradNorm balancing (Chen et al., 2018)
  - config: ConFIG (handled separately)

Author: Farzana Ahmad
Date: 2026-01-07
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class MultiObjectiveOptimizer:
    """
    Base class for multi-objective optimization methods.
    
    Each method implements a combine() function that takes loss tensors
    and returns a combined loss plus diagnostic information.
    """
    
    def __init__(self, model, optimizer):
        """
        Args:
            model: PyTorch model (for accessing parameters)
            optimizer: PyTorch optimizer (for adding learnable parameters)
        """
        self.model = model
        self.optimizer = optimizer
        self.device = next(model.parameters()).device
    
    def combine(self, loss_tensors: Dict[str, torch.Tensor], 
                use_flags: Dict[str, bool]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Combine losses using this method's strategy.
        
        Args:
            loss_tensors: Dict of {loss_name: loss_tensor}
            use_flags: Dict of {loss_name: should_use_bool}
        
        Returns:
            total_loss: Combined loss for backward()
            mo_info: Dict of diagnostics for logging
        """
        raise NotImplementedError


class WeightedSumMethod(MultiObjectiveOptimizer):
    """
    Simple weighted sum with fixed weights (λ parameters).
    
    Total Loss = λ₁L₁ + λ₂L₂ + ... + λₙLₙ
    
    Pros: Simple, stable, interpretable
    Cons: Requires manual tuning
    """
    
    def __init__(self, model, optimizer, weights: Dict[str, float]):
        super().__init__(model, optimizer)
        self.weights = weights
    
    def combine(self, loss_tensors: Dict[str, torch.Tensor],
                use_flags: Dict[str, bool]) -> Tuple[torch.Tensor, Dict[str, float]]:
        mo_info = {}
        total_loss = torch.zeros((), device=self.device)
        
        for loss_name, loss_tensor in loss_tensors.items():
            if use_flags.get(loss_name, True):
                weight = self.weights.get(loss_name, 1.0)
                weighted_loss = weight * loss_tensor
                total_loss = total_loss + weighted_loss
                mo_info[f'{loss_name}_contribution'] = weighted_loss.item()
                mo_info[f'{loss_name}_weight'] = weight
        
        return total_loss, mo_info


class GradNormMethod(MultiObjectiveOptimizer):
    """
    Gradient Norm Balancing: Scale losses to match gradient magnitudes.
    
    Automatically balances losses by ensuring their gradients have
    similar magnitudes.
    
    Pros: Automatic balancing, no manual tuning
    Cons: Expensive (requires extra backward passes)
    """
    
    def __init__(self, model, optimizer, target_loss: str = 'diffusion_loss'):
        super().__init__(model, optimizer)
        self.target_loss = target_loss
    
    def combine(self, loss_tensors: Dict[str, torch.Tensor],
                use_flags: Dict[str, bool]) -> Tuple[torch.Tensor, Dict[str, float]]:
        mo_info = {}
        
        # Compute gradient norm of target loss
        target_grad_norm = self._compute_grad_norm(loss_tensors[self.target_loss])
        mo_info[f'grad_norm_{self.target_loss}'] = target_grad_norm
        
        total_loss = loss_tensors[self.target_loss].clone()
        
        # Scale other losses
        for loss_name, loss_tensor in loss_tensors.items():
            if loss_name == self.target_loss:
                continue
                
            if use_flags.get(loss_name, True):
                aux_grad_norm = self._compute_grad_norm(loss_tensor)
                
                # Scale factor to match target gradient norm
                if aux_grad_norm > 1e-8:
                    scale = target_grad_norm / aux_grad_norm
                else:
                    scale = 1.0
                
                weighted_loss = scale * loss_tensor
                total_loss = total_loss + weighted_loss
                
                # Logging
                mo_info[f'grad_norm_{loss_name}'] = aux_grad_norm
                mo_info[f'grad_scale_{loss_name}'] = scale
                mo_info[f'{loss_name}_contribution'] = weighted_loss.item()
        
        # Reset gradients
        self.optimizer.zero_grad(set_to_none=True)
        
        return total_loss, mo_info
    
    def _compute_grad_norm(self, loss: torch.Tensor) -> float:
        """Compute L2 norm of gradients for a given loss."""
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        
        grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        
        return np.sqrt(grad_norm)


class UncertaintyWeightingMethod(MultiObjectiveOptimizer):
    """
    Homoscedastic Uncertainty Weighting (Kendall et al., CVPR 2018).
    
    Learns task-dependent uncertainty to automatically weight losses.
    Loss_i = (1/2σ²_i) * L_i + log(σ_i)
    
    Pros: Learns optimal weights automatically
    Cons: Adds learnable parameters
    
    Reference: https://arxiv.org/abs/1705.07115
    """
    
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)
        self.log_vars = nn.ParameterDict()
    
    def combine(self, loss_tensors: Dict[str, torch.Tensor],
                use_flags: Dict[str, bool]) -> Tuple[torch.Tensor, Dict[str, float]]:
        mo_info = {}
        total_loss = torch.zeros((), device=self.device)
        
        for loss_name, loss_tensor in loss_tensors.items():
            if use_flags.get(loss_name, True):
                # Initialize log-variance if needed
                if loss_name not in self.log_vars:
                    self.log_vars[loss_name] = nn.Parameter(
                        torch.zeros(1, device=self.device)
                    )
                    # Add to optimizer
                    self.optimizer.add_param_group({
                        'params': [self.log_vars[loss_name]]
                    })
                
                # Uncertainty weighting: precision * loss + log_var
                precision = torch.exp(-self.log_vars[loss_name])
                weighted_loss = precision * loss_tensor + self.log_vars[loss_name]
                total_loss = total_loss + weighted_loss
                
                # Logging
                uncertainty = torch.exp(self.log_vars[loss_name])
                mo_info[f'uncertainty_{loss_name}'] = uncertainty.item()
                mo_info[f'{loss_name}_contribution'] = weighted_loss.item()
        
        return total_loss, mo_info


class DynamicWeightAverageMethod(MultiObjectiveOptimizer):
    """
    Dynamic Weight Average (Liu et al., CVPR 2019).
    
    Adapts weights based on the rate of loss decrease.
    Faster decreasing losses get lower weights.
    
    Pros: Adapts over time automatically
    Cons: Requires loss history tracking
    
    Reference: https://arxiv.org/abs/1803.10704
    """
    
    def __init__(self, model, optimizer, temperature: float = 2.0):
        super().__init__(model, optimizer)
        self.temperature = temperature
        self.loss_history = {}
    
    def combine(self, loss_tensors: Dict[str, torch.Tensor],
                use_flags: Dict[str, bool]) -> Tuple[torch.Tensor, Dict[str, float]]:
        mo_info = {}
        
        # Get current losses
        current_losses = {
            name: loss.item() 
            for name, loss in loss_tensors.items() 
            if use_flags.get(name, True)
        }
        
        # Compute weights based on loss rate
        if len(self.loss_history) > 0:
            weights = {}
            for name, current_val in current_losses.items():
                if name in self.loss_history:
                    # Rate: L(t) / L(t-1)
                    rate = current_val / (self.loss_history[name] + 1e-8)
                    weights[name] = np.exp(rate / self.temperature)
                else:
                    weights[name] = 1.0
            
            # Normalize
            weight_sum = sum(weights.values())
            weights = {k: v / weight_sum for k, v in weights.items()}
        else:
            # First call: equal weights
            K = len(current_losses)
            weights = {k: 1.0 / K for k in current_losses.keys()}
        
        # Apply weights
        total_loss = torch.zeros((), device=self.device)
        for loss_name, loss_tensor in loss_tensors.items():
            if loss_name in weights:
                weighted_loss = weights[loss_name] * loss_tensor
                total_loss = total_loss + weighted_loss
                
                mo_info[f'dwa_weight_{loss_name}'] = weights[loss_name]
                mo_info[f'{loss_name}_contribution'] = weighted_loss.item()
        
        # Update history
        self.loss_history = current_losses.copy()
        
        return total_loss, mo_info


# ========================================
# Factory Function
# ========================================

def create_mo_optimizer(method: str, model: nn.Module, 
                       optimizer, **kwargs) -> MultiObjectiveOptimizer:
    """
    Factory function to create multi-objective optimizer.
    
    Args:
        method: One of ['weighted_sum', 'grad_norm', 'uncertainty', 'dwa']
        model: PyTorch model
        optimizer: PyTorch optimizer
        **kwargs: Method-specific parameters
    
    Returns:
        MultiObjectiveOptimizer instance
    
    Examples:
        >>> mo_opt = create_mo_optimizer('weighted_sum', model, optimizer,
        ...                              weights={'diffusion_loss': 1.0, 'energy_loss': 0.1})
        >>> mo_opt = create_mo_optimizer('grad_norm', model, optimizer)
        >>> mo_opt = create_mo_optimizer('uncertainty', model, optimizer)
        >>> mo_opt = create_mo_optimizer('dwa', model, optimizer, temperature=2.0)
    """
    if method == 'weighted_sum':
        weights = kwargs.get('weights', {})
        return WeightedSumMethod(model, optimizer, weights)
    
    elif method == 'grad_norm':
        target_loss = kwargs.get('target_loss', 'diffusion_loss')
        return GradNormMethod(model, optimizer, target_loss)
    
    elif method == 'uncertainty':
        return UncertaintyWeightingMethod(model, optimizer)
    
    elif method == 'dwa':
        temperature = kwargs.get('temperature', 2.0)
        return DynamicWeightAverageMethod(model, optimizer, temperature)
    
    else:
        raise ValueError(f"Unknown MO method: {method}. "
                        f"Choose from: weighted_sum, grad_norm, uncertainty, dwa")