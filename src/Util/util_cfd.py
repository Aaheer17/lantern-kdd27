
from __future__ import annotations
from typing import Optional, Union
import os
import torch

# Must match your dataset pipeline
from transforms import (
    Reshape, ScaleEnergy, LogEnergy,
    StandardizeFromFile, ExclusiveLogitTransform
)


 def _reverse_to_u(self, x_proc: torch.Tensor, e_proc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Undo ONLY: Reshape → ScaleEnergy → LogEnergy → StandardizeFromFile → ExclusiveLogitTransform.
        Returns:
          u_raw: (B,45)   # stick-breaking params in raw scale
          e_raw: (B,)     # raw incident energy (GeV)
        """
        layers, energy = x_proc, e_proc
        for fn in reversed(self.inv_fns):
            layers, energy = fn(layers, energy, rev=True)
        if layers.ndim == 3 and layers.shape[-1] == 1:
            layers = layers.squeeze(-1)
        if energy.ndim > 1:
            energy = energy.view(-1)
        return layers, energy

    @staticmethod
    def _u_to_layerE(u: torch.Tensor, e_inc: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct per-layer energies via stick-breaking (matches NormalizeByElayer.rev):
           total_E = e_inc * u0
           layer[i] = (total_E - sum_{k<i} layer[k]) * u_{i+1} for i=0..L-2
           layer[L-1] = total_E - sum_{k<L-1} layer[k]
        u:     (B,L) in (delta, 1-delta)
        e_inc: (B,)
        Returns: (B,L)
        """
        B, L = u.shape
        #print("shape of B: ",B, e_inc.shape)
        assert e_inc.shape[0] == B
        # Clamp u_{i>0} to [0,1] to mirror reverse logic and avoid numeric overshoot
        u = u.clone()
        if L > 1:
            u[:, 1:] = u[:, 1:].clamp_(0.0, 1.0)
        total_E = e_inc.view(-1) * u[:, 0]
        cum = torch.zeros_like(total_E)
        layers = []
        for i in range(L - 1):
            Ei = (total_E - cum) * u[:, i + 1]
            layers.append(Ei)
            cum = cum + Ei
        layers.append(total_E - cum)
        return torch.stack(layers, dim=1)

    @staticmethod
    def _corr(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Correlation across the batch (features D along dim=1).
        X: (B,D)
        Returns: (D,D)
        """
        Xc  = X - X.mean(dim=0, keepdim=True)
        cov = (Xc.transpose(0, 1) @ Xc) / (Xc.shape[0] - 1 + eps)
        var = cov.diag().clamp_min(eps)
        inv = var.rsqrt()
        return (cov * (inv[:, None] * inv[None, :])).clamp(-1.0, 1.0)