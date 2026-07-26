import numpy as np
import torch
from scipy.integrate import solve_ivp
import Networks
from Util.util import get
from Models.ModelBase import GenerativeModel
import Networks
import Models
from torchdiffeq import odeint
import math
from challenge_files import *
from torch import linalg as LA
# from .auxiliary_losses_shape import AuxiliaryLossManager, EnergyConservationLoss
# from energy_loss_transforms import (
#     Reshape_X,
#     AddFeaturesToCond_X,
#     ScaleEnergy_X,
#     LogEnergy_X,
#     StandardizeFromFile_X,
#     ExclusiveLogitTransform_X,
#     SelectiveUniformNoise_X,
#     CutValues_X,
#     NormalizeByElayer_X,
#     ScaleVoxels_X
# )
from challenge_files import XMLHandler
import torch.nn.functional as F
torch.cuda.empty_cache()

class TBD_DIFF(GenerativeModel):
    """
    Class for Trajectory Based Diffusion - converted to proper diffusion model
    Inheriting from the GenerativeModel BaseClass
    """

    def __init__(self, params, device, doc):
        super().__init__(params, device, doc)
        print("\n" + "="*60)
        print("INITIALIZING DIFFUSION MODEL")
        print("="*60)
        
        # ========================================
        # DIFFUSION PARAMETERS
        # ========================================
        self.timesteps = get(self.params, "num_timesteps", 1000)
        self.beta_start = get(self.params, "beta_start", 0.0001)
        self.beta_end = get(self.params, "beta_end", 0.02)
        self.beta_schedule = get(self.params, "beta_schedule", "linear")
        self.prediction_type = get(self.params, "prediction_type", "noise")
        
        print(f"\nDiffusion Config:")
        print(f"  Timesteps: {self.timesteps}")
        print(f"  Beta schedule: {self.beta_schedule} [{self.beta_start:.6f}, {self.beta_end:.6f}]")
        print(f"  Prediction type: {self.prediction_type}")
        
        # Initialize noise schedule (betas, alphas, etc.)
        self.setup_noise_schedule()
        
        # ========================================
        # DATA PATHS & PREPROCESSING
        # ========================================
        self.stats_dir = self.doc.basedir
        self.xml_file = get(
            self.params, 
            'xml_filename',
            '/project/biocomplexity/fa7sa/calo_dreamer/src/challenge_files/binning_dataset_2.xml'
        )
        # Load once at startup, move to GPU immediately
        # In tbd_Diff.__init__ — replace your current bare torch.load with this
        lap_path = self.params.get(
            'laplacian_path',
            '/scratch/fa7sa/LANTERN_rewrite/LANTERN_20_FEB/LANTERN_new/src/calo_laplacian.pt'
        )
        lambda_max_path = self.params.get(
            'lambda_max_path',
            '/scratch/fa7sa/LANTERN_rewrite/LANTERN_20_FEB/LANTERN_new/src/lambda_max.pt'
        )
        _L = torch.load(lap_path, weights_only=True)
        #self.L = _L.to(self.device)

        self.L = _L.to(dtype=torch.get_default_dtype(), device=self.device)
        _ones = torch.ones(self.L.shape[0], device=self.device)
        _res  = torch.sparse.mm(self.L, _ones.unsqueeze(1)).abs().max()
        assert _res < 1e-4, f"Laplacian corrupted: max|L·1|={_res:.2e}"
        print(f"[Laplacian] Loaded: shape={self.L.shape}, nnz={self.L._nnz()}")
        lambda_max = torch.load(lambda_max_path, weights_only=True).item()
        self.lambda_max_sq = torch.tensor(lambda_max ** 2, dtype= torch.float32, device=self.device) #torch.get_default_dtype(
        print(f"[Laplacian] λ_max={lambda_max:.4f},λ_max²{self.lambda_max_sq.item():.4f}")
        print(f"\nData Paths:")
        print(f"  Stats directory: {self.stats_dir}")
        print(f"  XML file: {self.xml_file}")
     
        # ========================================
        # SPARSITY-SPECIFIC PARAMETERS
        # ========================================
      
        self.occ_tau = get(self.params, 'occ_tau', 0.0)
        self.occ_temp = get(self.params, 'occ_temp', 1.0)
        print(f"  Sparsity Parameters:")
        print(f"    Occupancy tau: {self.occ_tau}")
        print(f"    Occupancy temperature: {self.occ_temp}")
        
        # ========================================
        # NOISE INJECTION (for training stability)
        # ========================================
        self.add_noise = get(self.params, "add_noise", False)
        self.gamma = get(self.params, "gamma", 1e-4)
        
        if self.add_noise:
            print(f"\nNoise Injection:")
            print(f"  Enabled: True")
            print(f"  Gamma: {self.gamma}")
        
        # ========================================
        # OTHER PARAMETERS
        # ========================================
        self.C = get(self.params, "C", 1)
        self.bayesian = get(self.params, "bayesian", False)
        
        if self.C != 1:
            print(f"\nChannel multiplier C: {self.C}")
        
        if self.bayesian:
            print(f"Bayesian mode: Enabled")
        
        print("="*60)
        print("INITIALIZATION COMPLETE")
        print("="*60 + "\n")


        
    def setup_noise_schedule(self):
        """Setup the noise schedule for diffusion"""
        if self.beta_schedule == "linear":
            betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        elif self.beta_schedule == "cosine":
            betas = self.cosine_beta_schedule(self.timesteps)
        elif self.beta_schedule == "sigmoid":
            betas = self.sigmoid_beta_schedule(self.timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        
        self.betas = betas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        
        # Log calculation clipped because the posterior variance is 0 at the beginning
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        )
        
    def cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, min = 0, max = 0.999)
    
    def sigmoid_beta_schedule(self, timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
        """Sigmoid schedule in https://arxiv.org/abs/2206.00364"""
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        v_start = torch.sigmoid(torch.tensor(start / tau)).item()
        v_end = torch.sigmoid(torch.tensor(end / tau)).item()
        alphas_cumprod = torch.sigmoid((t * (end - start) + start) / tau)
        alphas_cumprod = (v_end - alphas_cumprod) / (v_end - v_start)
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, min = clamp_min, max = 0.999)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Sample from q(x_t | x_0) using the reparameterization trick
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(x_start.device)[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(x_start.device)[t]
        
        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
        while len(sqrt_one_minus_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return (sqrt_alphas_cumprod_t * x_start + 
                sqrt_one_minus_alphas_cumprod_t * noise)

    def build_net(self):
        """
        Build the network
        """
        network = get(self.params, "network", "Resnet")
        try:
            return getattr(Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")

    def get_condition_and_input(self, input):
        """
        :param input: model input + conditional input
        :return: model input, conditional input
        """
        condition = input[1] if isinstance(input, (list, tuple)) and len(input) > 1 else None
        weights = None
        return input[0] if isinstance(input, (list, tuple)) else input, condition, weights
    def compute_poisson_voxel_loss(
        self,
        voxels_pred,        # (B,1,45,16,9) or (B,45,16,9) — network's predicted x0, preprocessed space
        voxels_true,        # same shape — true x0, preprocessed space
        x_real,             # (B,1,45,16,9) preprocessed true sample (inverted -> physical MeV for the weight)
        condition,          # carries the TRUE u-features for x_real  (verified)
        apply_lambda=False, # if True, multiply by self.get_voxel_loss_weight() before returning
    ):
        """
        Poisson-weighted Huber voxel loss.
    
        Residual stays in preprocessed (logit-standardized) space, so the gradient
        flows through the network's native output with no inverse-logit Jacobian.
        The weight is a DETACHED per-voxel scalar built from true physical energy
        E_MeV (recovered via apply_inverse_transforms), so 1/sqrt(E) is the genuine
        Poisson fluctuation scale and is monotone in brightness across all layers.
        """
        eps_mev = self.params.get('voxel_loss_eps_mev', 1.0e-2)  # MeV floor (matches verified notebook)
        delta   = self.params.get('voxel_loss_delta',   1.0)
    
        # normalize shapes to (B,45,16,9)
        vp = voxels_pred.squeeze(1) if voxels_pred.dim() == 5 else voxels_pred
        vt = voxels_true.squeeze(1) if voxels_true.dim() == 5 else voxels_true
    
        # ---- detached physical-MeV weight from the TRUE sample ----
        with torch.no_grad():
            # clone: apply_inverse_transforms mutates in place (NormalizeByElayer)
            vox_true_phys, _, _ = self.apply_inverse_transforms(x_real.clone(), condition)  # (B,6480) MeV
            vox_true_phys = vox_true_phys.reshape(vt.shape).clamp_min(0.0)                  # (B,45,16,9)
    
            s = torch.sqrt(vox_true_phys + eps_mev)   # Poisson scale sqrt(E)
            w = 1.0 / s                               # dim voxel -> large weight
            active = vox_true_phys > eps_mev
            w_mean = w[active].mean() if active.any() else w.mean()
            w = (w / w_mean).detach()                 # mean-1 over active voxels, detached
    
        # ---- residual in preprocessed space (carries the gradient) ----
        residuals = vp - vt
        per_vox_huber = F.huber_loss(
            residuals, torch.zeros_like(residuals), delta=delta, reduction='none'
        )
        loss = (w * per_vox_huber).mean()             # raw term
    
        info = {
            'voxel_energy_loss_unweighted': loss.item(),
            'voxel_weight_mean':  w.mean().item(),
            'voxel_weight_max':   w.max().item(),
            'phys_active_frac':   active.float().mean().item(),
        }
    
        if apply_lambda:
            lam  = self.get_voxel_loss_weight()
            loss = loss * lam
            info['voxel_energy_weight'] = lam
            info['voxel_energy_loss']   = loss.item()
    
        return loss, info

    def compute_sparsity_occupancy(
        self,
        real: torch.Tensor,
        gen: torch.Tensor,
    ):
        """
        Compute sparsity and soft-occupancy diagnostics in preprocessed space.
    
        Args:
            real: real voxel data x, shape (B, 1, 45, H, W)
            gen: generated voxel data x0_pred, same shape as real
    
        Returns:
            dict with scalar tensors:
                - sparsity_l1_real
                - sparsity_l1_gen
                - sparsity_match_loss
                - occ_real
                - occ_gen
                - occ_match_loss
        """
    
        device = real.device
        dtype  = real.dtype
    
        # -------- initialize outputs (safe defaults) --------
        out = {
            "sparsity_l1_real": torch.zeros((), device=device, dtype=dtype),
            "sparsity_l1_gen":  torch.zeros((), device=device, dtype=dtype),
            "sparsity_match_loss": torch.zeros((), device=device, dtype=dtype),
            "occ_real": torch.zeros((), device=device, dtype=dtype),
            "occ_gen":  torch.zeros((), device=device, dtype=dtype),
            "occ_match_loss": torch.zeros((), device=device, dtype=dtype),
        }
    
        # -------- L1 sparsity --------
        sparsity_l1_real = real.abs().mean()
        sparsity_l1_gen  = gen.abs().mean()
    
        out["sparsity_l1_real"] = sparsity_l1_real
        out["sparsity_l1_gen"]  = sparsity_l1_gen
        out["sparsity_match_loss"] = (sparsity_l1_gen - sparsity_l1_real).pow(2)
    
        # -------- soft occupancy --------
        tau  = float(getattr(self, "occ_tau", 0.0))
        temp = max(float(getattr(self, "occ_temp", 1.0)), 1e-6)
    
        act_real = torch.sigmoid((real.abs() - tau) / temp)
        act_gen  = torch.sigmoid((gen.abs()  - tau) / temp)
    
        occ_real = act_real.mean()
        occ_gen  = act_gen.mean()
    
        out["occ_real"] = occ_real
        out["occ_gen"]  = occ_gen
        out["occ_match_loss"] = (occ_gen - occ_real).pow(2)
    
        return out
    def get_ground_truth_layer_energies(self, condition):
        """
        Extract ground truth layer energies from condition
        
        Args:
            condition: (B, 46) where condition[:, 0] = E_inc, condition[:, 1:46] = u_features
        
        Returns:
            layer_energies: (B, 45)
        """
        # Extract u features
        u_features = condition[:, 1:46]  # (B, 45)
        E_inc = condition[:, 0:1]  # (B, 1)
        
        # Reconstruct layer energies from u features
        # (Same logic as in NormalizeByElayer reverse)
        u_clipped = u_features.clone()
        u_clipped[:, 1:] = torch.clip(u_clipped[:, 1:], 0, 1)
        
        layer_Es = []
        total_E = E_inc.flatten() * u_clipped[:, 0]
        cum_sum = torch.zeros_like(total_E)
        
        for i in range(44):  # 0 to 43
            layer_E = (total_E - cum_sum) * u_clipped[:, i+1]
            layer_Es.append(layer_E)
            cum_sum += layer_E
        layer_Es.append(total_E - cum_sum)
        
        layer_energies = torch.stack(layer_Es, dim=1)  # (B, 45)
        
        return layer_energies

    def compute_energy_loss(self, x_0_hat, condition):
        """
        Compute layer-wise energy loss
        
        Args:
            x_0_hat: (B, 1, 45, 16, 9) - predicted clean sample
            condition: (B, 46) - [E_inc, u_1, ..., u_45]
        
        Returns:
            energy_loss: scalar
        """
        # Get predicted layer energies from generated samples
        # Line 306 - FIX
        voxels_pred, energy_pred, E_true = self.apply_inverse_transforms(x_0_hat, condition)
        E_pred = self.compute_layer_energies_from_voxels(voxels_pred)  # (B, 45)
        
        # Get ground truth layer energies from condition
        
        # ========================================
        # Safety check: Make sure E_true was captured
        # ========================================
        if E_true is None:
            raise RuntimeError(
                "Layer energies not captured from NormalizeByElayer! "
                "Make sure NormalizeByElayer has 'self._last_layer_energies = layer_Es.detach()'"
            )
        # Compute Huber loss
        #energy_loss = F.huber_loss(E_pred, E_true, reduction='mean', delta=1.0)

        # Compute relative error: (predicted - true) / true
        relative_error = (E_pred - E_true) / (E_true.abs() + 1e-6)
        
        # Mean squared relative error
        energy_loss = torch.mean(relative_error ** 2)
        
        return energy_loss
    
    def _get_hlf_coords(self):
        """
        Load XMLHandler coordinates once and cache them on the model.
        Must be called before _shower_shape_guidance_loss.
        """
        if hasattr(self, '_hlf_eta') and self._hlf_eta is not None:
            return
    
        from challenge_files.XMLHandler import XMLHandler
        xml = XMLHandler('electron', filename=self.xml_file)
    
        self._hlf_bin_edges      = xml.GetBinEdges()
        eta_all, phi_all         = xml.GetEtaPhiAllLayers()
        self._hlf_eta_all_layers = eta_all
        self._hlf_phi_all_layers = phi_all
        self._hlf_layers_binned  = xml.GetLayersWithBinningInAlpha()
        self._hlf_eta            = True   # mark as loaded
        print(f"[HLF coords] Loaded {len(self._hlf_layers_binned)} active layers")
    
    
    def _shower_shape_guidance_loss(self, x0_pred, condition):
        """
        Differentiable shower shape loss for inference-time guidance.
    
        Mirrors HighLevelFeatures formulas exactly but stays in PyTorch
        so autograd can trace back to x_t.
    
        Computes per-layer EC Eta/Phi, Width Eta/Phi, Sparsity in
        physical MeV space and compares against precomputed Geant4
        reference stats loaded via load_reference_stats().
    
        Args:
            x0_pred   : (B, 1, 45, 16, 9)  predicted clean sample, log space
            condition : (B, 46)             conditioning tensor
    
        Returns:
            scalar loss tensor
        """
        self._get_hlf_coords()
    
        # Invert transforms to physical MeV space — graph stays intact
        voxels_phys, _, _ = self.apply_inverse_transforms(x0_pred, condition)
        voxels_phys = voxels_phys.clamp(min=0.0)   # (B, 6480) physical MeV
    
        B      = voxels_phys.shape[0]
        device = voxels_phys.device
        eps    = 1e-16
    
        loss     = torch.zeros((), device=device)
        n_layers = 0
    
        for l in self._hlf_layers_binned:
            start = self._hlf_bin_edges[l]
            end   = self._hlf_bin_edges[l + 1]
            E_l   = voxels_phys[:, start:end]   # (B, n_vox_l)
    
            if E_l.shape[1] == 0:
                continue
    
            # Physical mm coordinates for this layer
            eta_l = torch.tensor(
                self._hlf_eta_all_layers[l],
                dtype=voxels_phys.dtype, device=device
            ).unsqueeze(0)   # (1, n_vox_l)
    
            phi_l = torch.tensor(
                self._hlf_phi_all_layers[l],
                dtype=voxels_phys.dtype, device=device
            ).unsqueeze(0)
    
            E_sum = E_l.sum(dim=-1, keepdim=True) + eps   # (B, 1)
    
            # EC — mirrors HighLevelFeatures._calculate_EC
            ec_eta = (eta_l * E_l).sum(dim=-1) / E_sum.squeeze(-1)   # (B,)
            ec_phi = (phi_l * E_l).sum(dim=-1) / E_sum.squeeze(-1)
    
            # Width — mirrors HighLevelFeatures.GetECandWidths
            eta2_mean = (eta_l ** 2 * E_l).sum(dim=-1) / E_sum.squeeze(-1)
            phi2_mean = (phi_l ** 2 * E_l).sum(dim=-1) / E_sum.squeeze(-1)
            w_eta = torch.sqrt((eta2_mean - ec_eta ** 2).clamp(min=0.0) + eps)
            w_phi = torch.sqrt((phi2_mean - ec_phi ** 2).clamp(min=0.0) + eps)
    
            # Sparsity — differentiable approximation of (E > 0).mean()
            # sigmoid with small temp ≈ hard threshold but differentiable
            sp_temp    = float(self.params.get('shower_shape_sparsity_temp', 0.01))
            sparsity_l = torch.sigmoid(E_l / sp_temp).mean(dim=-1)   # (B,)
    
            # Compare against Geant4 reference values for this layer
            loss += (ec_eta    - self.ref_ec_eta[l]  ).pow(2).mean()
            loss += (ec_phi    - self.ref_ec_phi[l]  ).pow(2).mean()
            loss += (w_eta     - self.ref_w_eta[l]   ).pow(2).mean()
            loss += (w_phi     - self.ref_w_phi[l]   ).pow(2).mean()
            loss += (sparsity_l - self.ref_sparsity[l]).pow(2).mean()
            n_layers += 1
    
        return loss / max(n_layers, 1)

    def _voxel_cfd_guidance_loss(self, x0_pred, condition):
        """
        Differentiable voxel-CFD surrogate for inference-time guidance.
    
        Computes batch-wise voxel correlations between consecutive layers
        in physical MeV space and compares against precomputed Geant4
        reference voxel correlations.
    
        Args:
            x0_pred   : (B, 1, 45, 16, 9) predicted clean sample in model space
            condition : (B, 46)
    
        Returns:
            scalar loss tensor
        """
        eps = 1e-8
    
        # Invert transforms to physical MeV space
        voxels_phys, _, _ = self.apply_inverse_transforms(x0_pred, condition)
        voxels_phys = voxels_phys.clamp(min=0.0)
    
        # Convert to (B, 45, 16, 9)
        if voxels_phys.ndim == 2:
            voxels_phys = voxels_phys.reshape(voxels_phys.shape[0], 45, 16, 9)
        elif voxels_phys.ndim == 5:
            voxels_phys = voxels_phys[:, 0]
        elif voxels_phys.ndim == 4:
            pass
        else:
            raise ValueError(f"Unexpected voxels_phys shape: {voxels_phys.shape}")
    
        # Optional stabilization for huge physical dynamic range
        use_log1p = bool(self.params.get("voxel_cfd_guidance_log1p", True))
        if use_log1p:
            voxels_for_corr = torch.log1p(voxels_phys)
        else:
            voxels_for_corr = voxels_phys
    
        # Consecutive layer voxel pairs: (B, 44, 16, 9)
        x_i   = voxels_for_corr[:, :-1, :, :]
        x_ip1 = voxels_for_corr[:, 1:,  :, :]
    
        # Center over batch
        x_i_c   = x_i   - x_i.mean(dim=0, keepdim=True)
        x_ip1_c = x_ip1 - x_ip1.mean(dim=0, keepdim=True)
    
        cov = (x_i_c * x_ip1_c).mean(dim=0)
    
        std_i   = torch.sqrt((x_i_c ** 2).mean(dim=0) + eps)
        std_ip1 = torch.sqrt((x_ip1_c ** 2).mean(dim=0) + eps)
    
        corr_gen = cov / (std_i * std_ip1 + eps)
    
        corr_ref = self.ref_voxel_cfd.to(
            device=corr_gen.device,
            dtype=corr_gen.dtype
        )
    
        if hasattr(self, "ref_voxel_cfd_mask"):
            mask = self.ref_voxel_cfd_mask.to(device=corr_gen.device)
        else:
            mask = torch.isfinite(corr_ref)
    
        diff = corr_gen[mask] - corr_ref[mask]
    
        return (diff ** 2).mean()

    def load_voxel_cfd_reference(self, path):
        #import h5py
    
        with h5py.File(path, "r") as f:
            C_ref = f["C_ref_voxel"][:]
    
        C_ref = torch.tensor(C_ref, dtype=next(self.net.parameters()).dtype)
    
        self.ref_voxel_cfd = C_ref
        self.ref_voxel_cfd_mask = torch.isfinite(C_ref) & (C_ref.abs() > 1e-8)
    
        print("Loaded voxel CFD reference:", self.ref_voxel_cfd.shape)
        
    def load_reference_stats(self, ref_path: str):
        """
        Load precomputed Geant4 reference statistics for inference-time guidance.
        Call once after model.load(), before ddim_sample_guided().
    
        ref_path: path to .pt file produced by extract_reference_stats.py
        """
        ref = torch.load(ref_path, map_location=self.device, weights_only=True)
        self.ref_ec_eta      = ref['ec_eta'].to(self.device)
        self.ref_ec_phi      = ref['ec_phi'].to(self.device)
        self.ref_w_eta       = ref['w_eta'].to(self.device)
        self.ref_w_phi       = ref['w_phi'].to(self.device)
        self.ref_sparsity    = ref['sparsity'].to(self.device)
        self.ref_layers_used = ref['layers_binned']
        print(f"[Guidance] Reference stats loaded from {ref_path}")
        print(f"           {ref['n_samples']} Geant4 showers  |  "
              f"{len(ref['layers_binned'])} active layers  |  "
              f"space: {ref['space']}")
        
    def get_voxel_shape_loss_weight(self):
        """
        Weight schedule for voxel shape loss (separate from voxel energy).
        Mirrors get_voxel_loss_weight(), but uses different config keys.
        """
        if not self.params.get("use_voxel_shape_loss", False):
            return 0.0

        # ramp from 0 -> lambda_voxel_shape over warmup steps
        lambda_max = float(self.params.get("lambda_voxel_shape", 0.0))
        warmup_steps = int(self.params.get("voxel_shape_warmup_steps", 0))

        if warmup_steps <= 0:
            return lambda_max

        step = int(getattr(self, "opt_step", 0))
        frac = min(max(step / warmup_steps, 0.0), 1.0)
        return lambda_max * frac




    def compute_voxel_shape_loss_distribution(
        self,
        voxels_pred_phys,
        voxels_true_phys,
        H=16, W=9, L=45,
        eps=1e-12,
        frac_min=1e-4,
        huber_delta=1e-2,
        weighting_mode="sqrt",   # "linear", "sqrt", "uniform", "clamped"
        detach_pred_denom=True,  # True = safer/stabler gradients
        pred_denom_min=1e-8,     # clamp for S_pred to avoid blow-ups
        reduce="per_sample",     # "per_sample" | "global"
    ):
        """
        Distribution-style (shape-only) voxel loss.
    
        Core idea:
          - Normalize pred by its OWN layer sum (S_pred), true by its OWN layer sum (S_true)
          - Compare normalized maps per layer (Huber on voxel distributions)
          - Weight layers using TRUE energy fraction (frac_true) with flexible schemes
          - Mask layers with tiny true fraction (frac_true <= frac_min)
    
        Reduction:
          - reduce="per_sample": average per-sample normalized loss (more stable)
          - reduce="global":     one global normalization across batch (original behavior)
        """
    
        def to_BLHW(v):
            if v.dim() == 5:
                # (B, 1, L, H, W) -> (B, L, H, W)
                v = v.squeeze(1)
            elif v.dim() == 2:
                # (B, 6480) -> (B, L, H, W)
                v = v.view(v.size(0), L, H, W)
            elif v.dim() == 4:
                # already (B, L, H, W)
                pass
            else:
                raise ValueError(f"Unexpected shape: {tuple(v.shape)}")
            return v
    
        vp = to_BLHW(voxels_pred_phys)
        vt = to_BLHW(voxels_true_phys)
    
        # physical non-negativity
        vp = torch.clamp(vp, min=0.0)
        vt = torch.clamp(vt, min=0.0)
    
        # layer sums
        S_true = vt.sum(dim=(-1, -2))  # (B, L)
        S_pred = vp.sum(dim=(-1, -2))  # (B, L)
    
        # true energy fractions for weighting + masking
        E_true = S_true.sum(dim=1, keepdim=True)  # (B, 1)
        frac_true = S_true / (E_true + eps)       # (B, L)
        mask = (frac_true > frac_min).to(vt.dtype)  # (B, L)
    
        # distribution normalization (shape-only)
        denom_true = S_true.unsqueeze(-1).unsqueeze(-1) + eps  # (B, L, 1, 1)
    
        denom_pred = S_pred
        if detach_pred_denom:
            denom_pred = denom_pred.detach()
        denom_pred = denom_pred.clamp(min=pred_denom_min).unsqueeze(-1).unsqueeze(-1) + eps  # (B, L, 1, 1)
    
        P = vp / denom_pred
        Q = vt / denom_true
    
        # voxelwise huber, then average over spatial dims => (B, L)
        per_elem = F.huber_loss(P, Q, delta=huber_delta, reduction="none")
        per_layer = per_elem.mean(dim=(-1, -2))  # (B, L)
    
        # flexible layer weights (based on TRUE fractions)
        if weighting_mode == "linear":
            w = frac_true
        elif weighting_mode == "sqrt":
            w = torch.sqrt(frac_true + eps)
        elif weighting_mode == "uniform":
            w = torch.ones_like(frac_true)
        elif weighting_mode == "clamped":
            w = torch.clamp(frac_true, min=0.05, max=0.4)
        else:
            raise ValueError(f"Unknown weighting_mode: {weighting_mode}")
    
        weighted = per_layer * w * mask  # (B, L)
    
        if reduce == "per_sample":
            denom_per_sample = (w * mask).sum(dim=1).clamp(min=1.0)  # (B,)
            loss_per_sample = weighted.sum(dim=1) / denom_per_sample # (B,)
            loss = loss_per_sample.mean()
        elif reduce == "global":
            denom_global = (w * mask).sum().clamp(min=1.0)
            loss = weighted.sum() / denom_global
        else:
            raise ValueError(f"Unknown reduce: {reduce}. Use 'per_sample' or 'global'.")
    
        return loss


    def compute_voxel_shape_loss(self, x0_pred, x_real, condition):
        """
        Shape-only voxel loss in physical space:
        - invert transforms
        - compare normalized per-layer voxel distributions
        """
    
        voxels_pred_phys, _, _ = self.apply_inverse_transforms(x0_pred, condition)
        with torch.no_grad():
            voxels_true_phys, _, _ = self.apply_inverse_transforms(x_real, condition)
    
        H, W, L = 16, 9, 45
        eps = float(self.params.get("voxel_shape_eps", 1e-12))
        frac_min = float(self.params.get("voxel_shape_frac_min", 1e-4))
        huber_delta = float(self.params.get("voxel_shape_huber_delta", 1e-2))
        weighting_mode = str(self.params.get("voxel_shape_weighting_mode", "sqrt"))
        detach_pred_denom = bool(self.params.get("voxel_shape_detach_pred_denom", True))
        pred_denom_min = float(self.params.get("voxel_shape_pred_denom_min", 1e-8))
        reduce = str(self.params.get("voxel_shape_reduce", "per_sample"))
    
        return self.compute_voxel_shape_loss_distribution(
            voxels_pred_phys,
            voxels_true_phys,
            H=H, W=W, L=L,
            eps=eps,
            frac_min=frac_min,
            huber_delta=huber_delta,
            weighting_mode=weighting_mode,
            detach_pred_denom=detach_pred_denom,
            pred_denom_min=pred_denom_min,
            reduce=reduce,
        )


    def _normalize_by_layer_energy(self, voxels, layer_energies, eps=1e-8):
        """
        voxels: (B, 6480) or (B, 45, 16, 9) or (B, 1, 45, 16, 9)
        layer_energies: (B, 45)
        returns normalized voxels in shape (B, 45, 16, 9)
        """
        if voxels.dim() == 5:
            # (B, 1, 45, 16, 9) -> (B, 45, 16, 9)
            vox = voxels.squeeze(1)
        elif voxels.dim() == 2:
            # (B, 6480) -> (B, 45, 16, 9)
            vox = voxels.view(voxels.size(0), 45, 16, 9)
        elif voxels.dim() == 4:
            vox = voxels
        else:
            raise ValueError(f"Unexpected voxels shape: {tuple(voxels.shape)}")
    
        denom = layer_energies.unsqueeze(-1).unsqueeze(-1)  # (B, 45, 1, 1)
        return vox / (denom + eps)


    def compute_voxel_energy_loss(self, x_0_pred, x_true, condition, timestep=None):
        """
        Voxel-wise energy loss with configurable loss function
        
        Args:
            x_0_pred: (B, C, L, H, W) - predicted clean sample in preprocessed space
            x_true: (B, C, L, H, W) - true clean sample in preprocessed space
            condition: (B, 46) - [E_inc, u_1, ..., u_45]
            timestep: (B,) or (B, 1) - optional timestep for timestep-weighted loss
        
        Returns:
            voxel_loss: scalar loss value
        
        Supported loss types (set via config 'voxel_loss_type'):
            - 'huber': Huber loss (default) - robust to outliers, smooth gradients
            - 'mse': Mean Squared Error - strongest gradients
            - 'mae': Mean Absolute Error - robust, interpretable
            - 'relative': Relative error - scale-invariant, physically meaningful
            - 'log': Log-transformed MAE - scale-invariant (not recommended)
            - 'relative_mse': Relative MSE - scale-invariant with stronger gradients
            - 'weighted_mse': Energy-weighted MSE - emphasizes high-energy voxels
        """
        #import torch.nn.functional as F
        
        # ========================================
        # STEP 1: Convert to physical space
        # ========================================
        voxels_pred, _, layerE_pred = self.apply_inverse_transforms(x_0_pred, condition)
        
        with torch.no_grad():
            voxels_true, _, layerE_true = self.apply_inverse_transforms(x_true, condition)

        # 2) Normalize by per-layer energy (shape-focused)
        eps = self.params.get("voxel_loss_epsilon", 1e-8)

        v_pred_norm = self._normalize_by_layer_energy(voxels_pred, layerE_true, eps=eps)
        v_true_norm = self._normalize_by_layer_energy(voxels_true, layerE_true, eps=eps)
        
        
        # # ========================================
        # # STEP 2: Get loss function configuration
        # # ========================================
        loss_type = self.params.get('voxel_loss_type', 'huber')
        
        # ========================================
        # STEP 3: Compute loss based on type
        # ========================================
        # if loss_type == 'huber':
        #     delta = self.params.get('voxel_loss_delta', 1.0)
        #     voxel_loss = F.huber_loss(v_pred_norm, v_true_norm, delta=delta, reduction='mean')
    
        # elif loss_type == 'mse':
        #     voxel_loss = F.mse_loss(v_pred_norm, v_true_norm, reduction='mean')
    
        # elif loss_type == 'mae':
        #     voxel_loss = F.l1_loss(v_pred_norm, v_true_norm, reduction='mean')
    
        # else:
        #     raise ValueError(f"Unsupported with layer-normalization: {loss_type}")
    
        # # 4) Optional timestep weighting (keep as-is)
        # if self.params.get('use_voxel_timestep_weighting', False) and timestep is not None:
        #     if timestep.dim() == 2:
        #         timestep = timestep.squeeze(1)
        #     alpha_bar = self.alphas_cumprod.to(timestep.device)[timestep]
        #     t_weight = (1.0 - alpha_bar).mean()
        #     voxel_loss = voxel_loss * t_weight
    
        # return voxel_loss
        if loss_type == 'huber':
            # Huber Loss: Combines L1 and L2
            # For |error| < delta: L2 loss (smooth gradients)
            # For |error| > delta: L1 loss (robust to outliers)
            delta = self.params.get('voxel_loss_delta', 1.0)
            voxel_loss = F.huber_loss(voxels_pred, voxels_true, delta=delta, reduction='mean')
        
        elif loss_type == 'mse':
            # Mean Squared Error: (pred - true)^2
            # Strongest gradients, but sensitive to outliers
            voxel_loss = F.mse_loss(voxels_pred, voxels_true, reduction='mean')
        
        elif loss_type == 'mae':
            # Mean Absolute Error: |pred - true|
            # Robust to outliers, interpretable
            voxel_loss = F.l1_loss(voxels_pred, voxels_true, reduction='mean')
        
        elif loss_type == 'relative':
            # Relative Error: |pred - true| / |true|
            # Scale-invariant, physically meaningful
            epsilon = self.params.get('voxel_loss_epsilon', 1e-8)
            relative_error = torch.abs(voxels_pred - voxels_true) / (torch.abs(voxels_true) + epsilon)
            voxel_loss = torch.mean(relative_error)
        
        elif loss_type == 'log':
            # Log-Transformed MAE: |log(1 + pred) - log(1 + true)|
            # WARNING: Vanishing gradients for large values!
            log_pred = torch.log(1.0 + voxels_pred)
            log_true = torch.log(1.0 + voxels_true)
            voxel_loss = torch.mean(torch.abs(log_pred - log_true))
        
        elif loss_type == 'relative_mse':
            # Relative MSE: ((pred - true) / true)^2
            # Scale-invariant with stronger gradients than relative MAE
            epsilon = self.params.get('voxel_loss_epsilon', 1e-8)
            relative_error = (voxels_pred - voxels_true) / (torch.abs(voxels_true) + epsilon)
            voxel_loss = torch.mean(relative_error ** 2)
        
        elif loss_type == 'weighted_mse':
            # Weighted MSE: emphasizes high-energy voxels
            # Weight by normalized true energy
            weights = voxels_true / (voxels_true.sum() + 1e-10)
            squared_error = (voxels_pred - voxels_true) ** 2
            voxel_loss = torch.sum(weights * squared_error)
         
        elif loss_type == 'poisson':
            #-----------  NEW  -----------#
            # Poisson-Weighted Huber Loss
            # Motivation: Calorimeter energy deposits follow Poisson statistics,
            # where sigma ~ sqrt(E). Normalizing residuals by sqrt(E_true) gives
            # chi-like residuals that are physically meaningful.
            #
            # Effective weighting: voxels with E_true = 100 MeV are penalized
            # 10x more than voxels with E_true = 10000 MeV (between flat-E and logE).
            #
            # Sits between:
            #   - 'huber'        (flat, uniform weighting)
            #   - 'relative'     (1/E weighting, like logE)
            # and is grounded in detector physics.
            
            epsilon = self.params.get('voxel_loss_epsilon', 0.01)
            delta = self.params.get('voxel_loss_delta', 1.0)
            
            # Poisson std: sigma_i = sqrt(E_true_i)
            # Normalize residuals by expected fluctuation scale
            poisson_std = torch.sqrt(torch.abs(voxels_true) + epsilon)
            
            # Normalized residuals: (pred - true) / sqrt(true)
            normalized_residuals = (voxels_pred - voxels_true) / poisson_std
            
            # Apply Huber loss on normalized residuals for robustness
            # (pure MSE on normalized residuals would be chi-squared like)
            voxel_loss = F.huber_loss(
                normalized_residuals,
                torch.zeros_like(normalized_residuals),  # target is zero residual
                delta=delta,
                reduction='mean'
            )        
        
        else:
            raise ValueError(
                f"Unknown voxel_loss_type: {loss_type}. "
                f"Options: 'huber', 'mse', 'mae', 'relative', 'log', "
                f"'relative_mse', 'weighted_mse'"
            )
        
        # ========================================
        # STEP 4: Optional timestep weighting
        # ========================================
        if self.params.get('use_voxel_timestep_weighting', False):
            if timestep is not None:
                if timestep.dim() == 2:
                    timestep = timestep.squeeze(1)
                
                alpha_bar = self.alphas_cumprod.to(timestep.device)[timestep]
                t_weight = (1.0 - alpha_bar).mean()
                voxel_loss = voxel_loss * t_weight
        
        return voxel_loss
        
    def get_voxel_loss_weight(self):
        lambda_max   = float(self.params.get('lambda_voxel_energy',     0.12))
        lambda_min   = float(self.params.get('lambda_voxel_energy_min', 0.0))
        warmup_steps = int(self.params.get('voxel_loss_warmup_steps',   20850))
        full_start   = int(self.params.get('lambda_full_start_step',   104250))
        decay_start  = int(self.params.get('lambda_decay_start_step',  229350))
        decay_end    = int(self.params.get('lambda_decay_end_step',    312750))
        current_step = int(getattr(self, 'global_step', 0))
    
        # Phase 1: pure diffusion — no voxel loss (epochs 0-50)
        if current_step < warmup_steps:
           
            return 0.0
    
        # Phase 2: cosine ramp-up 0 -> lambda_max (epochs 50-250)
        if current_step < full_start:
            progress = (current_step - warmup_steps) / max(full_start - warmup_steps, 1)
            return lambda_max * 0.5 * (1.0 - np.cos(np.pi * progress))
    
        # Phase 3: full lambda — constant (epochs 250-550)
        if current_step < decay_start:
            return lambda_max
    
        # Phase 4: cosine ramp-down lambda_max -> lambda_min (epochs 550-750)
        if current_step < decay_end:
            progress = (current_step - decay_start) / max(decay_end - decay_start, 1)
            return lambda_min + (lambda_max - lambda_min) * 0.5 * (1.0 + np.cos(np.pi * progress))
    
        # Phase 5: pure diffusion fine-tune (epochs 750-800)
        return lambda_min
    # def get_voxel_loss_weight(self):
    #     """
    #     Dynamic lambda with warmup AND decay
    #     """
        #     # Your current config values
    #     lambda_max = self.params.get('lambda_voxel_energy', 1.0e-4)
    #     warmup_steps = self.params.get('voxel_loss_warmup_steps', 50000)
        
    #     # NEW: Add decay parameters
    #     decay_start = self.params.get('lambda_decay_start_step', 200000)  # ~epoch 200 depends
    #     decay_end = self.params.get('lambda_decay_end_step', 600000)      # ~epoch 600
    #     lambda_min = self.params.get('lambda_voxel_energy_min', 2.0e-5)  # Final value

    #     # Get current step inside the function
    #     current_step = getattr(self, 'global_step', 0)  # ← Add this line

    #     # Phase 1: Warmup (0 to warmup_steps)
    #     if current_step < warmup_steps:
    #         warmup_progress = current_step / warmup_steps
    #         weight = lambda_max * warmup_progress
        
    #     # Phase 2: Constant (warmup_steps to decay_start)
    #     elif current_step < decay_start:
    #         weight = lambda_max
        
    #     # Phase 3: Decay (decay_start to decay_end)
    #     elif current_step < decay_end:
    #         decay_progress = (current_step - decay_start) / (decay_end - decay_start)
    #         # Linear decay
    #         weight = lambda_max - (lambda_max - lambda_min) * decay_progress
    #         # Or cosine decay for smoother transition:
    #         # weight = lambda_min + (lambda_max - lambda_min) * 0.5 * (1 + np.cos(np.pi * decay_progress))
        
    #     # Phase 4: Min value (after decay_end)
    #     else:
    #         weight = lambda_min
        
    #     return weight
        
    def compute_layer_energies_from_voxels(self, voxels):
        """
        Sum voxels per layer to get layer energies
        
        Args:
            voxels: (B, 6480) - fully reconstructed shower voxels in physical space
        
        Returns:
            layer_energies: (B, 45) - energy sum per layer
        """
        from itertools import pairwise
        
        layer_energies = []
        
        # Load layer boundaries
        
        xml = XMLHandler.XMLHandler('electron', XMLHandler)
        layer_boundaries = np.unique(xml.GetBinEdges())
        
        # Sum each layer
        for start, end in pairwise(layer_boundaries):
            layer = voxels[:, start:end]  # Select layer voxels
            layer_energy = layer.sum(dim=-1)  # Sum across voxels in layer
            layer_energies.append(layer_energy)
        
        layer_energies = torch.stack(layer_energies, dim=1)  # (B, 45)
        
        return layer_energies

    def apply_inverse_transforms(self, x, condition=None):
        """
        Apply inverse preprocessing transforms to get layer energies
        
        Args:
            x: (B, C, L, H, W) - preprocessed data from model (e.g., B, 1, 45, 16, 9)
            condition: (B, 46) - [E_inc, u_1, u_2, ..., u_45] where:
                - condition[:, 0] is incident energy (after ScaleEnergy and LogEnergy)
                - condition[:, 1:46] are the 45 u features (after StandardizeFromFile)
        
        Returns:
            voxels of shape(B,6480) and energy (B,1)
        
        This applies reverse transforms up to NormalizeByElayer which directly
        computes layer energies from the u features.
        """
        
        if condition is None:
            raise ValueError("Condition is required for inverse transforms!")
        
        shower = x
        energy = condition
        
        # ========================================
        # REVERSE TRANSFORMATION PIPELINE
        # ========================================
        
        # 10. Reshape (reverse): (B, 1, 45, 16, 9) → (B, 6480)
        transform = Reshape_X(shape=[1, 45, 16, 9])
        shower, energy = transform(shower, energy, rev=True)
        # shower: (B, 6480), energy: (B, 46)
        
        # 9. AddFeaturesToCond (reverse): Add u features back to shower
        transform = AddFeaturesToCond_X(split_index=6480)
        
        shower, energy = transform(shower, energy, rev=True)
        # shower: (B, 6525) = (B, 6480 + 45), energy: (B, 1)
        
        # 8. ScaleEnergy (reverse): Unnormalize incident energy
        transform = ScaleEnergy_X(e_min=6.907755, e_max=13.815510)
        shower, energy = transform(shower, energy, rev=True)
        # energy is now in log-space
        
        # 7. LogEnergy (reverse): exp(energy) to get linear energy
        transform = LogEnergy_X()
        shower, energy = transform(shower, energy, rev=True)
        # energy is now in linear space (MeV)
        
        # 6. StandardizeFromFile (reverse): Unstandardize last 45 features
        # Note: You need to provide the model_dir where stats are saved
        
        transform = StandardizeFromFile_X(model_dir=self.stats_dir)
        shower, energy = transform(shower, energy, rev=True)
        # shower[:, -45:] are now unstandardized u features
        
        # 5. ExclusiveLogitTransform (reverse): Apply sigmoid with rescaling
        transform = ExclusiveLogitTransform_X(delta=1.0e-6, rescale=True)
        shower, energy = transform(shower, energy, rev=True)
        # u features are now in [0, 1] range
        
        # 4. SelectiveUniformNoise (reverse): Remove noise (cut applied)
        transform = SelectiveUniformNoise_X(
            noise_width=0.0e-6,
            cut=True,
            exclusions=[-45, -44, -43, -42, -41, -40, -39, -38, -37,
                        -36, -35, -34, -33, -32, -31, -30, -29, -28,
                        -27, -26, -25, -24, -23, -22, -21, -20, -19,
                        -18, -17, -16, -15, -14, -13, -12, -11, -10,
                        -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1]
        )
        shower, energy = transform(shower, energy, rev=True)
        
        # 3. CutValues (reverse): Restore cut values
        transform = CutValues_X(cut=1.0e-7)
        shower, energy = transform(shower, energy, rev=True)
        
        # 2. NormalizeByElayer (reverse): COMPUTE LAYER ENERGIES
        # This is the KEY step - it reconstructs layer energies from u features
        
        transform = NormalizeByElayer_X(
            ptype='electron',
            xml_file=self.xml_file,
            return_layer_energies=False
        )
        voxels, energy = transform(shower, energy, rev=True)

        if hasattr(transform, '_last_layer_energies'):
            layer_energies = transform._last_layer_energies
        else:
            print("Warning: Could not capture layer energies from NormalizeByElayer")
        # Continue with remaining transforms...
        transform = ScaleVoxels_X(factor=0.35)
        samples, energy = transform(voxels, energy, rev=True)
        
        return samples, energy,layer_energies

    def compute_auxiliary_losses(self, x0_pred, x_real, condition):
        """
        Compute ALL auxiliary losses (always, regardless of flags)
        Flags only control whether they're used in optimization, not computation!
        
        Args:
            x0_pred: (B, 1, 45, 16, 9) - predicted clean sample
            x_real: (B, 1, 45, 16, 9) - real clean sample
            condition: (B, 46) - [E_inc, u_1, ..., u_45] or None
        
        Returns:
            aux_loss_tensors: dict of loss tensors (for gradient computation)
            aux_loss_scalars: dict of scalar values (for logging)
        """
        device = x_real.device
        dtype = x_real.dtype
        
        aux_loss_tensors = {}
        aux_loss_scalars = {}
        
        # ========================================
        # 1. ENERGY CONSERVATION LOSS (ALWAYS compute)
        # ========================================
        # energy_loss = torch.zeros((), device=device, dtype=dtype)
        
        # if condition is not None:  # Only check if condition exists
        #     try:
        #         energy_loss = self.compute_energy_loss(x0_pred, condition)
        #         aux_loss_tensors['energy_loss'] = energy_loss
        #         aux_loss_scalars['energy_loss'] = energy_loss.item()
        #     except Exception as e:
        #         print(f"Warning: Energy loss computation failed: {e}")
        #         aux_loss_scalars['energy_loss'] = 0.0
        # else:
        #     aux_loss_scalars['energy_loss'] = 0.0
        
        # ========================================
        # 2. MOMENT MATCHING LOSS (ALWAYS compute)
        # ========================================
        voxels_pred, _, _ = self.apply_inverse_transforms(x0_pred, condition)
        
        with torch.no_grad():
            voxels_true, _, _ = self.apply_inverse_transforms(x_real, condition)
        mu_real = voxels_true.mean(dim=(-1, -2))
        mu_gen = voxels_pred.mean(dim=(-1, -2))
        var_real = voxels_true.var(dim=(-1, -2), unbiased=False)
        var_gen = voxels_pred.var(dim=(-1, -2), unbiased=False)
        
        moment_loss = (torch.mean((mu_gen - mu_real) ** 2) + 
                      torch.mean((var_gen - var_real) ** 2))
        
        aux_loss_tensors['moment_loss'] = moment_loss
        aux_loss_scalars['moment_loss'] = moment_loss.item()
        
        # ========================================
        # 3. SPARSITY & OCCUPANCY LOSS (ALWAYS compute)
        # ========================================
        sparsity_stats = self.compute_sparsity_occupancy(real=x_real, gen=x0_pred)
        sparsity_loss = sparsity_stats['sparsity_match_loss']

        # ========================================
        # 4. Voxel Energy LOSS (ALWAYS compute)
        # ========================================  
        if condition is not None:
            try:
                # Call without timestep (will use default=None)
                voxel_energy_loss = self.compute_voxel_energy_loss(
                    x0_pred, x_real, condition  # Only 3 args, no timestep!
                )

                # Get warmup weight (increases from 0 to lambda_max)
                weight = self.get_voxel_loss_weight()
               
                # Store weighted loss for optimization
                # This is what backprop will use
                # Compute weighted loss once
                weighted_voxel_loss = voxel_energy_loss * weight
                aux_loss_scalars['voxel_energy_loss_unweighted'] = voxel_energy_loss.item()
                
                # Store for optimization
                aux_loss_tensors['voxel_energy_loss'] = weighted_voxel_loss

                # Optional: also log the ratio to verify multiplication
                if weight > 0:
                    aux_loss_scalars['voxel_energy_loss_ratio'] = weighted_voxel_loss.item() / voxel_energy_loss.item()
                
                # Store for logging
                aux_loss_scalars['voxel_energy_loss'] = weighted_voxel_loss.item()
               
                aux_loss_scalars['voxel_energy_weight'] = weight  # Track weight schedule
            except Exception as e:
                print(f"Warning: Voxel energy loss computation failed: {e}")
                import traceback
                traceback.print_exc()
                aux_loss_scalars['voxel_energy_loss'] = 0.0
        else:
            aux_loss_scalars['voxel_energy_loss'] = 0.0
                # ========================================
        # 5. Voxel SHAPE LOSS (ALWAYS compute; weight may be 0)
        # ========================================
        if condition is not None:
            try:
                voxel_shape_loss = self.compute_voxel_shape_loss(x0_pred, x_real, condition)

                shape_w = self.get_voxel_shape_loss_weight()
                weighted_voxel_shape = voxel_shape_loss * shape_w

                # store tensor for optimization (even if weight=0, keeps pattern consistent)
                aux_loss_tensors["voxel_shape_loss"] = weighted_voxel_shape

                # logging
                aux_loss_scalars["voxel_shape_loss_unweighted"] = voxel_shape_loss.item()
                aux_loss_scalars["voxel_shape_loss"] = weighted_voxel_shape.item()
                aux_loss_scalars["voxel_shape_weight"] = shape_w

            except Exception as e:
                print(f"Warning: Voxel shape loss computation failed: {e}")
                import traceback
                traceback.print_exc()
                aux_loss_scalars["voxel_shape_loss_unweighted"] = 0.0
                aux_loss_scalars["voxel_shape_loss"] = 0.0
                aux_loss_scalars["voxel_shape_weight"] = 0.0
        else:
            aux_loss_scalars["voxel_shape_loss_unweighted"] = 0.0
            aux_loss_scalars["voxel_shape_loss"] = 0.0
            aux_loss_scalars["voxel_shape_weight"] = 0.0

        
        aux_loss_tensors['sparsity_loss'] = sparsity_loss
        
        # Log all sparsity statistics
        aux_loss_scalars['sparsity_l1_real'] = sparsity_stats['sparsity_l1_real'].item()
        aux_loss_scalars['sparsity_l1_gen'] = sparsity_stats['sparsity_l1_gen'].item()
        aux_loss_scalars['sparsity_match_loss'] = sparsity_stats['sparsity_match_loss'].item()
        aux_loss_scalars['occ_real'] = sparsity_stats['occ_real'].item()
        aux_loss_scalars['occ_gen'] = sparsity_stats['occ_gen'].item()
        aux_loss_scalars['occ_match_loss'] = sparsity_stats['occ_match_loss'].item()


        
              
        return aux_loss_tensors, aux_loss_scalars

    #### if anything bad happens, reason would be the following codes #####

    def compute_auxiliary_losses_log_space(self, x0_pred, x_real, condition):
        """
        Compute ALL auxiliary losses in preprocessed (log) space,
        consistent with the diffusion loss scale.
        
        Args:
            x0_pred: (B, 1, 45, 16, 9) - predicted clean sample
            x_real:  (B, 1, 45, 16, 9) - real clean sample
            condition: (B, 46) - [E_inc, u_1, ..., u_45] or None
        
        Returns:
            aux_loss_tensors: dict of loss tensors (for gradient computation)
            aux_loss_scalars: dict of scalar values (for logging)
        """
        device = x_real.device
        dtype  = x_real.dtype

        aux_loss_tensors = {}
        aux_loss_scalars = {}

        # Squeeze channel dim: (B, 1, 45, 16, 9) → (B, 45, 16, 9)
        voxels_pred = x0_pred.squeeze(1)
        with torch.no_grad():
            voxels_true = x_real.squeeze(1)

        # ========================================
        # 1. MOMENT MATCHING LOSS
        # ========================================
        mu_real  = voxels_true.mean(dim=(-1, -2))
        mu_gen   = voxels_pred.mean(dim=(-1, -2))
        var_real = voxels_true.var(dim=(-1, -2), unbiased=False)
        var_gen  = voxels_pred.var(dim=(-1, -2), unbiased=False)

        moment_loss = (torch.mean((mu_gen  - mu_real) ** 2) +
                       torch.mean((var_gen - var_real) ** 2))

        aux_loss_tensors['moment_loss'] = moment_loss
        aux_loss_scalars['moment_loss'] = moment_loss.item()

        # ========================================
        # 2. SPARSITY & OCCUPANCY LOSS
        # ========================================
        sparsity_stats = self.compute_sparsity_occupancy(real=x_real, gen=x0_pred)
        sparsity_loss  = sparsity_stats['sparsity_match_loss']

        # ========================================
        # 3. VOXEL ENERGY LOSS (Poisson-Weighted Huber in preprocessed space)
        # BUG FIX 1: use voxels_pred/voxels_true (squeezed), not x0_pred/x0_real
        # BUG FIX 2: epsilon, delta, weight were undefined — fetch from params
        # ========================================
        epsilon = self.params.get('voxel_loss_epsilon', 1e-6)   # FIX 3
        delta   = self.params.get('voxel_loss_delta',   1.0)    # FIX 4
        weight  = self.get_voxel_loss_weight()                   # FIX 5

        residuals        = voxels_pred - voxels_true             # FIX 1: (B, 45, 16, 9)
        poisson_std      = torch.sqrt(torch.abs(voxels_true) + epsilon)  # FIX 2
        normalized_resid = residuals / poisson_std

        voxel_energy_loss = F.huber_loss(
            normalized_resid,
            torch.zeros_like(normalized_resid),
            delta=delta,
            reduction='mean'
        )

        weighted_voxel_loss = voxel_energy_loss * weight
        aux_loss_tensors['voxel_energy_loss_unweighted'] = voxel_energy_loss  # raw tensor for grad
        aux_loss_tensors['voxel_energy_loss']            = weighted_voxel_loss
        aux_loss_scalars['voxel_energy_loss_unweighted'] = voxel_energy_loss.item()
        aux_loss_scalars['voxel_energy_loss']            = weighted_voxel_loss.item()
        aux_loss_scalars['voxel_energy_weight']          = weight

        if weight > 0:
            aux_loss_scalars['voxel_energy_loss_ratio'] = (
                weighted_voxel_loss.item() / voxel_energy_loss.item()
            )
        # ========================================
        # 4. LAPLACIAN SMOOTHNESS LOSS
        # ========================================
        if self.L is not None:
            
            lap_eps    = self.params.get('laplacian_loss_eps',    0.1)
            #lap_weight = self.params.get('laplacian_loss_weight', 0.01)
            lap_weight = self.get_laplacian_loss_weight()
        
            # Pass x0_pred and x_real directly — function handles the squeeze internally
            lap_loss = self.laplacian_loss_log(voxels_pred, voxels_true, self.L)
            weighted_lap_loss = lap_loss * lap_weight
            
            aux_loss_tensors['laplacian_loss_unweighted'] = lap_loss
            aux_loss_tensors['laplacian_loss']            = weighted_lap_loss
            aux_loss_scalars['laplacian_loss_unweighted'] = lap_loss.item()
            aux_loss_scalars['laplacian_loss']            = weighted_lap_loss.item()
            aux_loss_scalars['laplacian_loss_weight']     = lap_weight
        else:
            aux_loss_scalars['laplacian_loss_unweighted'] = 0.0
            aux_loss_scalars['laplacian_loss']            = 0.0
            aux_loss_scalars['laplacian_loss_weight']     = 0.0
            
        # ========================================
        # Sparsity logging
        # ========================================
        aux_loss_tensors['sparsity_loss']       = sparsity_loss
        aux_loss_scalars['sparsity_l1_real']    = sparsity_stats['sparsity_l1_real'].item()
        aux_loss_scalars['sparsity_l1_gen']     = sparsity_stats['sparsity_l1_gen'].item()
        aux_loss_scalars['sparsity_match_loss'] = sparsity_stats['sparsity_match_loss'].item()
        aux_loss_scalars['occ_real']            = sparsity_stats['occ_real'].item()
        aux_loss_scalars['occ_gen']             = sparsity_stats['occ_gen'].item()
        aux_loss_scalars['occ_match_loss']      = sparsity_stats['occ_match_loss'].item()

        return aux_loss_tensors, aux_loss_scalars

    def laplacian_loss_log(self, x_pred, x_true, L):
        """
        x_pred, x_true : (B, 45, 16, 9) — already in log/preprocessed space
        L               : preloaded sparse Laplacian on GPU, shape (6480, 6480)
        """
        B = x_pred.shape[0]
        N = 45 * 16 * 9
    
        # Data is already preprocessed — use directly
        pred_flat = x_pred.reshape(B, N)   # (B, N)
        true_flat = x_true.reshape(B, N)   # (B, N)
    
        # sparse mm: (N, N) @ (N, B) → (N, B) → transpose → (B, N)
        Lx_pred = torch.sparse.mm(L, pred_flat.T).T
        Lx_true = torch.sparse.mm(L, true_flat.T).T
    
        return F.mse_loss(Lx_pred, Lx_true)/self.lambda_max_sq

    def get_laplacian_loss_weight(self):
        """
        Laplacian loss weight.

        Flat by default (returns laplacian_loss_weight, the original behavior).
        Set use_laplacian_schedule: true in the config to enable a voxel-style
        5-phase cosine schedule using the lambda_lap_* keys below. To revert to
        flat weight, set use_laplacian_schedule: false (or remove it).
        """
        flat_weight = float(self.params.get('laplacian_loss_weight', 0.01))

        # default OFF: identical to the original flat-weight behavior
        if not self.params.get('use_laplacian_schedule', False):
            return flat_weight

        # scheduled mode: mirrors get_voxel_loss_weight, laplacian-specific keys
        lambda_max   = float(self.params.get('lambda_laplacian_max',
                                             flat_weight))
        lambda_min   = float(self.params.get('lambda_laplacian_min',   0.0))
        warmup_steps = int(self.params.get('laplacian_warmup_steps',   20850))
        full_start   = int(self.params.get('laplacian_full_start_step', 104250))
        decay_start  = int(self.params.get('laplacian_decay_start_step', 229350))
        decay_end    = int(self.params.get('laplacian_decay_end_step',  312750))
        current_step = int(getattr(self, 'global_step', 0))

        # Phase 1: pure diffusion, no laplacian loss
        if current_step < warmup_steps:
            return 0.0
        # Phase 2: cosine ramp-up 0 -> lambda_max
        if current_step < full_start:
            progress = (current_step - warmup_steps) / max(full_start - warmup_steps, 1)
            return lambda_max * 0.5 * (1.0 - np.cos(np.pi * progress))
        # Phase 3: full lambda, constant
        if current_step < decay_start:
            return lambda_max
        # Phase 4: cosine ramp-down lambda_max -> lambda_min
        if current_step < decay_end:
            progress = (current_step - decay_start) / max(decay_end - decay_start, 1)
            return lambda_min + (lambda_max - lambda_min) * 0.5 * (1.0 + np.cos(np.pi * progress))
        # Phase 5: pure diffusion fine-tune
        return lambda_min
        
    def batch_loss(self, x):
        """
        Calculate batch loss for diffusion model
        
        Returns:
            total_loss: Combined diffusion + auxiliary losses for optimization
            loss_dict: Dictionary of all loss components for logging
        """
        # ========================================
        # SETUP
        # ========================================
        x, condition, weights = self.get_condition_and_input(x)
        
        if self.latent:
            x = self.ae.encode(x, condition)
            if self.ae.kl:
                x = self.ae.reparameterize(x[0], x[1])
        
        assert torch.all(torch.isfinite(x)), "NaN or Inf detected in input x"
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device).long()
        t = t.unsqueeze(1)
        

        
        # Add noise to input if enabled
        if self.add_noise:
            x = x + self.gamma * torch.randn_like(x, device=x.device, dtype=x.dtype)
        
        # ========================================
        # DIFFUSION FORWARD PASS
        # ========================================
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise=noise)
        self.net.kl = 0
        
        # Get prediction and x0_pred based on prediction type
        if self.prediction_type == "noise":
            predicted_noise = self.net(x_noisy, t, condition)
            target = noise
            x0_pred = self.predict_start_from_noise(x_noisy, t, predicted_noise)
            diffusion_loss = torch.mean((predicted_noise - target) ** 2)
            
        elif self.prediction_type == "x0":
            predicted_x0 = self.net(x_noisy, t, condition)
            target = x
            x0_pred = predicted_x0
            diffusion_loss = torch.mean((predicted_x0 - target) ** 2)
            
        elif self.prediction_type == "v":
            alpha_bar = self.alphas_cumprod.to(t.device)[t].view(-1, *[1] * (x_noisy.ndim - 1))
            sqrt_alpha = torch.sqrt(alpha_bar)
            sqrt_one_minus_alpha = torch.sqrt(1. - alpha_bar)
            
            v_target = sqrt_alpha * noise + sqrt_one_minus_alpha * x
            v_pred = self.net(x_noisy, t, condition)
            target = v_target
            x0_pred = sqrt_alpha * x_noisy - sqrt_one_minus_alpha * v_pred
            diffusion_loss = torch.mean((v_pred - target) ** 2)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # ========================================
        # AUXILIARY LOSSES
        # ========================================
        # aux_loss_tensors, aux_loss_scalars = self.compute_auxiliary_losses(
        #     x0_pred=x0_pred,
        #     x_real=x,
        #     condition=condition
        # )

        # aux_loss_tensors = {}
        # aux_loss_scalars = {}
            
        # if self.params.get('use_raw_poisson_weight', False):
        #     vox_loss, vox_info = self.compute_poisson_voxel_loss(
        #         voxels_pred=x0_pred,    # predicted clean sample, preprocessed space
        #         voxels_true=x,          # true clean sample, preprocessed space
        #         x_real=x,               # inverted internally -> physical MeV for the weight
        #         condition=condition,
        #         apply_lambda=False,     # raw term; grad-blend/DWA applies the weight
        #     )
        #     aux_loss_tensors['voxel_energy_loss'] = vox_loss
        #     aux_loss_scalars.update(vox_info)
        # else:
        aux_loss_tensors, aux_loss_scalars = self.compute_auxiliary_losses_log_space(
            x0_pred=x0_pred,
            x_real=x,
            condition=condition
        )
        
        # ========================================
        # COMBINE INTO DICTIONARIES
        # ========================================
        loss_tensors = {
            'diffusion_loss': diffusion_loss,
            **aux_loss_tensors  # energy_loss, moment_loss, sparsity_loss
        }
        
        loss_scalars = {
            'diffusion_loss': diffusion_loss.item(),
            'diffusion_loss_tracked': diffusion_loss.item(),
            **aux_loss_scalars
        }
        
        return loss_tensors, loss_scalars  
        
       
 
    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from noise prediction
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(x_t.device)[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(x_t.device)[t]
        
        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_t.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
        while len(sqrt_one_minus_alphas_cumprod_t.shape) < len(x_t.shape):
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        posterior_mean_coef1 = self.posterior_mean_coef1.to(x_t.device)[t]
        posterior_mean_coef2 = self.posterior_mean_coef2.to(x_t.device)[t]
        posterior_variance = self.posterior_variance.to(x_t.device)[t]
        
        # Reshape for broadcasting
        while len(posterior_mean_coef1.shape) < len(x_t.shape):
            posterior_mean_coef1 = posterior_mean_coef1.unsqueeze(-1)
        while len(posterior_mean_coef2.shape) < len(x_t.shape):
            posterior_mean_coef2 = posterior_mean_coef2.unsqueeze(-1)
        while len(posterior_variance.shape) < len(x_t.shape):
            posterior_variance = posterior_variance.unsqueeze(-1)
        
        posterior_mean = (
            posterior_mean_coef1 * x_start + 
            posterior_mean_coef2 * x_t
        )
        #added to get rid of autoregressive error
        posterior_variance = posterior_variance.expand_as(x_t)
        
        return posterior_mean, posterior_variance

    def p_mean_variance(self, x_t, t, condition=None):
        """
        Apply the model to get p(x_{t-1} | x_t)
        """
        #model_output = self.net(condition, x_t, t,x=None, rev=False,  )
        
        if getattr(self, 'cfg', False):  # only use CFG if explicitly enabled
            eps_cond = self.net(x_t, t, condition)
            eps_uncond = self.net(x_t, t, None)
            guidance_scale = getattr(self, 'cfg_scale', 1.0)
            model_output = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            model_output = self.net(x_t, t, condition)
            
        if self.prediction_type == "noise":
            # Model predicts noise
            x_start = self.predict_start_from_noise(x_t, t, model_output)
        elif self.prediction_type == "x0":
            # Model predicts x_0
            x_start = model_output
        elif self.prediction_type == "v":
            self.alphas_cumprod = self.alphas_cumprod.to(self.device)
            alpha_t = self.alphas_cumprod[t].view(-1, *[1] * (x_t.ndim - 1))
            sqrt_alpha = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha = torch.sqrt(1. - alpha_t)
            x_start = sqrt_alpha * x_t - sqrt_one_minus_alpha * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        model_mean, model_variance = self.q_posterior_mean_variance(x_start, x_t, t)
        
        return model_mean, model_variance, x_start

    def p_sample(self, x_t, t, condition=None):
        """
        Sample x_{t-1} from the model
        """
        model_mean, model_variance, x_start = self.p_mean_variance(x_t, t, condition)
        
        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = (t != 0).float()
        while len(nonzero_mask.shape) < len(x_t.shape):
            nonzero_mask = nonzero_mask.unsqueeze(-1)
        
        sample = model_mean + nonzero_mask * torch.sqrt(model_variance) * noise
        return sample

    @torch.inference_mode()
    def sample_batch(self, batch):
        """
        Generate samples using the reverse diffusion process
        """
        #print("AM I here??")
        dtype = batch.dtype
        device = batch.device
        
        # Start from pure noise
        x = torch.randn((batch.shape[0], *self.shape), dtype=dtype, device=device)
        #print("shape of x: ",x.shape)
        
        # Reverse diffusion process
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch.shape[0],), i, device=device, dtype=torch.long)
            t=t.unsqueeze(1)
            x = self.p_sample(x, t, batch if isinstance(batch, torch.Tensor) else None)
        
        if self.latent:
            # decode the generated sample
            x = self.ae.decode(x, batch)
            
        return x
    

    def ddim_sample(self, condition, eta: float = 0.0, ddim_steps: int = 50):
        """
        DDIM sampling (epsilon/noise prediction) consistent with your p_mean_variance() path.
    
        Args:
            condition: torch.Tensor of shape (B, 46) for conditional sampling
            eta: 0.0 => deterministic DDIM, >0 => stochastic DDIM
            ddim_steps: number of DDIM steps (<= self.timesteps)
        """
        assert isinstance(condition, torch.Tensor), "condition must be a torch.Tensor"
        assert condition.dim() == 2 and condition.shape[1] == 46, \
            f"condition must have shape (B,46), got {tuple(condition.shape)}"
        assert ddim_steps > 0, "ddim_steps must be positive"
    
        device = condition.device
        dtype = condition.dtype
        T = int(self.timesteps)
    
        if ddim_steps > T:
            ddim_steps = T
    
        # Start from noise
        x = torch.randn((condition.shape[0], *self.shape), device=device, dtype=dtype)
    
        # Schedules on correct device (DO NOT assume they are already on GPU)
        alphas_cumprod = self.alphas_cumprod.to(device=device)
    
        # dtype-safe scalar constants
        one = torch.ones((), device=device, dtype=alphas_cumprod.dtype)
        eps = 1e-12  # numeric stability
    
        # Build decreasing timesteps in [T-1 ... 0]
        ts = torch.linspace(T - 1, 0, steps=ddim_steps, device=device).round().long()
    
        # Remove consecutive duplicates (linspace->round->long can duplicate indices)
        if ts.numel() > 1:
            keep = torch.ones_like(ts, dtype=torch.bool)
            keep[1:] = ts[1:] != ts[:-1]
            ts = ts[keep]
    
        # Ensure ends with 0
        if ts[-1].item() != 0:
            ts = torch.cat([ts, torch.tensor([0], device=device, dtype=torch.long)], dim=0)
    
        # Ensure strictly decreasing schedule (debug-time guardrail)
        if ts.numel() > 1:
            assert torch.all(ts[:-1] > ts[1:]), \
                f"DDIM schedule not strictly decreasing. Head={ts[:10].tolist()} Tail={ts[-10:].tolist()}"
    
        # DDIM reverse process
        for i in range(ts.numel()):
            t_int = int(ts[i].item())
    
            # "Previous" timestep in schedule
            if i < ts.numel() - 1:
                t_prev_int = int(ts[i + 1].item())
                a_prev = alphas_cumprod[t_prev_int]
            else:
                # For the final update, conceptually alpha_prev = 1
                a_prev = one
    
            a_t = alphas_cumprod[t_int]
    
            # timestep tensor shaped (B,1) to match sample_batch()/p_mean_variance()
            t_batch = torch.full(
                (condition.shape[0],),
                t_int,
                device=device,
                dtype=torch.long
            ).unsqueeze(1)
    
            # Predict eps (respect CFG flag like p_mean_variance)
            if getattr(self, "cfg", False):
                eps_cond = self.net(x, t_batch, condition)
                eps_uncond = self.net(x, t_batch, None)
                guidance_scale = getattr(self, "cfg_scale", 1.0)
                eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = self.net(x, t_batch, condition)
    
            # Predict x0 using the same helper as DDPM
            x0_pred = self.predict_start_from_noise(x, t_batch, eps_pred)
    
            # DDIM sigma:
            # sigma_t = eta * sqrt((1-a_prev)/(1-a_t)) * sqrt(1 - a_t/a_prev)
            one_minus_a_t = torch.clamp(one - a_t, min=eps)
            one_minus_a_prev = torch.clamp(one - a_prev, min=0.0)
            ratio = torch.clamp(a_t / torch.clamp(a_prev, min=eps), min=0.0, max=1.0)
    
            sigma_t = eta * torch.sqrt(one_minus_a_prev / one_minus_a_t) * torch.sqrt(
                torch.clamp(one - ratio, min=0.0)
            )
    
            # Direction coefficient: sqrt(1 - a_prev - sigma_t^2)
            coeff_eps = torch.sqrt(torch.clamp(one - a_prev - sigma_t**2, min=0.0))
    
            # Noise (skip on final step)
            if eta > 0.0 and i < ts.numel() - 1:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
    
            # Update: x_prev = sqrt(a_prev)*x0_pred + coeff_eps*eps + sigma_t*z
            sqrt_a_prev = torch.sqrt(torch.clamp(a_prev, min=eps))
            x = sqrt_a_prev * x0_pred + coeff_eps * eps_pred + sigma_t * z
    
        # Decode if latent
        if getattr(self, "latent", False):
            x = self.ae.decode(x, condition)
    
        return x
    def ddim_sample_guided_v2(
        self,
        condition,
        eta=0.0,
        ddim_steps=50,
        guidance_scale=0.0,
        guidance_start_frac=0.5,
        guidance_schedule="linear",
        normalize_guidance=True,
        final_step_only=False,
    ):
        """
        DDIM sampling with per-layer shower-shape guidance.
    
        Implements three inference-time disciplines from the literature:
          1. Late-step-only gating         (PIDDM arXiv:2505.22391, Jensen's Gap)
          2. Time-dependent guidance ramp  (PIDM/DiffusionPDE heuristic)
          3. Per-step gradient normalization
    
        Plus an optional fourth mode for ablation:
          4. Final-step-only projection    (Christopher et al. ICLR 2026, arXiv:2510.14989)
    
        Args:
            condition           : (B, 46) E_incident + 45 layer energies
            eta                 : DDIM stochasticity (0 = deterministic)
            ddim_steps          : number of denoising steps (e.g. 50, 200, 1000)
            guidance_scale      : base scale s. 0 disables guidance.
            guidance_start_frac : in [0, 1]. Apply guidance only for the last
                                  (1 - frac) fraction of steps. 0 = guide every
                                  step; 0.5 = guide only second half; 1.0 = guide
                                  only the very last step (≈ final_step_only).
            guidance_schedule   : "constant" | "linear" | "snr"
                                  "constant" — flat scale (your current behavior)
                                  "linear"   — ramps from 0 at start of guided
                                               window to guidance_scale at t=0
                                  "snr"      — scale ∝ √(α_bar_t / (1 - α_bar_t)),
                                               i.e. proportional to signal-to-noise
            normalize_guidance  : per-sample gradient norm normalization. Decouples
                                  guidance_scale from the raw gradient magnitude
                                  so the same scale works across schedules.
            final_step_only     : if True, ignore guidance during the trajectory
                                  and apply a single projection step at t=0.
                                  Strongest Jensen's Gap mitigation (Fioretto
                                  protein paper). Use for ablation.
    
        Returns:
            x : (B, *self.shape) generated samples in model space
        """
        if guidance_scale > 0.0:
            assert hasattr(self, "ref_ec_eta"), (
                "Call load_reference_stats() before ddim_sample_guided_v2()"
            )
    
        model_dtype = next(self.net.parameters()).dtype
        device = condition.device
        T = int(self.timesteps)
    
        if ddim_steps > T:
            ddim_steps = T
    
        condition = condition.to(model_dtype)
    
        x = torch.randn(
            (condition.shape[0], *self.shape),
            device=device,
            dtype=model_dtype,
        )
    
        alphas_cumprod = self.alphas_cumprod.to(device=device, dtype=model_dtype)
        one = torch.ones((), device=device, dtype=model_dtype)
        eps_n = 1e-12
    
        # Build DDIM timestep schedule
        ts = torch.linspace(T - 1, 0, steps=ddim_steps, device=device).round().long()
        if ts.numel() > 1:
            keep = torch.ones_like(ts, dtype=torch.bool)
            keep[1:] = ts[1:] != ts[:-1]
            ts = ts[keep]
        if ts[-1].item() != 0:
            ts = torch.cat([ts, torch.tensor([0], device=device, dtype=torch.long)])
    
        n_steps = ts.numel()
        guidance_start_idx = int(guidance_start_frac * n_steps)
    
        # ----- Trajectory loop -----
        for i in range(n_steps):
            t_int = int(ts[i].item())
            t_prev_int = int(ts[i + 1].item()) if i < n_steps - 1 else 0
            a_t = alphas_cumprod[t_int]
            a_prev = alphas_cumprod[t_prev_int] if i < n_steps - 1 else one
    
            t_batch = torch.full(
                (condition.shape[0],),
                t_int,
                device=device,
                dtype=torch.long,
            ).unsqueeze(1)
    
            # Guidance gating: only after start_idx, and never if final_step_only
            apply_guidance = (
                guidance_scale > 0.0
                and not final_step_only
                and i >= guidance_start_idx
            )
    
            if apply_guidance:
                x = x.detach().requires_grad_(True)
    
            with torch.set_grad_enabled(apply_guidance):
                eps_pred = self.net(x, t_batch, condition)
                x0_pred = self.predict_start_from_noise(x, t_batch, eps_pred)
    
                if apply_guidance:
                    guide_loss = self._shower_shape_guidance_loss(x0_pred, condition)
                    grad = torch.autograd.grad(guide_loss, x)[0]
    
                    if normalize_guidance:
                        grad_flat = grad.reshape(grad.shape[0], -1)
                        grad_norm = torch.linalg.vector_norm(
                            grad_flat, dim=1
                        ).view(-1, *([1] * (grad.ndim - 1)))
                        grad = grad / (grad_norm + 1e-8)
    
                    # Time-dependent scale schedule
                    if guidance_schedule == "constant":
                        s_t = guidance_scale
                    elif guidance_schedule == "linear":
                        n_guided = max(n_steps - guidance_start_idx, 1)
                        progress = (i - guidance_start_idx) / max(n_guided - 1, 1)
                        s_t = guidance_scale * progress
                    elif guidance_schedule == "snr":
                        snr = torch.sqrt(a_t / torch.clamp(one - a_t, min=eps_n))
                        snr_max = torch.sqrt(
                            alphas_cumprod[0] / torch.clamp(one - alphas_cumprod[0], min=eps_n)
                        )
                        s_t = guidance_scale * (snr / snr_max).item()
                    else:
                        raise ValueError(f"Unknown guidance_schedule: {guidance_schedule}")
    
                    x = x.detach()
                else:
                    grad = None
                    s_t = 0.0
    
            # Standard DDIM update (unchanged)
            one_minus_a_t = torch.clamp(one - a_t, min=eps_n)
            one_minus_a_prev = torch.clamp(one - a_prev, min=0.0)
            ratio = torch.clamp(
                a_t / torch.clamp(a_prev, min=eps_n), min=0.0, max=1.0
            )
            sigma_t = (
                eta
                * torch.sqrt(one_minus_a_prev / one_minus_a_t)
                * torch.sqrt(torch.clamp(one - ratio, min=0.0))
            )
            coeff_eps = torch.sqrt(torch.clamp(one - a_prev - sigma_t ** 2, min=0.0))
            sqrt_a_prev = torch.sqrt(torch.clamp(a_prev, min=eps_n))
    
            z = (
                torch.randn_like(x)
                if (eta > 0.0 and i < n_steps - 1)
                else torch.zeros_like(x)
            )
    
            x = sqrt_a_prev * x0_pred + coeff_eps * eps_pred + sigma_t * z
    
            # Apply guidance nudge after DDIM update
            if grad is not None:
                x = x - s_t * grad
    
        # ----- Optional final-step projection (final_step_only mode) -----
        if final_step_only and guidance_scale > 0.0:
            # Run a small inner optimization on x to minimize the shape loss.
            # No diffusion noise involved here — pure projection at t=0.
            x = x.detach().requires_grad_(True)
            proj_steps = int(self.params.get("final_proj_steps", 20))
            proj_lr = float(self.params.get("final_proj_lr", 1e-2))
    
            opt = torch.optim.Adam([x], lr=proj_lr)
            for _ in range(proj_steps):
                opt.zero_grad()
                # Treat x as already-clean; pass t=0 noise of zeros into x0 predictor.
                # The network's own t=0 estimate of x0 from x is just x.
                loss = self._shower_shape_guidance_loss(x, condition)
                loss.backward()
                opt.step()
    
            x = x.detach()
    
        return x
    def ddim_sample_guided(self, condition, eta=0.0, ddim_steps=50,
                       guidance_scale=0.0):
        """
        DDIM sampling with optional shower shape guidance.
    
        guidance_scale=0.0  →  identical output to ddim_sample
        guidance_scale>0.0  →  steers each step toward Geant4 reference stats
    
        Requires load_reference_stats() to have been called first.
        Suggested starting range from feasibility check: [0.1, 1.0, 5.0]
    
        Args:
            condition      : (B, 46)
            eta            : DDIM stochasticity (0 = deterministic)
            ddim_steps     : number of denoising steps
            guidance_scale : strength of shower shape guidance
        """
        if guidance_scale > 0.0:
            assert hasattr(self, 'ref_ec_eta'), \
                "Call load_reference_stats() before ddim_sample_guided()"
    
        # Use model's actual dtype — not condition dtype
        # (avoids float32/float64 mismatch from evaluate.py import)
        model_dtype = next(self.net.parameters()).dtype
    
        device = condition.device
        T      = int(self.timesteps)
    
        if ddim_steps > T:
            ddim_steps = T
    
        # Cast condition to model dtype
        condition = condition.to(model_dtype)
    
        x = torch.randn(
            (condition.shape[0], *self.shape),
            device=device, dtype=model_dtype
        )
    
        alphas_cumprod = self.alphas_cumprod.to(device=device, dtype=model_dtype)
        one   = torch.ones((), device=device, dtype=model_dtype)
        eps_n = 1e-12
    
        # Build DDIM timestep schedule
        ts = torch.linspace(T-1, 0, steps=ddim_steps, device=device).round().long()
        if ts.numel() > 1:
            keep     = torch.ones_like(ts, dtype=torch.bool)
            keep[1:] = ts[1:] != ts[:-1]
            ts       = ts[keep]
        if ts[-1].item() != 0:
            ts = torch.cat([ts, torch.tensor([0], device=device, dtype=torch.long)])
    
        for i in range(ts.numel()):
            t_int      = int(ts[i].item())
            t_prev_int = int(ts[i+1].item()) if i < ts.numel()-1 else 0
            a_t        = alphas_cumprod[t_int]
            a_prev     = alphas_cumprod[t_prev_int] if i < ts.numel()-1 else one
    
            t_batch = torch.full(
                (condition.shape[0],), t_int,
                device=device, dtype=torch.long
            ).unsqueeze(1)
    
            # Need grad on x for guidance
            if guidance_scale > 0.0:
                x = x.detach().requires_grad_(True)
    
            with torch.set_grad_enabled(guidance_scale > 0.0):
                eps_pred = self.net(x, t_batch, condition)
                x0_pred  = self.predict_start_from_noise(x, t_batch, eps_pred)
    
            # Compute guidance gradient
            if guidance_scale > 0.0:
                guide_loss = self._shower_shape_guidance_loss(x0_pred, condition)
                grad       = torch.autograd.grad(guide_loss, x)[0]
                x          = x.detach()
            else:
                grad = None
    
            # Standard DDIM update
            one_minus_a_t    = torch.clamp(one - a_t,    min=eps_n)
            one_minus_a_prev = torch.clamp(one - a_prev, min=0.0)
            ratio            = torch.clamp(
                a_t / torch.clamp(a_prev, min=eps_n), min=0.0, max=1.0
            )
            sigma_t    = eta * torch.sqrt(one_minus_a_prev / one_minus_a_t) \
                           * torch.sqrt(torch.clamp(one - ratio, min=0.0))
            coeff_eps  = torch.sqrt(torch.clamp(one - a_prev - sigma_t**2, min=0.0))
            sqrt_a_prev = torch.sqrt(torch.clamp(a_prev, min=eps_n))
    
            z = torch.randn_like(x) \
                if (eta > 0.0 and i < ts.numel()-1) else torch.zeros_like(x)
    
            x = sqrt_a_prev * x0_pred + coeff_eps * eps_pred + sigma_t * z
    
            # Apply guidance nudge
            if grad is not None:
                x = x - guidance_scale * grad
    
        return x
        
    def ddim_sample_cfd_guided(self, condition, eta=0.0, ddim_steps=50,
                           guidance_scale=0.0, guidance_start_frac=0.75,
                           normalize_guidance=True):
        """
        DDIM sampling with optional voxel-CFD guidance.
    
        guidance_scale=0.0  -> identical to unguided DDIM sampling
        guidance_scale>0.0  -> late-step voxel-CFD guidance toward Geant4 reference CFD
    
        Requires load_voxel_cfd_reference() to have been called first.
    
        Args:
            condition           : (B, 46)
            eta                 : DDIM stochasticity; 0.0 = deterministic
            ddim_steps          : number of denoising steps
            guidance_scale      : voxel-CFD guidance strength
            guidance_start_frac : fraction of DDIM steps after which guidance starts.
                                  0.75 means only last 25% of steps are guided.
            normalize_guidance  : if True, normalize guidance gradient per sample
        """
        if guidance_scale > 0.0:
            assert hasattr(self, "ref_voxel_cfd"), \
                "Call load_voxel_cfd_reference() before ddim_sample_cfd_guided()"
    
        model_dtype = next(self.net.parameters()).dtype
        device = condition.device
        T = int(self.timesteps)
    
        if ddim_steps > T:
            ddim_steps = T
    
        condition = condition.to(model_dtype)
    
        x = torch.randn(
            (condition.shape[0], *self.shape),
            device=device,
            dtype=model_dtype
        )
    
        alphas_cumprod = self.alphas_cumprod.to(device=device, dtype=model_dtype)
        one = torch.ones((), device=device, dtype=model_dtype)
        eps_n = 1e-12
    
        # Build DDIM timestep schedule
        ts = torch.linspace(T - 1, 0, steps=ddim_steps, device=device).round().long()
    
        if ts.numel() > 1:
            keep = torch.ones_like(ts, dtype=torch.bool)
            keep[1:] = ts[1:] != ts[:-1]
            ts = ts[keep]
    
        if ts[-1].item() != 0:
            ts = torch.cat([ts, torch.tensor([0], device=device, dtype=torch.long)])
    
        guidance_start_idx = int(guidance_start_frac * ts.numel())
    
        for i in range(ts.numel()):
            t_int = int(ts[i].item())
            t_prev_int = int(ts[i + 1].item()) if i < ts.numel() - 1 else 0
    
            a_t = alphas_cumprod[t_int]
            a_prev = alphas_cumprod[t_prev_int] if i < ts.numel() - 1 else one
    
            t_batch = torch.full(
                (condition.shape[0],),
                t_int,
                device=device,
                dtype=torch.long
            ).unsqueeze(1)
    
            apply_guidance = (
                guidance_scale > 0.0
                and i >= guidance_start_idx
            )
    
            # Need gradient on x only during guided late steps
            if apply_guidance:
                x = x.detach().requires_grad_(True)
    
            with torch.set_grad_enabled(apply_guidance):
                eps_pred = self.net(x, t_batch, condition)
                x0_pred = self.predict_start_from_noise(x, t_batch, eps_pred)
    
                if apply_guidance:
                    guide_loss = self._voxel_cfd_guidance_loss(x0_pred, condition)
                    grad = torch.autograd.grad(guide_loss, x)[0]
    
                    if normalize_guidance:
                        grad_flat = grad.reshape(grad.shape[0], -1)
                        grad_norm = torch.linalg.vector_norm(
                            grad_flat, dim=1
                        ).view(-1, *([1] * (grad.ndim - 1)))
    
                        grad = grad / (grad_norm + 1e-8)
    
                    x = x.detach()
                else:
                    grad = None
    
            # Standard DDIM update
            one_minus_a_t = torch.clamp(one - a_t, min=eps_n)
            one_minus_a_prev = torch.clamp(one - a_prev, min=0.0)
    
            ratio = torch.clamp(
                a_t / torch.clamp(a_prev, min=eps_n),
                min=0.0,
                max=1.0
            )
    
            sigma_t = (
                eta
                * torch.sqrt(one_minus_a_prev / one_minus_a_t)
                * torch.sqrt(torch.clamp(one - ratio, min=0.0))
            )
    
            coeff_eps = torch.sqrt(
                torch.clamp(one - a_prev - sigma_t ** 2, min=0.0)
            )
    
            sqrt_a_prev = torch.sqrt(torch.clamp(a_prev, min=eps_n))
    
            z = (
                torch.randn_like(x)
                if (eta > 0.0 and i < ts.numel() - 1)
                else torch.zeros_like(x)
            )
    
            x = sqrt_a_prev * x0_pred + coeff_eps * eps_pred + sigma_t * z
    
            # Apply voxel-CFD guidance nudge after DDIM update
            if grad is not None:
                x = x - guidance_scale * grad
    
        return x
        
    def sample_n_evolution(self, n_samples, cond=None, use_ddim=False, ddim_steps=50, eta=0.0):
        
        """
        Generate n_samples.
    
        Args:
            n_samples: int
            cond: None or torch.Tensor of shape (n_samples, 46).
                  If None, uses zeros conditioning of shape (B, 46).
            use_ddim: bool, if True uses DDIM sampler, else uses DDPM (sample_batch)
            ddim_steps: int, number of DDIM steps (only used if use_ddim=True)
            eta: float, DDIM stochasticity (0.0 is deterministic)
        """
        if self.net.bayesian:
            self.net.map = get(self.params, "fix_mu", False)
            for bay_layer in self.net.bayesian_layers:
                bay_layer.random = None
    
        # Put modules in eval mode
        self.eval()
        self.net.eval()
    
        batch_size = get(self.params, "batch_size", 64)
        cond_dim = 46
    
        # Basic input validation
        if cond is not None:
            if not isinstance(cond, torch.Tensor):
                raise TypeError(f"cond must be a torch.Tensor or None, got {type(cond)}")
            if cond.dim() != 2 or cond.shape[1] != cond_dim:
                raise ValueError(f"cond must have shape (N, {cond_dim}), got {tuple(cond.shape)}")
            if cond.shape[0] < n_samples:
                raise ValueError(f"cond has only {cond.shape[0]} rows, but n_samples={n_samples}")
    
        samples = []
        with torch.inference_mode():
            for i in range(0, n_samples, batch_size):
                cur_bs = min(batch_size, n_samples - i)
    
                # Build conditioning for this batch
                if cond is None:
                    batch_cond = torch.zeros(
                        (cur_bs, cond_dim),
                        device=self.device,
                        dtype=torch.float32,
                    )
                else:
                    batch_cond = cond[i:i + cur_bs].to(self.device)
                    if not torch.is_floating_point(batch_cond):
                        batch_cond = batch_cond.float()
    
                # Sample
                if use_ddim:
                    batch_samples = self.ddim_sample(batch_cond, eta=eta, ddim_steps=ddim_steps)
                else:
                    batch_samples = self.sample_batch(batch_cond)
    
                samples.append(batch_samples.detach().cpu())
    
        return torch.cat(samples, dim=0)[:n_samples]


    def invert_n(self, samples):
        """
        Invert samples through the forward process (for analysis)
        """
        if self.net.bayesian:
            self.net.map = get(self.params, "fix_mu", False)
            for bay_layer in self.net.bayesian_layers:
                bay_layer.random = None
        
        self.eval()
        batch_size = get(self.params, "batch_size", 64)
        n_samples = samples.shape[0]
        
        inverted_samples = []
        with torch.inference_mode():
            for i in range(0, n_samples, batch_size):
                current_batch_size = min(batch_size, n_samples - i)
                batch_samples = samples[i:i+current_batch_size].to(self.device)
                
                # Add noise gradually (forward process)
                t = torch.randint(0, self.timesteps, (current_batch_size,), device=self.device)
                noise = torch.randn_like(batch_samples)
                noisy_samples = self.q_sample(batch_samples, t, noise)
                
                inverted_samples.append(noisy_samples.detach().cpu())
        
        return torch.cat(inverted_samples, dim=0)[:n_samples]
    
