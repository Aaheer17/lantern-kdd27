# standard python libraries
import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from matplotlib.backends.backend_pdf import PdfPages
from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import apply_gradient_vector, get_gradient_vector
from conflictfree.length_model import TrackSpecific
from conflictfree.grad_operator import PCGradOperator, IMTLGOperator
import sys
import h5py
# Other functions of project
from Util.util import *
from datasets import *
from documenter import Documenter
from plotting_util import *
from transforms import *
from challenge_files import *
from challenge_files import evaluate # avoid NameError: 'evaluate' is not defined
from challenge_files.compute_metrics_helper import calculate_correlation_voxel

import Models
from Models import *
import time
from itertools import islice

import pandas as pd

from Models.multi_objective import create_mo_optimizer

class GenerativeModel(nn.Module):
    """
    Base Class for Generative Models to inherit from.
    Children classes should overwrite the individual methods as needed.
    Every child class MUST overwrite the methods:

    def build_net(self): should register some NN architecture as self.net
    def batch_loss(self, x): takes a batch of samples as input and returns the loss
    def sample_n_parallel(self, n_samples): generates and returns n_samples new samples

    See tbd.py for an example of child class

    Structure:

    __init__(params)      : Read in parameters and register the important ones
    build_net()           : Create the NN and register it as self.net
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    prepare_training()    : Read in the appropriate parameters and prepare the model for training
                            Currently this is called from run_training(), so it should not be called on its own
    run_training()        : Run the actual training.
                            Necessary parameters are read in and the training is performed.
                            This calls on the methods train_one_epoch() and validate_one_epoch()
    train_one_epoch()     : Performs one epoch of model training.
                            This calls on the method batch_loss(x)
    validate_one_epoch()  : Performs one epoch of validation.
                            This calls on the method batch_loss(x)
    batch_loss(x)         : Takes one batch of samples as input and returns the loss.
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    sample_n(n_samples)   : Generates and returns n_samples new samples as a numpy array
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    sample_and_plot       : Generates n_samples and makes plots with them.
                            This is meant to be used during training if intermediate plots are wanted

    """
    def __init__(self, params, device, doc):
        """
        :param params: file with all relevant model parameters
        """
        super().__init__()
        self.doc = doc
        self.params = params
        self.device = device
        self.shape = self.params['shape']
        self.conditional = get(self.params,'conditional',False)
        self.single_energy = get(self.params, 'single_energy', None)
        self.eval_mode = get(self.params, 'eval_mode', 'all')

        self.batch_size = self.params["batch_size"]
        self.batch_size_sample = get(self.params, "batch_size_sample", 10_000)
        print("in modelbase")
        self.net = self.build_net()
        param_count = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f'init model: Model has {param_count} parameters')
        self.params['parameter_count'] = param_count

        self.epoch = get(self.params, "total_epochs", 0)
        self.iterations = get(self.params,"iterations", 1)
        self.regular_loss = []
        self.kl_loss = []
        self.global_step = 0
        self.runs = get(self.params, "runs", 0)
        self.iterate_periodically = get(self.params, "iterate_periodically", False)
        self.validate_every = get(self.params, "validate_every", 50)

        # ── Gradient conflict diagnostic frequency ────────────────────────────
        # Every grad_diag_every optimizer steps, compute per-loss gradient
        # vectors and log conflict statistics to CSV/TensorBoard.
        # Set to 0 to disable entirely.
        self.grad_diag_every = get(self.params, 'grad_diag_every', 50)

        # augment data
        self.aug_transforms = get(self.params, "augment_batch", False)

        # load autoencoder for latent modelling
        self.ae_dir = get(self.params, "autoencoder", None)
        if self.ae_dir is None:
            self.transforms = get_transformations(
                params.get('transforms', None), doc=self.doc
            )
            self.latent = False
        else:
            self.ae = self.load_other(self.ae_dir)
            self.transforms = self.ae.transforms
            self.latent = True

        # ========================================
        # Voxel-wise CFD validation config
        # ========================================
        self.validate_voxel_CFD = get(self.params, "validate_voxel_CFD", False)
        self.validate_voxel_CFD_every = get(self.params, "validate_voxel_CFD_every", 0)

        self.use_laplacian_loss = get(self.params, 'use_laplacian_loss', False)
        self.lambda_laplacian   = get(self.params, 'lambda_laplacian', 0.01)

        self.val_voxel_cfd_epoch = np.array([], dtype=np.float64)
        if self.validate_voxel_CFD:
            self.val_CFD_hdf5_file = get(self.params, "val_CFD_hdf5_file", None)
            self.voxel_CFD_ref_file = get(self.params, "voxel_CFD_ref", None)

            if self.val_CFD_hdf5_file is None or self.voxel_CFD_ref_file is None:
                raise ValueError(
                    "validate_voxel_CFD=True but CFD paths not fully provided."
                )

            print(
                f"[Voxel CFD] Enabled | every {self.validate_voxel_CFD_every} epochs\n"
                f"[Voxel CFD] val_CFD_hdf5_file = {self.val_CFD_hdf5_file}\n"
                f"[Voxel CFD] voxel_CFD_ref    = {self.voxel_CFD_ref_file}"
            )

            with h5py.File(self.voxel_CFD_ref_file, "r") as f:
                self.C_ref_voxel = f["C_ref_voxel"][:]
                fro = f["fro_norm_ref"][:]

            self.fro_norm_ref = float(fro[0] if fro.ndim > 0 else fro)
            self._build_voxel_cfd_cond_loader()
        else:
            self.val_CFD_hdf5_file = None
            self.voxel_CFD_ref_file = None
            self.C_ref_voxel = None
            self.fro_norm_ref = None

        # Multi-objective optimization setup
        self.mo_method = get(self.params, 'mo_method', 'weighted_sum')
        self.use_energy_loss = get(self.params, 'use_energy_loss', False)
        self.use_moment_matching = get(self.params, 'use_moment_matching', False)
        self.use_sparsity = get(self.params, 'use_sparsity', False)
        self.use_voxel_energy_loss = get(self.params, 'use_voxel_energy_loss', False)
        self.lambda_energy = get(self.params, 'lambda_energy', 0.1)
        self.lambda_moment = get(self.params, 'lambda_moment', 0.0)
        self.lambda_sparsity = get(self.params, 'lambda_sparsity', 1e-4)
        self.lambda_voxel_energy = get(self.params, 'lambda_voxel_energy', 1e-4)
        self.mo_optimizer = None

        self.use_voxel_shape_loss = get(self.params, 'use_voxel_shape_loss', False)
        self.lambda_voxel_shape  = get(self.params, 'lambda_voxel_shape', 1e-4)

        # ── GradNorm ──────────────────────────────────────────────────────────
        self.gradnorm_alpha    = get(self.params, 'gradnorm_alpha', 1.5)
        self.gradnorm_lr       = get(self.params, 'gradnorm_lr',   1e-3)
        self.gradnorm_balancer = None

        # ── Gradient Blending ─────────────────────────────────────────────────
        self.blend_alpha                  = get(self.params, 'blend_alpha',                  1.0)
        self.blend_fallback_threshold     = get(self.params, 'blend_fallback_threshold',     150.0)
        self.blend_alpha_decay_start      = get(self.params, 'blend_alpha_decay_start',      120.0)
        self.blend_use_gradnorm           = get(self.params, 'blend_use_gradnorm',           True)
        self.use_unweighted_loss_for_grad = get(self.params, 'use_unweighted_loss_for_grad', False)

        # ── Uncertainty weighting (Kendall et al. 2018) ───────────────────────
        self.use_uncertainty_weighting = get(self.params, 'use_uncertainty_weighting', False)
        self.uw_use_schedule_envelope  = get(self.params, 'uw_use_schedule_envelope', True)
        if self.use_uncertainty_weighting:
            self.s_diff = nn.Parameter(torch.zeros((), device=self.device))
            self.s_aux  = nn.Parameter(torch.zeros((), device=self.device))

    def build_net(self):
        pass

    def prepare_training(self):

        print("train_model: Preparing model training")

        self.train_loader, self.val_loader, self.bounds = get_loaders(
            self.params.get('hdf5_file'),
            self.params.get('particle_type'),
            self.params.get('xml_filename'),
            self.params.get('val_frac'),
            self.params.get('batch_size'),
            self.transforms,
            self.params.get('eps', 1.e-10),
            device=self.device,
            shuffle=True,
            width_noise=self.params.get('width_noise', 1.e-6),
            single_energy=self.params.get('single_energy', None),
            aug_transforms=self.aug_transforms
        )

        self.use_scheduler = get(self.params, "use_scheduler", False)

        self.n_trainbatches = len(self.train_loader)
        self.n_traindata = self.n_trainbatches * self.batch_size
        self.set_optimizer(steps_per_epoch=self.n_trainbatches)

        # ========================================
        # Multi-Objective Optimizer
        # ========================================
        has_aux_losses = (
            self.use_energy_loss or
            self.use_moment_matching or
            self.use_sparsity or
            self.use_voxel_energy_loss or
            self.use_voxel_shape_loss or
            self.use_laplacian_loss
        )

        if not has_aux_losses:
            print("✓ No auxiliary losses enabled - using simple diffusion loss only")
            self.mo_optimizer = None
            self.mo_method = 'none'
        elif self.mo_method not in ('none', 'config', 'grad_blend', 'pcgrad', 'imtlg', 'gradnorm','uncertainty'):
            mo_kwargs = {}

            if self.mo_method == 'weighted_sum':
                mo_kwargs['weights'] = {
                    'diffusion_loss':    get(self.params, 'lambda_diffusion', 1.0),
                    'energy_loss':       self.lambda_energy,
                    'moment_loss':       self.lambda_moment,
                    'sparsity_loss':     self.lambda_sparsity,
                    'voxel_energy_loss': self.lambda_voxel_energy,
                    'voxel_shape_loss':  self.lambda_voxel_shape,
                    'laplacian_loss':    self.lambda_laplacian,
                }
            elif self.mo_method == 'dwa':
                mo_kwargs['temperature'] = get(self.params, 'dwa_temperature', 2.0)
            elif self.mo_method == 'grad_norm':
                mo_kwargs['target_loss'] = 'diffusion_loss'

            self.mo_optimizer = create_mo_optimizer(
                self.mo_method,
                self.net,
                self.optimizer,
                **mo_kwargs
            )
            print(f"✓ Multi-objective optimizer: {self.mo_method}")
        else:
            if self.mo_method == 'config':
                print(f"✓ Multi-objective optimizer: ConFIG (gradient-based)")
            elif self.mo_method == 'grad_blend':
                print(f"✓ Multi-objective optimizer: Gradient Blending "
                      f"(professor's method, alpha={self.blend_alpha})")
                if self.blend_use_gradnorm:
                    from Models.gradnorm import GradNormBalancer
                    _blend_use_flags = {
                        'diffusion_loss':    True,
                        'voxel_energy_loss': get(self.params, 'use_voxel_energy_loss', False),
                        'laplacian_loss':    get(self.params, 'use_laplacian_loss',     False),
                        'energy_loss':       get(self.params, 'use_energy_loss',        False),
                        'moment_loss':       get(self.params, 'use_moment_matching',    False),
                        'sparsity_loss':     get(self.params, 'use_sparsity',           False),
                    }
                    _blend_objectives = [k for k, v in _blend_use_flags.items() if v]
                    self.gradnorm_balancer = GradNormBalancer(
                        num_tasks=len(_blend_objectives),
                        alpha=self.gradnorm_alpha,
                        lr=self.gradnorm_lr,
                        device=self.device,
                    )
                    self.gradnorm_objectives = _blend_objectives
                    print(f"  ↳ GradNorm magnitude balancer: tasks={_blend_objectives}")
                else:
                    print(f"  ↳ GradNorm magnitude balancer: disabled")
            elif self.mo_method == 'pcgrad':
                print(f"✓ Multi-objective optimizer: PCGrad (gradient surgery)")
            elif self.mo_method == 'imtlg':
                print(f"✓ Multi-objective optimizer: IMTL-G (impartial multi-task learning)")
            elif self.mo_method == 'uncertainty':
                print("✓ Multi-objective optimizer: Uncertainty Weighting "
                      "(Kendall et al. 2018, learned log-variances s_diff, s_aux)")
            elif self.mo_method == 'gradnorm':
                from Models.gradnorm import GradNormBalancer
                _gradnorm_use_flags = {
                    'diffusion_loss':    True,
                    'voxel_energy_loss': get(self.params, 'use_voxel_energy_loss', False),
                    'laplacian_loss':    get(self.params, 'use_laplacian_loss',     False),
                    'energy_loss':       get(self.params, 'use_energy_loss',        False),
                    'moment_loss':       get(self.params, 'use_moment_matching',    False),
                    'sparsity_loss':     get(self.params, 'use_sparsity',           False),
                }
                objectives = [k for k, v in _gradnorm_use_flags.items() if v]

                self.gradnorm_balancer = GradNormBalancer(
                    num_tasks=len(objectives),
                    alpha=self.gradnorm_alpha,
                    lr=self.gradnorm_lr,
                    device=self.device,
                )
                self.gradnorm_objectives = objectives
                print(f"✓ Multi-objective optimizer: GradNorm "
                      f"(alpha={self.gradnorm_alpha}, tasks={objectives})")

        if hasattr(self, 'setup_ema'):
            print("Setting up EMA model...")
            self.setup_ema()

        self.sample_periodically = get(self.params, "sample_periodically", False)
        if self.sample_periodically:
            self.sample_every = get(self.params, "sample_every", 1)
            self.sample_every_n_samples = get(self.params, "sample_every_n_samples", 100000)
            print(f'train_model: sample_periodically set to True. Sampling {self.sample_every_n_samples} every'
                  f' {self.sample_every} epochs. This may significantly slow down training!')

        self.log = get(self.params, "log", True)
        if self.log:
            log_dir = self.doc.basedir
            self.logger = SummaryWriter(log_dir)
            print(f"train_model: Logging to log_dir {log_dir}")
        else:
            print("train_model: log set to False. No logs will be written")

    def set_optimizer(self, steps_per_epoch=1, params=None):
        if params is None:
            params = self.params

        uw_params = []
        if getattr(self, 'use_uncertainty_weighting', False):
            uw_params = [self.s_diff, self.s_aux]

        self.optimizer = torch.optim.AdamW(
            list(self.net.parameters()) + uw_params,
            lr=params.get("lr", 0.0002),
            betas=params.get("betas", [0.9, 0.999]),
            eps=params.get("optimizer_eps", 1e-6),
            weight_decay=params.get("weight_decay", 0.)
        )

        # self.optimizer = torch.optim.AdamW(
        #     self.net.parameters(),
        #     lr=params.get("lr", 0.0002),
        #     betas=params.get("betas", [0.9, 0.999]),
        #     eps=params.get("optimizer_eps", 1e-6),
        #     weight_decay=params.get("weight_decay", 0.)
        # )
        self.scheduler = set_scheduler(self.optimizer, params, steps_per_epoch, last_epoch=-1)

    def _build_voxel_cfd_cond_loader(self):
        batch_size_sample = get(self.params, "batch_size_sample", 100)

        ds = CaloChallengeDataset(
            self.val_CFD_hdf5_file,
            self.params.get('particle_type'),
            self.params.get('xml_filename'),
            transform=self.transforms,
            device=self.device,
            single_energy=self.single_energy
        )

        transformed_cond = ds.energy

        self.voxel_cfd_cond = DataLoader(
            dataset=transformed_cond,
            batch_size=batch_size_sample,
            shuffle=False
        )

        try:
            n = len(transformed_cond)
            cdim = transformed_cond.shape[1] if hasattr(transformed_cond, "shape") and transformed_cond.dim() == 2 else None
            print(f"[Voxel CFD] Condition tensor: N={n}, cond_dim={cdim}, batch_size_sample={batch_size_sample}")
            print(f"[Voxel CFD] Using CFD HDF5: {self.val_CFD_hdf5_file}")
        except Exception:
            print(f"[Voxel CFD] Built condition loader from: {self.val_CFD_hdf5_file}")

    def _compute_voxel_cfd_scalar(self, x_gen_all, cond_all):
        if not isinstance(x_gen_all, torch.Tensor):
            raise TypeError(f"x_gen_all must be a torch.Tensor, got {type(x_gen_all)}")
        if not isinstance(cond_all, torch.Tensor):
            raise TypeError(f"cond_all must be a torch.Tensor, got {type(cond_all)}")

        x = x_gen_all.to(self.device)
        c = cond_all.to(self.device)
        if not torch.is_floating_point(c):
            c = c.float()

        for fn in self.transforms[::-1]:
            x, c = fn(x, c, rev=True)

        if x.dim() != 2 or x.shape[1] != 45 * 16 * 9:
            raise ValueError(f"Expected physical showers shape (N,6480), got {tuple(x.shape)}")

        vox = x.detach().cpu().numpy().reshape(-1, 45, 16, 9)
        C_gen = calculate_correlation_voxel(vox)

        diff = C_gen - self.C_ref_voxel
        fro_diff = float(np.sqrt(np.sum(diff * diff)))
        voxel_cfd = fro_diff / (float(self.fro_norm_ref) + 1e-12)

        return voxel_cfd

    def run_training(self):

        self.prepare_training()

        samples = []
        n_epochs = get(self.params, "n_epochs", 100)

        past_epochs = get(self.params, "total_epochs", 0)
        if past_epochs != 0:
            self.load(epoch=past_epochs)
            self.scheduler = set_scheduler(self.optimizer, self.params, self.n_trainbatches, last_epoch=self.params.get("total_epochs", -1)*self.n_trainbatches)
        print(f"train_model: Model has been trained for {past_epochs} epochs before.")
        print(f"train_model: Beginning training. n_epochs set to {n_epochs}")
        t_0 = time.time()
        for e in range(n_epochs):
            t0 = time.time()

            self.epoch = past_epochs + e
            self.net.train()
            one_s = time.time()
            self.train_one_epoch()
            print("One epoch time: ", time.time()-one_s)
            if (self.epoch + 1) % self.validate_every == 0:
                self.eval()
                self.validate_one_epoch()

            if self.sample_periodically:
                if (self.epoch + 1) % self.sample_every == 0:
                    self.eval()
                    if get(self.params, "reconstruct", False):
                        samples, c = self.reconstruct_n()
                    else:
                        samples, c = self.sample_n()
                    self.plot_samples(samples=samples, conditions=c, name=self.epoch, energy=self.single_energy)

            voxel_cfd_value = np.nan

            if self.validate_voxel_CFD and self.validate_voxel_CFD_every > 0:
                if (self.epoch + 1) % self.validate_voxel_CFD_every == 0:
                    self.eval()
                    self.net.eval()

                    ddim_steps = get(self.params, "voxel_CFD_ddim_steps", 400)
                    eta = get(self.params, "voxel_CFD_ddim_eta", 0.0)

                    gen_chunks = []
                    cond_chunks = []

                    with torch.inference_mode():
                        for cond_batch in self.voxel_cfd_cond:
                            if not isinstance(cond_batch, torch.Tensor):
                                raise TypeError(f"Expected Tensor from voxel_cfd_cond loader, got {type(cond_batch)}")

                            cond_batch = cond_batch.to(self.device)
                            if not torch.is_floating_point(cond_batch):
                                cond_batch = cond_batch.float()

                            x_gen = self.ddim_sample(cond_batch, eta=eta, ddim_steps=ddim_steps)

                            gen_chunks.append(x_gen.detach().cpu())
                            cond_chunks.append(cond_batch.detach().cpu())

                    x_gen_all = torch.cat(gen_chunks, dim=0)
                    cond_all  = torch.cat(cond_chunks, dim=0)

                    voxel_cfd_value = self._compute_voxel_cfd_scalar(
                        x_gen_all=x_gen_all,
                        cond_all=cond_all
                    )

                    print(f"[Voxel CFD] Epoch {self.epoch+1}: voxel_CFD = {voxel_cfd_value:.6f}")

            self.val_voxel_cfd_epoch = np.append(self.val_voxel_cfd_epoch, voxel_cfd_value)

            if get(self.params, "save_periodically", False):
                if (self.epoch + 1) % get(self.params, "save_every", 10) == 0 or self.epoch == 0:
                    self.save(epoch=f"{self.epoch}")
            print(f"Epoch {self.epoch}: LR = {self.scheduler.get_last_lr()[0]}")
            if e == 0:
                t1 = time.time()
                dtEst = (t1-t0) * n_epochs
                print(f"Training time estimate: {dtEst/60:.2f} min = {dtEst/60**2:.2f} h")
            sys.stdout.flush()
        t_1 = time.time()
        traintime = t_1 - t_0
        self.params['train_time'] = traintime
        print(
            f"train_model: Finished training {n_epochs} epochs after {traintime:.2f} s = {traintime / 60:.2f} min = {traintime / 60 ** 2:.2f} h.", flush=True)

        print("train_model: Saving diagnostics", flush=True)
        self.save_all_metrics()

        self.save()
        if get(self.params, "sample", True):
            if get(self.params, "reconstruct", False):
                samples, c = self.reconstruct_n()
            else:
                samples, c = self.sample_n()
            self.plot_samples(samples=samples, conditions=c, energy=self.single_energy)

    def combine_losses(self, loss_tensors):
        """
        Combine losses using the selected multi-objective method.
        """

        if getattr(self, 'use_uncertainty_weighting', False):
            L_diff = loss_tensors['diffusion_loss']            # already unweighted
            
            if self.use_laplacian_loss:
                L_aux = loss_tensors.get('laplacian_loss_unweighted',
                                         loss_tensors['laplacian_loss'])
                lam_t = self.get_laplacian_loss_weight()
            elif self.use_voxel_energy_loss:
                L_aux = loss_tensors.get('voxel_energy_loss_unweighted',
                                         loss_tensors['voxel_energy_loss'])
                lam_t = self.get_voxel_loss_weight()
            else:
                return L_diff, {}
    
            p_diff = torch.exp(-self.s_diff)
            p_aux  = torch.exp(-self.s_aux)
    
            aux_term = 0.5 * p_aux * L_aux + 0.5 * self.s_aux
            if self.uw_use_schedule_envelope:
                aux_term = lam_t * aux_term        # same warmup/decay timing as other methods
    
            total_loss = 0.5 * p_diff * L_diff + 0.5 * self.s_diff + aux_term
    
            mo_info = {
                'uw_weight_diff':   float((0.5 * p_diff).detach()),
                'uw_weight_aux':    float((0.5 * p_aux).detach()),
                'uw_lambda_env':    float(lam_t),
                's_diff':           float(self.s_diff.detach()),
                's_aux':            float(self.s_aux.detach()),
            }
            return total_loss, mo_info
        
        if self.mo_method == 'none' or self.mo_optimizer is None:
            return loss_tensors['diffusion_loss'], {}

        if self.mo_method in ('config', 'grad_blend', 'pcgrad', 'imtlg'):
            return loss_tensors['diffusion_loss'], {}

        use_flags = {
            'diffusion_loss':    True,
            'energy_loss':       self.use_energy_loss,
            'moment_loss':       self.use_moment_matching,
            'sparsity_loss':     self.use_sparsity,
            'voxel_energy_loss': self.use_voxel_energy_loss,
            'voxel_shape_loss':  self.use_voxel_shape_loss,
            'laplacian_loss':    self.use_laplacian_loss,
        }

        total_loss, mo_info = self.mo_optimizer.combine(loss_tensors, use_flags)

        return total_loss, mo_info

    def _get_config_objectives(self, loss_tensors: dict) -> list[str]:
        """
        Decide which objectives participate in gradient-based MO methods.
        """
        cfg = getattr(self, "params", {}) or {}
        requested = cfg.get("config_objectives", None)

        if requested is not None:
            return [k for k in requested if k in loss_tensors]

        default_order = [
            "diffusion_loss",
            "residual_loss",
            "energy_loss",
            "moment_loss",
            "sparsity_loss",
            'voxel_energy_loss',
            'voxel_shape_loss',
            "laplacian_loss",
        ]
        return [k for k in default_order if k in loss_tensors]

    # ============================================================
    # GRADIENT CONFLICT DIAGNOSTICS
    # ============================================================

    def _compute_conflict_diagnostics_only(self, loss_tensors: dict) -> dict:
        """
        Compute gradient conflict diagnostics WITHOUT affecting optimization.

        Calls backward(retain_graph=True) on each loss separately, collects
        flat gradient vectors, computes norms / cosines / conflict flags,
        then zeros all gradients. The computation graph remains intact for
        the main backward() call that follows.

        Keys returned use '_' not '/' so CSV column names are valid pandas
        identifiers. All keys are prefixed 'grad_diag_' for easy filtering.

        Losses diagnosed (in order, only those present in loss_tensors):
            diffusion_loss, voxel_energy_loss, laplacian_loss,
            energy_loss, moment_loss, sparsity_loss

        Returns
        -------
        dict  — empty dict if fewer than 2 losses are present or if any
                backward() call fails (training is not interrupted).
        """
        params = [p for p in self.net.parameters() if p.requires_grad]

        def _get_grad_vec(loss):
            # zero before backward
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            loss.backward(retain_graph=True)
            vecs = []
            for p in params:
                if p.grad is not None:
                    vecs.append(p.grad.detach().clone().flatten())
                else:
                    vecs.append(torch.zeros(p.numel(), device=loss.device))
            # zero after collection — leave graph alive
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            return torch.cat(vecs)

        # canonical order: diffusion always first
        diag_order = [
            "diffusion_loss",
            "voxel_energy_loss",
            "laplacian_loss",
            "energy_loss",
            "moment_loss",
            "sparsity_loss",
        ]
        present = [k for k in diag_order if k in loss_tensors]

        if len(present) < 2:
            return {}

        # collect gradient vectors
        grads = {}
        for name in present:
            try:
                grads[name] = _get_grad_vec(loss_tensors[name])
            except Exception as e:
                print(f"[GradDiag] backward failed for {name}: {e}")
                return {}

        info = {}
        g_diff    = grads["diffusion_loss"]
        norm_diff = float(g_diff.norm().item()) + 1e-10
        info["grad_diag_norm_diffusion"] = norm_diff

        # ── diffusion vs each auxiliary ───────────────────────────────────────
        for name in present[1:]:
            g_aux    = grads[name]
            norm_aux = float(g_aux.norm().item())
            dot      = float(torch.dot(g_diff, g_aux).item())
            cos      = float(np.clip(dot / (norm_diff * (norm_aux + 1e-10)), -1.0, 1.0))
            angle    = float(np.degrees(np.arccos(cos)))

            # sanitize name for CSV: replace any special chars
            tag = name.replace("/", "_")

            info[f"grad_diag_norm_{tag}"]              = norm_aux
            info[f"grad_diag_ratio_{tag}_over_diff"]   = norm_aux / norm_diff
            info[f"grad_diag_cos_diff__{tag}"]         = cos
            info[f"grad_diag_angle_deg_diff__{tag}"]   = angle
            info[f"grad_diag_conflict_diff__{tag}"]    = float(cos < 0)

        # ── pairwise among auxiliary losses ───────────────────────────────────
        aux_names = present[1:]
        for i in range(len(aux_names)):
            for j in range(i + 1, len(aux_names)):
                na, nb   = aux_names[i], aux_names[j]
                ga, gb   = grads[na], grads[nb]
                na_n     = float(ga.norm().item()) + 1e-10
                nb_n     = float(gb.norm().item()) + 1e-10
                cos_ab   = float(np.clip(
                    float(torch.dot(ga, gb).item()) / (na_n * nb_n), -1.0, 1.0
                ))
                tag_a = na.replace("/", "_")
                tag_b = nb.replace("/", "_")
                info[f"grad_diag_cos_{tag_a}__{tag_b}"]      = cos_ab
                info[f"grad_diag_conflict_{tag_a}__{tag_b}"] = float(cos_ab < 0)

        return info

    # ============================================================
    # GRADIENT STEP HELPERS
    # ============================================================

    def _compute_per_objective_grads(self, loss_tensors: dict, objectives: list):
        """
        Backpropagate each objective separately and collect flat gradient vectors.

        Returns
        -------
        grads    : list[Tensor]  one flat gradient vector per objective
        grad_ok  : bool          False if any gradient contains NaN or Inf
        """
        grads   = []
        grad_ok = True

        for i, name in enumerate(objectives):
            self.optimizer.zero_grad(set_to_none=True)
            retain = (i < len(objectives) - 1)
            loss_tensors[name].backward(retain_graph=retain)
            g = get_gradient_vector(self.net)
            grads.append(g)

            if torch.isnan(g).any() or torch.isinf(g).any():
                grad_ok = False

        return grads, grad_ok

    def _compute_gradient_diagnostics(self, grads: list, objectives: list,
                                       g_combined=None) -> dict:
        """
        Compute gradient diagnostics. All angles in degrees, cosines clipped to [-1,1].
        Norm ratios use objectives[0] (diffusion_loss) as denominator.
        """
        info  = {}
        norms = [float(torch.linalg.vector_norm(g).item()) for g in grads]

        for name, n in zip(objectives, norms):
            info[f"gnorm_{name}"] = n

        norm_diff = norms[0] + 1e-10
        for name, n in zip(objectives[1:], norms[1:]):
            info[f"gnorm_ratio_{name}_over_diff"] = n / norm_diff

        for i in range(len(objectives)):
            for j in range(i + 1, len(objectives)):
                denom  = (norms[i] * norms[j]) + 1e-10
                cos_ij = float(np.clip(
                    float(torch.dot(grads[i], grads[j]).item()) / denom,
                    -1.0, 1.0
                ))
                info[f"cos_{objectives[i]}__{objectives[j]}"]       = cos_ij
                info[f"angle_deg_{objectives[i]}__{objectives[j]}"] = float(np.degrees(np.arccos(cos_ij)))

        if g_combined is not None:
            norm_combined = float(torch.linalg.vector_norm(g_combined).item())
            info["gnorm_combined"] = norm_combined
            for name, g, n in zip(objectives, grads, norms):
                cos_ci = float(np.clip(
                    float(torch.dot(g_combined, g).item()) / ((norm_combined + 1e-10) * (n + 1e-10)),
                    -1.0, 1.0
                ))
                info[f"cos_combined__{name}"]       = cos_ci
                info[f"angle_deg_combined__{name}"] = float(np.degrees(np.arccos(cos_ci)))

        # ── Unified MO comparison aliases (identical keys across all methods) ──
        # mo_conflict_angle_{i}_{j} is the angle between objectives[i] and
        # objectives[j], indexed by position in the objectives list.
        # objectives[0] is always diffusion_loss, so mo_conflict_angle_0_1 is
        # diffusion vs the first physics loss. mo_conflict_angle_primary is the
        # diffusion-vs-first-aux shortcut. mo_magnitude_ratio is the applied
        # combined-step norm relative to the pure diffusion gradient norm.
        for i in range(len(objectives)):
            for j in range(i + 1, len(objectives)):
                info[f"mo_conflict_angle_{i}_{j}"] = \
                    info[f"angle_deg_{objectives[i]}__{objectives[j]}"]

        if len(objectives) >= 2:
            info["mo_conflict_angle_primary"] = \
                info[f"angle_deg_{objectives[0]}__{objectives[1]}"]

        if g_combined is not None:
            info["mo_magnitude_ratio"] = info["gnorm_combined"] / (norms[0] + 1e-10)

        # Write the index->name legend once so the numeric columns are
        # self-documenting. Assumes the enabled objective set is fixed for the
        # run (it is, since it comes from the config flags).
        if not getattr(self, "_mo_legend_logged", False):
            self._mo_legend_logged = True
            legend = {
                "objective_index": {i: name for i, name in enumerate(objectives)},
                "conflict_pairs": {
                    f"mo_conflict_angle_{i}_{j}": f"{objectives[i]}__{objectives[j]}"
                    for i in range(len(objectives))
                    for j in range(i + 1, len(objectives))
                },
                "mo_conflict_angle_primary": (
                    f"{objectives[0]}__{objectives[1]}" if len(objectives) >= 2 else None
                ),
            }
            try:
                import json
                with open(self.doc.get_file("mo_conflict_angle_legend.json"), "w") as _f:
                    json.dump(legend, _f, indent=2)
                print(f"[MO legend] objectives order: {list(enumerate(objectives))}")
                print(f"[MO legend] saved mo_conflict_angle_legend.json")
            except Exception as _e:
                print(f"[MO legend] could not save legend: {_e}")

        return info

    # ─────────────────────────────────────────────────────────────────────────
    # ConFIG
    # ─────────────────────────────────────────────────────────────────────────

    def _step_with_config(self, loss_tensors: dict) -> dict:
        """
        ConFIG update: combined gradient is the unit bisector of all objective
        gradients, scaled by the length model.
        """
        info       = {}
        objectives = self._get_config_objectives(loss_tensors)

        assert objectives[0] == "diffusion_loss", (
            f"ConFIG requires diffusion_loss at index 0, got: {objectives}"
        )
        if len(objectives) < 2:
            return {"config_used": 0.0}

        grads, grad_ok = self._compute_per_objective_grads(loss_tensors, objectives)

        if not grad_ok:
            apply_gradient_vector(self.net, grads[0])
            return {"config_used": 0.0}

        g_config = ConFIG_update(grads, use_least_square=False, rtol=0.0001)
        apply_gradient_vector(self.net, g_config)
        _verify_info = self._verify_mo_applied(grads, g_config, "config")
        

        # ── pinv diagnostic + gradient saver ────────────────────────────────
        step_log_epoch_diag = getattr(self, "params", {}).get("step_log_epoch", -1)
        current_epoch_diag  = getattr(self, "epoch", -1)
        if step_log_epoch_diag >= 0 and current_epoch_diag == step_log_epoch_diag:
            import math as _math, os as _os
            with torch.no_grad():
                _grads_st = torch.stack(grads)
                _norms    = _grads_st.norm(dim=1)
                _units    = torch.nan_to_num(_grads_st / _norms.unsqueeze(1), 0)
                _best     = torch.linalg.pinv(_units) @ torch.ones(2, device=_units.device)
                _dot_u0   = float(torch.dot(_best, _units[0]).item())
                _dot_u1   = float(torch.dot(_best, _units[1]).item())
                _cos_u0u1 = float(torch.dot(_units[0], _units[1]).item())
                _angle_u  = float(_math.acos(max(-1.0, min(1.0, _cos_u0u1))) * 180 / _math.pi)
                _norm_u1  = float(_units[1].norm().item())
                _is_healthy = abs(_dot_u0) > 0.5 and abs(_dot_u1) > 0.5
                print(
                    f"[pinv_diag] step={self.global_step:6d} | "
                    f"angle_units={_angle_u:7.3f}° | "
                    f"norm_u1={_norm_u1:.6f} | "
                    f"dot(best,u0)={_dot_u0:+.6f} | "
                    f"dot(best,u1)={_dot_u1:+.6f} | "
                    f"{'HEALTHY' if _is_healthy else 'BROKEN '}"
                )
                # Save gradients for up to 5 healthy and 5 broken steps
                _save_root = self.doc.get_file("grad_saves")
                _os.makedirs(_save_root, exist_ok=True)
                _category  = "healthy" if _is_healthy else "broken"
                _cat_dir   = _os.path.join(_save_root, _category)
                _os.makedirs(_cat_dir, exist_ok=True)
                _existing  = len([f for f in _os.listdir(_cat_dir) if f.endswith('.pt')])
                if _existing < 5:
                    _save_path = _os.path.join(_cat_dir, f"step_{self.global_step:06d}.pt")
                    torch.save({
                        "step":         self.global_step,
                        "angle_units":  _angle_u,
                        "dot_best_u0":  _dot_u0,
                        "dot_best_u1":  _dot_u1,
                        "category":     _category,
                        "g_diff":       grads[0].cpu(),
                        "g_vox":        grads[1].cpu(),
                        "g_config":     g_config.cpu(),
                        "u_diff":       _units[0].cpu(),
                        "u_vox":        _units[1].cpu(),
                        "best_dir":     _best.cpu(),
                        "norm_diff":    float(_norms[0].item()),
                        "norm_vox":     float(_norms[1].item()),
                    }, _save_path)
                    print(f"[grad_save]  Saved {_category} step {self.global_step} -> {_save_path}")
        # ─────────────────────────────────────────────────────────────────────

        info = self._compute_gradient_diagnostics(grads, objectives, g_combined=g_config)
        info["config_used"]           = 1.0
        info["config_num_objectives"] = float(len(objectives))
        info.update(_verify_info)

        # ── Step-level angle logger (set step_log_epoch in YAML to enable) ──
        step_log_epoch = getattr(self, "params", {}).get("step_log_epoch", -1)
        current_epoch  = getattr(self, "epoch", -1)
        if step_log_epoch >= 0 and current_epoch == step_log_epoch:
            import csv
            log_path = self.doc.get_file(f"step_angles_epoch{step_log_epoch}.csv")
            write_header = not os.path.exists(log_path)
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "global_step",
                        "angle_diff_vox",
                        "angle_combined_diff",
                        "angle_combined_vox",
                        "bisector_gap",
                        "norm_diff",
                        "norm_vox",
                        "norm_combined",
                    ])
                writer.writerow([
                    self.global_step,
                    info.get("angle_deg_diffusion_loss__voxel_energy_loss", float("nan")),
                    info.get("angle_deg_combined__diffusion_loss",          float("nan")),
                    info.get("angle_deg_combined__voxel_energy_loss",       float("nan")),
                    info.get("angle_deg_combined__diffusion_loss",          float("nan"))
                    - info.get("angle_deg_combined__voxel_energy_loss",     float("nan")),
                    info.get("gnorm_diffusion_loss",    float("nan")),
                    info.get("gnorm_voxel_energy_loss", float("nan")),
                    info.get("gnorm_combined",          float("nan")),
                ])
        # ────────────────────────────────────────────────────────────────────

        return info

    # ─────────────────────────────────────────────────────────────────────────
    # Gradient Blending
    # ─────────────────────────────────────────────────────────────────────────

    def _step_with_grad_blend(self, loss_tensors: dict) -> dict:
        """
        Gradient blending with:
          - Adaptive alpha: bisector direction with linear decay toward diffusion
            as conflict grows between blend_alpha_decay_start and blend_fallback_threshold.
          - GradNorm adaptive magnitude: w_diff*||g_diff|| + w_vox*||g_vox||.
          - Unweighted loss for gradient: uses raw loss tensor when
            use_unweighted_loss_for_grad=True (avoids lambda schedule bias).
          - Conflict fallback: pure diffusion gradient above blend_fallback_threshold.
          - NaN logging: tracks grad_ok failures for diagnostics.
        """
        info       = {}
        objectives = self._get_config_objectives(loss_tensors)

        assert objectives[0] == "diffusion_loss", (
            f"grad_blend requires diffusion_loss at index 0, got: {objectives}"
        )
        if len(objectives) < 2:
            return {"blend_used": 0.0}

        # ── Use unweighted losses for gradient computation if flag set ────────
        use_unweighted = getattr(self, 'use_unweighted_loss_for_grad', False)
        # ── Step 1: GradNorm FIRST (before graph is freed) ──────────────────
        # GradNorm uses autograd.grad with retain_graph=True internally,
        # so the graph stays intact. _compute_per_objective_grads runs after
        # and frees the graph normally.
        gradnorm_info  = {}
        beta_magnitude = None   # filled after norms are computed below
        self.blend_anchor_magnitude_to_diff = get(self.params, 'blend_anchor_magnitude_to_diff', False)
        
        if self.gradnorm_balancer is not None:
            losses_for_gradnorm = []
            for k in objectives:
                unweighted_key = k + '_unweighted'
                if use_unweighted and unweighted_key in loss_tensors:
                    losses_for_gradnorm.append(loss_tensors[unweighted_key])
                else:
                    losses_for_gradnorm.append(loss_tensors[k])
            shared_params = list(self.net.parameters())
            _, gradnorm_info, _ = self.gradnorm_balancer.step(
                losses=losses_for_gradnorm,
                objectives=objectives,
                shared_params=shared_params,
                all_params=None,
            )
        # ── Step 2: Per-objective grads (frees graph) ─────────────────────────
        grad_loss_tensors = {}
        for k in objectives:
            unweighted_key = k + '_unweighted'
            if use_unweighted and unweighted_key in loss_tensors:
                grad_loss_tensors[k] = loss_tensors[unweighted_key]
            else:
                grad_loss_tensors[k] = loss_tensors[k]

        grads, grad_ok = self._compute_per_objective_grads(grad_loss_tensors, objectives)

        if not grad_ok:
            if not hasattr(self, '_grad_nan_count'):
                self._grad_nan_count = 0
            self._grad_nan_count += 1
            apply_gradient_vector(self.net, grads[0])
            return {
                "blend_used":     0.0,
                "grad_nan":       1.0,
                "grad_nan_total": float(self._grad_nan_count),
            }

        alpha = self.blend_alpha
        norms = [float(torch.linalg.vector_norm(g).item()) for g in grads]
        units = [g / (n + 1e-10) for g, n in zip(grads, norms)]

        # ── Step 3a: Magnitude control ────────────────────────────────────────
        use_gref_clipping = get(self.params, 'blend_use_gref_clipping', False)

        if use_gref_clipping:
            # professor's framework: g_ref = ||g_diff|| x cos(theta_combined_diff)
            # compute combined unit direction first (before clipping)
            g_dir_pre = units[0].clone()
            aux_w_pre = self.blend_alpha / max(len(objectives) - 1, 1)
            for u in units[1:]:
                g_dir_pre = g_dir_pre + aux_w_pre * u
            dir_norm_pre    = float(torch.linalg.vector_norm(g_dir_pre).item())
            g_combined_unit = g_dir_pre / (dir_norm_pre + 1e-10)

            cos_diff = float(torch.dot(units[0],
                             g_combined_unit).clamp(-1, 1).item())
            g_ref    = norms[0] * max(cos_diff, 0.05)   # floor at 0.05

            # clip both gradients to g_ref scale
            scale_diff = min(1.0, g_ref / (norms[0] + 1e-8))
            scale_vox  = min(1.0, g_ref / (norms[1] + 1e-8))

            grads[0] = grads[0] * scale_diff
            grads[1] = grads[1] * scale_vox

            # recompute norms and units after clipping
            norms = [float(torch.linalg.vector_norm(g).item()) for g in grads]
            units = [g / (n + 1e-10) for g, n in zip(grads, norms)]

            beta_magnitude = norms[0]   # magnitude now controlled by clipping
            #beta_magnitude = min(norms[0], norms[1])

            # log clipping diagnostics
            info['blend_gref']       = g_ref
            info['blend_scale_diff'] = scale_diff
            info['blend_scale_vox']  = scale_vox
            info['blend_cos_diff']   = cos_diff

        elif self.gradnorm_balancer is not None and not self.blend_anchor_magnitude_to_diff:
            w = self.gradnorm_balancer.weights.detach()
            beta_magnitude = float(sum(w[i] * norms[i] for i in range(len(norms))))
        else:
            beta_magnitude = norms[0]

        # ── ensure gref keys always exist for consistent CSV columns ──────────
        info.setdefault('blend_gref',       float('nan'))
        info.setdefault('blend_scale_diff', float('nan'))
        info.setdefault('blend_scale_vox',  float('nan'))
        info.setdefault('blend_cos_diff',   float('nan'))   
        # save gref info before overwrite
        gref_info = {k: info[k] for k in
                     ['blend_gref','blend_scale_diff','blend_scale_vox','blend_cos_diff']
                     if k in info}
          
        # ── Step 3b: magnitude ratio logging ─────────────────────────────────
        _mag_ratio  = beta_magnitude / (norms[0] + 1e-8)
        _mag_capped = 0.0   # clipping handled in Step 3a
                
        # ── Step 4: Compute conflict angle ────────────────────────────────────
        import math as _math
        cos_theta = float(torch.dot(units[0], units[1]).clamp(-1, 1).item())
        theta_deg = _math.acos(cos_theta) * 180.0 / _math.pi

        fallback_threshold = getattr(self, 'blend_fallback_threshold', 150.0)
        alpha_decay_start  = getattr(self, 'blend_alpha_decay_start',  120.0)

        # ── Step 5: Adaptive alpha ────────────────────────────────────────────
        if theta_deg <= alpha_decay_start:
            effective_alpha = alpha
        elif theta_deg <= fallback_threshold:
            t = (theta_deg - alpha_decay_start) / (fallback_threshold - alpha_decay_start)
            effective_alpha = alpha * (1.0 - t)
        else:
            effective_alpha = 0.0

        g_direction = units[0].clone()
        aux_weight  = effective_alpha / max(len(objectives) - 1, 1)
        for u in units[1:]:
            g_direction = g_direction + aux_weight * u

        dir_norm = float(torch.linalg.vector_norm(g_direction).item())

        # ── Step 6: Conflict fallback ─────────────────────────────────────────
        if theta_deg > fallback_threshold or dir_norm < 1e-3:
            g_blend = norms[0] * units[0]
            apply_gradient_vector(self.net, g_blend) 
            info = self._compute_gradient_diagnostics(grads, objectives, g_combined=g_blend)
            info.update(gref_info)
            info["blend_used"]              = 1.0
            info["blend_alpha"]             = alpha
            info["blend_effective_alpha"]   = 0.0
            info["blend_beta"]              = norms[0]
            info["blend_theta_deg"]         = theta_deg
            info["blend_conflict_fallback"] = 1.0
            info["config_num_objectives"]   = float(len(objectives))
            info["grad_nan"]                = 0.0
            info["grad_nan_total"]          = float(getattr(self, '_grad_nan_count', 0))
            info["blend_magnitude_ratio"]  = _mag_ratio
            info["blend_magnitude_capped"] = _mag_capped
            info.update({f"blend_{k}": v for k, v in gradnorm_info.items()})
            return info

        beta    = beta_magnitude / (dir_norm + 1e-10)
        g_blend = beta_magnitude * (g_direction / (dir_norm + 1e-10))
        apply_gradient_vector(self.net, g_blend)

        info = self._compute_gradient_diagnostics(grads, objectives, g_combined=g_blend)
        info.update(gref_info)
        info["blend_used"]              = 1.0
        info["blend_alpha"]             = alpha
        info["blend_effective_alpha"]   = effective_alpha
        info["blend_beta"]              = beta
        info["blend_theta_deg"]         = theta_deg
        info["config_num_objectives"]   = float(len(objectives))
        info["blend_conflict_fallback"] = 0.0
        info["grad_nan"]                = 0.0
        info["grad_nan_total"]          = float(getattr(self, '_grad_nan_count', 0))
        info["blend_magnitude_ratio"]  = _mag_ratio
        info["blend_magnitude_capped"] = _mag_capped
        info.update({f"blend_{k}": v for k, v in gradnorm_info.items()})

        return info

    # ─────────────────────────────────────────────────────────────────────────
    # PCGrad
    # ─────────────────────────────────────────────────────────────────────────
    def _verify_mo_applied(self, grads, g_combined, method):
        """
        Confirm the gradient sitting in p.grad equals g_combined (the method's
        output) and quantify how far it is from pure diffusion (grads[0]).
        Returns a dict of verify_* scalars for the CSV. Cheap-ish: one flatten.
        """
        if not get(self.params, "verify_mo", False):
            return {}
        import torch
        with torch.no_grad():
            applied = torch.cat([
                (p.grad.detach().flatten()
                 if p.grad is not None
                 else torch.zeros(p.numel(), device=g_combined.device))
                for p in self.net.parameters()
            ])
            g_diff = grads[0]

            match = torch.allclose(applied, g_combined, atol=1e-6, rtol=1e-4)
            rel_to_combined = float((applied - g_combined).norm()
                                    / (g_combined.norm() + 1e-12))
            cos_applied_diff = float(torch.dot(applied, g_diff)
                                     / (applied.norm() * g_diff.norm() + 1e-12))
            rel_to_diff = float((applied - g_diff).norm()
                                / (g_diff.norm() + 1e-12))
            # raw conflict of the first aux objective for context
            cos_diff_aux = float("nan")
            if len(grads) > 1:
                cos_diff_aux = float(torch.dot(grads[0], grads[1])
                                     / (grads[0].norm() * grads[1].norm() + 1e-12))

            tag = "OK" if match else "MISMATCH!!"
            print(f"[verify_mo:{method}] step={self.global_step:6d} {tag} | "
                  f"applied==combined: {match} (rel={rel_to_combined:.2e}) | "
                  f"vs g_diff: cos={cos_applied_diff:+.4f} rel={rel_to_diff:.3f} | "
                  f"cos(diff,aux)={cos_diff_aux:+.4f}")

            return {
                "verify_applied_eq_combined": 1.0 if match else 0.0,
                "verify_rel_to_combined":     rel_to_combined,
                "verify_cos_applied_diff":    cos_applied_diff,
                "verify_rel_to_diff":         rel_to_diff,
                "verify_cos_diff_aux":        cos_diff_aux,
            }
    def _step_with_pcgrad(self, loss_tensors: dict) -> dict:
        """
        PCGrad — Gradient Surgery (Yu et al., NeurIPS 2020).

        For each gradient g_i, remove the projection onto g_j when they conflict:
            if dot(g_i, g_j) < 0:
                g_i = g_i - ( dot(g_i, g_j) / ||g_j||^2 ) * g_j

        Projected gradients are summed and applied.
        """
        info       = {}
        objectives = self._get_config_objectives(loss_tensors)

        assert objectives[0] == "diffusion_loss", (
            f"pcgrad requires diffusion_loss at index 0, got: {objectives}"
        )
        if len(objectives) < 2:
            return {"pcgrad_used": 0.0}

        grads, grad_ok = self._compute_per_objective_grads(loss_tensors, objectives)

        if not grad_ok:
            apply_gradient_vector(self.net, grads[0])
            return {"pcgrad_used": 0.0}

        g_pc = PCGradOperator().calculate_gradient(grads)
        apply_gradient_vector(self.net, g_pc)

        info = self._compute_gradient_diagnostics(grads, objectives, g_combined=g_pc)

        # pre-projection conflict count (how many pairs PCGrad had to handle)
        n_conflicts = sum(
            1 for i in range(len(grads)) for j in range(i + 1, len(grads))
            if float(torch.dot(grads[i], grads[j]).item()) < 0
        )
        info["pcgrad_used"]           = 1.0
        info["pcgrad_conflict_pairs"] = float(n_conflicts)
        info["config_num_objectives"] = float(len(objectives))

        return info

    # ─────────────────────────────────────────────────────────────────────────
    # IMTL-G
    # ─────────────────────────────────────────────────────────────────────────

    def _step_with_imtlg(self, loss_tensors: dict) -> dict:
        """
        IMTL-G — Impartial Multi-Task Learning via Gradient (Liu et al., ICLR 2021).
        Finds weights α_i such that û_i · g_combined is equal for all i.
        """
        info       = {}
        objectives = self._get_config_objectives(loss_tensors)

        assert objectives[0] == "diffusion_loss", (
            f"imtlg requires diffusion_loss at index 0, got: {objectives}"
        )
        if len(objectives) < 2:
            return {"imtlg_used": 0.0}

        grads, grad_ok = self._compute_per_objective_grads(loss_tensors, objectives)

        if not grad_ok:
            apply_gradient_vector(self.net, grads[0])
            return {"imtlg_used": 0.0}

        g_imtlg = IMTLGOperator().calculate_gradient(grads)
        apply_gradient_vector(self.net, g_imtlg)

        info = self._compute_gradient_diagnostics(grads, objectives, g_combined=g_imtlg)

        # impartiality check: û_i · g_imtlg should be equal for all i
        grads_stack = torch.stack(grads)
        norms_vec   = grads_stack.norm(dim=1, keepdim=True).clamp(min=1e-10)
        units       = grads_stack / norms_vec
        for name, u in zip(objectives, units):
            info[f"imtlg_proj_{name}"] = float(torch.dot(u, g_imtlg).item())

        info["imtlg_used"]            = 1.0
        info["config_num_objectives"] = float(len(objectives))

        return info

    # ─────────────────────────────────────────────────────────────────────────
    # GradNorm
    # ─────────────────────────────────────────────────────────────────────────

    def _step_with_gradnorm(self, loss_tensors: dict) -> dict:
        """
        GradNorm adaptive task weighting (Chen et al., ICML 2018).

        Logging matched to grad_blend: emits the shared gradient diagnostics
        (gnorm_*, gnorm_ratio_*, cos_*__*, angle_deg_*__*, gnorm_combined,
        cos_combined__*) via _compute_gradient_diagnostics, the per-task
        GradNorm weights (gradnorm_weight_*), config_num_objectives, and the
        balancer's own info (prefixed gradnorm_*).

        The applied gradient is sum_i w_i * g_i, identical to backprop through
        L_total = sum_i w_i * L_i with detached w_i. It is built explicitly so
        the same per-objective flat gradients used by ConFIG / PCGrad / IMTL-G
        / grad_blend are available for the diagnostics.
        """
        if self.gradnorm_balancer is None:
            loss_tensors['diffusion_loss'].backward()
            return {"gradnorm_used": 0.0}

        objectives = getattr(self, 'gradnorm_objectives', None)
        if objectives is None:
            loss_tensors['diffusion_loss'].backward()
            return {"gradnorm_used": 0.0}

        objectives = [k for k in objectives if k in loss_tensors]

        if not objectives or objectives[0] != "diffusion_loss":
            loss_tensors['diffusion_loss'].backward()
            return {"gradnorm_used": 0.0}

        if len(objectives) < 2:
            loss_tensors['diffusion_loss'].backward()
            return {"gradnorm_used": 0.0}

        if len(objectives) != self.gradnorm_balancer.num_tasks:
            loss_tensors['diffusion_loss'].backward()
            return {"gradnorm_used": 0.0, "gradnorm_task_mismatch": 1.0}

        losses = [loss_tensors[k] for k in objectives]

        # ── Step 1: GradNorm weight update FIRST (keeps graph alive) ─────────
        # balancer.step uses autograd.grad(retain_graph=True) internally, so
        # the graph stays intact for the per-objective backward calls below.
        # shared_params is the full network, matching _step_with_grad_blend so
        # the baseline and grad_blend update task weights on the same params.
        shared_params = list(self.net.parameters())

        _, gradnorm_info, _ = self.gradnorm_balancer.step(
            losses=losses,
            objectives=objectives,
            shared_params=shared_params,
            all_params=None,
        )

        # ── Step 2: Per-objective full-network grads (frees graph) ───────────
        grads, grad_ok = self._compute_per_objective_grads(loss_tensors, objectives)

        if not grad_ok:
            if not hasattr(self, '_grad_nan_count'):
                self._grad_nan_count = 0
            self._grad_nan_count += 1
            apply_gradient_vector(self.net, grads[0])
            info = {
                "gradnorm_used":  0.0,
                "grad_nan":       1.0,
                "grad_nan_total": float(self._grad_nan_count),
            }
            info.update({f"gradnorm_{k}": v for k, v in gradnorm_info.items()})
            return info

        # ── Step 3: Build GradNorm-weighted combined gradient ────────────────
        w = self.gradnorm_balancer.weights.detach()
        g_combined = torch.zeros_like(grads[0])
        for i in range(len(objectives)):
            g_combined = g_combined + float(w[i]) * grads[i]

        apply_gradient_vector(self.net, g_combined)

        # ── Step 4: Diagnostics matched to grad_blend ────────────────────────
        info = self._compute_gradient_diagnostics(grads, objectives, g_combined=g_combined)
        info["gradnorm_used"]         = 1.0
        info["config_num_objectives"] = float(len(objectives))
        info["grad_nan"]              = 0.0
        info["grad_nan_total"]        = float(getattr(self, '_grad_nan_count', 0))
        for i, name in enumerate(objectives):
            info[f"gradnorm_weight_{name}"] = float(w[i])
        info.update({f"gradnorm_{k}": v for k, v in gradnorm_info.items()})

        return info

    # ============================================================
    # TRAINING LOOP
    # ============================================================

    def train_one_epoch(self):
        """Train for one epoch. batch_loss must return (loss_tensors:dict, loss_scalars:dict)."""
        self.net.train()

        batch_metrics = {}
        skipped  = 0
        n_batches = 0

        mo_method = getattr(self, "mo_method", "weighted_sum")

        for batch_id, x in enumerate(self.train_loader):

            n_batches += 1
            self.global_step += 1
            self.optimizer.zero_grad(set_to_none=True)

            loss_tensors, loss_scalars = self.batch_loss(x)

            loss_scalars = {
                k: (float(v.detach().item()) if hasattr(v, "detach") else float(v))
                for k, v in (loss_scalars or {}).items()
            }

            did_step = 0.0
            mo_info  = {}

            # ==========================================
            # Branch 1: NO MULTI-OBJECTIVE — pure diffusion
            # Gradient conflict diagnostics run here
            # every grad_diag_every optimizer steps.
            # ==========================================
            if mo_method == "none":
                total_loss = loss_tensors['diffusion_loss']
                loss_scalars["total_loss"] = float(total_loss.detach().item())

                if np.isfinite(loss_scalars["total_loss"]):

                    # ── gradient conflict diagnostics ────────────────────────
                    # Fires every grad_diag_every steps (0 disables).
                    # Calls backward(retain_graph=True) per aux loss, then
                    # zeros grads. Graph stays alive for main backward below.
                    if (self.grad_diag_every > 0
                            and self.global_step % self.grad_diag_every == 0):
                        diag = self._compute_conflict_diagnostics_only(loss_tensors)
                        # keys already use '_' not '/' — safe CSV column names
                        loss_scalars.update(diag)
                    # ─────────────────────────────────────────────────────────

                    total_loss.backward()

                    clip = self.params.get("clip_gradients_to", None)
                    if clip:
                        nn.utils.clip_grad_norm_(self.net.parameters(), clip)

                    self.optimizer.step()
                    did_step = 1.0

                    if hasattr(self, "update_ema"):
                        self.update_ema()

                    if getattr(self, "use_scheduler", False) and getattr(self, "scheduler_step_per_batch", True):
                        self.scheduler.step()
                else:
                    skipped += 1
                    print(f"Unstable loss at epoch {self.epoch}, batch {batch_id} (skipping step)")

            # ==========================================
            # Branch 2: Weighted sum
            # ==========================================
            elif mo_method == "weighted_sum":
                total_loss, mo_info = self.combine_losses(loss_tensors)

                mo_info = {
                    k: (float(v.detach().item()) if hasattr(v, "detach") else float(v))
                    for k, v in (mo_info or {}).items()
                }

                total_loss_scalar = float(total_loss.detach().item())
                loss_scalars["total_loss"] = total_loss_scalar
                loss_scalars.update(mo_info)

                if np.isfinite(total_loss_scalar):

                    # ── pre-backward conflict diagnostics ────────────────────
                    # Computes per-loss gradients via retain_graph=True, logs
                    # raw conflict stats, then zeros grads before the combined
                    # backward below. Produces identical grad_diag_* columns
                    # as Branch 1 for direct comparison across runs.
                    if (self.grad_diag_every > 0
                            and self.global_step % self.grad_diag_every == 0):
                        diag = self._compute_conflict_diagnostics_only(loss_tensors)
                        loss_scalars.update(diag)
                    # ────────────────────────────────────────────────────────

                    total_loss.backward()

                    clip = self.params.get("clip_gradients_to", None)
                    if clip:
                        nn.utils.clip_grad_norm_(self.net.parameters(), clip)

                    self.optimizer.step()
                    did_step = 1.0

                    if hasattr(self, "update_ema"):
                        self.update_ema()

                    if getattr(self, "use_scheduler", False) and getattr(self, "scheduler_step_per_batch", True):
                        self.scheduler.step()
                else:
                    skipped += 1
                    print(f"Unstable loss at epoch {self.epoch}, batch {batch_id} (skipping step)")
            # ==========================================
            # Branch: Uncertainty Weighting (Kendall et al. 2018)
            # ==========================================
            elif mo_method == "uncertainty":
                total_loss, mo_info = self.combine_losses(loss_tensors)

                mo_info = {
                    k: (float(v.detach().item()) if hasattr(v, "detach") else float(v))
                    for k, v in (mo_info or {}).items()
                }

                total_loss_scalar = float(total_loss.detach().item())
                loss_scalars["total_loss"] = total_loss_scalar
                loss_scalars.update(mo_info)

                if np.isfinite(total_loss_scalar):

                    if (self.grad_diag_every > 0
                            and self.global_step % self.grad_diag_every == 0):
                        diag = self._compute_conflict_diagnostics_only(loss_tensors)
                        loss_scalars.update(diag)

                    total_loss.backward()

                    clip = self.params.get("clip_gradients_to", None)
                    if clip:
                        nn.utils.clip_grad_norm_(self.net.parameters(), clip)

                    self.optimizer.step()
                    did_step = 1.0

                    if hasattr(self, "update_ema"):
                        self.update_ema()

                    if getattr(self, "use_scheduler", False) and getattr(self, "scheduler_step_per_batch", True):
                        self.scheduler.step()
                else:
                    skipped += 1
                    print(f"Unstable loss at epoch {self.epoch}, batch {batch_id} (skipping step)")
            # ==========================================
            # Branch 3: ConFIG
            # ==========================================
            elif mo_method == "config":
                report_total, report_info = self.combine_losses(loss_tensors)
                report_info = {
                    k: (float(v.detach().item()) if hasattr(v, "detach") else float(v))
                    for k, v in (report_info or {}).items()
                }
                loss_scalars["total_loss"] = float(report_total.detach().item())
                loss_scalars.update(report_info)

                if np.isfinite(loss_scalars["total_loss"]):

                    # ── pre-method conflict diagnostics ──────────────────────
                    # Logs raw gradient conflict BEFORE ConFIG transforms the
                    # gradients. config_info below logs the post-method stats
                    # (gnorm_combined, cos_combined__*) separately.
                    if (self.grad_diag_every > 0
                            and self.global_step % self.grad_diag_every == 0):
                        diag = self._compute_conflict_diagnostics_only(loss_tensors)
                        loss_scalars.update(diag)
                    # ────────────────────────────────────────────────────────

                    config_info = self._step_with_config(loss_tensors)

                    clip = self.params.get("clip_gradients_to", None)
                    if clip:
                        nn.utils.clip_grad_norm_(self.net.parameters(), clip)

                    self.optimizer.step()
                    did_step = 1.0

                    if hasattr(self, "update_ema"):
                        self.update_ema()

                    if getattr(self, "use_scheduler", False) and getattr(self, "scheduler_step_per_batch", True):
                        self.scheduler.step()

                    loss_scalars.update(config_info)
                else:
                    skipped += 1
                    loss_scalars["config_used"] = 0.0
                    print(f"Unstable total_loss at epoch {self.epoch}, batch {batch_id} (skipping step)")

            # ==========================================
            # Branch 4: Gradient Blending
            # ==========================================
            elif mo_method == "grad_blend":
                report_total, report_info = self.combine_losses(loss_tensors)
                report_info = {
                    k: (float(v.detach().item()) if hasattr(v, "detach") else float(v))
                    for k, v in (report_info or {}).items()
                }
                loss_scalars["total_loss"] = float(report_total.detach().item())
                loss_scalars.update(report_info)

                if np.isfinite(loss_scalars["total_loss"]):

                    # ── pre-method conflict diagnostics ──────────────────────
                    # Logs raw gradient conflict BEFORE blending transforms the
                    # gradients. blend_info below logs post-method stats
                    # (gnorm_combined, cos_combined__*, blend_beta) separately.
                    if (self.grad_diag_every > 0
                            and self.global_step % self.grad_diag_every == 0):
                        diag = self._compute_conflict_diagnostics_only(loss_tensors)
                        loss_scalars.update(diag)
                    # ────────────────────────────────────────────────────────

                    blend_info = self._step_with_grad_blend(loss_tensors)

                    clip = self.params.get("clip_gradients_to", None)
                    if clip:
                        nn.utils.clip_grad_norm_(self.net.parameters(), clip)

                    self.optimizer.step()
                    did_step = 1.0

                    if hasattr(self, "update_ema"):
                        self.update_ema()

                    if getattr(self, "use_scheduler", False) and getattr(self, "scheduler_step_per_batch", True):
                        self.scheduler.step()

                    loss_scalars.update(blend_info)
                else:
                    skipped += 1
                    loss_scalars["blend_used"] = 0.0
                    print(f"Unstable total_loss at epoch {self.epoch}, batch {batch_id} (skipping step)")

            # ==========================================
            # Branch 5: PCGrad
            # ==========================================
            elif mo_method == "pcgrad":
                report_total, report_info = self.combine_losses(loss_tensors)
                report_info = {
                    k: (float(v.detach().item()) if hasattr(v, "detach") else float(v))
                    for k, v in (report_info or {}).items()
                }
                loss_scalars["total_loss"] = float(report_total.detach().item())
                loss_scalars.update(report_info)

                if np.isfinite(loss_scalars["total_loss"]):
                    pcgrad_info = self._step_with_pcgrad(loss_tensors)

                    clip = self.params.get("clip_gradients_to", None)
                    if clip:
                        nn.utils.clip_grad_norm_(self.net.parameters(), clip)

                    self.optimizer.step()
                    did_step = 1.0

                    if hasattr(self, "update_ema"):
                        self.update_ema()

                    if getattr(self, "use_scheduler", False) and getattr(self, "scheduler_step_per_batch", True):
                        self.scheduler.step()

                    loss_scalars.update(pcgrad_info)
                else:
                    skipped += 1
                    loss_scalars["pcgrad_used"] = 0.0
                    print(f"Unstable total_loss at epoch {self.epoch}, batch {batch_id} (skipping step)")

            # ==========================================
            # Branch 6: IMTL-G
            # ==========================================
            elif mo_method == "imtlg":
                report_total, report_info = self.combine_losses(loss_tensors)
                report_info = {
                    k: (float(v.detach().item()) if hasattr(v, "detach") else float(v))
                    for k, v in (report_info or {}).items()
                }
                loss_scalars["total_loss"] = float(report_total.detach().item())
                loss_scalars.update(report_info)

                if np.isfinite(loss_scalars["total_loss"]):
                    imtlg_info = self._step_with_imtlg(loss_tensors)

                    clip = self.params.get("clip_gradients_to", None)
                    if clip:
                        nn.utils.clip_grad_norm_(self.net.parameters(), clip)

                    self.optimizer.step()
                    did_step = 1.0

                    if hasattr(self, "update_ema"):
                        self.update_ema()

                    if getattr(self, "use_scheduler", False) and getattr(self, "scheduler_step_per_batch", True):
                        self.scheduler.step()

                    loss_scalars.update(imtlg_info)
                else:
                    skipped += 1
                    loss_scalars["imtlg_used"] = 0.0
                    print(f"Unstable total_loss at epoch {self.epoch}, batch {batch_id} (skipping step)")

            # ==========================================
            # Branch 7: GradNorm
            # ==========================================
            elif mo_method == "gradnorm":
                report_total, report_info = self.combine_losses(loss_tensors)
                report_info = {
                    k: (float(v.detach().item()) if hasattr(v, "detach") else float(v))
                    for k, v in (report_info or {}).items()
                }
                loss_scalars["total_loss"] = float(report_total.detach().item())
                loss_scalars.update(report_info)

                if np.isfinite(loss_scalars["total_loss"]):
                    gradnorm_info = self._step_with_gradnorm(loss_tensors)

                    clip = self.params.get("clip_gradients_to", None)
                    if clip:
                        nn.utils.clip_grad_norm_(self.net.parameters(), clip)

                    self.optimizer.step()
                    did_step = 1.0

                    if hasattr(self, "update_ema"):
                        self.update_ema()

                    if getattr(self, "use_scheduler", False) and getattr(self, "scheduler_step_per_batch", True):
                        self.scheduler.step()

                    loss_scalars.update(gradnorm_info)
                else:
                    skipped += 1
                    loss_scalars["gradnorm_used"] = 0.0
                    print(f"Unstable total_loss at epoch {self.epoch}, batch {batch_id} (skipping step)")

            else:
                raise ValueError(f"Unknown mo_method: '{mo_method}'. "
                     f"Valid options: none, weighted_sum, config, grad_blend, pcgrad, imtlg, gradnorm, uncertainty")

            loss_scalars["did_step"] = float(did_step)

            for k, v in loss_scalars.items():
                batch_metrics.setdefault(k, []).append(float(v))

        # Epoch means
        epoch_stats = {
            k: float(np.mean(v)) if len(v) > 0 else float("nan")
            for k, v in batch_metrics.items()
        }
        epoch_stats["num_batches"] = float(n_batches)
        epoch_stats["num_skipped"] = float(skipped)

        self._append_epoch_metrics(prefix="train", epoch_stats=epoch_stats)

        if getattr(self, "use_scheduler", False) and not getattr(self, "scheduler_step_per_batch", True):
            self.scheduler.step()

        if getattr(self, "log", False):
            for k, v in epoch_stats.items():
                self.logger.add_scalar(f"train/{k}", v, self.epoch)
            if getattr(self, "use_scheduler", False):
                try:
                    self.logger.add_scalar("train/learning_rate", self.scheduler.get_last_lr()[0], self.epoch)
                except Exception:
                    pass

        if hasattr(self, "_print_epoch_summary"):
            self._print_epoch_summary(epoch_stats)

        return epoch_stats

    def _append_epoch_metrics(self, prefix: str, epoch_stats: dict):
        """
        Store metrics into numpy arrays with names like:
          train_total_loss_epoch, train_diffusion_loss_epoch, ...
        """
        for k, v in epoch_stats.items():
            attr = f"{prefix}_{k}_epoch"
            if not hasattr(self, attr):
                setattr(self, attr, np.array([], dtype=np.float64))
            arr = getattr(self, attr)
            setattr(self, attr, np.append(arr, float(v)))

    @torch.inference_mode()
    def validate_one_epoch(self):
        self.net.eval()
        batch_metrics = {}
        n_batches = 0

        for batch_id, x in enumerate(self.val_loader):
            n_batches += 1
            loss_tensors, loss_scalars = self.batch_loss(x)

            loss_scalars = {k: float(v.item() if hasattr(v, "item") else v)
                           for k, v in loss_scalars.items()}

            mo_method = getattr(self, "mo_method", "weighted_sum")

            if mo_method == "none":
                total_loss = loss_tensors['diffusion_loss']
                loss_scalars["total_loss"] = float(total_loss.item())
            else:
                total_loss, mo_info = self.combine_losses(loss_tensors)
                mo_info = {k: float(v.item() if hasattr(v, "item") else v)
                          for k, v in mo_info.items()}
                loss_scalars["total_loss"] = float(total_loss.item())
                loss_scalars.update(mo_info)

            for k, v in loss_scalars.items():
                batch_metrics.setdefault(k, []).append(float(v))

        epoch_stats = {k: np.mean(v) for k, v in batch_metrics.items()}
        epoch_stats["num_batches"] = n_batches

        self._append_epoch_metrics("val", epoch_stats)

        if self.log:
            for k, v in epoch_stats.items():
                self.logger.add_scalar(f"val/{k}", v, self.epoch)

        return epoch_stats

    def save_all_metrics(self):
        """Save all training/validation metrics to CSV."""
        import pandas as pd

        metrics = {}
        for attr_name in self.__dict__:
            if attr_name.endswith('_epoch') and not attr_name.startswith('_'):
                try:
                    arr = getattr(self, attr_name)
                    if isinstance(arr, np.ndarray) and arr.size > 0:
                        metrics[attr_name] = arr.reshape(-1).astype(np.float64)
                except Exception:
                    pass

        if not metrics:
            return

        max_len = max(len(v) for v in metrics.values())
        padded = {
            k: np.concatenate([v, np.full(max_len - len(v), np.nan)])
            if len(v) < max_len else v
            for k, v in metrics.items()
        }

        padded['epoch'] = np.arange(max_len)

        df = pd.DataFrame(padded)
        train_cols = sorted([c for c in df.columns if c.startswith('train_')])
        val_cols   = sorted([c for c in df.columns if c.startswith('val_')])
        other_cols = sorted([c for c in df.columns if c not in train_cols + val_cols + ['epoch']])
        df = df[['epoch'] + train_cols + val_cols + other_cols]

        csv_path = self.doc.get_file("train_val_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved {len(df.columns)-1} metrics to CSV")

    def batch_loss(self, x):
        pass

    def generate_Einc_ds1(self, energy=None, sample_multiplier=1000):
        ret = np.logspace(8, 18, 11, base=2)
        ret = np.tile(ret, 10)
        ret = np.array(
            [*ret, *np.tile(2. ** 19, 5), *np.tile(2. ** 20, 3), *np.tile(2. ** 21, 2), *np.tile(2. ** 22, 1)])
        ret = np.tile(ret, sample_multiplier)
        if energy is not None:
            ret = ret[ret == energy]
        np.random.shuffle(ret)
        return ret

    @torch.inference_mode()
    def sample_trained_model_energy(self, size=10**5, sampling_type='ddim'):
        self.eval()
        energy_model = self.load_other(self.params['energy_model'])
        t_0 = time.time()

        Einc = torch.tensor(
            10**np.random.uniform(3, 6, size=get(self.params, "n_samples", 10**5))
            if self.params['eval_dataset'] in ['2', '3'] else
            self.generate_Einc_ds1(energy=self.single_energy),
            dtype=torch.get_default_dtype(),
            device=self.device
        ).unsqueeze(1)

        dummy, transformed_cond = None, Einc
        print("before starting the loop Einc min and max: ", Einc.min(), Einc.max())
        for fn in self.transforms:
            if hasattr(fn, 'cond_transform'):
                dummy, transformed_cond = fn(dummy, transformed_cond)
                print("Einc min and max: ", transformed_cond.min(), transformed_cond.max())

        batch_size_sample = get(self.params, "batch_size_sample", 10000)
        transformed_cond_loader = DataLoader(
            dataset=transformed_cond, batch_size=batch_size_sample, shuffle=False
        )
        sample = torch.vstack([energy_model.sample_batch(c, sampling_type=sampling_type).cpu()
                               for c in transformed_cond_loader])

        t_1 = time.time()
        sampling_time = t_1 - t_0
        self.params["sample_time"] = sampling_time
        print(f"generate_samples: Finished generating {len(sample)} samples after {sampling_time} s.", flush=True)

        return sample, transformed_cond.cpu()

    @torch.inference_mode()
    def sample_n(self, size=10**5):
        print("Inside sample_n")
        self.eval()

        t_0 = time.time()

        Einc = torch.tensor(
            10**np.random.uniform(3, 6, size=get(self.params, "n_samples", 10**3))
            if self.params['eval_dataset'] in ['2', '3'] else
            self.generate_Einc_ds1(energy=self.single_energy),
            dtype=torch.get_default_dtype(),
            device=self.device
        ).unsqueeze(1)

        dummy, transformed_cond = None, Einc
        print("creating dummy E_inc!!!!!!!!!")
        for fn in self.transforms:
            if hasattr(fn, 'cond_transform'):
                print("fn: ",fn)
                dummy, transformed_cond = fn(dummy, transformed_cond)

        batch_size_sample = get(self.params, "batch_size_sample", 100)
        transformed_cond_loader = DataLoader(
            dataset=transformed_cond, batch_size=batch_size_sample, shuffle=False
        )

        if self.params['model_type'] == 'shape':
            energy_model = self.load_other(self.params['energy_model'])

            if self.params.get('sample_us', False):
                print("starting from energy model........")
                u_samples = torch.vstack([
                    energy_model.sample_batch(c) for c in transformed_cond_loader
                ])

                dummy = torch.empty(1, 1)
                for fn in energy_model.transforms[::-1]:
                    if fn.__class__.__name__ == 'StandardizeFromFile':
                        fn.n_features = u_samples.shape[1]
                        u_samples, dummy = fn(u_samples, dummy, rev=True)

                # if self.latent:
                #     dummy = torch.empty(1, 1)
                #     for fn in energy_model.transforms[:0:-1]:
                #         u_samples, dummy = fn(u_samples, dummy, rev=True)
                transformed_cond = torch.cat([transformed_cond, u_samples], dim=1)
                # print("starting from raw.........")
                # transformed_cond_real = CaloChallengeDataset(
                #     self.params.get('eval_hdf5_file'),
                #     self.params.get('particle_type'),
                #     self.params.get('xml_filename'),
                #     transform=self.transforms,
                #     device=self.device,
                #     single_energy=self.single_energy
                # ).energy

                
                # # ── debug: save both conditions and stop ──
                # torch.save(transformed_cond.detach().cpu(),
                #            self.doc.get_file("debug_transformed_cond_sampled.pt"))
                # torch.save(transformed_cond_real.detach().cpu(),
                #            self.doc.get_file("debug_transformed_cond_real.pt"))
                # print(f"[debug] sampled cond: shape={tuple(transformed_cond.shape)}, "
                #       f"min={transformed_cond.min().item():.4f}, max={transformed_cond.max().item():.4f}")
                # print(f"[debug] real cond:    shape={tuple(transformed_cond_real.shape)}, "
                #       f"min={transformed_cond_real.min().item():.4f}, max={transformed_cond_real.max().item():.4f}")
                # raise SystemExit("[debug] Stopped after saving condition tensors.")
                
            else:
                print("generating from real data")
                transformed_cond = CaloChallengeDataset(
                    self.params.get('eval_hdf5_file'),
                    self.params.get('particle_type'),
                    self.params.get('xml_filename'),
                    transform=self.transforms,
                    device=self.device,
                    single_energy=self.single_energy
                ).energy

            transformed_cond_loader = DataLoader(
                dataset=transformed_cond, batch_size=batch_size_sample, shuffle=False
            )

        sample = torch.vstack([self.sample_batch(c).cpu() for c in transformed_cond_loader])

        t_1 = time.time()
        sampling_time = t_1 - t_0
        self.params["sample_time"] = sampling_time
        print(f"generate_samples: Finished generating {len(sample)} samples after {sampling_time} s.", flush=True)

        return sample, transformed_cond.cpu()

    def reconstruct_n(self):
        print("inside reconstruct_n")
        if not hasattr(self, 'train_loader'):
            self.train_loader, self.val_loader, self.bounds = get_loaders(
                self.params.get('hdf5_file'),
                self.params.get('particle_type'),
                self.params.get('xml_filename'),
                self.params.get('val_frac'),
                self.params.get('batch_size_sample'),
                self.transforms,
                self.params.get('eps', 1.e-10),
                device=self.device,
                shuffle=False,
                width_noise=self.params.get('width_noise', 1.e-6),
                single_energy=self.params.get('single_energy', None)
            )

        recos = []
        energies = []

        self.eval()
        for n, x in enumerate(self.train_loader):
            reco, cond = self.sample_batch(x)
            recos.append(reco)
            energies.append(cond)
        for n, x in enumerate(self.val_loader):
            reco, cond = self.sample_batch(x)
            recos.append(reco)
            energies.append(cond)

        recos    = torch.vstack(recos)
        energies = torch.vstack(energies)
        return recos, energies

    def sample_batch(self, batch):
        pass

    def plot_samples_old(self, sample_path, reference_path, doc):
        samples   = torch.load(sample_path)
        reference = torch.load(reference_path)

        samples[:,1:]   = torch.clip(samples[:,1:],   min=0., max=1.)
        reference[:,1:] = torch.clip(reference[:,1:], min=0., max=1.)

        samples_np   = samples.detach().cpu().numpy()
        reference_np = reference.detach().cpu().numpy()

        print("LBL....", self.params['LBL'])

        plot_ui_dists(samples_np, reference_np, documenter=doc, LBL=self.params['LBL'])
        evaluate.eval_ui_dists(samples_np, reference_np, documenter=doc, params=self.params)

    def plot_samples(self, samples, conditions, name="", energy=None, doc=None):
        print(f"Samples requires_grad {samples.requires_grad}")
        transforms = self.transforms
        if doc is None: doc = self.doc

        if self.params['model_type'] == 'energy':
            reference = CaloChallengeDataset(
                self.params.get('eval_hdf5_file'),
                self.params.get('particle_type'),
                self.params.get('xml_filename'),
                transform=transforms,
                device=self.device,
                single_energy=self.single_energy
            ).layers

            print("*******  Starting Reverse Transformation *******")
            for fn in transforms[::-1]:
                if fn.__class__.__name__ == 'NormalizeByElayer':
                    break
                print("Sampling information")
                samples, _ = fn(samples, conditions, rev=True)
                print("Geant4 information")
                reference, _ = fn(reference, conditions, rev=True)
            torch.save(samples, doc.get_file("samples.pt"))
            torch.save(reference, doc.get_file("reference.pt"))

            samples[:,1:]   = torch.clip(samples[:,1:],   min=0., max=1.)
            reference[:,1:] = torch.clip(reference[:,1:], min=0., max=1.)

            print("LBL....", self.params['LBL'])
            plot_ui_dists(
                samples.detach().cpu().numpy(),
                reference.detach().cpu().numpy(),
                documenter=doc,
                LBL=self.params['LBL']
            )
            print("before passing to the evaluate: ", samples.requires_grad, reference.requires_grad)
            evaluate.eval_ui_dists(
                samples.detach().cpu().numpy(),
                reference.detach().cpu().numpy(),
                documenter=doc,
                params=self.params,
            )
        else:
            if self.latent:
                self.save_sample(samples, conditions, name=name+'_latent', doc=doc)

            print("\n" + "="*80)
            print("Transform Tracking")
            print("="*80)
            print(f"{'Step':<5} {'Transform':<30} {'Shape':<20} {'Min':<12} {'Max':<12}")
            print("-"*80)

            print(f"{'Init':<5} {'Model Output':<30} {str(samples.shape):<20} {samples.min().item():<12.6f} {samples.max().item():<12.6f}")
            torch.save(samples.cpu(), self.doc.get_file('debug_samples_0_initial.pt'))

            for i, fn in enumerate(transforms[::-1]):
                samples, conditions = fn(samples, conditions, rev=True)

            print("="*80 + "\n")

            samples    = samples.detach().cpu().numpy()
            conditions = conditions.detach().cpu().numpy()

            self.save_sample(samples, conditions, name=name, doc=doc)

            evaluate.run_from_py(samples, conditions, doc, self.params, name=name)

    def plot_saved_samples(self, name="", energy=None, doc=None):
        if doc is None: doc = self.doc
        mode = self.params.get('eval_mode', 'all')
        script_args = (
            f"-i {doc.basedir}/ "
            f"-r {self.params['eval_hdf5_file']} -m {mode} --cut {self.params['eval_cut']} "
            f"-d {self.params['eval_dataset']} --output_dir {doc.basedir}/final/ --save_mem"
        ) + (f" --energy {energy}" if energy is not None else '')
        evaluate.main(script_args.split())

    def save_sample(self, sample, energies, name="", doc=None):
        if doc is None: doc = self.doc
        save_file = h5py.File(doc.get_file(f'samples_{name}.hdf5'), 'w')
        save_file.create_dataset('incident_energies', data=energies)
        save_file.create_dataset('showers', data=sample)
        save_file.close()

    def save(self, epoch=""):
        save_dict = {
            "opt":       self.optimizer.state_dict(),
            "net":       self.net.state_dict(),
            "epoch":     self.epoch,
            "scheduler": self.scheduler.state_dict()
        }
        if self.gradnorm_balancer is not None:
            save_dict["gradnorm_log_w"]   = self.gradnorm_balancer.log_w.data
            save_dict["gradnorm_L0"]      = self.gradnorm_balancer.L0
            save_dict["gradnorm_w_optim"] = self.gradnorm_balancer.w_optim.state_dict()

        if hasattr(self, 'ema_model') and self.ema_model is not None:
            save_dict["ema_net"] = self.ema_model.state_dict()

        torch.save(save_dict, self.doc.get_file(f"model{epoch}.pt"))

    def load(self, epoch=""):
        epoch=""
        name = self.doc.get_file(f"model{epoch}.pt")
        state_dicts = torch.load(name, map_location=self.device, weights_only=False)
        self.net.load_state_dict(state_dicts["net"])

        if "losses" in state_dicts:
            self.train_losses_epoch = state_dicts.get("losses", {})
        if "epoch" in state_dicts:
            self.epoch = state_dicts.get("epoch", 0)

        if "gradnorm_log_w" in state_dicts and self.gradnorm_balancer is not None:
            self.gradnorm_balancer.log_w.data = state_dicts["gradnorm_log_w"]
            self.gradnorm_balancer.L0         = state_dicts["gradnorm_L0"]
            self.gradnorm_balancer.w_optim.load_state_dict(state_dicts["gradnorm_w_optim"])

        self.net.to(self.device)

        if "ema_net" in state_dicts and hasattr(self, 'ema_model') and self.ema_model is not None:
            self.ema_model.load_state_dict(state_dicts["ema_net"])

    def load_other(self, model_dir):
        with open(os.path.join(model_dir, 'params.yaml')) as f:
            params_old = yaml.load(f, Loader=yaml.FullLoader)

        model_class = params_old['model']
        if model_class == 'TBD':
            Model = self.__class__
        elif model_class == 'TransfusionAR':
            from Models import TransfusionAR
            Model = TransfusionAR
        elif model_class == 'AE':
            from Models import AE
            Model = AE
        elif model_class == 'TransfusionDDPM':
            from Models import TransfusionDDPM
            Model = TransfusionDDPM
        elif model_class == 'TBD_DIFF':
            from Models import TBD_DIFF
            Model = TBD_DIFF
        else:
            raise ValueError(f"Unknown model_class: '{model_class}'")

        doc_trained = Documenter(None, existing_run=model_dir, read_only=True)
        other = Model(params_old, self.device, doc_trained)
        state_dicts = torch.load(
            os.path.join(model_dir, 'model.pt'), map_location=self.device, weights_only=False
        )
        other.net.load_state_dict(state_dicts["net"])

        other.eval()
        for p in other.parameters():
            p.requires_grad = False

        return other