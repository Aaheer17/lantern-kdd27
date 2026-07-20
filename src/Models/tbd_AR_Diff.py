import numpy as np
import torch
from scipy.integrate import solve_ivp
import Networks
from Util.util import get
from Models.ModelBase import GenerativeModel
import Networks
import Models
from einops import rearrange

import math
from typing import Type, Callable, Union, Optional
import torch
import torch.nn as nn

class TransfusionDDPM(GenerativeModel):

    def __init__(self, params: dict, device, doc):
        super().__init__(params, device, doc)
        self.params = params
        
        # DDPM specific parameters
        self.num_timesteps = get(self.params, "num_timesteps", 1000)
        self.beta_schedule = get(self.params, "beta_schedule", "linear")
        self.cfg= get(self.params, "cfg", False)
        self.drop_prob= get(self.params, "drop_prob", 0)
        
        # Initialize noise schedule
        self.setup_noise_schedule()
        
        self.dim_embedding = params["dim_embedding"]

    def setup_noise_schedule(self):
        """
        Setup the noise schedule for DDPM
        """
        if self.beta_schedule == "linear":
            beta_start = get(self.params, "beta_start", 1e-4)
            beta_end = get(self.params, "beta_end", 2e-2)
            self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        elif self.beta_schedule == "cosine":
            self.betas = self.cosine_beta_schedule(self.num_timesteps)
        elif self.beta_schedule == "sigmoid":
            self.betas = self.sigmoid_beta_schedule(self.num_timesteps) 
        else:
            raise NotImplementedError(f"Beta schedule {self.beta_schedule} not implemented")
        
        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # For sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
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


    def build_net(self):
        """
        Build the network - should output noise prediction
        """
        network = get(self.params, "network", "ARtransformer")
        try:
            return getattr(Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")

    def get_condition_and_input(self, input):
        """
        :param input: model input + conditional input
        :return: model input, conditional input
        """
        condition = input[1]
        weights = None
        return input[0], condition, weights

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
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

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and predicted noise
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
        posterior_mean_coef1_t = self.posterior_mean_coef1.to(x_t.device)[t]
        posterior_mean_coef2_t = self.posterior_mean_coef2.to(x_t.device)[t]
        posterior_variance_t = self.posterior_variance.to(x_t.device)[t]
        
        # Reshape for broadcasting
        while len(posterior_mean_coef1_t.shape) < len(x_t.shape):
            posterior_mean_coef1_t = posterior_mean_coef1_t.unsqueeze(-1)
        while len(posterior_mean_coef2_t.shape) < len(x_t.shape):
            posterior_mean_coef2_t = posterior_mean_coef2_t.unsqueeze(-1)
        while len(posterior_variance_t.shape) < len(x_t.shape):
            posterior_variance_t = posterior_variance_t.unsqueeze(-1)
        
        posterior_mean = (posterior_mean_coef1_t * x_start + 
                         posterior_mean_coef2_t * x_t)
        
        return posterior_mean, posterior_variance_t

    def batch_loss(self, input):
        """
        DDPM training loss
        Args:
            input: input tensor and conditions
        Returns:
            loss: batch loss
        """
        x, c, _ = self.get_condition_and_input(input)
        
        if self.latent:  # encode x into autoencoder latent space
            x = self.ae.encode(x, c)
            if self.ae.kl:
                x = self.ae.reparameterize(x[0], x[1])
            x = self.ae.unflatten_layer_from_batch(x)
        if self.cfg:
            if torch.rand(1).item() < self.drop_prob:
                print("unconditioned")
                c = None
        else:
            # add phantom layer dim to condition
            c = c.unsqueeze(-1)
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        
        # Sample noise
        noise = torch.randn_like(x)
        
        # Forward diffusion process
        x_t = self.q_sample(x, t, noise=noise)
        
        # Predict noise
        predicted_noise = self.net(c, x_t, t, x)
        
        # # Compute loss (simple MSE on noise prediction)
        # loss = ((predicted_noise - noise) ** 2).mean()
        
        # return loss

        # Compute loss (simple MSE on noise prediction)
        diffusion_loss = ((predicted_noise - noise) ** 2).mean()
    
        loss_tensors = {
            'diffusion_loss': diffusion_loss,
        }
        loss_scalars = {
            'diffusion_loss': diffusion_loss.item(),
            'diffusion_loss_tracked': diffusion_loss.item(),
        }
    
        return loss_tensors, loss_scalars

    def p_sample(self, x_t, t, c, x_orig=None):
        """
        Sample x_{t-1} from x_t
        """
        # Predict noise
        predicted_noise = self.net(c, x_t, t, x_orig)
        
        # Predict x_0
        x_start = self.predict_start_from_noise(x_t, t, predicted_noise)
        
        # Get posterior mean and variance
        posterior_mean, posterior_variance = self.q_posterior_mean_variance(x_start, x_t, t)
        
        if t == 0:
            return posterior_mean
        else:
            noise = torch.randn_like(x_t)
            return posterior_mean + torch.sqrt(posterior_variance) * noise

    def p_sample_loop(self, shape, c):
        """
        Full sampling loop
        """
        device = c.device
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, c)
            
        return x
    
    
    
    
    @torch.inference_mode()
    def sample_batch(self, c,sampling_type='ddpm'):
        """
        Generate samples using DDPM
        """
        #print("Sampling with DDPM: ", c.shape)
        #print("in sample_batch: ",self.params)
        diffusion_params = {
            'sqrt_alphas_cumprod': self.sqrt_alphas_cumprod,
            'sqrt_one_minus_alphas_cumprod': self.sqrt_one_minus_alphas_cumprod,
            'posterior_mean_coef1': self.posterior_mean_coef1,
            'posterior_mean_coef2': self.posterior_mean_coef2,
            'posterior_variance': self.posterior_variance,
        }
        
        c_unsqueezed = c.unsqueeze(-1)
        
        # Determine shape for sampling
        if hasattr(self, 'sample_shape'):
            shape = self.sample_shape
        else:
            # Infer shape from condition or use default
            shape = list(c.shape) + [self.dim_embedding]  # Adjust as needed
        
        # Run sampling loop
        #sample = self.p_sample_loop(shape, c_unsqueezed)
        sample = self.net(
            c, x_t=None, t=None, x=None, rev=True,
            diffusion_buffers=diffusion_params,
            sampling_type=sampling_type
        )
        
        if self.latent:  # decode the generated sample
            sample, c = self.ae.flatten_layer_to_batch(sample, c)
            sample = self.ae.decode(sample.squeeze(), c)
            
        return sample

    def ddim_sample(self, c, ddim_timesteps=50, eta=0.0):
        """
        DDIM sampling for faster inference
        """
        device = c.device
        batch_size = c.shape[0]
        c_unsqueezed = c.unsqueeze(-1)
        
        # Create DDIM timestep schedule
        step = self.num_timesteps // ddim_timesteps
        timesteps = np.arange(0, self.num_timesteps, step)
        timesteps = timesteps[:ddim_timesteps]
        timesteps = list(reversed(timesteps))
        
        # Start from pure noise
        if hasattr(self, 'sample_shape'):
            shape = self.sample_shape
        else:
            shape = list(c.shape) + [self.dim_embedding]
        
        x = torch.randn(shape, device=device)
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.net(c_unsqueezed, x, t_tensor)
            
            # Predict x_0
            alpha_cumprod_t = self.alphas_cumprod[t].to(device)
            alpha_cumprod_t_prev = self.alphas_cumprod[timesteps[i+1]].to(device) if i < len(timesteps) - 1 else torch.tensor(1.0).to(device)
            
            # Reshape for broadcasting
            while len(alpha_cumprod_t.shape) < len(x.shape):
                alpha_cumprod_t = alpha_cumprod_t.unsqueeze(-1)
            while len(alpha_cumprod_t_prev.shape) < len(x.shape):
                alpha_cumprod_t_prev = alpha_cumprod_t_prev.unsqueeze(-1)
            
            x_0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            # Direction to x_t
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - eta**2 * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)) * predicted_noise
            
            # Random noise
            noise = eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)) * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev) * torch.randn_like(x)
            
            # Update x
            x = torch.sqrt(alpha_cumprod_t_prev) * x_0_pred + dir_xt + noise
            
        if self.latent:  # decode the generated sample
            x, c = self.ae.flatten_layer_to_batch(x, c)
            x = self.ae.decode(x.squeeze(), c)
            
        return x