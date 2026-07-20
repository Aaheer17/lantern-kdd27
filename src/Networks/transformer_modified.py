import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat
from torchdiffeq import odeint
from typing import Optional
import inspect
import time


class CrossAttention(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
    def forward(self, x, cond):
        # x: [B, L_q, D], cond: [B, L_kv, D]
        # returns [B, L_q, D]
        return self.attn(x, cond, cond)[0]

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, activation="SiLU", dropout=0.0, normalization=None):
        super().__init__()
        layers = []

        layers.append(nn.Linear(dim, dim))
        if normalization:
            layers.append(getattr(nn, normalization)(dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(getattr(nn, activation)())
        
        layers.append(nn.Linear(dim, dim))  # Second layer in residual block

        self.block = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(dim)  # Or use BatchNorm1d if more appropriate

    def forward(self, x):
        #print("inside residual network")
        return self.norm(x + self.block(x))
class ARtransformerModified(nn.Module):

    def __init__(self, params):
        super().__init__()
        # Read in the network specifications from the params
        self.params = params
        self.dim_embedding = self.params["dim_embedding"]
        self.dims_in = self.params["shape"][0]
        self.dims_c = self.params["n_con"]
        self.bayesian = False
        self.layer_cond = self.params.get("layer_cond", False)
        self.gen_model=self.params['gen_model']
        #self.inference_steps=self.params.get('inference_steps',100)
        self.inference_steps=1000
        #print("self.gen_model: ",self.gen_model)
        self.c_embed = self.params.get("c_embed", None)
        self.x_embed = self.params.get("x_embed", None)

        self.encode_t_dim = self.params.get("encode_t_dim", 64)
        self.encode_t_scale = self.params.get("encode_t_scale", 30)
        
        # Support both CFM and DDPM modes
        #self.model_type = self.params.get("model_type", "cfm")  # 'cfm' or 'ddpm'
        
        #print("------in ARTRansformer: ",self.c_embed,self.x_embed, self.encode_t_dim, self.encode_t_scale)
        #print("Model type:", self.model_type)
        
        self.transformer = nn.Transformer(
            d_model=self.dim_embedding,
            nhead=params["n_head"],
            num_encoder_layers=params["n_encoder_layers"],
            num_decoder_layers=params["n_decoder_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params.get("dropout_transformer", 0.0),
            batch_first=True,
        )
        # print('checking transformer parameters')
        # print('d_model and also expected output of x_embed and c_embed: ',self.dim_embedding)
        # print('shape of dims_in (expected input of x_embed)',self.dims_in)
        # print('shape of dims_c (expected input of c_embed)',self.dims_c)
        
        if self.x_embed:
            self.x_embed = nn.Sequential(
                nn.Linear(1, self.dim_embedding),
                nn.Linear(self.dim_embedding, self.dim_embedding)
            )
        if self.c_embed:
            self.c_embed = nn.Sequential(
                nn.Linear(1, self.dim_embedding),
                nn.ReLU(),
                nn.Linear(self.dim_embedding, self.dim_embedding)
            )
        
        # Time embedding - support both continuous (CFM) and discrete (DDPM)
        if self.gen_model == "DDPM":
            # For DDPM discrete timesteps
            max_timesteps = self.params.get("num_timesteps", 1000)
            self.t_embed_discrete = nn.Embedding(max_timesteps, self.encode_t_dim)
            
        # Always keep continuous time embedding for CFM
        self.t_embed_continuous = nn.Sequential(
            GaussianFourierProjection(embed_dim=self.encode_t_dim, scale=self.encode_t_scale),
            nn.Linear(self.encode_t_dim, self.encode_t_dim)
        )
        
        # For backward compatibility, keep the original t_embed
        self.t_embed = self.t_embed_continuous  ## why would this work?
            
        self.subnet = self.build_subnet()
        self.positional_encoding = PositionalEncoding(
            d_model=self.dim_embedding, max_len=max(self.dims_in, self.dims_c) + 1, dropout=0.0
        )
        self.num_timesteps=self.params.get("num_timesteps",1000)
    def predict_start_from_noise(self, x_t, t, noise, schedule):
        sqrt_alpha = schedule["sqrt_alphas_cumprod"].to(x_t.device)[t]
        sqrt_one_minus = schedule["sqrt_one_minus_alphas_cumprod"].to(x_t.device)[t]

        while len(sqrt_alpha.shape) < len(x_t.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        while len(sqrt_one_minus.shape) < len(x_t.shape):
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        return (x_t - sqrt_one_minus * noise) / sqrt_alpha
    
    def q_posterior_mean_variance(self, x_start, x_t, t, schedule):
        coef1 = schedule["posterior_mean_coef1"].to(x_t.device)[t]
        coef2 = schedule["posterior_mean_coef2"].to(x_t.device)[t]
        var = schedule["posterior_variance"].to(x_t.device)[t]

        while len(coef1.shape) < len(x_t.shape):
            coef1 = coef1.unsqueeze(-1)
        while len(coef2.shape) < len(x_t.shape):
            coef2 = coef2.unsqueeze(-1)
        while len(var.shape) < len(x_t.shape):
            var = var.unsqueeze(-1)

        mean = coef1 * x_start + coef2 * x_t
        return mean, var
    def compute_embedding(
        self, p: torch.Tensor, dim: int, embedding_net: Optional[nn.Module]
    ) -> torch.Tensor:
        """
        Appends the one-hot encoded position to the momenta p. Then this is either zero-padded
        or an embedding net is used to compute the embedding of the correct dimension.
        """
        one_hot = torch.eye(dim, device=p.device, dtype=p.dtype)[
            None, : p.shape[1], :
        ].expand(p.shape[0], -1, -1)
        
        if embedding_net is None:
            n_rest = self.dim_embedding - dim - p.shape[-1]
            assert n_rest >= 0
            zeros = torch.zeros((*p.shape[:2], n_rest), device=p.device, dtype=p.dtype)
            # print("----->p shape: ", p.shape)
            # print("one_hot shape: ", one_hot.shape)
            # print("zeros shape: ", zeros.shape)
            if p.ndim == 2:
                p = p.unsqueeze(1)
            return torch.cat((p, one_hot, zeros), dim=2)
        else:
            #print("----inside positional_encoding----")
            return self.positional_encoding(embedding_net(p))

    def build_subnet(self):
        self.intermediate_dim = self.params.get("intermediate_dim", 512)
        self.dropout = self.params.get("dropout", 0.0)
        self.activation = self.params.get("activation", "SiLU")
        self.layers_per_block = self.params.get("layers_per_block", 8)
        self.normalization = self.params.get("normalization", None)
        self.residual = self.params.get('residual',None)
        self.cross_attention = self.params.get('cross_attention',None)

        cond_dim = self.encode_t_dim + self.dim_embedding
        if self.layer_cond:
            cond_dim += self.dims_in

        linear = nn.Linear(1+cond_dim, self.intermediate_dim)
        layers = [linear, getattr(nn, self.activation)()]

        for _ in range(1, self.layers_per_block - 1):
            if self.residual :
                layers.append(ResidualMLPBlock(self.intermediate_dim, activation=self.activation, 
                                       dropout=self.dropout, normalization=self.normalization))
                #layers.append(CrossAttention(self.intermediate_dim)) 
            else:
                linear = nn.Linear(self.intermediate_dim, self.intermediate_dim)
                layers.append(linear)
                if self.normalization is not None:
                    layers.append(getattr(nn, self.normalization)(self.intermediate_dim))
                if self.dropout is not None:
                    layers.append(nn.Dropout(p=self.dropout))
                layers.append(getattr(nn, self.activation)())

        linear = nn.Linear(self.intermediate_dim, 1)
        layers.append(linear)

        return nn.Sequential(*layers)
    ## When is this function called
    def get_time_embedding(self, t):
        """
        Get time embedding based on model type and input format
        """
        if self.gen_model == "DDPM":
            # For DDPM, expect integer timesteps
            if t.dtype == torch.long:
                return self.t_embed_discrete(t)
            else:
                # If float timesteps are passed for DDPM, convert to int
                return self.t_embed_discrete(t.long())
        else:
            # For CFM, use continuous time embedding
            return self.t_embed_continuous(t.float())

    def sample_dimension(self, c: torch.Tensor, t=None):
        """
        Original CFM sampling with ODE integration
        """
        #print("in sample_dimension (CFM mode)")
        batch_size = c.size(0)
        dtype = c.dtype
        device = c.device

        net = self.subnet
        x_0 = torch.randn((batch_size, 1), device=device, dtype=dtype)

        # NN wrapper to pass into ODE solver
        def net_wrapper(t, x_t):
            t_torch = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)
            t_torch = self.t_embed_continuous(t_torch)
            v = net(torch.cat([x_t, t_torch.reshape(batch_size, -1), c.squeeze()], dim=-1))
            return v

        # Solve ODE from t=0 to t=1
        with torch.inference_mode():
            x_t = odeint(
                net_wrapper, x_0, torch.tensor([0, 1], dtype=dtype, device=device),
                **self.params.get("solver_kwargs", {})
            )
        # Extract generated samples
        x_1 = x_t[-1]
        #print('in sample_dimension before returning------>',x_1.shape)

        return x_1.unsqueeze(1)

    def sample_dimension_ddim(self, c: torch.Tensor, scheduler=None):
        """
        DDIM 1D sampling by looping over DDIM steps (not necessarily all timesteps).
        Returns a generated sample x_0
        """
        #print("Shape of c:", c.shape)
        batch_size = c.size(0)
        device = c.device
        dtype = c.dtype

        # Start from pure noise
        x_t = torch.randn((batch_size, 1), device=device, dtype=dtype)

        # Use ddim_steps spaced evenly across self.num_timesteps
        step_indices = torch.linspace(0, self.num_timesteps - 1, self.inference_steps, dtype=torch.long).flip(0).to(device)
        #c = c.view(c.size(0), -1)
        t0=time.time()
        for i in range(len(step_indices) - 1):
            t = step_indices[i]
            t_prev = step_indices[i + 1]

            t_batch = t.expand(batch_size)
            t_prev_batch = t_prev.expand(batch_size)

            # Predict noise
            eps_pred = self.subnet(torch.cat([
                x_t,
                self.get_time_embedding(t_batch).view(batch_size, -1),
                c.view(batch_size,-1)
            ], dim=-1))
            

            # Get alpha values
            alpha_t = scheduler['sqrt_alphas_cumprod'].to(x_t.device)[t_batch]
            alpha_prev = scheduler['sqrt_alphas_cumprod'].to(x_t.device)[t_prev_batch]
            sqrt_one_minus_alpha_t = scheduler['sqrt_one_minus_alphas_cumprod'].to(x_t.device)[t_batch]
            
            alpha_t = alpha_t.view(-1, 1)  # Shape: (1000, 1)
            alpha_prev = alpha_prev.view(-1, 1)  # Shape: (1000, 1)
            sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.view(-1, 1)  # Shape: (1000, 1)
            #print("shape of alpha_t, alpha_prev and other: ",alpha_t.shape, alpha_prev.shape, sqrt_one_minus_alpha_t.shape)

            # Predict x0
            x_0_pred = (x_t - sqrt_one_minus_alpha_t * eps_pred) / alpha_t
            
            # Compute x_{t-1} using DDIM update rule (deterministic, eta = 0)
            sigma = 0.0
            dir_xt = torch.sqrt(1.0 - alpha_prev ** 2 - sigma**2) * eps_pred
            x_t = alpha_prev * x_0_pred + dir_xt  # + sigma * noise ← skipped since sigma=0
            
        
        # Final step (t=0)
        t = step_indices[-1].expand(batch_size)
        eps_pred = self.subnet(torch.cat([
            x_t,
            self.get_time_embedding(t).view(batch_size, -1),
            c.view(batch_size,-1)
        ], dim=-1))
       
        alpha_t = scheduler['sqrt_alphas_cumprod'].to(x_t.device)[t].view(-1, 1)
        sqrt_one_minus_alpha_t = scheduler['sqrt_one_minus_alphas_cumprod'].to(x_t.device)[t].view(-1, 1)
        
        x_0_pred = (x_t - sqrt_one_minus_alpha_t * eps_pred) / alpha_t
        t1=time.time()
        
        return x_0_pred.unsqueeze(1)
    
    def sample_dimension_plms(self, c: torch.Tensor, scheduler=None, plms_steps=100):
        """
        PLMS 1D sampling using multistep noise prediction.
        Returns a generated sample x_0.
        """
        batch_size = c.size(0)
        device = c.device
        dtype = c.dtype

        # Start from pure noise
        x_t = torch.randn((batch_size, 1), device=device, dtype=dtype)

        # Precompute time indices (reversed)
        step_indices = torch.linspace(0, self.num_timesteps - 1, plms_steps, dtype=torch.long).flip(0).to(device)

        prev_eps = []
        t0=time.time()
        for i in range(len(step_indices) - 1):
            t = step_indices[i]
            t_prev = step_indices[i + 1]
            t_batch = t.expand(batch_size)
            t_prev_batch = t_prev.expand(batch_size)

            # Predict noise
            eps_pred = self.subnet(torch.cat([
                x_t,
                self.get_time_embedding(t_batch).view(batch_size, -1),
                c.view(batch_size, -1)
            ], dim=-1))

            # Store current eps_pred
            prev_eps.append(eps_pred)
            if len(prev_eps) > 4:
                prev_eps.pop(0)

            # Compute multistep epsilon estimate
            if len(prev_eps) == 1:
                eps_hat = prev_eps[-1]
            elif len(prev_eps) == 2:
                eps_hat = (3 * prev_eps[-1] - prev_eps[-2]) / 2
            elif len(prev_eps) == 3:
                eps_hat = (23 * prev_eps[-1] - 16 * prev_eps[-2] + 5 * prev_eps[-3]) / 12
            else:
                eps_hat = (55 * prev_eps[-1] - 59 * prev_eps[-2] + 37 * prev_eps[-3] - 9 * prev_eps[-4]) / 24

            # Get alpha values
            alpha_t = scheduler['sqrt_alphas_cumprod'].to(x_t.device)[t_batch].view(-1, 1)
            alpha_prev = scheduler['sqrt_alphas_cumprod'].to(x_t.device)[t_prev_batch].view(-1, 1)
            sqrt_one_minus_alpha_t = scheduler['sqrt_one_minus_alphas_cumprod'].to(x_t.device)[t_batch].view(-1, 1)
            one_minus_alpha_prev_sq = 1.0 - alpha_prev**2

            # Predict x0
            x_0_pred = (x_t - sqrt_one_minus_alpha_t * eps_hat) / alpha_t

            # DDIM-style update with PLMS-predicted eps
            x_t = alpha_prev * x_0_pred + torch.sqrt(one_minus_alpha_prev_sq) * eps_hat

        # Final step to recover x0
        t = step_indices[-1].expand(batch_size)
        eps_pred = self.subnet(torch.cat([
            x_t,
            self.get_time_embedding(t).view(batch_size, -1),
            c.view(batch_size, -1)
        ], dim=-1))
        alpha_t = scheduler['sqrt_alphas_cumprod'].to(x_t.device)[t].view(-1, 1)
        sqrt_one_minus_alpha_t = scheduler['sqrt_one_minus_alphas_cumprod'].to(x_t.device)[t].view(-1, 1)
        x_0_pred = (x_t - sqrt_one_minus_alpha_t * eps_pred) / alpha_t
        t1=time.time()
        #print("Intermediate time for one layer inside plms: ",t1-t0)
        return x_0_pred.unsqueeze(1)


    
    def sample_dimension_ddpm(self, c: torch.Tensor,scheduler=None):
        """
        DDPM 1D sampling by looping over time steps.
        Returns a generated sample x_0
        """
        #print("Shape of c:", c.shape)
        #print("in sample_dimension_ddpm")
        batch_size = c.size(0)
        device = c.device
        dtype = c.dtype

        # Start from pure noise
        x_t = torch.randn((batch_size, 1), device=device, dtype=dtype)
        self.inference_steps=1000
        #print("self.inference_steps: ",self.inference_steps)
        for i in reversed(range(self.inference_steps)):
            t = torch.full((batch_size,), i, dtype=torch.long, device=device)

            # Predict noise
            eps_pred = self.subnet(torch.cat([
                x_t,
                self.get_time_embedding(t).view(batch_size, -1),
                c.view(batch_size, -1)
            ], dim=-1))
            #print("shape of eps_pred: ",eps_pred.shape, c.shape)
            # Denoise
            x_0_pred = self.predict_start_from_noise(x_t, t, eps_pred,scheduler)
            mu, var = self.q_posterior_mean_variance(x_0_pred, x_t, t,scheduler)

            if i > 0:
                noise = torch.randn_like(x_t)
                x_t = mu + torch.sqrt(var) * noise
            else:
                x_t = mu

        return x_t.unsqueeze(1)


    def forward(self, c, x_t=None, t=None, x=None, rev=False, diffusion_buffers=None, sampling_type='ddpm'):
        
        # print('initial inputs to the functions where autoregressiveness happens')
        # print(f'shape of c {c.shape}, shape of x_t {"none" if x_t is None else x_t.shape}')
        # print(f'shape of t {"none" if t is None else t.shape} shape of x {"none" if x is None else x.shape}')
        # print(f'status of rev {rev}, model_type: {self.model_type}')
        
        if not rev:
            #print(f'Forward pass for {self.model_type} training')
            # Autoregressive masking for training
            xp = nn.functional.pad(x[:, :-1], (0, 0, 1, 0))
            # print('shape of xp',xp.shape)
            # print('shape of condition c', c.shape)
            
            src_transformer = self.compute_embedding(c, dim=self.dims_c, embedding_net=self.c_embed)
            tgt_transformer = self.compute_embedding(xp, dim=self.dims_in + 1, embedding_net=self.x_embed)
            xformer_tgt_mask = torch.ones((xp.shape[1], xp.shape[1]), device=x.device, dtype=torch.bool).triu(diagonal=1)
            
            # print('shape of transformer inputs to create embeddings')
            # print('src transformer shape (encoder sequence) ',src_transformer.shape)
            # print('tgt transformer shape (decoder sequence) ',tgt_transformer.shape)
            # print('tgt mask shape for transformer',xformer_tgt_mask.shape)
            
            embedding = self.transformer(
                src=src_transformer,
                tgt=tgt_transformer,
                tgt_mask=xformer_tgt_mask,
            )
            
            #print('after autoregressive computation in xformer embedding shape is',embedding.shape)

            if self.layer_cond:
                #print('self.layer cond is true')
                layer_one_hot = repeat(
                    torch.eye(self.dims_in, device=x.device), '... -> b ...', b=len(c)
                )
                #print('shape of layer_one_hot',layer_one_hot.shape)
                embedding = torch.cat([embedding, layer_one_hot], dim=2)
                #print('after concat embedding shape is',embedding.shape)

            # Get appropriate time embedding
            t_embedded = self.get_time_embedding(t)
            t_embedded = t_embedded.unsqueeze(1).expand(-1, x_t.size(1), -1)
            
            #print(f'followings are going to subnet ({self.model_type} mode)')
            # print('xt shape',x_t.shape)
            # print('shape of t embedding',t_embedded.shape)
            # print('shape of condition ',embedding.shape)
            
            # Predict output (velocity for CFM, noise for DDPM)
            pred = self.subnet(torch.cat([x_t, t_embedded, embedding], dim=-1))
            #print("shape of prediction in training: ",pred.shape)
            # if self.gen_model == "cfm":
            #     #print('shape of velocity prediction after subnet:',pred.shape)
            # else:
            #     #print('shape of noise prediction after subnet:',pred.shape)

        else:
            
            
            #print(f"Sampling mode ({self.model_type}): ",rev, c.shape)
            x = torch.zeros((c.shape[0], 1, 1), device=c.device, dtype=c.dtype)
            c_embed = self.compute_embedding(c, dim=self.dims_c, embedding_net=self.c_embed)
            #print("dims_in and c_embed: ",c_embed.shape, self.dims_in)
            
            for i in range(self.dims_in):
                #print("in each loop: ",i, c_embed.shape)
                embedding = self.transformer(
                    src=c_embed,
                    tgt=self.compute_embedding(x, dim=self.dims_in + 1, embedding_net=self.x_embed),
                    tgt_mask=torch.ones(
                        (x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool
                    ).triu(diagonal=1),
                )
                #print("i and embedding shape in reverse: ",i, embedding.shape)
                
                if self.layer_cond:
                    #print("what is self.layer_cond: ", self.layer_cond)
                    layer_one_hot = repeat(
                        F.one_hot(torch.tensor(i, device=x.device), self.dims_in),
                        'd -> b 1 d', b=len(c)
                    )
                    embedding = torch.cat([embedding[:, -1:,:], layer_one_hot], dim=2)
                
                # Choose sampling method based on model type
                #print("self.gen_model: ",self.gen_model)
                if sampling_type == "cfm":
                    # Use ODE integration for CFM
                    x_new = self.sample_dimension(embedding[:, -1:, :])
                elif sampling_type == "ddim":
                    
                    x_new = self.sample_dimension_ddim(embedding[:, -1:, :], diffusion_buffers)
                    #print("shape of x and x_new: ",x.shape, x_new.shape)
                elif sampling_type == "plms":
                    x_new = self.sample_dimension_plms(embedding[:, -1:, :], diffusion_buffers)
                else:
                    
                    x_new = self.sample_dimension_ddpm(embedding[:, -1:, :], diffusion_buffers)
                    #print("shape of x and x_new: ",x.shape, x_new.shape)
                
                x = torch.cat((x, x_new), dim=1)
                #print("in reverse shape of x and x_new: ",x.shape,x_new.shape)

            pred = x[:, 1:]
            pred = pred.squeeze()
            #print(f"pred shape for sampling: ",pred.shape)
            
        return pred

    

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        #print("inside the GaussianFourier...",x.shape)
        if x.ndim == 1:
            x = x[:, None]  # Make sure x is (B, 1)

        x_proj = x * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim==2:
            x=x.unsqueeze(1)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)