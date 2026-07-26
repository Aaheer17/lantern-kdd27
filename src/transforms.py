import torch
import numpy as np
import os

from challenge_files import *
from challenge_files import XMLHandler

import os
import pwd

def get_username():
    return pwd.getpwuid(os.getuid())[0]

if get_username()=='zm8bh':
    from more_itertools import pairwise
else:
    from itertools import pairwise

def logit(array, alpha=1.e-6, inv=False):
    if inv:
        z = torch.sigmoid(array)
        z = (z-alpha)/(1-2*alpha)
    else:
        z = array*(1-2*alpha) + alpha
        z = torch.logit(z)
    return z

class Standardize(object):
    """
    Standardize features 
        mean: vector of means 
        std: vector of stds
    """
    def __init__(self, means, stds):
        self.means = means
        self.stds = stds

    def __call__(self, shower, energy, rev=False):
        if rev:
            transformed = shower*self.stds + self.means
        else:
            transformed = (shower - self.means)/self.stds
        return transformed, energy

# class StandardizeFromFile(object):
#     """
#     Standardize features 
#         mean_path: path to `.npy` file containing means of the features 
#         std_path: path to `.npy` file containing standard deviations of the features
#         create: whether or not to calculate and save mean/std based on first call
#     """

#     def __init__(self, model_dir):

#         self.model_dir = model_dir
#         #print('In StandardizeFromFile: ',model_dir)
#         self.mean_path = os.path.join(model_dir, 'means.npy')
#         self.std_path = os.path.join(model_dir, 'stds.npy')
#         self.dtype = torch.get_default_dtype()
#         try:
#             # load from file
#             self.mean = torch.from_numpy(np.load(self.mean_path)).to(self.dtype)
#             self.std = torch.from_numpy(np.load(self.std_path)).to(self.dtype)
#             self.written = True
#         except FileNotFoundError:
#             self.written = False

#     def write(self, shower, energy):
#         self.mean = shower.mean(axis=0)
#         self.std = shower.std(axis=0)
#         np.save(self.mean_path, self.mean.detach().cpu().numpy())
#         np.save(self.std_path, self.std.detach().cpu().numpy())
#         self.written = True

#     def __call__(self, shower, energy, rev=False):
#         if rev:
#             transformed = shower*self.std.to(shower.device) + self.mean.to(shower.device)
#         else:
#             if not self.written:
#                 self.write(shower, energy)
#             print("Shape of shower and mean: ",shower.shape, self.mean.shape)
#             transformed = (shower - self.mean.to(shower.device))/self.std.to(shower.device)
#         if shower is not None and energy is not None:
#             print("standard: ", transformed.shape, transformed.min(), transformed.max(), energy.shape,energy.min(), energy.max())
#         return transformed, energy
class StandardizeFromFile(object):
    """
    Standardize ONLY the voxel features (first 6480), leave u-features unchanged.
    """

    def __init__(self, model_dir, n_features=6480, eps=1e-12):
        self.model_dir = model_dir
        self.mean_path = os.path.join(model_dir, 'means.npy')
        self.std_path = os.path.join(model_dir, 'stds.npy')
        self.dtype = torch.get_default_dtype()
        self.n_features = int(n_features)
        self.eps = float(eps)
        
        try:
            # Load from file
            self.mean = torch.from_numpy(np.load(self.mean_path)).to(self.dtype)
            self.std = torch.from_numpy(np.load(self.std_path)).to(self.dtype)
            self.written = True
            print(f"Loaded stats: mean shape={self.mean.shape}, std shape={self.std.shape}")
        except FileNotFoundError:
            self.mean = None
            self.std = None
            self.written = False
            print(f"No existing stats found, will compute on first call")

    def write(self, shower, energy):
        # Only compute stats on first n_features (voxels)
        head = shower[:, :self.n_features]
        self.mean = head.mean(dim=0)
        self.std = head.std(dim=0).clamp(min=self.eps)  # Prevent std=0
        
        np.save(self.mean_path, self.mean.detach().cpu().numpy())
        np.save(self.std_path, self.std.detach().cpu().numpy())
        self.written = True
        print(f"Computed and saved stats: mean shape={self.mean.shape}, std shape={self.std.shape}")

    def __call__(self, shower, energy, rev=False):
        # Check if shower has enough features
        if shower.shape[1] < self.n_features:
            raise ValueError(f"Shower has {shower.shape[1]} features but expected at least {self.n_features}")
        
        if (not rev) and (not self.written):
            self.write(shower, energy)
        
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardizeFromFile: mean/std not initialized")
        
        # Verify shapes match
        if self.mean.shape[0] != self.n_features:
            raise ValueError(
                f"Loaded stats have {self.mean.shape[0]} features but n_features={self.n_features}. "
                f"Delete {self.mean_path} and {self.std_path} to recompute."
            )
        
        # Move to device
        mean = self.mean.to(device=shower.device, dtype=shower.dtype)
        std = self.std.to(device=shower.device, dtype=shower.dtype)
        
        # Split: standardize first n_features, leave rest unchanged
        head = shower[:, :self.n_features]
        tail = shower[:, self.n_features:]  # u-features (if any)
        
        if rev:
            # Unstandardize voxels
            head_out = head * std + mean
        else:
            # Standardize voxels
            head_out = (head - mean) / std
        
        # Recombine
        if tail.shape[1] > 0:
            transformed = torch.cat([head_out, tail], dim=1)
        else:
            transformed = head_out
        
        return transformed, energy
class SelectDims(object):
    """
    Selects a subset of the features 
        start: start of range of indices to keep
        end:   end of range of indices to keep (exclusive)
    """

    def __init__(self, start, end):
        #print("inside selectDims")
        self.indices = torch.arange(start, end)
    def __call__(self, shower, energy, rev=False):
        if shower is not None and energy is not None:
            print("Inside SelectDims: ",shower.shape, shower.min(), shower.max(), energy.shape, energy.min(), energy.max() )
        if rev:
           return shower, energy
        transformed = shower[..., self.indices]
        if energy is not None and transformed is not None:
            
            print("transformed: ",transformed.shape, energy.shape)
            print("inside selectDims, transformed and energy: ",transformed.shape, transformed.min(), transformed.max(), energy.shape,energy.min(), energy.max())
        return transformed, energy

class AddFeaturesToCond(object):
    """
    Transfers a subset of the input features to the condition
        split_index: Index at which to split input. Features past the index will be moved
    """

    def __init__(self, split_index):
        self.split_index = split_index
    
    def __call__(self, x, c, rev=False):
        
        if rev:
            c_, split = c[:, :1], c[:, 1:]
            x_ = torch.cat([x, split], dim=1)
        else:
            x_, split = x[:, :self.split_index], x[:, self.split_index:]
            c_ = torch.cat([c, split], dim=1)
        return x_, c_
    
class LogEnergy(object):
    """
    Log transform incident energies
        alpha: Optional regularization for the log
    """            
    def __init__(self, alpha=0.):
        self.alpha = alpha
        self.cond_transform = True
        
    def __call__(self, shower, energy, rev=False):
        if shower is not None and energy is not None:
            print("Inside LogEnergy: ",shower.shape, shower.min(), shower.max(), energy.shape, energy.min(), energy.max() )
        
        if rev:
            transformed = torch.exp(energy) - self.alpha
        else:
            #print("alpha: ",self.alpha, energy.shape)
            transformed = torch.log(energy + self.alpha)
            if shower is not None and transformed is not None:
                print("Inside LogEnergy: ",shower.shape, shower.min(), shower.max(), transformed.shape, transformed.min(), transformed.max())
        #print("Log energy shower and transformed: ",shower.shape,transformed.shape)
        return shower, transformed              

class ScaleVoxels(object):
    """
    Apply a multiplicative factor to the voxels.
        factor: Number to multiply voxels
    """
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, shower, energy, rev=False):
        print("inside Scale voxels: ",shower.shape, shower.min(), shower.max(), energy.shape, energy.min(), energy.max())
        if rev:
            transformed = shower/self.factor
        else:
            transformed = shower*self.factor
        return transformed, energy

class ScaleTotalEnergy(object):
    """
    Scale only E_tot/E_inc by a factor f.
    The effect is the same of ScaleVoxels but
    it is applied in a different position in the
    preprocessing chain.
    """
    def __init__(self, factor, n_layers=45):
        self.factor = factor
        self.n_layers = n_layers

    def __call__(self, shower, energy, rev=False):
        
        if rev:
            shower[..., -self.n_layers] /= self.factor
        else:
            shower[..., -self.n_layers] *= self.factor
        if shower is not None and energy is not None:
            print("inside Scale Total Energy: ",shower.shape, shower.min(), shower.max(), energy.shape, energy.min(), energy.max())
        return shower, energy

class ScaleEnergy(object):
    """
    Scale incident energies to lie in the range [0, 1]
        e_min: Expected minimum value of the energy
        e_max: Expected maximum value of the energy
    """
    def __init__(self, e_min, e_max):
        self.e_min = e_min
        self.e_max = e_max
        self.cond_transform = True

    def __call__(self, shower, energy, rev=False):
        if shower is not None and energy is not None:
            print("Inside ScaleEnergy: ",shower.shape, shower.min(), shower.max(), energy.shape, energy.min(), energy.max() )
        if rev:
            transformed = energy * (self.e_max - self.e_min)
            transformed += self.e_min
        else:
            transformed = energy - self.e_min
            transformed /= (self.e_max - self.e_min)
        if shower is not None and transformed is not None:
            print("Inside ScaleEnergy: ",shower.shape, shower.min(), shower.max(), transformed.shape, transformed.min(), transformed.max())
        return shower, transformed

class LogTransform(object):
    """
    Take log of input data
        alpha: regularization
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, shower, energy, rev=False):
        if rev:
            transformed = torch.exp(shower) - self.alpha
        else:
            transformed = torch.log(shower + self.alpha)
        if shower is not None and transformed is not None:
            print("Inside LogTransform: ",energy.shape, transformed.shape)
        return transformed, energy

class SelectiveLogTransform(object):
    """
    Take log of input data
        alpha: regularization
        exclusions: list of indices for features that should not be transformed
    """
    def __init__(self, alpha, exclusions=None):
        self.alpha = alpha
        self.exclusions = exclusions

    def __call__(self, shower, energy, rev=False):
        if rev:
            transformed = torch.exp(shower) - self.alpha
        else:
            transformed = torch.log(shower + self.alpha)
        if self.exclusions is not None:
            transformed[..., self.exclusions] = shower[..., self.exclusions]
        return transformed, energy

class ExclusiveLogTransform(object):
    """
    Take log of input data
        delta: regularization
        exclusions: list of indices for features that should not be transformed
    """

    def __init__(self, delta, exclusions=None):
        self.delta = delta
        self.exclusions = exclusions

    def __call__(self, shower, energy, rev=False):
        if rev:
            transformed = torch.exp(shower) - self.delta
        else:
            transformed = torch.log(shower + self.delta)
        if self.exclusions is not None:
            transformed[..., self.exclusions] = shower[..., self.exclusions] 
        return transformed, energy
 
class ExclusiveLogitTransform(object):
    """
    Take log of input data
        delta: regularization
        exclusions: list of indices for features that should not be transformed
    """

    def __init__(self, delta, exclusions=None, rescale=False):
        self.delta = delta
        self.exclusions = exclusions
        self.rescale = rescale

    def __call__(self, shower, energy, rev=False):
        if shower is not None and energy is not None:
            print("inside exclusive, shower and energy: ",shower.shape, shower.min(), shower.max(), energy.shape,energy.min(), energy.max())
        if rev:
            if self.rescale:
                transformed = logit(shower, alpha=self.delta, inv=True)
            else:
                transformed = torch.special.expit(shower)
        else:
            if self.rescale:
                transformed = logit(shower, alpha=self.delta)
            else:
                transformed = torch.logit(shower, eps=self.delta)

        if self.exclusions is not None:
            transformed[..., self.exclusions] = shower[..., self.exclusions]
        if energy is not None and transformed is not None:
           
            print("inside exclusive, transformed and energy: ",transformed.shape, transformed.min(), transformed.max(), energy.shape,energy.min(), energy.max())
        return transformed, energy
    

class AddNoise(object):
    """
    Add noise to input data
        func: torch distribution used to sample from
        width_noise: noise rescaling
    """
    def __init__(self, noise_width, cut=False):
        #self.func = func
        self.func = torch.distributions.Uniform(torch.tensor(0.0), torch.tensor(1.0))
        self.noise_width = noise_width
        self.cut = cut # apply cut if True

    def __call__(self, shower, energy, rev=False):
        if rev:
            mask = (shower < self.noise_width)
            transformed = shower
            if self.cut:
                transformed[mask] = 0.0 
        else:
            noise = self.func.sample(shower.shape)*self.noise_width
            transformed = shower + noise.reshape(shower.shape).to(shower.device)
        return transformed, energy


class SmoothUPeaks(object):
    """
    Smooth voxels equal to 0 or 1 using uniform noise
        w0: noise width for zeros
        w1: noise width for ones
        eps: threshold below which values are considered zero
    """

    def __init__(self, w0, w1, eps=1.e-10):
        self.func = torch.distributions.Uniform(
            torch.tensor(0.0), torch.tensor(1.0))
        self.w0 = w0
        self.w1 = w1
        self.scale = 1 + w0 + w1
        self.eps = eps

    def __call__(self, u, energy, rev=False):
        if rev:
            # undo scaling
            transformed = u*self.scale - self.w0
            # clip to [0, 1]
            transformed = torch.clip(transformed, min=0., max=1.)
            # restore u0
            transformed[:, 0] = u[:, 0]
        else:
            # sample noise values
            n0 = self.w0*self.func.sample(u.shape).to(u.device)
            n1 = self.w1*self.func.sample(u.shape).to(u.device)
            # add noise to us
            transformed = u - n0*(u<=self.eps) + n1*(u>=1-self.eps)
            # scale to [0,1] in preparation for logit
            transformed = (transformed + self.w0)/self.scale
            # restore u0
            transformed[:, 0] = u[:, 0]
            
        return transformed, energy

class SelectiveUniformNoise(object):
    """
    Add noise to input data with the option to exlude some features
        func: torch distribution used to sample from
        width_noise: noise rescaling
        exclusions: list of indices for features that should not be transformed
    """
    def __init__(self, noise_width, exclusions = None, cut=False):
        #self.func = func
        self.func = torch.distributions.Uniform(torch.tensor(0.0), torch.tensor(1.0))
        self.noise_width = noise_width
        self.exclusions = exclusions
        self.cut = cut # apply cut if True

    def __call__(self, shower, energy, rev=False):
        if rev:
            mask = (shower < self.noise_width)
            if self.exclusions:
                mask[:, self.exclusions] = False
            transformed = shower
            if self.cut:
                transformed[mask] = 0.0 
        else:
            noise = self.func.sample(shower.shape)*self.noise_width
            if self.exclusions:
                noise[:, self.exclusions] = 0.0
            transformed = shower + noise.reshape(shower.shape).to(shower.device)
        return transformed, energy        

class SetToVal(object):
    """
    Masks voxels to zero in the reverse transformation
        cut: threshold value for the mask
    """
    def __init__(self, val=0.):
        self.val = val

    def __call__(self, shower, energy, rev=False):
        if rev:
            mask = (shower < self.val)
            transformed = shower
            if self.val:
                transformed[mask] = self.val
        else:
            mask = (shower < self.val)
            transformed = shower
            if self.val:
                transformed[mask] = self.val
        return transformed, energy        

class CutValues(object):
    """
    Cut in Normalized space
        cut: threshold value for the cut
        n_layers: number of layers to avoid cutting on the us
    """
    def __init__(self, cut=0., n_layers=45):
        self.cut = cut
        self.n_layers = n_layers
        print(f'cut has been initialized with threshold {self.cut}')

    def __call__(self, shower, energy, rev=False):
        if rev:
            mask = (shower <= self.cut)
            mask[:, -self.n_layers:] = False
            transformed = shower
            if self.cut:
                print(f'Cut is being applied with {self.cut}')
                transformed[mask] = 0.0 
        else:
            transformed = shower
        return transformed, energy        

class CutBothValues(object):
    def __init__(self, cut=0.):
        self.cut = cut

    def __call__(self, shower, energy, rev=False):
        if rev:
            mask = (shower <= self.cut)
            transformed = shower
            if self.cut:
                transformed[mask] = 0.0 
        else:
            mask = (shower <= self.cut)
            transformed = shower
            if self.cut:
                transformed[mask] = 0.0 
        return transformed, energy        

class ZeroMask(object):
    """
    Masks voxels to zero in the reverse transformation
        cut: threshold value for the mask
    """
    def __init__(self, cut=0.):
        self.cut = cut

    def __call__(self, shower, energy, rev=False):
        if rev:
            mask = (shower < self.cut)
            transformed = shower
            if self.cut:
                transformed[mask] = 0.0 
        else:
            transformed = shower
        return transformed, energy        

class AddPowerlawNoise(object):
    """
    Add noise to input data following a power law distribution:
        eps ~ k x^(k-1)
        k   -- The power parameter of the distribution
        cut -- The value below which voxels will be masked to zero in the reverse transformation
    """
    def __init__(self, k, cut=None):
        self.k = k
        self.cut = cut

    def __call__(self, shower, energy, rev=False):
        if rev:
            mask = (shower < self.cut)
            transformed = shower
            if self.cut is not None:
                transformed[mask] = 0.0
        else:
            noise = torch.from_numpy(np.random.power(self.k, shower.shape)).to(shower.dtype)
            transformed = shower + noise.reshape(shower.shape).to(shower.device)
        return transformed, energy

class NormalizeByEinc(object):
    """
    Normalize each shower by the incident energy

    """
    def __call__(self, shower, energy, rev=False):
        if rev:
            transformed = shower*energy
        else:
            transformed = shower/energy
        return transformed, energy

class Reshape(object):
    """
    Reshape the shower as specified. Flattens batch in the reverse transformation.
        shape -- Tuple representing the desired shape of a single example
    """

    def __init__(self, shape):
        self.shape = torch.Size(shape)

    def __call__(self, shower, energy, rev=False):
        if shower is not None and energy is not None:
            print("Before reshaping: shower and energy: ",shower.shape, shower.min(), shower.max(),energy.shape,energy.min(),energy.max())
        if rev:
            shower = shower.reshape(-1, self.shape.numel())
        else:
            print("Inside the reshape function and shape of shower: ", shower.shape)
            shower = shower.reshape(-1, *self.shape)
        if shower is not None and energy is not None:
            
            print("after reshaping: shower and energy: ",shower.shape, shower.min(), shower.max(),energy.shape,energy.min(),energy.max())
        return shower, energy
 
class Reweight(object):
    """
    Reweight voxels
    """

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, shower, energy, rev=False):
        if rev:
            shower = shower**(1/self.factor)
        else:
            shower = shower**(self.factor)
        return shower, energy

class NormalizeByElayer(object):
    """
    Normalize each shower by the layer energy
    This will change the shower shape to N_voxels+N_layers
       layer_boundaries: ''
       eps: numerical epsilon
    """
    def __init__(self, ptype, xml_file, cut=0.0, eps=1.e-10):
        self.eps = eps
        print("what is transforms: ",xml_file)
        self.xml = XMLHandler.XMLHandler(xml_file, ptype)
        self.layer_boundaries = np.unique(self.xml.GetBinEdges())
        self.n_layers = len(self.layer_boundaries) - 1
        self.cut = cut

    def __call__(self, shower, energy, rev=False):
        print("Inside normalizeByELayer: Before", shower.shape, shower.min(), shower.max(), energy.shape, energy.min(), energy.max())
        if rev:

            # select u features
            us = shower[:, -self.n_layers:]
            
            # clip u_{i>0} into [0,1]
            us[:, (-self.n_layers+1):] = torch.clip(
                us[:, (-self.n_layers+1):],
                min=torch.tensor(0., device=shower.device),
                max=torch.tensor(1., device=shower.device)
            ) 

            # calculate unnormalised energies from the u's
            layer_Es = []
            total_E = torch.multiply(energy.flatten(), us[:,0]) # Einc * u_0
            cum_sum = torch.zeros_like(total_E)
            for i in range(us.shape[-1]-1):
                layer_E = (total_E - cum_sum) * us[:,i+1]
                layer_Es.append(layer_E)
                cum_sum += layer_E
            layer_Es.append(total_E - cum_sum)
            layer_Es = torch.vstack(layer_Es).T
            #next two lines are done by Farzana
            # print("shape of layer_Es: ", layer_Es.shape)
            #transformed=layer_Es  #needed only for energy model sample generation
            # select voxels
            # ------ commented by Farzana------ to get the layer energy only.
            shower = shower[:, :-self.n_layers]

            # Normalize each layer and multiply it with its original energy
            transformed = torch.zeros_like(shower)
            for l, (start, end) in enumerate(pairwise(self.layer_boundaries)):
                layer = shower[:, start:end] # select layer
                layer /= layer.sum(-1, keepdims=True) + self.eps # normalize to unity
                mask = (layer <= self.cut)
                layer[mask] = 0.0 # apply normalized cut
                transformed[:, start:end] = layer * layer_Es[:,[l]] # scale to layer energy

        else:
            # compute layer energies
            layer_Es = []
            for start, end in pairwise(self.layer_boundaries):
                layer_E = torch.sum(shower[:, start:end], dim=1, keepdims=True)
                shower[:, start:end] /= layer_E + self.eps # normalize to unity
                layer_Es.append(layer_E) # store layer energy
            layer_Es = torch.cat(layer_Es, dim=1).to(shower.device)
            print("in in ",layer_Es.shape)

            # compute generalized extra dimensions
            extra_dims = [torch.sum(layer_Es, dim=1, keepdim=True) / energy]
            for l in range(layer_Es.shape[1]-1):
                remaining_E = torch.sum(layer_Es[:, l:], dim=1, keepdim=True)
                extra_dim = layer_Es[:, [l]] / (remaining_E+ self.eps)
                extra_dims.append(extra_dim)
            extra_dims = torch.cat(extra_dims, dim=1)
            print("extra_dims concatenated: ",extra_dims.shape)
            transformed = torch.cat((shower, extra_dims), dim=1)
        if energy is not None and transformed is not None:
            
            print("inside Normalizeby Elayer Transform and energy: ",transformed.shape, transformed.min(),transformed.max(), energy.shape,energy.min(), energy.max())
        return transformed, energy

class AddCoordChannels(object):
    """
    Add channel to image containing the coordinate value along particular
    dimension. This breaks the translation symmetry of the convoluitons,
    as discussed in arXiv:2308.03876

        dims -- List of dimensions for which should have a coordinate channel
                should be created.
    """

    def __init__(self, dims):
        self.dims = dims

    def __call__(self, shower, energy, rev=False):

        if rev:
            transformed = shower # generated shower already only has 1 channel
        else:
            coords = []
            for d in self.dims:
                bcst_shp = [1] * shower.ndim
                bcst_shp[d] = -1
                size = shower.size(d)
                coords.append(torch.ones_like(shower) / size *
                              torch.arange(size).view(bcst_shp))
            transformed = torch.cat([shower] + coords, dim=1)
        return transformed, energy
