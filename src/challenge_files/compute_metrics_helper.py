import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import challenge_files.HighLevelFeatures as HLF
#from utils import *
from scipy.stats import wasserstein_distance
import math
import pandas as pd
import jetnet
import vector

def separation_power(hist1, hist2, bins):
    """ computes the separation power aka triangular discrimination (cf eq. 15 of 2009.03796)
        Note: the definition requires Sum (hist_i) = 1, so if hist1 and hist2 come from
        plt.hist(..., density=True), we need to multiply hist_i by the bin widhts
    """
    hist1, hist2 = hist1*np.diff(bins), hist2*np.diff(bins)
    ret = (hist1 - hist2)**2
    ret /= hist1 + hist2 + 1e-16
    return 0.5 * ret.sum()


def get_emd(dist1, dist2):
    # compute the emd score aka wasserstein distance-1
    emd_score = wasserstein_distance(dist1, dist2)
    return emd_score

def mean_energy(xy, data):
    if xy == 'r':
        axis = (1, 2)
    if xy == 'a':
        axis = (1, 3)
    if xy == 'z':
        axis = (2, 3)
    print("In mean_energy ",data.shape) 
    result = np.sum(data, axis=axis)/1000 # GeV
    mean_result = np.mean(result, axis=0)
    
    return mean_result


def calculate_frob_norm(corr_geant, corr_gen):
    """
    computing frobenius norm between two numpy array manually.
    """
    return np.sqrt(np.sum((corr_geant - corr_gen) ** 2))


def calculate_correlation_layer(layer_data):
    """
    computing correlation between layer i and layer i+1.
    """
    dim = layer_data.shape[1]
    correlation_matrix = np.ones((dim, dim))
    p_value_matrix = np.zeros((dim, dim))

    # Loop through each pair of layers and compute correlations and p-values
    for i in range(dim):
        for j in range(dim):
            if i != j:  # Exclude self-correlation (diagonal elements)
                corr, p_value = pearsonr(layer_data[:, i], layer_data[:, j])
                correlation_matrix[i, j] = corr
                p_value_matrix[i, j] = p_value
                
                
    return correlation_matrix, p_value_matrix


def calculate_correlation_voxel(data):
    """
    Generate voxel-wise correlation matrices for consecutive layers.
    That means, correlation between layer i and layer i+1 for voxel j
    
    Parameters:
        data (ndarray): Input data of shape (100000, 45, 16, 9).
        
    Returns:
        correlation_matrices (list of ndarray): A list of correlation matrices (16x9) for each layer pair. Final shape is (44,16,9)
    """
    n_layers = data.shape[1]
    correlation_matrices = []
    a_bin = data.shape[2]
    r_bin = data.shape[3]
    # Iterate over each pair of consecutive layers
    for i in range(n_layers - 1):
        # Extract data for the two layers
        layer_data_1 = data[:, i, :, :].reshape(-1, a_bin, r_bin)  # Shape: (100000, 16, 9)
        layer_data_2 = data[:, i + 1, :, :].reshape(-1, a_bin, r_bin)  # Shape: (100000, 16, 9)

        # Initialize the correlation matrix for this layer pair
        correlation_matrix = np.zeros((a_bin, r_bin))

        # Compute voxel-wise correlations
        for row in range(a_bin):
            for col in range(r_bin):
                corr, _ = pearsonr(layer_data_1[:, row, col], layer_data_2[:, row, col])
                correlation_matrix[row, col] = corr

        correlation_matrices.append(correlation_matrix)

    return np.array(correlation_matrices)


def calculate_correlation_group(data):
    """
    Generate voxel-wise correlation matrices for consecutive layers.
    
    Parameters:
        data (ndarray): Input data of shape (100000, group#, radial_bins).
        
    Returns:
        correlation_matrices (list of ndarray): A list of correlation matrices  for each layer pair.
    """
    n_layers = data.shape[1]
    correlation_matrices = []
   
    r_bin = data.shape[2]
    # Iterate over each pair of consecutive layers
    for i in range(n_layers - 1):
        # Extract data for the two layers
        layer_data_1 = data[:, i, :].reshape(-1, r_bin)  # Shape: (100000, 16, 9)
        layer_data_2 = data[:, i + 1, :].reshape(-1,  r_bin)  # Shape: (100000, 16, 9)

        # Initialize the correlation matrix for this layer pair
        correlation_matrix = np.zeros((9))

        # Compute voxel-wise correlations

        for col in range(r_bin):
            corr, _ = pearsonr(layer_data_1[:, col], layer_data_2[:, col])
            correlation_matrix[col] = corr

        correlation_matrices.append(correlation_matrix)

    return np.array(correlation_matrices)

def evaluate_fpd_kpd(Es, Showers, HLFs, model_names, files, args):
    """Calculates FPD and KPD scores """
    fpd_vals = {}
    kpd_vals = {}
    fpd_errs = {}
    kpd_errs = {}
    g_index=model_names.index('Geant4')
            
    reference_HLF = HLFs[g_index]
    reference_file = h5py.File(files[g_index],'r')
    reference_array = prepare_high_data_for_classifier(reference_file, reference_HLF, 1)
    reference_array=reference_array[:, :-1]

    for j in range(len(model_names)):
        if j != g_index:
            source_file = h5py.File(files[j], 'r')
            source_array = prepare_high_data_for_classifier(source_file, HLFs[j], 0)
            source_array = source_array[:, :-1]
            fpd_val, fpd_err = jetnet.evaluation.fpd(reference_array, source_array)
            kpd_val, kpd_err = jetnet.evaluation.kpd(reference_array, source_array)
            name = "dataset_" + str(args.dataset_num) + "_particle_" + args.particle_type + "_model_names"+model_names[j]
            fpd_vals[name] = fpd_val
            kpd_vals[name] = kpd_val
            fpd_errs[name] = fpd_err
            kpd_errs[name] = kpd_err
        print("done with:", model_names[j])
        
        
    filename = "fpd_val_" + str(args.dataset_num) + "_" + args.particle_type + ".txt"
    save_path = os.path.join(args.output_dir, filename)
    write_dict_to_txt(fpd_vals, save_path)
    filename = "kpd_val_" + str(args.dataset_num) + "_" + args.particle_type + ".txt"
    save_path = os.path.join(args.output_dir, filename)
    write_dict_to_txt(kpd_vals,save_path)
    filename = "fpd_errs_" + str(args.dataset_num) + "_" + args.particle_type + ".txt"
    save_path = os.path.join(args.output_dir, filename)
    write_dict_to_txt(fpd_errs, save_path)
    filename = "kpd_errs_" + str(args.dataset_num) + "_" + args.particle_type + ".txt"
    save_path = os.path.join(args.output_dir, filename)
    write_dict_to_txt(kpd_errs, save_path)
