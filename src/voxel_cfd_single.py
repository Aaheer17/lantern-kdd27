#!/usr/bin/env python3
"""voxel_cfd_single.py — voxel CFD for a single generated HDF5 file.

Computes the voxel-wise Correlation Frobenius Distance (CFD) between one file
of generated showers and the Geant4 reference file. This is the minimal,
dependency-light version of the batch harness `voxel_cfd.py`: no runs CSV,
no folder scanning, no loss-type/variant filtering — just one generated file
in, one CFD number out.

Voxel CFD measures how well a model reproduces the longitudinal correlation
between a voxel and the voxel at the same transverse position in the next
layer. For every consecutive layer pair (i, i+1) and every transverse
position (a, r) we take the Pearson correlation across showers, giving a
correlation tensor of shape (L-1, A, R). The CFD is the normalized Frobenius
distance between the generated and reference tensors, restricted to positions
that carry nonzero variance in both.

Usage:
    python voxel_cfd_single.py \
        --gen  path/to/samples.hdf5 \
        --ref  path/to/reference.hdf5 \
        [--out result.csv] \
        [--match-count] \
        [--verbose]

Notes:
  * --gen and --ref both required: CFD is a distance to Geant4, so a single
    file cannot be scored in isolation.
  * --match-count truncates the generated set to the number of reference
    showers (mirrors the batch harness, which did gen_data[:n_ref]).
  * The correlation and CFD math is identical to voxel_cfd.py, so results
    reproduce the paper's numbers exactly.
"""

import argparse
from pathlib import Path

import numpy as np
import h5py
from scipy.stats import pearsonr


# ------------------ data loading (identical to voxel_cfd.py) ------------------

def load_showers(h5_path, dataset=2, verbose=False):
    if dataset != 2:
        raise ValueError("This loader currently supports dataset=2 only.")
    expected_flat = 45 * 16 * 9
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        if verbose:
            print("HDF5 keys:", keys)
        candidates = []
        for k in keys:
            shape = f[k].shape
            if len(shape) == 2 and shape[1] == expected_flat:
                candidates.append((k, shape, "flat"))
            elif len(shape) == 4 and shape[1:] == (45, 16, 9):
                candidates.append((k, shape, "voxel"))
            elif len(shape) == 5 and shape[1:] == (1, 45, 16, 9):
                candidates.append((k, shape, "voxel_ch"))
        if not candidates:
            shapes = {k: f[k].shape for k in keys}
            raise ValueError(
                f"Could not find shower dataset in {h5_path}. "
                f"Available keys/shapes: {shapes}")
        priority = {"voxel": 0, "voxel_ch": 1, "flat": 2}
        candidates.sort(key=lambda x: priority[x[2]])
        key, shape, kind = candidates[0]
        data = f[key][:]
    if verbose:
        print(f"Selected key '{key}' with shape {shape} ({kind}) from {h5_path}")
    if kind == "flat":
        data = data.reshape(-1, 45, 16, 9)
    elif kind == "voxel_ch":
        data = data[:, 0, :, :, :]
    if data.ndim != 4 or data.shape[1:] != (45, 16, 9):
        raise ValueError(f"After reshape, got {data.shape}, expected (N,45,16,9)")
    return data


# ------------------ correlation + CFD (identical to voxel_cfd.py) -------------

def calculate_correlation_voxel_stable(data, eps=1e-12):
    N, n_layers, A, R = data.shape
    corr_mats = np.zeros((n_layers - 1, A, R))
    active_mask = np.zeros_like(corr_mats, dtype=bool)
    for i in range(n_layers - 1):
        x1 = data[:, i]
        x2 = data[:, i + 1]
        for a in range(A):
            for r in range(R):
                v1 = x1[:, a, r]
                v2 = x2[:, a, r]
                if not np.isfinite(v1).all() or not np.isfinite(v2).all():
                    continue
                std1 = np.std(v1)
                std2 = np.std(v2)
                if std1 < eps and std2 < eps:
                    continue
                if std1 < eps or std2 < eps:
                    continue
                corr, _ = pearsonr(v1, v2)
                if np.isfinite(corr):
                    corr_mats[i, a, r] = corr
                    active_mask[i, a, r] = True
    return corr_mats, active_mask


def compute_voxel_cfd(gen_data, ref_corr, ref_mask):
    gen_corr, gen_mask = calculate_correlation_voxel_stable(gen_data)
    valid_mask = gen_mask & ref_mask
    if valid_mask.sum() == 0:
        return np.nan, 0
    diff = gen_corr[valid_mask] - ref_corr[valid_mask]
    frob = np.sqrt(np.sum(diff ** 2))
    ref_norm = max(np.sqrt(np.sum(ref_corr[valid_mask] ** 2)), 1e-12)
    return frob / ref_norm, int(valid_mask.sum())


# ------------------ main ------------------

def main():
    parser = argparse.ArgumentParser(
        description="Voxel CFD for a single generated HDF5 file vs. the reference.")
    parser.add_argument("--gen", required=True,
                        help="Generated showers HDF5 file to score.")
    parser.add_argument("--ref", required=True,
                        help="Geant4 reference HDF5 file.")
    parser.add_argument("--out", default=None,
                        help="Optional CSV path to append the result to.")
    parser.add_argument("--match-count", action="store_true",
                        help="Truncate generated showers to the reference count "
                             "(mirrors the batch harness).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("Loading reference showers...")
    ref_data = load_showers(args.ref, dataset=2, verbose=args.verbose)
    n_ref = len(ref_data)
    print(f"  reference: {n_ref} showers")
    print("Computing reference correlation tensor...")
    ref_corr, ref_mask = calculate_correlation_voxel_stable(ref_data)
    del ref_data

    print("Loading generated showers...")
    gen_data = load_showers(args.gen, dataset=2, verbose=args.verbose)
    print(f"  generated: {len(gen_data)} showers")
    if args.match_count:
        gen_data = gen_data[:n_ref]
        print(f"  truncated to {len(gen_data)} to match the reference count")

    print("Computing voxel CFD...")
    cfd, n_active = compute_voxel_cfd(gen_data, ref_corr, ref_mask)

    if np.isnan(cfd):
        print("\nVoxel CFD: NaN (no jointly-active positions)")
    else:
        print(f"\nVoxel CFD: {cfd:.4f}")
    print(f"Active voxel positions: {n_active}")

    if args.out:
        import csv
        out_path = Path(args.out)
        write_header = not out_path.exists()
        with out_path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if write_header:
                w.writerow(["gen_file", "ref_file", "voxel_cfd",
                            "active_voxels", "n_gen", "n_ref"])
            w.writerow([args.gen, args.ref,
                        "" if np.isnan(cfd) else f"{cfd:.6f}",
                        n_active, len(gen_data), n_ref])
        print(f"appended result to {out_path}")


if __name__ == "__main__":
    main()
