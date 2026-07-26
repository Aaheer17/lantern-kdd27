# Lantern: Conflict-Aware Gradient Blending for Physics-Guided Diffusion Models in Calorimeter Simulation

This repository contains the code for the paper
**"Lantern: Conflict-Aware Gradient Blending for Physics-Guided Diffusion Models in Calorimeter Simulation."**

## Overview

Lantern is a physics-guided diffusion framework for fast, high-fidelity calorimeter
shower simulation. Training a diffusion model with additional physics-based objectives
(e.g. voxel residual and Laplacian/smoothness losses) alongside the standard
diffusion loss is difficult, because these objectives often produce *conflicting*
gradients that degrade one another when naively summed.

Lantern addresses this with **conflict-aware gradient blending**: instead of simply
adding the losses, it inspects the gradients of the diffusion objective and each
physics objective, detects when they conflict, and blends them so that the
physics-guided signals improve sample fidelity without destabilizing the core
generative training. The result is a diffusion model that respects the underlying
physics of electromagnetic showers.

## Dataset

We use **Dataset 2** from the
[Fast Calorimeter Simulation Challenge 2022 (CaloChallenge)](https://calochallenge.github.io/homepage/),
which consists of simulated electron showers in a granular calorimeter with a
voxel geometry of 45 radial layers × 16 × 9 (6,480 voxels per shower).

- **Dataset homepage:** https://calochallenge.github.io/homepage/
- **Dataset 2 download (Zenodo):** https://zenodo.org/records/6366271

Download the Dataset 2 HDF5 files from the Zenodo link above and place them where your
config files expect them (see the `hdf5_file` / `eval_hdf5_file` paths in
`configs/`). Note that the raw data files are **not** tracked in this repository.

## Installation

We recommend setting up the environment with **Miniforge** (a minimal conda
distribution that uses the community `conda-forge` channel by default).

### 1. Install Miniforge

If you don't already have it, install Miniforge by following the instructions at
https://github.com/conda-forge/miniforge, or on Linux:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

Restart your shell (or `source ~/.bashrc`) so the `conda` command is available.

### 2. Create and activate the environment

```bash
conda create -n lantern python=<PYTHON_VERSION>   # e.g. 3.10
conda activate lantern
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch/CUDA:** `requirements.txt` pins the package versions this project
> was tested with. If you need a specific CUDA build of PyTorch, install it from the
> official index first (see https://pytorch.org/get-started/locally/) and then run the
> `pip install -r requirements.txt` above.

## Training

The architecture builds on [CaloDREAM](https://github.com/luigifvr/calo_dreamer).
Lantern has two networks: an **energy network** (layer energies) and a **shape
network** (voxel shower). They can be trained simultaneously.

**Config locations:**
- Energy network: `configs/d2_energy_model_DDPM.yaml`
- Shape network: the remaining configs (`configs/seed_runs/`, `configs/uw_yamls/`, `configs/ws_yamls/`)

### 1. Set your data paths

In each config you train, update these to your own paths:

```yaml
hdf5_file:      /path/to/dataset_2/dataset_2_1.hdf5   # training file
eval_hdf5_file: /path/to/dataset_2/dataset_2_2.hdf5   # evaluation file
xml_filename:   /path/to/binning_dataset_2.xml        # binning XML
```

### 2. Train the energy network

Point the config path in `src/smoketest.sh` at `configs/d2_energy_model_DDPM.yaml`,
then run:

```bash
bash src/smoketest.sh 1
```

### 3. Train the shape network

Set `CONFIG_DIR` in `src/seed_runs_full.sh` to your `configs/seed_runs` folder,
then run (seed, then `voxel`, `laplacian`, or `all`):

```bash
bash src/seed_runs_full.sh 1 all
```

Check jobs with `squeue -u $USER`.

### 4. Generate samples

Sampling needs the trained energy network. In the shape network's config, set:

```yaml
energy_model: /path/to/results/<your_trained_energy_model_dir>/
sample_us:    true
```

(along with the data paths from Step 1), then submit the shape config as in Step 3.

## Evaluation

### Voxel CFD

`src/voxel_cfd_single.py` computes the voxel-wise **Correlation Frobenius
Distance (CFD)** between a file of generated showers and the Geant4 reference.
Both files are required.

```bash
python src/voxel_cfd_single.py \
    --gen  path/to/generated_samples.hdf5 \
    --ref  path/to/dataset_2_2.hdf5 \
    --out  results/voxel_cfd.csv \
    --match-count
```

Prints the voxel CFD (lower is better) and the number of active voxel positions.
The generated showers must be in the same units and preprocessing as the reference.

## Repository structure

```
lantern-kdd27/
├── configs/        # experiment / seed-run configuration files
├── src/            # model, training, gradient-blending, and evaluation code
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

<!-- TODO: replace with the final BibTeX once the paper is on arXiv / published. -->
```bibtex
@inproceedings{<CITEKEY>,
  title     = {Lantern: Conflict-Aware Gradient Blending for Physics-Guided Diffusion Models in Calorimeter Simulation},
  author    = {<AUTHORS>},
  booktitle = {<VENUE>},
  year      = {<YEAR>}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
