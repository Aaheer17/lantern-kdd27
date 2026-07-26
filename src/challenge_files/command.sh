#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=mgmn
##SBATCH --exclusive
#SBATCH -t 9:00:00
#SBATCH --mem=150000
#SBATCH -p bii-gpu
#SBATCH --gres=gpu
#SBATCH -A bii_nssac
module load miniforge
module load texlive

source activate torch_gpu_renew
export PATH=~/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
which dvipng

python3 evaluate_fpd_kpd.py \
  --dataset_path /project/biocomplexity/fa7sa/calo_dreamer/fpd_kpd_data \
  --output_dir /project/biocomplexity/fa7sa/calo_dreamer/fpd_kpd_data \
  --binning_file binning_dataset_2.xml \
  --particle_type electron \
  --dataset_num 2 \
  --metrics fpd-kpd
# python3 evaluate_updated.py -i /project//biocomplexity/fa7sa/calo_dreamer/results/20250707_100712_d2_vit_epoch_400_diffusion_SHOW_ONLY_DIFFUSION/ -r /project/biocomplexity/fa7sa/calochallenge_datasets/dataset_2/dataset_2_2.hdf5 -d 2 --output_dir /project/biocomplexity/fa7sa/calo_dreamer/results/20250707_100712_d2_vit_epoch_400_diffusion_SHOW_ONLY_DIFFUSION/
# --which_cuda --cut 0.01515 --mode 'cls-low'


# python3 evaluate_updated.py -i /project/biocomplexity/fa7sa/calo_dreamer/results/20250108_144726_d2_vit_800_full_3161_nonoise_learnemb_scale_clip/ -r /project/biocomplexity/fa7sa/calochallenge_datasets/dataset_2/dataset_2_2.hdf5 -d 2 --output_dir /project/biocomplexity/fa7sa/calo_dreamer/results/20250108_144726_d2_vit_800_full_3161_nonoise_learnemb_scale_clip/eval/
# --which_cuda --cut 0.01515