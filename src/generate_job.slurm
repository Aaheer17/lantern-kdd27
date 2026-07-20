#!/bin/bash
#SBATCH -n 1
#SBATCH -t 72:00:00
#SBATCH --mem=100000
#SBATCH -p bii-gpu
#SBATCH --gres=gpu:1
#SBATCH -A bii_nssac
module load miniforge
module load texlive
source activate torch_gpu_renew
export PATH=~/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR=$1
python3 main.py ${RUN_DIR}/params.yaml --use_cuda --plot --generate -d ${RUN_DIR} -ep ''