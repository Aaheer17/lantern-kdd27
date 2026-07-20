#!/bin/bash
# Submit all WEIGHTED-SUM configs:
#   {full, high, mid, low} x {laplacian, voxel}
# Fixed weights: voxel 0.12, laplacian 0.005
# Usage: bash submit_ws.sh

CONFIG_DIR="/sfs/weka/scratch/fa7sa/LANTERN_rewrite/LANTERN_20_FEB/LANTERN_new/configs/ws_yamls"

CONFIGS=(
    # ---- full ----
    'ws_full_laplacian.yaml'
    'ws_full_voxel.yaml'
    # ---- high ----
    'ws_high_laplacian.yaml'
    'ws_high_voxel.yaml'
    # ---- mid ----
    'ws_mid_laplacian.yaml'
    'ws_mid_voxel.yaml'
    # ---- low ----
    'ws_low_laplacian.yaml'
    'ws_low_voxel.yaml'
)

mkdir -p logs

for CONFIG in "${CONFIGS[@]}"; do
    CONFIG_PATH="${CONFIG_DIR}/${CONFIG}"
    JOB_NAME="${CONFIG%.yaml}"

    if [[ ! -f "${CONFIG_PATH}" ]]; then
        echo "SKIP (missing): ${CONFIG_PATH}"
        continue
    fi

    echo "Submitting: ${JOB_NAME}"
    sbatch \
        --job-name="${JOB_NAME}" \
        --output="logs/${JOB_NAME}_%j.out" \
        --error="logs/${JOB_NAME}_%j.err" \
        run_job.slurm "${CONFIG_PATH}"
done

echo ""
echo "All ${#CONFIGS[@]} weighted-sum jobs submitted. Check status with: squeue -u \$USER"