#!/bin/bash
# Submit all UNCERTAINTY-WEIGHTING configs:
#   {sched, pure} x {full, high, mid, low} x {laplacian, voxel}
# Usage: bash submit_uw.sh

CONFIG_DIR="/sfs/weka/scratch/fa7sa/LANTERN_rewrite/LANTERN_20_FEB/LANTERN_new/configs/uw_yamls"

CONFIGS=(
    # ---- full ----
    'uw_sched_full_laplacian.yaml'
    'uw_pure_full_laplacian.yaml'
    'uw_sched_full_voxel.yaml'
    'uw_pure_full_voxel.yaml'
    # ---- high ----
    'uw_sched_high_laplacian.yaml'
    'uw_pure_high_laplacian.yaml'
    'uw_sched_high_voxel.yaml'
    'uw_pure_high_voxel.yaml'
    # ---- mid ----
    'uw_sched_mid_laplacian.yaml'
    'uw_pure_mid_laplacian.yaml'
    'uw_sched_mid_voxel.yaml'
    'uw_pure_mid_voxel.yaml'
    # ---- low ----
    'uw_sched_low_laplacian.yaml'
    'uw_pure_low_laplacian.yaml'
    'uw_sched_low_voxel.yaml'
    'uw_pure_low_voxel.yaml'
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
echo "All ${#CONFIGS[@]} UW jobs submitted. Check status with: squeue -u \$USER"