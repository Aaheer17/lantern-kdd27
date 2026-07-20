#!/bin/bash
# Submit seed_runs configs for a given seed.
# Usage: bash submit_seed_runs.sh <SEED> [voxel|laplacian|all]
#   e.g. bash submit_seed_runs.sh 1 voxel
#        bash submit_seed_runs.sh 1 all

SEED=$1
WHICH=${2:-all}

if [[ -z "${SEED}" ]]; then
    echo "ERROR: no seed given. Usage: bash submit_seed_runs.sh <SEED> [voxel|laplacian|all]"
    exit 1
fi

CONFIG_DIR="/sfs/weka/scratch/fa7sa/LANTERN_rewrite/LANTERN_20_FEB/LANTERN_new/configs/seed_runs"

# VOXEL_CONFIGS=(
#     'grad_blend_full_voxel.yaml'
#     'pcgrad_full_voxel.yaml'
#     'imtlg_full_voxel.yaml'
#     'gradnorm_full_voxel.yaml'
#     'config_full_voxel.yaml'
# )

# LAPLACIAN_CONFIGS=(
#     'grad_blend_full_laplacian.yaml'
#     'pcgrad_full_laplacian.yaml'
#     'imtlg_full_laplacian.yaml'
#     'gradnorm_full_laplacian.yaml'
#     'config_full_laplacian.yaml'
# )
VOXEL_CONFIGS=(
    'grad_blend_full_voxel_full_on.yaml'
    'grad_blend_full_voxel_full_on_with_late_decay.yaml'
    'grad_blend_full_voxel_no_late_decay.yaml'
    'grad_blend_full_voxel_decay_550.yaml'
    'grad_blend_full_voxel_decay_650.yaml'
)
LAPLACIAN_CONFIGS=(
    'grad_blend_full_laplacian.yaml'
    'grad_blend_full_laplacian_no_late_decay.yaml'
    'grad_blend_full_laplacian_late_decay_550.yaml'
    'grad_blend_full_laplacian_late_decay_650.yaml'

)

#BASELINE_CONFIG='baseline_diffusion_full.yaml'
BASELINE_CONFIG=''
# Build the submission list based on the WHICH argument.
CONFIGS=()
case "${WHICH}" in
    voxel)
        CONFIGS=("${VOXEL_CONFIGS[@]}" "${BASELINE_CONFIG}")
        ;;
    laplacian)
        CONFIGS=("${LAPLACIAN_CONFIGS[@]}" "${BASELINE_CONFIG}")
        ;;
    all)
        CONFIGS=("${VOXEL_CONFIGS[@]}" "${LAPLACIAN_CONFIGS[@]}" "${BASELINE_CONFIG}")
        ;;
    *)
        echo "ERROR: second arg must be voxel, laplacian, or all (got '${WHICH}')"
        exit 1
        ;;
esac

mkdir -p logs
for CONFIG in "${CONFIGS[@]}"; do
    CONFIG_PATH="${CONFIG_DIR}/${CONFIG}"
    JOB_NAME="${CONFIG%.yaml}_s${SEED}"
    if [[ ! -f "${CONFIG_PATH}" ]]; then
        echo "SKIP (missing): ${CONFIG_PATH}"
        continue
    fi
    echo "Submitting: ${JOB_NAME}"
    sbatch \
        --job-name="${JOB_NAME}" \
        --output="logs/${JOB_NAME}_%j.out" \
        --error="logs/${JOB_NAME}_%j.err" \
        run_job.slurm "${CONFIG_PATH}" "${SEED}"
done

echo ""
echo "Submitted ${#CONFIGS[@]} jobs (set=${WHICH}, seed=${SEED}). Check: squeue -u \$USER"