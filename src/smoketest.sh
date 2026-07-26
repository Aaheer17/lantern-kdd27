#!/bin/bash
# Smoke test: submit ONE short config to verify the pipeline runs from this folder.
# Usage: bash smoketest.sh [SEED]
#   e.g. bash smoketest.sh 1

SEED=${1:-1}

# --- EDIT THIS to point at the configs/seed_runs dir in your NEW folder ---
CONFIG_DIR="/sfs/weka/scratch/fa7sa/lantern-kdd27/configs/seed_runs"
CONFIG="smoketest.yaml"
# -------------------------------------------------------------------------

CONFIG_PATH="${CONFIG_DIR}/${CONFIG}"
JOB_NAME="smoketest_s${SEED}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: config not found: ${CONFIG_PATH}"
    exit 1
fi

mkdir -p logs

echo "Submitting: ${JOB_NAME}"
sbatch \
    --job-name="${JOB_NAME}" \
    --output="logs/${JOB_NAME}_%j.out" \
    --error="logs/${JOB_NAME}_%j.err" \
    run_job.slurm "${CONFIG_PATH}" "${SEED}"

echo ""
echo "Submitted smoke test (seed=${SEED}). Check: squeue -u \$USER"
echo "Logs: logs/${JOB_NAME}_*.out  (stdout)  /  .err  (errors)"
