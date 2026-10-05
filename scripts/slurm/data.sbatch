#!/usr/bin/env bash
#SBATCH --job-name=eeg-data
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
stage="${STAGE:-prepare}"
config="${CONFIG:-configs/openbmi.yaml}"
seed="${SEED:-42}"
case "$stage" in
  cache) "$python_bin" -m eeg_task_fusion.cli cache --config "$config" --seed "$seed" --download ;;
  prepare|verify-api|splits) "$python_bin" -m eeg_task_fusion.cli "$stage" --config "$config" --seed "$seed" ;;
  *) printf 'Unknown stage: %s\n' "$stage" >&2; exit 2 ;;
esac