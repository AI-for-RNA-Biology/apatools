#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileCopyrightText: Copyright © 2026 University of Bern
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@idiap.ch>
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@unibe.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
#SBATCH --time 04:00:00
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8G
#SBATCH --output=logs/iCLIP-inte-proximal%A\_%a.out
#SBATCH --error=logs/iCLIP-inte-proximal%A\_%a.err
#SBATCH --array=1-18 # job array index
#SBATCH --account gratis

# Paths and environment: config.sh of this folder (copy of config.example.sh). sbatch runs a copy of this script, so
# the folder of the job scripts is given by launch_all_clip_scripts.sh (APATOOLS_SLURM_DIR) or, when this script is
# submitted directly with `sbatch slurm_scripts/<script>` from the repository root, is <submission dir>/slurm_scripts.
SLURM_DIR="${APATOOLS_SLURM_DIR:-${SLURM_SUBMIT_DIR}/slurm_scripts}"
source "${SLURM_DIR}/common.sh"

# iCLIP files, without the ones excluded in config.sh (ICLIP_EXCLUDED)
EXCLUDE_ARGS=()
for name in "${ICLIP_EXCLUDED[@]}"; do EXCLUDE_ARGS+=(-not -name "${name}"); done
CLIPFILE=$(find "${CLIP_DIR}/iCLIP" -maxdepth 1 -type f -name "*.gff" "${EXCLUDE_ARGS[@]}" | awk -v num="${SLURM_ARRAY_TASK_ID}" 'NR==num')

CONFIG_FILE=$1

python3 "${APATOOLS_DIR}/scripts/clip_integration.py" --config_file "${CONFIG_FILE}" --iso_type proximal --n_jobs 1 --clip_file "${CLIPFILE}"
