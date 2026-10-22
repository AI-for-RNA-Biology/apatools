#
# SPDX-FileCopyrightText: Copyright © 2026 University of Bern
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@unibe.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
# Sourced by the *_job.sh scripts, with SLURM_DIR = this folder: loads config.sh, activates the conda environment and
# sets APATOOLS_DIR (the repository root).

if [[ ! -f "${SLURM_DIR}/config.sh" ]]; then
    echo "ERROR: ${SLURM_DIR}/config.sh not found: copy ${SLURM_DIR}/config.example.sh to config.sh and set its paths." >&2
    exit 1
fi
source "${SLURM_DIR}/config.sh"

APATOOLS_DIR="$(cd "${SLURM_DIR}/.." && pwd)"

if [[ -n "${CONDA_ENV}" ]]; then
    [[ -n "${CONDA_MODULE}" ]] && module load "${CONDA_MODULE}"
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi
