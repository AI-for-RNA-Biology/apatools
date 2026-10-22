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

# Example usage:
# ./launch_all_clip_scripts.sh config_file_for_eclip config_file_for_iclip

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# The job scripts read their paths from ${SCRIPT_DIR}/config.sh (copy of config.example.sh); sbatch runs copies of
# them, so they are told where this folder is.
export APATOOLS_SLURM_DIR="${SCRIPT_DIR}"

sbatch ${SCRIPT_DIR}/eclip_HepG2_int_prox_job.sh $1
sbatch ${SCRIPT_DIR}/eclip_HepG2_int_dist_job.sh $1
sbatch ${SCRIPT_DIR}/eclip_K652_int_prox_job.sh $1
sbatch ${SCRIPT_DIR}/eclip_K652_int_dist_job.sh $1
sbatch ${SCRIPT_DIR}/iclip_int_prox_job.sh $2
sbatch ${SCRIPT_DIR}/iclip_int_dist_job.sh $2