#
# SPDX-FileCopyrightText: Copyright © 2026 University of Bern
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@unibe.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
# Settings of the SLURM job scripts of this folder (*_job.sh, launch_all_clip_scripts.sh).
# Copy this file to config.sh (git-ignored) and set the values of your system:
#   cp slurm_scripts/config.example.sh slurm_scripts/config.sh

# Folder of the CLIP cross-link files: eCLIP/HepG2/*.sum.gff, eCLIP/K652/*.sum.gff and iCLIP/*.gff
CLIP_DIR=/path/to/CLIP/hg38

# Conda environment in which apatools is installed ("" = use the python already on the PATH)
CONDA_ENV=apatools

# Environment module that provides conda on the cluster ("" = conda is already available)
CONDA_MODULE=Anaconda3

# iCLIP files of ${CLIP_DIR}/iCLIP that are not integrated (file names). With the 3 exclusions below, 18 iCLIP files
# remain (the --array size of the iclip_*_job.sh scripts).
ICLIP_EXCLUDED=(
    "group_3950_all-SmB-Cal51-hg38_sum_G_hg38--ensembl59_from_3802-3803-3804-3805-3806-3807Graph-cDNA-hits-in-genome.sorted.gff"
    "group_4014_all-U2AF65-Hela-wt1-hg38_sum_G_hg38--ensembl59_from_1685-1686-3413-3414Graph-cDNA-hits-in-genome.sorted.gff"
    "group_4037_all-hnRNPA1-Hela-hg38_sum_G_hg38--ensembl59_from_3420-3421-3436-3437-3638-3639-3640Graph-cDNA-hits-in-genome.sorted.gff"
)
