<!--
#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileCopyrightText: Copyright © 2026 University of Bern
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@idiap.ch>
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@unibe.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
-->
# Project

This project aims at creating a package with multiple tools to analyze Alternative Polyadenylation (APA) from RNA sequencing data (bulk and single-cell).

## Origin

apatools was first developed at the [Idiap Research Institute](https://www.idiap.ch), where [version 0.1](https://github.com/idiap/apatools) was released. It has been developed since at the University of Bern: this repository (version 0.2.0 and later) continues it.

## Installation

To install this library in another Python project, execute simply:

```bash
pip install git+https://github.com/AI-for-RNA-Biology/apatools.git
```
or
```bash
git clone https://github.com/AI-for-RNA-Biology/apatools.git
cd apatools
pip install .
```

## Scripts
### `bulkapa`
`bulkapa.py` aims at performing 3' UTR analysis of bulk RNA sequencing data. To run it, udpate the file `config/bulkapa_template.yaml` with the right paths. You can then run the script as follows:
```bash
python3 scripts/bulkapa.py --config_file config/bulkapa_template.yaml
```

The config file should contain:
- `df_counts_3UTR_path`: Path to the matrix of 3' UTR isoforms (output of quantification).
- `gtf_file_focus: GTF` file used for quantification of the 3' UTR isoforms -- usually it corresponds to a focus on the last nucleotides of the original 3' UTR isoform annotation.
- `gtf_file_no_focus`: Original GTF file describing the 3' UTR isoforms (no focus on the last nucleotides).
- `df_filtered_genes_path`: Path to the reliably expressed genes dataframe. Should contain gene names in the first column. This matrix is used to filter the reliably expressed 3' UTR isoforms.
- `extension_figures`: Extension you want to save your figures with, e.g "pdf", "png",...
- `treatments`: dictionnary with name of the treatments in keys and list of corresponding sample names in keys.
- `path_to_results`: Directory where to save the results.

### `clip_integration`
`clip_integration.py` intersects one CLIP file (`.gff` of cross-link sites) with the regions around the PAS of every proximal or distal 3' UTR isoform of a `bulkapa` table of pairs. For each region it compares the PUD of cross-linked vs non cross-linked pairs (Welch test) and tests the association between cross-linking and significant shifts (Fisher test). It saves one `<RBP>_results_<iso_type>.pkl` per CLIP file. It requires the `cliptools` package. Update `config/clip_integration_template.yaml` with the right paths, then run:
```bash
python3 scripts/clip_integration.py --config_file config/clip_integration_template.yaml --iso_type proximal --clip_file /path/to/one_clip_file.gff --n_jobs 1
```

`--iso_type` is `proximal`, `distal` or `stop_codon`: windows around the PAS of the proximal or distal isoforms, or around the 3' UTR start (first nucleotide after the stop codon; each pair counted once, under its distal isoform).

The config file should contain:
- `filepath_to_gtf_isoforms`: GTF file describing the 3' UTR isoforms (no focus on the last nucleotides).
- `filepath_to_table_of_pairs`: `table_of_pairs.csv` output of `bulkapa`.
- `filepath_to_sig_pairs`: list of significant shift files (`shifts/pairs_*.csv` outputs of `bulkapa`).
- `path_to_results`: Directory where to save the results.
- `treatments`: list of the treatments to compare.
- `nt_before` / `nt_after`: number of nucleotides upstream/downstream of the PAS to analyze.
- `length_region`: length of each region within `[nt_before, nt_after]`.
- `split_name_before` / `split_number_before` / `split_name_after` / `split_number_after`: how to parse the RBP name from the CLIP file name.
- `bootstrap`: whether to compute a bootstrap null distribution.
- `filepath_to_CLIP` (optional): CLIP file or directory, used only if `--clip_file` is not given.

### SLURM scripts
`slurm_scripts/` runs `clip_integration.py` on every CLIP dataset as job arrays (one task per CLIP file), for eCLIP HepG2, eCLIP K562 and iCLIP, on proximal and distal isoforms.

The paths and the environment of your system are set in `slurm_scripts/config.sh`, which is not versioned: create it from the template and edit it:
```bash
cp slurm_scripts/config.example.sh slurm_scripts/config.sh
```
It defines `CLIP_DIR` (folder with `eCLIP/HepG2/*.sum.gff`, `eCLIP/K652/*.sum.gff` and `iCLIP/*.gff`), `CONDA_ENV` (conda environment with apatools installed; empty = the current `python3`), `CONDA_MODULE` (environment module providing conda; empty = none) and `ICLIP_EXCLUDED` (iCLIP files not integrated). The job scripts load it through `slurm_scripts/common.sh`. Adapt the `#SBATCH` lines of the job scripts (account, time, memory, `--array` = number of CLIP files) to your cluster and CLIP collection.

To submit all six arrays (the first config is used for the eCLIP files, the second for the iCLIP files):
```bash
bash slurm_scripts/launch_all_clip_scripts.sh config_eCLIP.yaml config_iCLIP.yaml
```
A single array can also be submitted from the repository root, e.g. `sbatch slurm_scripts/iclip_int_prox_job.sh config_iCLIP.yaml`.

## Modules
### `TableOfPairs`
`apatools.TableOfPairs.TableOfPairs` wraps a table of proximal/distal isoform pairs (PUD per sample) for downstream statistics and plots (per-treatment comparisons, linear mixed models, ...). It requires the `omicstools` package (`omicstools.helper_plot`, `omicstools.LMM_helper`).
