#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileCopyrightText: Copyright © 2026 University of Bern
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@idiap.ch>
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@unibe.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import yaml
from bulkanalysis.filtering import filter_samples_based_on_thresholds
from bulkanalysis.filtering import gmm_threshold
from bulkanalysis.loader import bulk_aggregate_featureCounts_outputs
from bulkanalysis.normalization import quantile_normalize
from bulkanalysis.plot import plot_gene_expression_distribution
from gtfparse import read_gtf
from kallistomanager.loader import merge_kallisto_counts

from apatools.apanalysis import create_table_of_pairs
from apatools.apanalysis import get_significant_shifts_given_fisher_and_scores
from apatools.apanalysis import plot_score_boxplot
from apatools.filtering import add_length_info
from apatools.filtering import bulk_correction
from apatools.filtering import check_proximal_abundances_in_raw_isoforms_counts
from apatools.filtering import correct_counts_for_small_proximals
from apatools.filtering import get_multi_utrs_isoforms_from_reliably_expressed_genes
from apatools.filtering import remove_length_outliers
from apatools.filtering import stats_raw

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def readjust_treatments(actual_samples, treatments):
    """Remove from each treatment of `treatments` ({treatment: sample names}) the samples absent from `actual_samples`.
    """
    # build new lists: removing items from the list being iterated skipped the sample following each removed one
    for treatment, samples in treatments.items():
        treatments[treatment] = [sample for sample in samples if sample in actual_samples]
    return treatments


def load_from_featureCounts(path_to_counts, data_list: str = None, sample_names=None):
    """3' UTR isoform count matrix: read from `path_to_counts` if it exists, otherwise aggregated from the featureCounts
    outputs of `data_list` (`bulkanalysis.loader.bulk_aggregate_featureCounts_outputs`) and saved to `path_to_counts`.

    Args:
        path_to_counts: CSV of the count matrix (read, or written after aggregation).
        data_list: featureCounts output files, one per sample.
        sample_names: sample names of the `data_list` files.

    Returns:
        pd.DataFrame: count matrix, isoforms in rows, samples in columns.

    Raises:
        ValueError: if `path_to_counts` does not exist and no `data_list` is given.
    """
    if os.path.exists(path_to_counts):
        df_counts = pd.read_csv(path_to_counts, index_col=0)
    elif data_list is None:
        raise ValueError(
            "Enter either a valid path to the matrix of counts, \
                         or a list of .txt files output from featureCounts to be aggregated."
        )
    else:
        df_counts = bulk_aggregate_featureCounts_outputs(
            files=data_list, names=sample_names, save_path=path_to_counts
        )

    return df_counts


def main():
    """3' UTR / APA analysis of bulk RNA-seq data, configured by a YAML file (see README and
    config/bulkapa_template.yaml): load or aggregate the isoform counts, correct and filter them, build the table of
    proximal/distal pairs and its scores, and find the significant shifts between treatments. The stdout of the run is
    written to `<path_to_results>/message.log`.
    """
    FLAGS = argparse.ArgumentParser(description="Length analysis for bulk RNAseq data.")
    FLAGS.add_argument("--config_file", help="Location of config file")
    args = FLAGS.parse_args()
    with open(args.config_file, "r") as stream:
        config = yaml.safe_load(stream)
        
        
    if "bulk_correction" in config.keys():
        correction = config["bulk_correction"]
    else:
        correction = True
        
    if "keep_only_long" in config.keys():
        keep_only_long = config["keep_only_long"]
    else:
        keep_only_long = False
        
    if correction:
        path_to_figures = os.path.join(config["path_to_results"], "Figures")
        path_to_results = os.path.join(config["path_to_results"])
    else:
        path_to_figures = os.path.join(config["path_to_results"], "no_bulk_correction", "Figures")
        path_to_results = os.path.join(config["path_to_results"], "no_bulk_correction")


    if not os.path.exists(path_to_results):
        os.makedirs(path_to_results)

    if not os.path.exists(path_to_figures):
        os.makedirs(path_to_figures)


    old_stdout = sys.stdout
    log_file = open(os.path.join(path_to_results, "message.log"), "w")
    sys.stdout = log_file

    # Load
    if "data_list" in config.keys():
        data_list = config["data_list"]
    else:
        data_list = None
    if "data_list" in config.keys():
        data_list = config["data_list"]
        if "sample_names" not in config:
            sample_names = [
                file.split("_")[-1].split(".")[0] for file in config["data_list"]
            ]
        else:
            sample_names = config["sample_names"]
    else:
        data_list = None
        sample_names = None

    if "df_kallisto_path" in config.keys():
        merge_kallisto_counts(
            save_dir=config["df_kallisto_path"],
            data_list=data_list,
            data_path=config["df_kallisto_path"],
        )

        df_counts = pd.read_csv(
            os.path.join(config["df_kallisto_path"], "counts_matrix.csv"), index_col=0
        )
    elif "df_counts_3UTR_path" in config.keys():
        df_counts = load_from_featureCounts(
            path_to_counts=config["df_counts_3UTR_path"],
            data_list=data_list,
            sample_names=sample_names,
        )
    else:
        raise ValueError(
            "Please provide either a valid path to the matrix of counts generated by featureCounts using the \
                --df_counts_3UTR_path argument, \
                or a path to the kallisto quantification results using the --df_kallisto_path argument."
        )
        

    if "strategy" in config.keys():
        strategy = config["strategy"]
    else:
        strategy = "mean"

    df_filtered_genes = pd.read_csv(config["df_filtered_genes_path"], index_col=0)
    print(f"df_counts columns {df_counts.columns}")
    print(f"df counts index {df_counts.index}")
    print(f"df_genes_columns {df_filtered_genes.columns}")

    df_counts = df_counts[df_filtered_genes.columns]

    treatments = readjust_treatments(
        actual_samples=list(df_counts.columns), treatments=config["treatments"]
    )

    print(f"Before filtering, we have {len(df_counts)} isoforms.")
    reps = df_counts.columns

    gtf_quantification = read_gtf(config["gtf_file_focus"])
    print(gtf_quantification.columns)
    if "gene_name" not in gtf_quantification.columns:
        gtf_quantification['gene_name'] = gtf_quantification['gene_id']
    print(f"GTF quantification: {gtf_quantification['gene_name'].values}")

    # Correct counts for isoforms whose annotation is smaller than 300nt length
    df_counts = correct_counts_for_small_proximals(
        df_counts=df_counts,
        reps=reps,
        gtf_file_quantification=config["gtf_file_focus"],
        focus_nt=300,
        minimal_required_length=100,
    )

    # Remove the genes with outlier lengths (probable annotation error)
    df_counts = remove_length_outliers(df_counts[reps])

    stats_raw(
        df_counts,
        save_file=os.path.join(
            path_to_figures, f"reads_per_sample.{config['figures_extension']}"
        ),
    )

    # Keep the reliably expressed multi-UTR genes only
    df_counts_iso = get_multi_utrs_isoforms_from_reliably_expressed_genes(
        df_counts, df_filtered_genes
    )

    print(
        f"Before filtering, we have {len(df_counts_iso)} isoforms and {len(df_counts_iso.groupby('gene').sum())} genes."
    )

    df_counts_iso = add_length_info(
        df_counts_iso, path_gtf_no_length_focus=config["gtf_file_no_focus"]
    )
    
    print(df_counts_iso.head())
    


    insane_genes, df_counts_iso = check_proximal_abundances_in_raw_isoforms_counts(
        df_counts_iso=df_counts_iso,
        path_to_results=path_to_results,
        reps=reps,
    )
    
    # Convert list to DataFrame before saving
    insane_genes_df = pd.DataFrame(insane_genes, columns=['gene'])
    insane_genes_df.to_csv(
        os.path.join(path_to_results, "insane_genes_proximal_abundance.csv")
    )

    # Correct the counts according to bulk profile
    if correction:
        df_counts_iso_corr = bulk_correction(
            df_counts_iso,
            columns_to_correct=list(reps),
            replace_col=True,
            gene_id="gene",
            length_id="length",
            clip=True,
        )
    else:
        df_counts_iso_corr = df_counts_iso.copy()
    print(df_counts_iso_corr.columns)
    

    df_counts_iso_log = np.log2(
        df_counts_iso_corr.drop(columns=["gene", "length", "strand"], axis=1).set_index(
            "transcript_id"
        )
        + 1
    )
    # Compute the thresholds for reliably expressed ISOFORMS for each sample
    thresholds = {}
    for col in df_counts_iso_log.columns:
        threshold = gmm_threshold(
            data=np.array(df_counts_iso_log[df_counts_iso_log[col] != 0][col]),
            quantile_threshold=config["quantile_threshold"],
            background_quantile=config["background_quantile"],
            density=True,
            show_plot=False,
            data_name=col,
            saving_folder=path_to_figures,
            extension=config["figures_extension"],
            return_int=False,
        )
        thresholds[col] = threshold
    print(thresholds)

    df_counts_iso_corr_filtered = filter_samples_based_on_thresholds(
        df_genes=df_counts_iso_corr,
        treatments=treatments,
        thresholds=thresholds,
        pct_in_treatment=config["pct_in_treatment"],
        save_file=os.path.join(
            path_to_figures,
            f"isoform_exp_after_filtering.{config['figures_extension']}",
        ),
    )

    if "FOSL2.6" in df_counts_iso_corr_filtered.index:
        print("We have FOSL2.6 after filtering")
    else:
        print("We do not have FOSL2.6 after filtering")

    print(f"{df_counts_iso_corr_filtered.shape[0]} isoforms left after filtering.")


    df_counts_iso_corr_filtered.set_index("transcript_id", inplace=True)
    print(df_counts_iso_corr_filtered.head())

    if "quantile_normalization" in config.keys():
        quantile_norm = config["quantile_normalization"]
    else:
        quantile_norm = True

    if quantile_norm:
        df_counts_iso_corr_normalized = quantile_normalize(
            df_counts_iso_corr_filtered[reps]
        )
        plot_gene_expression_distribution(
            df_counts_iso_corr_normalized,
            title="Isoform expression after filtering and normalization",
            save_file=os.path.join(
                path_to_figures,
                f"isoform_expression_after_filtering_quantile.{config['figures_extension']}",
            ),
        )
        if "ENST" in df_counts_iso_corr_normalized.index[0]:
            # Mack's style gene naming
            df_counts_iso_corr_normalized["gene"] = [
                x.split(".")[0] for x in df_counts_iso_corr_normalized.index
            ]  # remove version number
        else:
            # classic isoform naming
            df_counts_iso_corr_normalized["gene"] = [
                x[:-2] for x in df_counts_iso_corr_normalized.index
            ]

        df_counts_iso_corr_normalized = add_length_info(
            df_counts_iso_corr_normalized,
            path_gtf_no_length_focus=config["gtf_file_no_focus"],
        )

    else:
        df_counts_iso_corr_normalized = df_counts_iso_corr_filtered.copy()

    df_counts_iso_corr_normalized = df_counts_iso_corr_normalized.reset_index()
    print(df_counts_iso_corr_normalized.columns)
    print(df_counts_iso_corr_normalized.head())
    
    
    if keep_only_long:
        print("Keeping only the longest and shortest isoforms for each gene in the table of pairs.")
        idx_min = df_counts_iso_corr_normalized.groupby("gene")["length"].idxmin()
        idx_max = df_counts_iso_corr_normalized.groupby("gene")["length"].idxmax()
        keep_idx = pd.Index(idx_min.tolist() + idx_max.tolist()).unique()
        df_counts_iso_corr_normalized = df_counts_iso_corr_normalized.loc[keep_idx].sort_index()
    
    table_of_pairs = create_table_of_pairs(
        df_counts_iso_corr_normalized,
        treatments=treatments,
        replicates=False,
        strategy=strategy,
    )

    print(f"We analyze {len(table_of_pairs)} pairs of isoforms.")

    for score in ["PUD", "RUD", "DUD"]:
        plot_score_boxplot(
            table_of_pairs=table_of_pairs,
            treatments=treatments,
            score=score,
            statistic="wilcoxon",
            save_file=os.path.join(
                path_to_figures,
                f"{score}_boxplot_distributions.{config['figures_extension']}",
            ),
        )

    table_of_pairs.to_csv(
        os.path.join(path_to_results, f"table_of_pairs_simplified.csv")
    )

    if "thresh_FDR" in config.keys():
        thresh_FDR = config["thresh_FDR"]
    else:
        thresh_FDR = 0.01

    if "thresh_PUD" in config.keys():
        thresh_PUD = config["thresh_PUD"]
    else:
        thresh_PUD = 0.15

    if "thresh_RUD" in config.keys():
        thresh_RUD = config["thresh_RUD"]
    else:
        thresh_RUD = 1

    if "thresh_pval" in config.keys():
        thresh_pval = config["thresh_pval"]
    else:
        thresh_pval = None
        
    if "treatment_pairs" in config.keys():
        treatment_pairs = [tuple(pair) for pair in config["treatment_pairs"]]
    else:
        treatment_pairs = None

    table_of_pairs = get_significant_shifts_given_fisher_and_scores(
        table_of_pairs,
        treatments=treatments,
        path_to_results=path_to_results,
        scores=["PUD", "RUD"],
        extension=config["figures_extension"],
        thresh_FDR=thresh_FDR,
        thresh_PUD=thresh_PUD,
        thresh_RUD=thresh_RUD,
        thresh_pval=thresh_pval,
        treatment_pairs=treatment_pairs,
    )

    table_of_pairs.to_csv(os.path.join(path_to_results, f"table_of_pairs.csv"))

    print(
        f"{df_counts_iso_corr_filtered.shape[0]} isoforms will be saved in the table."
    )
    df_counts_iso_corr_normalized.to_csv(
        os.path.join(
            path_to_results, f"df_counts_filtered_corrected_normalized.csv"
        )
    )

    sys.stdout = old_stdout

    log_file.close()


if __name__ == "__main__":
    main()
