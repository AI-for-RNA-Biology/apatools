#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileCopyrightText: Copyright © 2026 University of Bern
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@idiap.ch>
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@unibe.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gtfparse import read_gtf


def remove_length_outliers(
    df_counts: pd.DataFrame,
    outliers_list: list = [
        "ABCD3",
        "UPK3B",
        "ABCF3",
        "PTTG2",
        "ANKRD30A",
        "AL583828.1",
        "ZDHHC11",
    ],
):
    """Remove transcripts or genes from a count matrix according to a list of genes.
    Args:
        df_counts: DataFrame of read counts with gene names or transcript names in index.
        outliers_list: List of genes to remove. Defaults to [ "ABCD3", "UPK3B", "ABCF3", "PTTG2",
        "ANKRD30A", "AL583828.1", "ZDHHC11", ].

    Returns:
        pd.DataFrame: the filtered count matrix
    """
    for outlier in outliers_list:
        df_counts = df_counts[~df_counts.index.str.startswith(outlier)]
    return df_counts


def add_length_info(df_counts: pd.DataFrame, path_gtf_no_length_focus: str):
    """From a transcript counts matrix and the GTF file used to produce it, add the length
    column to the count matrix.

    Args:
        df_counts: DataFrame of read counts with transcript names in index.
        path_gtf_no_length_focus: Path to the GTF file that has been used for quantification; i.e corresponding
        to the transcripts in df_counts.

    Returns:
        pd.DataFrame: the count matrix with a new column reporting transcript length.
    """
    df_counts_length_info = df_counts.copy()
    # Create the table of pairs
    annotations = read_gtf(path_gtf_no_length_focus)
    annotations["length"] = annotations["end"] - annotations["start"]

    df_counts_length_info = annotations[["transcript_id", "length", "strand"]].merge(
        df_counts, left_on="transcript_id", right_index=True
    )
    df_counts_length_info.drop_duplicates(keep="first", ignore_index=True, inplace=True)
    return df_counts_length_info


def stats_raw(df_counts: pd.DataFrame, save_file: str = None):
    """Plot the number of reads per sample and print the current number of isoforms and genes

    Args:
        df_counts: DataFrame of read counts, with samples in columns.
        save_file: Filename to save the figures

    """

    # Number of reads per sample
    plt.subplots(1, 1, figsize=(1 * len(df_counts.columns), 4))
    sns.barplot(
        x=np.log10(df_counts.sum()).index,
        y=np.log10(df_counts.sum()).values,
        color="gray",
    )
    plt.xticks(rotation=90)
    plt.ylabel("$log_{10}(reads)$")
    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight")

    nb_genes = len(list(set(list([x[:-2] for x in df_counts.index.values]))))

    print(f"Number of isoforms: {len(df_counts)}")
    print(f"Number of genes: {nb_genes}")


def get_multi_utrs_isoforms_from_reliably_expressed_genes(
    df_counts: pd.DataFrame, df_counts_genes: pd.DataFrame
):
    """Filter the read counts matrix of isoforms using the reliably expressed genes counts matrix.

    Args:
        df_counts: 3' UTR isoforms count matrix with transcript names in index.
        df_counts_genes: Reliably expressed genes count matrix with gene names in index.

    Returns:
        pd.DataFrame: the filtered 3' UTR counts matrix.
    """

    # Filter genes that are not reliably expressed
    reliably_expressed_genes = list(df_counts_genes.index)

    if "ENST" in df_counts.index[0]:
        # Mack's style gene naming
        df_counts["gene"] = [
            x.split(".")[0] for x in df_counts.index
        ]  # remove version number
    else:
        # classic isoform naming
        df_counts["gene"] = [x[:-2] for x in df_counts.index]
        
    df_counts_iso = df_counts[df_counts["gene"].isin(reliably_expressed_genes)]

    # Filter genes that do not express multiple isoforms
    multi_utr_genes = (
        df_counts_iso.groupby("gene")
        .count()[df_counts_iso.groupby("gene").count()[df_counts_iso.columns[0]] > 1]
        .index
    )
    df_counts_iso = df_counts_iso[df_counts_iso["gene"].isin(multi_utr_genes)]

    print(
        f"After filtering isoforms of genes that are not reliably expressed, \
          we have {len(df_counts_iso)} reliably expressed isoforms."
    )

    return df_counts_iso


# add in apatools.filtering
def bulk_correction(
    df_iso: pd.DataFrame,
    columns_to_correct: list,
    replace_col: bool = False,
    gene_id: str = "gene",
    length_id: str = "length",
    clip=True,
):
    """Correct the isoform counts by taking into account the fact that bulk
    coverage is flat.

    Args:
        df_iso: DataFrame of 3' UTRs raw counts. Isoforms in rows and samples in columns.
        You should also have a column with the names of the genes associated to the 3' UTR isoforms,
        as well as a column with the length of the isoforms.
        columns_to_correct: Name of the columns (i.e samples) you want to correct.
        replace_col: Whether to replace the columns with the corrected values or not.
        If False, it will create new columns with the suffix _corrected. Defaults to False.
        gene_id: Name of the column containing the gene names. Defaults to "gene".
        length_id: Name of the column containing the length of the 3' UTR isoforms. Defaults to "length".
        clip (bool, optional): Whether to clip the negative values to 0 or not. Defaults to True.

    Returns:
        pd.DataFrame: The new dataframe with the corrected 3' UTR isoform counts.
    """
    df_counts_iso = df_iso.copy()
    df_counts_iso.sort_values(
        [gene_id, length_id], ascending=(True, False), inplace=True
    )
    for column_to_correct in columns_to_correct:
        if replace_col:
            df_counts_iso[f"{column_to_correct}"] = df_counts_iso[
                f"{column_to_correct}"
            ] - df_counts_iso.groupby(gene_id)[column_to_correct].transform(
                "shift", fill_value=0
            )
            if clip:
                df_counts_iso[column_to_correct] = df_counts_iso[
                    column_to_correct
                ].clip(lower=0)
        else:
            df_counts_iso[f"{column_to_correct}_corrected"] = df_counts_iso[
                f"{column_to_correct}"
            ] - df_counts_iso.groupby(gene_id)[column_to_correct].transform(
                "shift", fill_value=0
            )
            if clip:
                df_counts_iso[f"{column_to_correct}_corrected"] = df_counts_iso[
                    f"{column_to_correct}_corrected"
                ].clip(lower=0)

    return df_counts_iso


def check_proximal_abundances_in_raw_isoforms_counts(
    df_counts_iso,
    path_to_results,
    reps,
    remove_non_max_proximals=False,
    threshold_reads=20,
):
    """Check, in each sample, whether the proximal (shortest) 3' UTR isoform of every gene is its most expressed one.

    A proximal isoform passes if its count plus `threshold_reads` reaches the count of the most expressed isoform of the
    gene. The genes that fail in a sample are written to
    `<path_to_results>/genes_utrs/<sample>_genes_prox_not_most_expressed.csv`.

    Args:
        df_counts_iso: isoform count matrix with a `gene` and a `length` column and one column per sample.
        path_to_results: directory where the `genes_utrs/` lists are written.
        reps: sample columns to check.
        remove_non_max_proximals: if True, remove from `df_counts_iso` every gene that fails in at least one sample.
            Defaults to False.
        threshold_reads: tolerance, in reads, of the comparison. Defaults to 20.

    Returns:
        tuple: the list of failing genes (concatenated over the samples, with repeats) and the count matrix
        (filtered if `remove_non_max_proximals`).
    """
    # Sanity check
    print("--------- Check if proximal isoform is the most expressed ---------")
    if not os.path.exists(os.path.join(path_to_results, "genes_utrs")):
        os.makedirs(os.path.join(path_to_results, "genes_utrs"))
    insane_genes = []
    for rep in reps:
        proximals = df_counts_iso.loc[
            df_counts_iso.groupby("gene").length.idxmin()
        ].drop(["length"], axis=1)
        max_occ = df_counts_iso.loc[df_counts_iso.groupby("gene")[rep].idxmax()]
        sane = proximals.loc[(proximals[rep] + threshold_reads) >= max_occ[rep].values]
        insane = proximals.loc[(proximals[rep] + threshold_reads) < max_occ[rep].values]

        print(
            f"{rep}: Among the {len(proximals)} proximal isoforms, {len(sane)} of them \
            are the ones with the more reads before correction ({np.round(len(sane)*100/len(proximals))})%"
        )

        insane_genes_rep = insane["gene"].values

        common_with_others = list(
            set([gene for gene in insane_genes_rep if gene in insane_genes])
        )
        print(
            f"{len(insane)} genes do not have the proximal isoform as most expressed one. "
        )
        if len(insane) > 0:
            print(
                f"Among them, {len(common_with_others)} are common with \
            the previously processed samples ({(len(common_with_others)*100/len(insane)):.2f}%)"
            )
        else:   
            print("/")
        pd.DataFrame(insane_genes_rep).to_csv(
            os.path.join(
                path_to_results,
                "genes_utrs",
                f"{rep}_genes_prox_not_most_expressed.csv",
            )
        )
        insane_genes += list(insane_genes_rep)

    if remove_non_max_proximals:
        print(f"Starting from {len(proximals)} genes.")
        print(
            "Removing genes for which the proximal isoform is not the most expressed in bulk in all replicates..."
        )
        df_counts_iso = df_counts_iso[~df_counts_iso["gene"].isin(insane_genes)]
        print(f"{len(list(set(insane_genes)))} genes removed.")
        print(
            f"In total, {len(list(set(insane_genes)))} genes removed when \
            removing the genes for which the proximal isoform in not the most expressed in bulk."
        )
    return insane_genes, df_counts_iso


def correct_counts_for_small_proximals(
    df_counts, reps, gtf_file_quantification, focus_nt=300, minimal_required_length=100
):
    """Correct the counts of short isoforms for the length of their counted region.

    Reads are counted on the last `focus_nt` nucleotides of each isoform (focused annotation), so the counted region of
    an isoform is `length` = its length in `gtf_file_quantification` (at most `focus_nt`). Counts are multiplied by
    `focus_nt / length`, which only changes isoforms shorter than `focus_nt`. Isoforms whose counted region is shorter
    than `minimal_required_length` are removed.

    Args:
        df_counts: isoform count matrix, isoform IDs in index (the `transcript_id` of `gtf_file_quantification`).
        reps: sample columns to correct.
        gtf_file_quantification: GTF used for the quantification (focused annotation); gives each isoform's length.
        focus_nt: length of the focus used to build the quantification annotation. Defaults to 300.
        minimal_required_length: minimal isoform length kept. Defaults to 100.

    Returns:
        pd.DataFrame: corrected counts of the `reps` columns, indexed by isoform ID.
    """
    gtf_quantification = read_gtf(gtf_file_quantification)
    gtf_quantification["length"] = (
        gtf_quantification["end"] - gtf_quantification["start"]
    )
    print(df_counts.columns)
    print(df_counts.index)
    print("---------------")
    df_counts = df_counts.merge(
        gtf_quantification[["transcript_id", "length"]],
        left_index=True,
        right_on="transcript_id",
    ).set_index("transcript_id")

    print(df_counts.columns)
    print(df_counts.index)
    # df_counts["length"] = gtf_quantification["length"].values
    df_counts = df_counts[df_counts["length"] > minimal_required_length]
    df_counts_corrected = df_counts[reps].apply(
        lambda rep: rep * focus_nt / df_counts["length"]
    )
    return df_counts_corrected
