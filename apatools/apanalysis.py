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
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
from statannotations.Annotator import Annotator
from statsmodels.stats.multitest import fdrcorrection


def find_significant_shifts(
    df_paired,
    col_prox_treatment1="proximal_occurence_parental",
    col_dist_treatment_1="distal_occurence_parental",
    col_prox_treatment2="proximal_occurence_resistant",
    col_dist_treatment2="distal_occurence_resistant",
    suffix="",
):
    """Compute fisher exact test to find significant association between proximal/distal isoform usage and treatment.
    Returns the original dataframe with added columns for the results of the fisher tests.
    Args:
        :param df_paired: DataFrame containing the bulk or pseudo bulk counts. A row represent a pair,
        and must contain the
        proximal and distal counts for both treatments.
        :param col_prox_treatment1: column name containing raw counts for the proximal isoforms in treatment 1
        :param col_dist_treatment1: column name containing raw counts for the distal isoforms  in treatment 1
        :param col_prox_treatment2: column name containing raw counts for the proximal isoforms in treatment 2
        :param col_dist_treatment2: column name containing raw counts for the distal isoforms  in treatment 2
    """

    # Create a fisher contingency table for each pair of isoform
    df_paired["fisher_table_" + suffix] = df_paired.apply(
        lambda row: np.array(
            [
                [row[col_prox_treatment1], row[col_dist_treatment_1]],
                [row[col_prox_treatment2], row[col_dist_treatment2]],
            ]
        ),
        axis=1,
    )

    # Compute Fisher's exact test on each contingency table
    df_paired["fisher_results_" + suffix] = df_paired["fisher_table_" + suffix].apply(
        lambda tab: fisher_exact(table=tab, alternative="two-sided")
    )

    # Store the p-value in a different column of the dataframe
    df_paired["fisher_pvalue_" + suffix] = df_paired["fisher_results_" + suffix].apply(
        lambda x: x[1]
    )

    df_paired["adjusted_fisher_pvalue_" + suffix] = fdrcorrection(
        df_paired["fisher_pvalue_" + suffix], alpha=0.01
    )[1]

    return df_paired


def signif_group_PUD_from_table(
    row,
    col_adj_fisher_pval="adjusted_fisher_pvalue",
    col_pvalue="fisher_pvalue",
    col_PUD_diff="PUD_diff",
    col_PUD_treatment1=None,
    col_PUD_treatment2=None,
    thresh_PUD=0.15,
    thresh_FDR=0.01,
    thresh_pval=None,
):
    """Use the criterion defined in Andreassi, Luisier et. al. (2021, Cell Reports)
    to define if a shift is significant or not.
    Returns the significance status for a given pair.
    Args:
        :param row: row of a Pandas Dataframe representing a pair
        :param col_adj_fisher_pval: column name containing the adjusted pvalue for the fisher count test
        :param col_PUD_diff: column name containing the PUD difference value between the two treatment
        :param col_PUD_treatment1: Optional. Needed if col_PUD_diff is None.
        column name containing the PUD score in treatment 1
        :param col_PUD_treatment2: Optional. Needed if col_PUD_diff is None.
        column name containing the PUD score in treatment 2
    """
    if col_PUD_treatment1 is not None and col_PUD_treatment2 is not None:
        PUD_diff = row[col_PUD_treatment1] - row[col_PUD_treatment2]
    elif col_PUD_diff is not None:
        PUD_diff = row[col_PUD_diff]
    else:
        raise ValueError(
            "Please enter either the column name for PUD difference, or column names for PUD scores in each treatment."
        )

    if (row[col_adj_fisher_pval] < thresh_FDR) & (PUD_diff <= -thresh_PUD):
        if thresh_pval is not None:
            if row[col_pvalue] < thresh_pval:
                return "Significant proximal shift"
            else:
                return "Non-Significant"
        else:
            return "Significant proximal shift"
    elif (row[col_adj_fisher_pval] < thresh_FDR) & (PUD_diff >= thresh_PUD):
        if thresh_pval is not None:
            if row[col_pvalue] < thresh_pval:
                return "Significant distal shift"
            else:
                return "Non-Significant"
        else:
            return "Significant distal shift"
    else:
        return "Non-Significant"


def signif_group_RUD_from_table(
    row,
    col_adj_fisher_pval="adjusted_fisher_pvalue",
    col_pvalue="fisher_pvalue",
    col_RUD_diff="RUD_diff",
    col_RUD_treatment1=None,
    col_RUD_treatment2=None,
    thresh_RUD=1,
    thresh_FDR=0.01,
    thresh_pval=None,
):
    """Use the criterion defined in Andreassi, Luisier et. al. (2021, Cell Reports)
    to define if a shift is significant or not.
    Returns the significance status for a given pair.
    Args:
        :param row: row of a Pandas Dataframe representing a pair
        :param col_adj_fisher_pval: column name containing the adjusted pvalue for the fisher count test
        :param col_RUD_diff: column name containing the RUD difference value between the two treatment
        :param col_RUD_treatment1: Optional. Needed if col_RUD_diff is None.
        column name containing the PUD score in treatment 1
        :param col_RUD_treatment2: Optional. Needed if col_RUD_diff is None.
        column name containing the RUD score in treatment 2
    """
    if col_RUD_treatment1 is not None and col_RUD_treatment2 is not None:
        RUD_diff = row[col_RUD_treatment1] - row[col_RUD_treatment2]
    elif col_RUD_diff is not None:
        RUD_diff = row[col_RUD_diff]
    else:
        raise ValueError(
            "Please enter either the column name for RUD difference, or column names for PUD scores in each treatment."
        )

    if (row[col_adj_fisher_pval] < thresh_FDR) & (RUD_diff <= -thresh_RUD):
        if thresh_pval is not None:
            if row[col_pvalue] < thresh_pval:
                return "Significant proximal shift"
            else:
                return "Non-Significant"
        else:
            return "Significant proximal shift"
    elif (row[col_adj_fisher_pval] < thresh_FDR) & (RUD_diff >= thresh_RUD):
        if thresh_pval is not None:
            if row[col_pvalue] < thresh_pval:
                return "Significant distal shift"
            else:
                return "Non-Significant"
        else:
            return "Significant distal shift"
    else:
        return "Non-Significant"


def signif_group_DUD_from_table(
    row,
    col_adj_fisher_pval="adjusted_fisher_pvalue",
    col_pvalue="fisher_pvalue",
    col_DUD_diff="DUD_diff",
    col_DUD_treatment1=None,
    col_DUD_treatment2=None,
    thresh_DUD=0.15,
    thresh_FDR=0.01,
    thresh_pval=None,
):
    """Use the criterion defined in Andreassi, Luisier et. al. (2021, Cell Reports)
    to define if a shift is significant or not.
    Returns the significance status for a given pair.
    Args:
        :param row: row of a Pandas Dataframe representing a pair
        :param col_adj_fisher_pval: column name containing the adjusted pvalue for the fisher count test
        :param col_DUD_diff: column name containing the DUD difference value between the two treatment
        :param col_DUD_treatment1: Optional. Needed if col_DUD_diff is None.
        column name containing the DUD score in treatment 1
        :param col_DUD_treatment2: Optional. Needed if col_DUD_diff is None.
        column name containing the DUD score in treatment 2
    """
    if col_DUD_treatment1 is not None and col_DUD_treatment2 is not None:
        DUD_diff = row[col_DUD_treatment1] - row[col_DUD_treatment2]
    elif col_DUD_diff is not None:
        DUD_diff = row[col_DUD_diff]
    else:
        raise ValueError(
            "Please enter either the column name for PUD difference, or column names for DUD scores in each treatment."
        )

    if (row[col_adj_fisher_pval] < thresh_FDR) & (DUD_diff >= thresh_DUD):
        if thresh_pval is not None:
            if row[col_pvalue] < thresh_pval:
                return "Significant proximal shift"
            else:
                return "Non-Significant"
        else:
            return "Significant proximal shift"
    elif (row[col_adj_fisher_pval] < thresh_FDR) & (DUD_diff <= -thresh_DUD):
        if thresh_pval is not None:
            if row[col_pvalue] < thresh_pval:
                return "Significant distal shift"
            else:
                return "Non-Significant"
        else:
            return "Significant distal shift"
    else:
        return "Non-Significant"


def get_PUD_one_pair(Ip, Id):
    """Proximal usage (PUD) of a proximal/distal isoform pair: Ip / (Ip + Id), or 0.5 when both counts are 0.

    Args:
        Ip: proximal isoform expression.
        Id: distal isoform expression.

    Returns:
        float: PUD, between 0 and 1.
    """
    if (Ip == 0) and (Id == 0):
        return 0.5
    else:
        return Ip / (Ip + Id)


def get_RUD_one_pair(Ip, Id):
    """RUD score of a proximal/distal isoform pair: log2(Ip / Id), counts of 0 replaced by 0.01.

    Args:
        Ip: proximal isoform expression.
        Id: distal isoform expression.

    Returns:
        float: log2 ratio of proximal to distal expression.
    """
    if Ip == 0:
        Ip = 0.01
    if Id == 0:
        Id = 0.01
    return np.log2(Ip / Id)


def get_DUD_one_pair(Ip, Id):
    """Distal usage (DUD) of a proximal/distal isoform pair: Id / (Ip + Id), or 0.5 when both counts are 0.

    Args:
        Ip: proximal isoform expression.
        Id: distal isoform expression.

    Returns:
        float: DUD, between 0 and 1 (1 - PUD).
    """
    if (Ip == 0) and (Id == 0):
        return 0.5
    else:
        return Id / (Ip + Id)


def get_categorical_one_pair(Ip, Id):
    """Expression category of a pair: 0 = neither isoform expressed, 1 = proximal only, 2 = both, 3 = distal only.

    Args:
        Ip: proximal isoform expression.
        Id: distal isoform expression.

    Returns:
        int: category, 0 to 3.
    """
    if Ip == 0:
        if Id == 0:
            return 0
        else:
            return 3
    else:
        if Id == 0:
            return 1
        else:
            return 2


def shift_due_to(shift_type, Ip_par, Ip_res, Id_par, Id_res):
    """Tell which isoform drives a significant shift between two conditions (parental -> resistant).

    For a proximal shift, compares the fold increase of the proximal isoform (Ip_res / Ip_par) with the fold decrease of
    the distal one (Id_par / Id_res); for a distal shift, the reverse. Counts of 0 in a denominator are replaced by 1.

    Args:
        shift_type: "proximal" or "distal".
        Ip_par, Ip_res: proximal isoform expression in the first (parental) and second (resistant) condition.
        Id_par, Id_res: distal isoform expression in the first and second condition.

    Returns:
        str: "proximal increase" or "distal decrease" for a proximal shift, "proximal decrease" or "distal increase"
        for a distal shift (None for another `shift_type`).
    """
    if shift_type == "proximal":
        if Ip_par == 0:
            Ip_par = 1
        if Id_res == 0:
            Id_res = 1
        if Ip_res / Ip_par >= Id_par / Id_res:
            return "proximal increase"
        else:
            return "distal decrease"
    elif shift_type == "distal":
        if Id_par == 0:
            Id_par = 1
        if Ip_res == 0:
            Ip_res = 1
        if Ip_par / Ip_res >= Id_res / Id_par:
            return "proximal decrease"
        else:
            return "distal increase"


def get_fisher_stats(
    df_paired, alternative="two-sided", treatments=[], combinations=None
):
    """Get Fisher statistics for each pair of treatments.

    Args:
        df_paired (pd.DataFrame): DataFrame with each row representing a pair.
        Should contain proximal and distal occurences for each treatment.
        treatments (list, optional): list of treatments to analyze. Defaults to [].

    Returns:
        pd.DataFrame: df_paired with the added columns.
    """
    if len(treatments) == 0:
        treatments = [
            col.split("proximal_occurence_")[1]
            for col in df_paired.columns
            if col.startswith("proximal_occurence")
        ]
    # Create a fisher contingency table for each pair of isoform
    if combinations is None:
        combinations = combinations(treatments, 2)

    for treatment1_name, treatment2_name in combinations:
        df_paired[
            "fisher_table_" + treatment1_name + "_VS_" + treatment2_name
        ] = df_paired.apply(
            lambda row: np.array(
                [
                    [
                        row["proximal_occurence_" + treatment1_name],
                        row["distal_occurence_" + treatment1_name],
                    ],
                    [
                        row["proximal_occurence_" + treatment2_name],
                        row["distal_occurence_" + treatment2_name],
                    ],
                ]
            ),
            axis=1,
        )

        # Compute Fisher's exact test on each contingency table
        df_paired[
            "fisher_results_" + treatment1_name + "_VS_" + treatment2_name
        ] = df_paired[
            "fisher_table_" + treatment1_name + "_VS_" + treatment2_name
        ].apply(
            lambda tab: fisher_exact(table=tab, alternative=alternative)
        )

        # Store the p-value in a different column of the dataframe
        df_paired[
            "fisher_pvalue_" + treatment1_name + "_VS_" + treatment2_name
        ] = df_paired[
            "fisher_results_" + treatment1_name + "_VS_" + treatment2_name
        ].apply(
            lambda x: x[1]
        )

        # FDR correction
        df_paired[
            "adjusted_fisher_pvalue_" + treatment1_name + "_VS_" + treatment2_name
        ] = fdrcorrection(
            df_paired["fisher_pvalue_" + treatment1_name + "_VS_" + treatment2_name],
            alpha=0.01,
        )[
            1
        ]

    return df_paired


def get_significances(
    df_paired,
    treatments=[],
    save_score="PUD_RUD",
    path_save=None,
    combinations=None,
    thresh_PUD=0.15,
    thresh_FDR=0.01,
    thresh_RUD=1,
):
    """Add columns to df_paired for PUD and RUD significances. Useful for the scatter plots.

    Args:
        df_paired (_type_): _description_
        treatments (list, optional): _description_. Defaults to [].
    """

    def signif_group_PUD(
        row, treatment1_name, treatment2_name, thresh_PUD=0.15, thresh_FDR=0.01
    ):
        """Significance call of one pair from its adjusted Fisher p-value and PUD difference.

        Returns "Significant proximal shift" if the PUD difference is <= -`thresh_PUD`, "Significant distal shift" if it
        is >= `thresh_PUD` (adjusted p-value < `thresh_FDR` in both cases), "Non-Significant" otherwise.
        """
        if (
            row["adjusted_fisher_pvalue_" + treatment1_name + "_VS_" + treatment2_name]
            < thresh_FDR
        ) & (
            row["PUD_diff_" + treatment1_name + "_VS_" + treatment2_name] <= -thresh_PUD
        ):
            return "Significant proximal shift"
        elif (
            row["adjusted_fisher_pvalue_" + treatment1_name + "_VS_" + treatment2_name]
            < thresh_FDR
        ) & (
            row["PUD_diff_" + treatment1_name + "_VS_" + treatment2_name] >= thresh_PUD
        ):
            return "Significant distal shift"
        else:
            return "Non-Significant"

    def signif_group_RUD(
        row, treatment1_name, treatment2_name, thresh_RUD=1, thresh_FDR=0.01
    ):
        """Significance call of one pair from its adjusted Fisher p-value and RUD difference.

        Returns "Significant proximal shift" if the RUD difference is <= -`thresh_RUD`, "Significant distal shift" if it
        is >= `thresh_RUD` (adjusted p-value < `thresh_FDR` in both cases), "Non-Significant" otherwise.
        """
        if (
            row["adjusted_fisher_pvalue_" + treatment1_name + "_VS_" + treatment2_name]
            < thresh_FDR
        ) & (
            row["RUD_diff_" + treatment1_name + "_VS_" + treatment2_name] <= -thresh_RUD
        ):
            return "Significant proximal shift"
        elif (
            row["adjusted_fisher_pvalue_" + treatment1_name + "_VS_" + treatment2_name]
            < thresh_FDR
        ) & (
            row["RUD_diff_" + treatment1_name + "_VS_" + treatment2_name] >= thresh_RUD
        ):
            return "Significant distal shift"
        else:
            return "Non-Significant"

    def signif_group_PUD_RUD(row, treatment1_name, treatment2_name):
        """Combined call of one pair: its PUD call if the PUD and RUD calls agree, "Non-Significant" otherwise."""
        if (
            row["PUD_Significance_" + treatment1_name + "_VS_" + treatment2_name]
            == row["RUD_Significance_" + treatment1_name + "_VS_" + treatment2_name]
        ):
            return row["PUD_Significance_" + treatment1_name + "_VS_" + treatment2_name]
        else:
            return "Non-Significant"

    if len(treatments) == 0:
        treatments = [
            col.split("proximal_occurence_")[1]
            for col in df_paired.columns
            if col.startswith("proximal_occurence")
        ]

    if combinations is None:
        combinations = combinations(treatments, 2)
    # Create a fisher contingency table for each pair of isoform
    for treatment1_name, treatment2_name in combinations:
        # Compute the difference of PUD between both conditions for each pair of isoforms
        df_paired["PUD_diff_" + treatment1_name + "_VS_" + treatment2_name] = (
            df_paired["PUD_" + treatment1_name] - df_paired["PUD_" + treatment2_name]
        )

        # Compute the significance of the difference between each pair of condition
        df_paired[
            "PUD_Significance_" + treatment1_name + "_VS_" + treatment2_name
        ] = df_paired.apply(
            lambda row: signif_group_PUD(
                row,
                treatment1_name,
                treatment2_name,
                thresh_FDR=thresh_FDR,
                thresh_PUD=thresh_PUD,
            ),
            axis=1,
        )

        df_paired["RUD_diff_" + treatment1_name + "_VS_" + treatment2_name] = (
            df_paired["RUD_" + treatment1_name] - df_paired["RUD_" + treatment2_name]
        )

        df_paired[
            "RUD_Significance_" + treatment1_name + "_VS_" + treatment2_name
        ] = df_paired.apply(
            lambda row: signif_group_RUD(
                row,
                treatment1_name,
                treatment2_name,
                thresh_FDR=thresh_FDR,
                thresh_RUD=thresh_RUD,
            ),
            axis=1,
        )

        df_paired[
            "PUD_RUD_Significance_" + treatment1_name + "_VS_" + treatment2_name
        ] = df_paired.apply(
            lambda row: signif_group_PUD_RUD(row, treatment1_name, treatment2_name),
            axis=1,
        )

        if path_save is not None:
            sig_pairs_to_proximal = df_paired[
                df_paired[
                    save_score
                    + "_Significance_"
                    + treatment1_name
                    + "_VS_"
                    + treatment2_name
                ]
                == "Significant proximal shift"
            ]
            sig_pairs_to_distal = df_paired[
                df_paired[
                    save_score
                    + "_Significance_"
                    + treatment1_name
                    + "_VS_"
                    + treatment2_name
                ]
                == "Significant distal shift"
            ]
            pd.DataFrame(sig_pairs_to_proximal["gene"]).to_csv(
                os.path.join(
                    path_save,
                    "genes_towards_prox_"
                    + save_score
                    + "_"
                    + treatment1_name.replace(" ", "_")
                    + "_VS_"
                    + treatment2_name.replace(" ", "_")
                    + ".csv",
                ),
                index=False,
                header=False,
            )
            pd.DataFrame(sig_pairs_to_distal["gene"]).to_csv(
                os.path.join(
                    path_save,
                    "genes_towards_dist_"
                    + save_score
                    + "_"
                    + treatment1_name.replace(" ", "_")
                    + "_VS_"
                    + treatment2_name.replace(" ", "_")
                    + ".csv",
                ),
                index=False,
                header=False,
            )
            pd.DataFrame(sig_pairs_to_proximal[["proximal_id", "distal_id"]]).to_csv(
                os.path.join(
                    path_save,
                    "pairs_towards_prox_"
                    + save_score
                    + "_"
                    + treatment1_name.replace(" ", "_")
                    + "_VS_"
                    + treatment2_name.replace(" ", "_")
                    + ".csv",
                ),
                index=False,
            )
            pd.DataFrame(sig_pairs_to_distal[["proximal_id", "distal_id"]]).to_csv(
                os.path.join(
                    path_save,
                    "pairs_towards_dist_"
                    + save_score
                    + "_"
                    + treatment1_name.replace(" ", "_")
                    + "_VS_"
                    + treatment2_name.replace(" ", "_")
                    + ".csv",
                ),
                index=False,
            )
    return df_paired


def create_table_of_pairs(
    df_counts_iso, treatments, strategy="mean", replicates=True, path_to_results=None
):
    """Build the table of proximal/distal isoform pairs and their PUD, RUD and DUD scores per sample and per treatment.

    The proximal isoform of a gene is its shortest one; every other isoform of the gene is paired with it as a distal
    isoform. Counts are summed per treatment (`<treatment>_prox`, `<treatment>_dist` columns).

    Args:
        df_counts_iso: isoform count matrix with `gene`, `transcript_id` and `length` columns and one column per sample.
        treatments: {treatment name: list of its sample columns}.
        strategy: how the treatment scores are obtained: "mean" = mean of the per-sample scores (the per-sample
            `PUD_<sample>`, `RUD_<sample>`, `DUD_<sample>` columns are also kept); "sum" = score of the summed counts.
            Defaults to "mean".
        replicates: if True, the samples of a treatment are the columns whose name contains the treatment name;
            otherwise they are taken from `treatments`. Defaults to True.
        path_to_results: if given, the table is also written to `<path_to_results>/table_of_pairs.csv`.

    Returns:
        pd.DataFrame: one row per pair (`gene`, `proximal_id`, `distal_id`, counts and `PUD_` / `RUD_` / `DUD_` scores);
        rows with missing values are dropped.
    """
    proximals = df_counts_iso.loc[df_counts_iso.groupby("gene").length.idxmin()].drop(
        ["length"], axis=1
    )
    distals = df_counts_iso[
        ~(df_counts_iso["transcript_id"].isin(proximals["transcript_id"]))
    ]
    table_of_pairs = proximals.merge(
        distals, left_on="gene", right_on="gene", suffixes=["_prox", "_dist"]
    ).rename(
        columns={"transcript_id_prox": "proximal_id", "transcript_id_dist": "distal_id"}
    )

    print(
        f"We analyze {str(len(table_of_pairs))} tandem pairs, \
            corresponding to {len(df_counts_iso.groupby('gene').sum(numeric_only=True))} genes."
    )

    # columns_scores = [col for col in df_counts_iso.columns if 'rep' in col]
    columns_scores = [
        col
        for col in df_counts_iso.columns
        if col not in ["gene", "transcript_id", "length", "strand", "index"]
    ]

    print(columns_scores)

    for treatment, samples in treatments.items():
        if replicates:
            cols_treatment_prox = [
                treatment_rep + "_prox"
                for treatment_rep in columns_scores
                if treatment in treatment_rep
            ]
            cols_treatment_dist = [
                treatment_rep + "_dist"
                for treatment_rep in columns_scores
                if treatment in treatment_rep
            ]
        else:
            cols_treatment_prox = [sample + "_prox" for sample in samples]
            cols_treatment_dist = [sample + "_dist" for sample in samples]
        table_of_pairs[treatment + "_prox"] = table_of_pairs[cols_treatment_prox].sum(
            axis=1
        )
        table_of_pairs[treatment + "_dist"] = table_of_pairs[cols_treatment_dist].sum(
            axis=1
        )

    if strategy == "mean":
        print("Mean strategy")
        for treatment_rep in columns_scores:
            table_of_pairs["PUD_" + treatment_rep] = table_of_pairs.apply(
                lambda row: get_PUD_one_pair(
                    Ip=row[treatment_rep + "_prox"], Id=row[treatment_rep + "_dist"]
                ),
                axis=1,
            )
            table_of_pairs["RUD_" + treatment_rep] = table_of_pairs.apply(
                lambda row: get_RUD_one_pair(
                    Ip=row[treatment_rep + "_prox"], Id=row[treatment_rep + "_dist"]
                ),
                axis=1,
            )
            table_of_pairs["DUD_" + treatment_rep] = table_of_pairs.apply(
                lambda row: get_DUD_one_pair(
                    Ip=row[treatment_rep + "_prox"], Id=row[treatment_rep + "_dist"]
                ),
                axis=1,
            )

        for treatment, samples in treatments.items():
            if replicates:
                cols_treatment_PUD = [
                    "PUD_" + treatment_rep
                    for treatment_rep in columns_scores
                    if treatment in treatment_rep
                ]
                cols_treatment_RUD = [
                    "RUD_" + treatment_rep
                    for treatment_rep in columns_scores
                    if treatment in treatment_rep
                ]
                cols_treatment_DUD = [
                    "DUD_" + treatment_rep
                    for treatment_rep in columns_scores
                    if treatment in treatment_rep
                ]
            else:
                cols_treatment_PUD = ["PUD_" + sample for sample in samples]
                cols_treatment_RUD = ["RUD_" + sample for sample in samples]
                cols_treatment_DUD = ["DUD_" + sample for sample in samples]
            table_of_pairs["PUD_" + treatment] = table_of_pairs[
                cols_treatment_PUD
            ].mean(axis=1)
            table_of_pairs["RUD_" + treatment] = table_of_pairs[
                cols_treatment_RUD
            ].mean(axis=1)
            table_of_pairs["DUD_" + treatment] = table_of_pairs[
                cols_treatment_DUD
            ].mean(axis=1)

    elif strategy == "sum":
        for treatment in treatments.keys():
            table_of_pairs["PUD_" + treatment] = table_of_pairs.apply(
                lambda row: get_PUD_one_pair(
                    Ip=row[treatment + "_prox"], Id=row[treatment + "_dist"]
                ),
                axis=1,
            )
            print(table_of_pairs.columns)
            table_of_pairs["RUD_" + treatment] = table_of_pairs.apply(
                lambda row: get_RUD_one_pair(
                    Ip=row[treatment + "_prox"], Id=row[treatment + "_dist"]
                ),
                axis=1,
            )
            table_of_pairs["DUD_" + treatment] = table_of_pairs.apply(
                lambda row: get_DUD_one_pair(
                    Ip=row[treatment + "_prox"], Id=row[treatment + "_dist"]
                ),
                axis=1,
            )
    else:
        raise ValueError(
            "Please enter a valid strategy for score computation across replicates: 'mean' or 'sum'."
        )
    table_of_pairs.dropna(inplace=True)
    if path_to_results is not None:
        table_of_pairs.to_csv(
            os.path.join(path_to_results, "table_of_pairs.csv"), index=False
        )
    return table_of_pairs


def plot_scatter_PUD(
    df_paired: pd.DataFrame,
    col_PUD_treatment1: str = "PUD_parental",
    col_PUD_treatment2: str = "PUD_resistant",
    col_PUD_sig: str = "PUD Significance",
    height: int = 8,
    save_path: str = None,
):
    """Scatter plot (with marginal distributions) of the PUD of every pair in two treatments, coloured by significance.

    Dashed lines mark the diagonal +/- 0.15 (the PUD threshold); the number of significant proximal and distal shifts is
    written on the plot. The figure is saved as `<save_path>/<treatment1>_vs_<treatment2>.png`.

    Args:
        df_paired: table of pairs.
        col_PUD_treatment1, col_PUD_treatment2: PUD columns of the two treatments (named `PUD_<treatment>`).
        col_PUD_sig: significance column ("Non-Significant", "Significant proximal shift" or "Significant distal
            shift").
        height: figure height (inches). Defaults to 8.
        save_path: directory where the figure is saved.
    """
    # fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    pl = sns.jointplot(
        data=df_paired,
        x=col_PUD_treatment1,
        y=col_PUD_treatment2,
        s=8,
        height=height,
        color="gray",
        joint_kws={"alpha": 0.5, "sizes": [2, 12, 500]},
    )
    pl.ax_joint.plot(
        np.linspace(0, 1, 10), np.linspace(0, 1, 10), linestyle="dashed", color="black"
    )
    pl.ax_joint.cla()
    sns.scatterplot(
        data=df_paired,
        x=col_PUD_treatment1,
        y=col_PUD_treatment2,
        ax=pl.ax_joint,
        hue=col_PUD_sig,
        size=col_PUD_sig,
        sizes=[5, 50, 50],
        palette={
            "Non-Significant": "black",
            "Significant proximal shift": "limegreen",
            "Significant distal shift": "salmon",
        },
        alpha=0.5,
    )
    pl.ax_joint.plot(
        np.linspace(0, 1, 10),
        np.linspace(0, 1, 10) + 0.15,
        linestyle="dashed",
        color="black",
    )
    pl.ax_joint.plot(
        np.linspace(0, 1, 10),
        np.linspace(0, 1, 10) - 0.15,
        linestyle="dashed",
        color="black",
    )
    pl.ax_joint.set_xlabel(
        f"{col_PUD_treatment1.split('PUD_')[1]} PUD: Ip/(Id+Id)", fontsize=15
    )
    pl.ax_joint.set_ylabel(
        f"{col_PUD_treatment2.split('PUD_')[1]} PUD: Ip/(Id+Id)", fontsize=15
    )
    # ax[1].set_title("Scatter plot of the PUD for each pair of isoform depending on the condition", weight='bold')
    pl.ax_joint.set_xlim(0, 1)
    pl.ax_joint.set_ylim(0, 1)
    pl.ax_joint.legend(fontsize=15, loc="lower center")

    pl.ax_joint.text(
        s=f"n={df_paired.value_counts(col_PUD_sig)['Significant proximal shift']}",
        x=0.2,
        y=0.8,
        color="black",
        fontsize=15,
    )
    pl.ax_joint.text(
        s=f"n={df_paired.value_counts(col_PUD_sig)['Significant distal shift']}",
        x=0.8,
        y=0.35,
        color="black",
        fontsize=15,
    )

    plt.savefig(
        os.path.join(
            save_path,
            f"{col_PUD_treatment1.split('PUD_')[1]}_vs_{col_PUD_treatment2.split('PUD_')[1]}.png",
        ),
        bbox_inches="tight",
    )


def plot_PUD_one_gene(gene_name, table_of_pairs, ax=None):
    """Bar plot of the PUD of the pairs of one gene in the parental and resistant treatments.

    Args:
        gene_name: gene to plot.
        table_of_pairs: table of pairs with `gene`, `PUD_parental` and `PUD_resistant` columns.
        ax: matplotlib axes to draw on (a new figure if None).

    Raises:
        ValueError: if the gene has no pair in the table.
    """
    df = table_of_pairs[table_of_pairs["gene"] == gene_name]
    if len(df) == 0:
        raise ValueError("No existing pair for the required gene.")
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 5))

    sns.barplot(
        df[["PUD_parental", "PUD_resistant"]].rename(
            columns={"PUD_parental": "parental", "PUD_resistant": "resistant"}
        ),
        ax=ax,
    )
    ax.set_ylabel("PUD=Ip/(Ip+Id)")
    ax.set_title(gene_name + " pairs", weight="bold", fontsize=15)
    ax.set_ylim(0, 1)


def plot_distributions_all_treatments_pairs(
    df_paired, treatments_order={}, score="PUD", combinations=None, linewidth=None
):
    """Plot the PUD or RUD or another score distributions for each pair of treatments.

    Args:
        df_paired (pd.DataFrame): pd.DataFrame with each row representing a pair with the proximal/distal ids,
        length, occurences, PUD and RUD scores in each treatment.
        Use the output of the function compute_df_paired.
        treatments_order (dict, optional): dictionnary with treatment names in keys and color in values.
        Used to define the order you want the treatment to appear and the colors associated to them.
        score (str, optional): Which score to plot. Defaults to 'PUD'.
    """

    if len(treatments_order) == 0:
        treatments = [
            col.split("proximal_occurence_")[1]
            for col in df_paired.columns
            if col.startswith("proximal_occurence")
        ]
        for i, treatment in enumerate(treatments):
            treatments_order[treatment] = sns.color_palette("colorblind")[i]

    if combinations is None:
        combinations = combinations(list(treatments_order.keys()), 2)
    nb_comb = len(list(combinations))

    if nb_comb <= 4:
        fig, ax = plt.subplots(1, nb_comb, figsize=(5 * nb_comb, 10), sharey=True)
    elif ((nb_comb / 2) % 1 == 0) and (nb_comb / 2 <= 4):
        fig, ax = plt.subplots(
            2, int(nb_comb / 2), figsize=(5 * int(nb_comb / 2), 20), sharey=True
        )
    else:
        fig, ax = plt.subplots(
            int(np.ceil(nb_comb / 4)),
            4,
            figsize=(20, 10 * int(np.ceil(nb_comb / 4))),
            sharey=True,
        )
    i = 0
    for treatment1_name, treatment2_name in combinations:
        if nb_comb == 1:
            axx = ax
        elif nb_comb <= 4:
            axx = ax[i]
        elif ((nb_comb / 2) % 1 == 0) and (nb_comb / 2 <= 4):
            axx = ax[int(np.floor(i / (nb_comb / 2)))][int(i % (int(nb_comb / 2)))]
        else:
            axx = ax[int(np.floor(i / (int(np.ceil(nb_comb / 4)))))][int(i % 4)]
        s, p = ttest_ind(
            df_paired[score + "_" + treatment1_name],
            df_paired[score + "_" + treatment2_name],
            equal_var=False,
            alternative="two-sided",
        )
        sns.set(font_scale=1.2, palette="colorblind")
        sns.boxplot(
            data=pd.melt(
                df_paired[
                    [score + "_" + treatment1_name, score + "_" + treatment2_name]
                ],
                value_name=score,
                var_name="treatment",
            ).replace(
                {
                    score + "_" + treatment1_name: treatment1_name,
                    score + "_" + treatment2_name: treatment2_name,
                }
            ),
            y=score,
            x="treatment",
            ax=axx,
            palette=[
                treatments_order[treatment1_name],
                treatments_order[treatment2_name],
            ],
            showfliers=False,
            linewidth=linewidth,
        )
        # ax[0].set_title("Boxplots of the PUD distribution among pairs of isoform", weight='bold')
        axx.set_xlabel("Treatment", fontsize=15)
        if score == "PUD":
            axx.set_ylabel("PUD: Ip/(Ip+Id)", fontsize=15)
            axx.text(
                x=0.1,
                y=axx.get_ylim()[1] - 0.02,
                s="p-value = " + "{:.2e}".format(p),
                fontsize=15,
                weight="bold",
            )
        elif score == "RUD":
            axx.set_ylabel("RUD: log2(Ip/Id)", fontsize=15)
            axx.text(
                x=0.1,
                y=axx.get_ylim()[1] - 0.5,
                s="p-value = " + "{:.2e}".format(p),
                fontsize=15,
                weight="bold",
            )
        else:
            axx.set_ylabel(score, fontsize=15)
        axx.set_title(
            treatment1_name.capitalize() + " VS " + treatment2_name.capitalize(),
            weight="bold",
        )
        i += 1
    plt.tight_layout()


def plot_scatter_pairs(
    df_paired, score_plot, score_hue, treatment1_name, treatment2_name
):
    """Scatter plot representing the score score_plot for each pair in both condition treatment1 and treatment2.
     Colored by score_hue.

    Args:
        df_paired (pd.DataFrame): df_paired with each row representing one pair.
        score_plot (str): Name of the score to plot the pairs.
        score_hue (str): Name of the score to color the pairs according to the significance.
        treatment1_name (str): Name of the first treatment.
        treatment2_name (str): Name of the second treatment.
    """
    if (
        score_hue + "_" + "Significance_" + treatment1_name + "_VS_" + treatment2_name
        not in df_paired.columns
    ) or (
        score_plot + "_" + "Significance_" + treatment1_name + "_VS_" + treatment2_name
        not in df_paired.columns
    ):
        df_paired = get_significances(
            df_paired, treatments=[treatment1_name, treatment2_name]
        )

    pl = sns.jointplot(
        data=df_paired,
        x=score_plot + "_" + treatment1_name,
        y=score_plot + "_" + treatment2_name,
        s=8,
        height=8,
        color="gray",
        joint_kws={"alpha": 0.5, "sizes": [2, 12, 500]},
    )
    pl.ax_joint.plot(
        np.linspace(0, 1, 10), np.linspace(0, 1, 10), linestyle="dashed", color="black"
    )
    pl.ax_joint.cla()
    sns.scatterplot(
        data=df_paired,
        x=score_plot + "_" + treatment1_name,
        y=score_plot + "_" + treatment2_name,
        ax=pl.ax_joint,
        hue=score_hue
        + "_"
        + "Significance_"
        + treatment1_name
        + "_VS_"
        + treatment2_name,
        size=score_hue
        + "_"
        + "Significance_"
        + treatment1_name
        + "_VS_"
        + treatment2_name,
        sizes=[5, 50, 50],
        palette={
            "Non-Significant": "black",
            "Significant distal shift": "salmon",
            "Significant proximal shift": "limegreen",
        },
        alpha=0.5,
    )
    pl.ax_joint.plot(
        np.linspace(0, 1, 10),
        np.linspace(0, 1, 10) + 0.15,
        linestyle="dashed",
        color="black",
    )
    pl.ax_joint.plot(
        np.linspace(0, 1, 10),
        np.linspace(0, 1, 10) - 0.15,
        linestyle="dashed",
        color="black",
    )
    nb_prox = len(
        df_paired[
            df_paired[
                score_hue
                + "_"
                + "Significance_"
                + treatment1_name
                + "_VS_"
                + treatment2_name
            ]
            == "Significant proximal shift"
        ]
    )
    nb_dist = len(
        df_paired[
            df_paired[
                score_hue
                + "_"
                + "Significance_"
                + treatment1_name
                + "_VS_"
                + treatment2_name
            ]
            == "Significant distal shift"
        ]
    )
    pl.ax_joint.text(
        s="n = " + str(nb_prox),
        x=0.1,
        y=0.9,
        color="limegreen",
        weight="bold",
        bbox=dict(facecolor="none", edgecolor="black", boxstyle="round"),
    )
    pl.ax_joint.text(
        s="n = " + str(nb_dist),
        x=0.8,
        y=0.05,
        color="salmon",
        weight="bold",
        bbox=dict(facecolor="none", edgecolor="black", boxstyle="round"),
    )

    if score_plot == "PUD":
        pl.ax_joint.set_xlabel(treatment1_name + " PUD: Ip/(Ip+Id)", fontsize=15)
        pl.ax_joint.set_ylabel(treatment2_name + " PUD: Ip/(Ip+Id)", fontsize=15)
    elif score_plot == "RUD":
        pl.ax_joint.set_xlabel(treatment1_name + " RUD: log2(Ip/Id)", fontsize=15)
        pl.ax_joint.set_ylabel(treatment2_name + " RUD: log2(Ip/Id)", fontsize=15)
    else:
        pl.ax_joint.set_xlabel(treatment1_name + score_plot, fontsize=15)
        pl.ax_joint.set_ylabel(treatment2_name + score_plot, fontsize=15)
    # ax[1].set_title("Scatter plot of the PUD for each pair of isoform depending on the condition", weight='bold')
    pl.ax_joint.set_xlim(0, 1)
    pl.ax_joint.set_ylim(0, 1)
    pl.ax_joint.legend(fontsize=15, loc="lower center")
    pl.fig.suptitle(
        treatment1_name.capitalize() + " VS " + treatment2_name.capitalize(),
        weight="bold",
    )
    pl.fig.tight_layout()


def plot_score_boxplot(
    table_of_pairs,
    treatments,
    score="PUD",
    statistic="t-test",
    save_file=None,
):
    """Box plots of a score (PUD, RUD or DUD) of all pairs per treatment, with the p-value of every pair of treatments.

    Args:
        table_of_pairs: table of pairs with one `<score>_<treatment>` column per treatment.
        treatments: {treatment name: samples}; only the names are used.
        score: "PUD", "RUD" or "DUD". Defaults to "PUD".
        statistic: test between two treatments: "t-test" (Welch's, two-sided) or "wilcoxon". Defaults to "t-test".
        save_file: if given, path where the figure is saved.

    Raises:
        NotImplementedError: for another `statistic`.
    """
    # Aggregate distributions
    df = pd.melt(
        table_of_pairs[
            [f"{score}_{treatment_name}" for treatment_name in treatments.keys()]
        ],
        value_name=score,
    )

    treatment_col = []
    for treatment in treatments.keys():
        treatment_col = treatment_col + [treatment] * len(table_of_pairs)
    df["treatment"] = treatment_col

    print(df.head())

    treatment_pairs = list(
        combinations([treatment for treatment in treatments.keys()], 2)
    )

    # compute p-values
    p_values = []
    for p1, p2 in treatment_pairs:
        if statistic == "t-test":
            p_values.append(
                ttest_ind(
                    table_of_pairs[f"{score}_{p1}"],
                    table_of_pairs[f"{score}_{p2}"],
                    equal_var=False,
                    alternative="two-sided",
                )[1]
            )
        elif statistic == "wilcoxon":
            p_values.append(
                wilcoxon(
                    table_of_pairs[f"{score}_{p1}"], table_of_pairs[f"{score}_{p2}"]
                )[1]
            )
        else:
            raise NotImplementedError(
                "Please enter a valid statistic among t-test and wilcoxon."
            )
    formatted_pvalues = [f"P={pvalue:.2e}" for pvalue in p_values]

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(
            1
            * len(
                [f"{score}_{treatment_name}" for treatment_name in treatments.keys()]
            ),
            6,
        ),
    )
    plotting_parameters = {"data": df, "x": "treatment", "y": score, "linewidth": 2.3}

    sns.boxplot(
        **plotting_parameters,
        showfliers=False,
        ax=ax,
        boxprops={"edgecolor": "k"},
        medianprops={"color": "black"},
        whiskerprops={"color": "k"},
        capprops={"color": "k"},
    )

    annotator = Annotator(ax, treatment_pairs, **plotting_parameters)
    annotator.configure(fontsize=12)

    annotator.set_custom_annotations(formatted_pvalues)
    annotator.annotate()
    plt.xticks(rotation=45)

    if save_file is not None:
        fig.savefig(save_file, bbox_inches="tight")


def plot_proportion_shifts(
    n_prox,
    n_dist,
    n_tot_pairs,
    treatment_comp_name="",
    path_to_save=None,
    fs=17,
    extension="pdf",
):
    """Bar plot of the percentage of proximal vs distal significant shifts, with a Fisher test against an even split.

    Args:
        n_prox: number of significant proximal shifts.
        n_dist: number of significant distal shifts.
        n_tot_pairs: number of pairs tested (for the title).
        treatment_comp_name: name of the comparison, used in the file name.
        path_to_save: if given, directory where `pct_shift_type_<treatment_comp_name>.<extension>` is saved.
        fs: font size. Defaults to 17.
        extension: figure file extension. Defaults to "pdf".
    """
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 3.5))
    tmp_dict = {"Proximal": n_prox, "Distal": n_dist}

    pairs = list(combinations(["Proximal", "Distal"], 2))

    p_value = fisher_exact(
        [[n_prox, n_dist], [(n_prox + n_dist) / 2, (n_prox + n_dist) / 2]]
    )[1]
    formatted_pvalues = [f"P={p_value:.2e}"]

    df_bar = pd.DataFrame.from_dict(tmp_dict, orient="index", columns=["# of events"])
    df_bar["% of events"] = df_bar["# of events"] * 100 / df_bar["# of events"].sum()
    df_bar.drop(["# of events"], axis=1, inplace=True)
    df_bar = df_bar.T

    sns.barplot(data=df_bar, ax=ax)
    ax.set_ylim(-10, 100)
    sns.despine()

    annotator = Annotator(ax, pairs, data=df_bar)
    annotator.configure(fontsize=fs)
    annotator.set_custom_annotations(formatted_pvalues)
    annotator.annotate()
    sns.despine(trim=True)

    hatch_patterns = ["..", ""]

    for i, bar in enumerate(ax.patches):
        hatch = hatch_patterns[i % len(hatch_patterns)]
        bar.set_hatch(hatch)
        bar.set_color("white")
        bar.set_edgecolor("black")
        bar.set_linewidth(1.2)

    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.ylabel("Percentage of events [%]", fontsize=fs)
    plt.xlabel("Type of shift", fontsize=fs)
    plt.title(
        f"Distribution of the {n_dist+n_prox} significant \n \
shifts ({np.round(((n_dist+n_prox)/n_tot_pairs*100))}% of total pairs)",
        fontsize=fs,
        y=1.1,
        weight="bold",
    )
    if path_to_save is not None:
        plt.savefig(
            os.path.join(
                path_to_save, f"pct_shift_type_{treatment_comp_name}.{extension}"
            ),
            bbox_inches="tight",
        )


def get_significant_shifts_given_fisher_and_scores(
    table_of_pairs,
    treatments,
    path_to_results,
    scores=["PUD", "RUD"],
    extension="pdf",
    thresh_PUD=0.15,
    thresh_RUD=1,
    thresh_FDR=0.01,
    thresh_pval=None,
    treatment_pairs=None,
):
    """Find the significant 3' UTR shifts between every pair of treatments, and write the shift lists and figures.

    For each pair of treatments: Fisher test of the proximal/distal counts (`find_significant_shifts`), then a PUD, RUD
    and DUD significance call per pair (adjusted p-value < `thresh_FDR`, optional raw p-value < `thresh_pval`, and a
    score difference above the threshold). A pair is a significant proximal (distal) shift if it is one for every score
    of `scores`. Writes, under `path_to_results`:
    - `shifts/pairs_{proximals,distals}_<treatment1>_vs_<treatment2>.csv` (proximal_id, distal_id) and
      `shifts/genes_{proximals,distals}_<treatment1>_vs_<treatment2>.csv` (gene names), proximal / distal = shift
      towards the proximal / distal isoform in treatment 2;
    - `Figures/`: proportion of proximal vs distal shifts and PUD scatter plot of each comparison.

    Args:
        table_of_pairs: output of `create_table_of_pairs`.
        treatments: {treatment name: samples}.
        path_to_results: output directory.
        scores: scores that must all call the shift significant. Defaults to ["PUD", "RUD"].
        extension: figure file extension. Defaults to "pdf".
        thresh_PUD: minimal absolute PUD (and DUD) difference. Defaults to 0.15.
        thresh_RUD: minimal absolute RUD difference. Defaults to 1.
        thresh_FDR: maximal adjusted Fisher p-value. Defaults to 0.01.
        thresh_pval: optional maximal raw Fisher p-value. Defaults to None (not used).
        treatment_pairs: list of (treatment1, treatment2) to compare. Defaults to every pair of treatments.

    Returns:
        pd.DataFrame: the table of pairs with the Fisher test results and the `<score>_significance_<t1>_<t2>` columns.
    """
    if treatment_pairs is None:
        treatment_pairs = list(
            combinations([treatment for treatment in treatments.keys()], 2)
        )

    for treatment1_name, treatment2_name in treatment_pairs:
        find_significant_shifts(
            table_of_pairs,
            col_prox_treatment1=treatment1_name + "_prox",
            col_dist_treatment_1=treatment1_name + "_dist",
            col_prox_treatment2=treatment2_name + "_prox",
            col_dist_treatment2=treatment2_name + "_dist",
            suffix=treatment1_name + "_" + treatment2_name,
        )
        print(f"{treatment1_name} vs. {treatment2_name} is done!")

    for treatment1_name, treatment2_name in treatment_pairs:
        # common_proximals_shift = []
        # common_distals_shift = []
        table_of_pairs[
            "PUD_significance_" + treatment1_name + "_" + treatment2_name
        ] = table_of_pairs.apply(
            lambda row: signif_group_PUD_from_table(
                row,
                col_adj_fisher_pval="adjusted_fisher_pvalue_"
                + treatment1_name
                + "_"
                + treatment2_name,
                col_pvalue="fisher_pvalue_" + treatment1_name + "_" + treatment2_name,
                col_PUD_treatment1="PUD_" + treatment1_name,
                col_PUD_treatment2="PUD_" + treatment2_name,
                thresh_FDR=thresh_FDR,
                thresh_PUD=thresh_PUD,
                thresh_pval=thresh_pval,
            ),
            axis=1,
        )
        table_of_pairs[
            "RUD_significance_" + treatment1_name + "_" + treatment2_name
        ] = table_of_pairs.apply(
            lambda row: signif_group_RUD_from_table(
                row,
                col_adj_fisher_pval="adjusted_fisher_pvalue_"
                + treatment1_name
                + "_"
                + treatment2_name,
                col_pvalue="fisher_pvalue_" + treatment1_name + "_" + treatment2_name,
                col_RUD_treatment1="RUD_" + treatment1_name,
                col_RUD_treatment2="RUD_" + treatment2_name,
                thresh_FDR=thresh_FDR,
                thresh_RUD=thresh_RUD,
                thresh_pval=thresh_pval,
            ),
            axis=1,
        )

        table_of_pairs[
            "DUD_significance_" + treatment1_name + "_" + treatment2_name
        ] = table_of_pairs.apply(
            lambda row: signif_group_DUD_from_table(
                row,
                col_adj_fisher_pval="adjusted_fisher_pvalue_"
                + treatment1_name
                + "_"
                + treatment2_name,
                col_pvalue="fisher_pvalue_" + treatment1_name + "_" + treatment2_name,
                col_DUD_treatment1="DUD_" + treatment1_name,
                col_DUD_treatment2="DUD_" + treatment2_name,
                thresh_FDR=thresh_FDR,
                thresh_DUD=thresh_PUD,
                thresh_pval=thresh_pval,
            ),
            axis=1,
        )
        for i, score in enumerate(scores):
            if i == 0:
                proxs = table_of_pairs[
                    table_of_pairs[
                        f"{score}_significance_{treatment1_name}_{treatment2_name}"
                    ]
                    == "Significant proximal shift"
                ]
                dists = table_of_pairs[
                    table_of_pairs[
                        f"{score}_significance_{treatment1_name}_{treatment2_name}"
                    ]
                    == "Significant distal shift"
                ]
            else:
                proxs = proxs[
                    proxs[f"{score}_significance_{treatment1_name}_{treatment2_name}"]
                    == "Significant proximal shift"
                ]
                dists = dists[
                    dists[f"{score}_significance_{treatment1_name}_{treatment2_name}"]
                    == "Significant distal shift"
                ]
        # common_proximals_shift.append(proxs)
        # common_distals_shift.append(dists)

        nb_sig_shifts = len(proxs) + len(dists)
        nb_prox_shift = len(proxs)

        if not os.path.exists(os.path.join(path_to_results, "Figures")):
            os.makedirs(os.path.join(path_to_results, "Figures"))

    
        try:
            plot_proportion_shifts(
                n_prox=nb_prox_shift,
                n_dist=len(dists),
                n_tot_pairs=len(table_of_pairs),
                treatment_comp_name=f"{treatment1_name}_vs_{treatment2_name}",
                path_to_save=os.path.join(path_to_results, "Figures"),
                extension=extension,
            )
        except Exception as e:
            print(f"Warning: plot_proportion_shifts failed: {e}")
        

        print(
            f"{treatment1_name} vs {treatment2_name}: Among the {nb_sig_shifts} \
                significant pairs: {nb_prox_shift} are towards a proximal use in \
                      {treatment2_name} cells ({np.round((nb_prox_shift * 100 / nb_sig_shifts))}%)"
        )

        # Plot them

        plot_scatter_PUD(
            table_of_pairs,
            col_PUD_treatment1="PUD_" + treatment1_name,
            col_PUD_treatment2="PUD_" + treatment2_name,
            col_PUD_sig="PUD_significance_" + treatment1_name + "_" + treatment2_name,
            height=6,
            save_path=os.path.join(path_to_results, "Figures"),
        )

        genes_proximals = proxs[["gene"]]
        genes_distals = dists[["gene"]]

        pairs_proximals = proxs[["proximal_id", "distal_id"]]
        pairs_distals = dists[["proximal_id", "distal_id"]]

        if not os.path.exists(os.path.join(path_to_results, "shifts")):
            os.makedirs(os.path.join(path_to_results, "shifts"))

        genes_proximals.to_csv(
            os.path.join(
                path_to_results,
                "shifts",
                f"genes_proximals_{treatment1_name}_vs_{treatment2_name}.csv",
            ),
            index=False,
            header=False,
        )
        genes_distals.to_csv(
            os.path.join(
                path_to_results,
                "shifts",
                f"genes_distals_{treatment1_name}_vs_{treatment2_name}.csv",
            ),
            index=False,
            header=False,
        )
        pairs_proximals.to_csv(
            os.path.join(
                path_to_results,
                "shifts",
                f"pairs_proximals_{treatment1_name}_vs_{treatment2_name}.csv",
            ),
            index=False,
        )
        pairs_distals.to_csv(
            os.path.join(
                path_to_results,
                "shifts",
                f"pairs_distals_{treatment1_name}_vs_{treatment2_name}.csv",
            ),
            index=False,
        )

    return table_of_pairs


def summed_log2_counts_pair(
    table_of_pairs: pd.DataFrame, treatments: dict, prox_id: str, dist_id: str
):
    """Plot the log2(sum of counts over all samples of a treatment+1) for a given pair.

    Args:
        table_of_pairs: DataFrame with proximal and distal occurence of each pair for all samples and treatments.
        Should contain the columns
        'proximal_id', 'distal_id', '{treatment}_prox' and '{treatment}_dist' for each treatment.
        treatments: Dictionnary with name of treatments in keys and sample names in values.
        prox_id: Name of the proximal isoform.
        dist_id: Name of the distal isoform.
    """
    fig, ax = plt.subplots(1, len(treatments), figsize=(3 * len(treatments), 4))
    for i, treatment in enumerate(treatments.keys()):
        sns.barplot(
            np.log2(
                table_of_pairs[
                    (table_of_pairs["proximal_id"] == prox_id)
                    & (table_of_pairs["distal_id"] == dist_id)
                ][[treatment + "_prox", treatment + "_dist"]]
                + 1
            ),
            ax=ax[i],
            palette=["limegreen", "salmon"],
        )
        ax[i].set_xticks(ticks=[0, 1], labels=["Ip", "Id"])
        ax[i].set_ylabel("log2(read count+1)")
        ax[i].set_xlabel("Isoform")
        ax[i].set_title(treatment, weight="bold")
    plt.suptitle(
        f"Summed counts over all samples for pair \n {prox_id}/{dist_id}", weight="bold"
    )
    plt.tight_layout()


def plot_pair_scores(table_of_pairs, treatments, prox_id, dist_id):
    """Plot the scores for all treatments for a given pair.

    Args:
        table_of_pairs: DataFrame with proximal and distal occurence of each pair for all samples and treatments.
        Should contain the columns
        'proximal_id', 'distal_id', '{treatment}_prox' and '{treatment}_dist' for each treatment.
        treatments: Dictionnary with name of treatments in keys and sample names in values.
        prox_id: Name of the proximal isoform.
        dist_id: Name of the distal isoform.
    """
    columns = (
        ["PUD_" + treatment for treatment in treatments]
        + ["RUD_" + treatment for treatment in treatments]
        + ["DUD_" + treatment for treatment in treatments]
    )

    df_barplot = pd.melt(
        table_of_pairs[
            (table_of_pairs["proximal_id"] == prox_id)
            & (table_of_pairs["distal_id"] == dist_id)
        ][columns],
        value_name="value",
        var_name="score_treatment",
    )

    df_barplot["score"] = df_barplot["score_treatment"].apply(lambda x: x.split("_")[0])
    df_barplot["treatment"] = df_barplot["score_treatment"].apply(
        lambda x: x.split("_")[1]
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(
        data=df_barplot[df_barplot["score"] == "RUD"],
        x="score",
        y="value",
        hue="treatment",
        ax=ax[0],
    )
    sns.move_legend(ax[0], "upper left", bbox_to_anchor=(1, 1))
    sns.barplot(
        data=df_barplot[df_barplot["score"] != "RUD"],
        x="score",
        y="value",
        hue="treatment",
        ax=ax[1],
    )
    sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
    plt.suptitle(
        f"Mean PUD across samples for pair \n {prox_id}/{dist_id}", weight="bold"
    )

    plt.tight_layout()


def per_sample_log2_counts_pair(table_of_pairs, treatments, prox_id, dist_id):
    """Plot the log2(counts+1) across all samples of a treatment for a given pair.

    Args:
        table_of_pairs: DataFrame with proximal and distal occurence of each pair for all samples and treatments.
        Should contain the columns
        'proximal_id', 'distal_id', '{treatment}_prox' and '{treatment}_dist' for each treatment.
        treatments: Dictionnary with name of treatments in keys and sample names in values.
        prox_id: Name of the proximal isoform.
        dist_id: Name of the distal isoform.
    """
    fig, ax = plt.subplots(1, len(treatments), figsize=(3 * len(treatments), 4))

    for i, (treatment, samples) in enumerate(treatments.items()):
        t = table_of_pairs[
            [col for col in table_of_pairs.columns if col.split("_")[0] in samples]
            + ["proximal_id", "distal_id"]
        ]
        tt = np.log2(
            t[(t["proximal_id"] == prox_id) & (t["distal_id"] == dist_id)][
                [
                    col
                    for col in t.columns
                    if col.startswith("2") and col.endswith("_prox")
                ]
            ]
            + 1
        ).T
        tt.columns = ["count"]
        tt["isoform"] = "Ip"
        tt_d = np.log2(
            t[(t["proximal_id"] == prox_id) & (t["distal_id"] == dist_id)][
                [
                    col
                    for col in t.columns
                    if col.startswith("2") and col.endswith("_dist")
                ]
            ]
            + 1
        ).T
        tt_d.columns = ["count"]
        tt_d["isoform"] = "Id"
        sns.barplot(
            data=pd.concat([tt, tt_d]),
            x="isoform",
            y="count",
            palette=["limegreen", "salmon"],
            ax=ax[i],
        )
        ax[i].set_ylabel("log2(read counts+1)")
        ax[i].set_title(treatment, weight="bold")
    plt.suptitle(
        f"Log2 read counts across all samples for pair \n {prox_id}/{dist_id}",
        weight="bold",
    )
    plt.tight_layout()


# Warning: this function is not generic and has to be updated
def get_PUD_distribution_one_gene(table_of_pairs, treatments, gene_name):
    """Get distribution of PUDs for each pair from a given gene across all samples of the treatments

    Args:
        table_of_pairs (_type_): _description_
        treatments (_type_): _description_
        gene_name (_type_): _description_
    """
    df_PUD = table_of_pairs[
        [
            "PUD_" + sample
            for sample in [x for xs in list(treatments.values()) for x in xs]
            if "PUD_" + sample in table_of_pairs.columns
        ]
    ]
    df_PUD.index = table_of_pairs["proximal_id"] + "/" + table_of_pairs["distal_id"]
    df_PUD.columns = [
        x
        for xs in list(treatments.values())
        for x in xs
        if "PUD_" + x in table_of_pairs.columns
    ]

    df_PUD_gene_name = df_PUD.loc[[idx for idx in df_PUD.index if gene_name in idx]].T
    df_PUD_gene_name["sample_name"] = [
        f"M{idx.split('-M')[1].split('_RNA')[0]}" for idx in df_PUD_gene_name.index
    ]

    df_PUD_gene_name["treatment"] = "unknown"
    for treatment, samples in treatments.items():
        df_PUD_gene_name.loc[
            [sample for sample in samples if sample in df_PUD_gene_name.index],
            "treatment",
        ] = treatment

    fig, ax = plt.subplots(
        1,
        len([idx for idx in df_PUD.index if gene_name in idx]),
        figsize=(8, 5),
        sharey=True,
    )
    for i, pair in enumerate([idx for idx in df_PUD.index if gene_name in idx]):
        sns.boxplot(data=df_PUD_gene_name, x="treatment", y=pair, ax=ax[i])
        ax[i].set_title(pair, weight="bold")
        ax[i].set_ylabel("PUD")
        ax[i].set_xticks(
            ticks=list(range(len(treatments))),
            labels=list(treatments.keys()),
            rotation=35,
        )
    plt.tight_layout()


def get_PUD_values_one_gene(table_of_pairs, treatments, gene_name):
    """Bar plot of the PUD of every pair of one gene in each treatment.

    Args:
        table_of_pairs: table of pairs with one `PUD_<treatment>` column per treatment.
        treatments: {treatment name: samples}; only the names are used (the plot assumes 3 treatments).
        gene_name: gene to plot.
    """
    table_of_pairs_gene = table_of_pairs[table_of_pairs["gene"] == gene_name]
    PUD_gene_pairs = pd.melt(
        table_of_pairs_gene[["PUD_" + treatment for treatment in treatments.keys()]],
        value_name="PUD",
        var_name="treatment",
    )
    PUD_gene_pairs["distal_id"] = list(table_of_pairs_gene["distal_id"].values) * 3
    PUD_gene_pairs["proximal_id"] = list(table_of_pairs_gene["proximal_id"].values) * 3

    PUD_gene_pairs["treatment"] = PUD_gene_pairs["treatment"].apply(
        lambda x: x.split("PUD_")[1]
    )
    PUD_gene_pairs["pair_name"] = (
        PUD_gene_pairs["proximal_id"] + "/" + PUD_gene_pairs["distal_id"]
    )

    ax = sns.barplot(data=PUD_gene_pairs, x="pair_name", y="PUD", hue="treatment")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.ylim(0, 1)
