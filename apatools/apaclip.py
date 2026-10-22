#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileCopyrightText: Copyright © 2026 University of Bern
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@idiap.ch>
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@unibe.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cliptools.processing import get_fisher_table
from scanalysis.plot import get_gene_expression
from scanalysis.plot import get_prop_cells_expressing_gene
from scanalysis.utils import get_statistics_from_dict
from scipy.stats import fisher_exact
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator


def plot_welch_dist_per_region_per_association(
    df_results,
    iso_type="proximal",
    treatments=["parental", "resistant"],
    p_val_sig=0.05,
    nb_nucleotide_upstream=-1000,
    nb_nucleotide_downstream=200,
    region_length=50,
    save_path=None,
    figsize=(20, 5),
    annotate_region=True,
    fs=12,
    path_to_table_of_pairs=None,
    ylim=None,
):
    """Compute and plot the distributions of welch's test results across all RBPs starting
    from the integrated CLIP data.

    Args:
        df_results (dict): dictionnary with RBP names in keys and pd.DataFrame in values. Output of the function
        *load_and_combine_integrated_CLIP*.
        iso (str, optional): Which isoform type you want to analyze. Either proximal or distal.
        Defaults to 'proximal'.
        treatments (list, optional): list of all possible treatments. Defaults to ['parental', 'resistant'].
        p_val_sig (float, optional): p-value threshold for significance. Defaults to 0.05.
        nb_nucleotide_downstream (int, optional): Number of nucleotides downstream of the polyadenylation site (PAS) you
        want to analyze. Defaults to 1000.
        nb_nucleotide_upstream (int, optional): Number of nucleotides upstream of the polyadenylation site (PAS) you
        want to analyze. Defaults to 200.
        region_length (int, optional): Length of the regions you want to analyze. Defaults to 50.
        save_path (_type_, optional): path where you want to save the generated figures. Defaults to None.

    Raises:
        ValueError: In case you don't provide a valid isoform type.

    Returns:
        dict: dictionnary with treatments as keys and pd.DataFrame as values.
        The dataframes contains regions in rows, and
        welch statistics in columns, namely:
                                            -p_val_pos: dictionnary with the RBPs for which a higher PUD score
                                            is observed in the cross-linked population compared
                                            to the non cross-linked population and their associated -log10(p-values)
                                            -p_val_neg: dictionnary with the RBPs for which a lower PUD score
                                            is observed in the cross-linked population compared
                                            to the non cross-linked population and their associated -log10(p-values)
                                            -z_score_pos: dictionnary with the RBPs for which a higher PUD score
                                            is observed in the cross-linked population compared
                                            to the non cross-linked population and their associated z-scores
                                            -z_score_neg: dictionnary with the RBPs for which a lower PUD score
                                            is observed in the cross-linked population compared
                                            to the non cross-linked population and their associated z-scores
                                            -Q1_pos: first quartile of the -log10(pvalues) distribution across
                                            the RBPs for which a higher PUD score is observed in the cross-linked
                                            population compared to the non cross-linked population
                                            -median_pos: median of the -log10(pvalues) distribution across
                                            the RBPs for which a higher PUD score is observed in the cross-linked
                                            population compared to the non cross-linked population
                                            -Q3_pos: third quartile of the -log10(pvalues) distribution across
                                              the RBPs for which a higher PUD score is observed in the cross-linked
                                              population compared to the non cross-linked population
                                            -Q1_neg: first quartile of the -log10(pvalues) distribution across
                                              the RBPs for which a lower PUD score is observed in the cross-linked
                                              population compared to the non cross-linked population
                                            -median_neg: median of the -log10(pvalues) distribution across
                                            the RBPs for which a lower PUD score is observed in the cross-linked
                                              population compared to the non cross-linked population
                                            -Q3_neg: third quartile of the -log10(pvalues) distribution across
                                            the RBPs for which a lower PUD score is observed in the cross-linked
                                            population compared
                                            to the non cross-linked population
                                            -region_range: middle region nucleotide with respect to the
                                            polyadenylation site (PAS)
    """
    if iso_type == "proximal":
        iso = "Ip"
    elif iso_type == "distal":
        iso = "Id"
    else:
        raise ValueError(
            "Please enter a valid isoform type. Either proximal either distal."
        )

    p_values_regions_treatment = {}
    z_scores_regions_treatment = {}
    for treatment in treatments:
        p_values_regions = {}
        z_scores_regions = {}
        nb_regions = int(
            np.floor(
                (-nb_nucleotide_upstream + nb_nucleotide_downstream) / region_length
            )
        )
        for region in range(nb_regions):
            p_values_pos = {}
            p_values_neg = {}
            z_scores_pos = {}
            z_scores_neg = {}
            for RBP_name, df_res in df_results.items():
                # Verify if we are dealing with positive or negative regulators
                if math.isinf(
                    df_res.loc[
                        "Region" + str(region + 1), "-log10(pval)*dPUD_" + treatment
                    ]
                ):
                    if path_to_table_of_pairs is None:
                        raise ValueError(
                            "Please enter a table of pairs in order to retrieve some values."
                        )
                    table_of_pairs = pd.read_csv(path_to_table_of_pairs)
                    if f"{iso_type}_name" in table_of_pairs.columns:
                        suffix = "name"
                    else:
                        suffix = "id"
                    cross_linked = table_of_pairs[
                        table_of_pairs[f"{iso_type}_{suffix}"].isin(
                            df_res.loc["Region" + str(region + 1), "cross_link"]
                        )
                    ]
                    non_cross_linked = table_of_pairs[
                        table_of_pairs[f"{iso_type}_{suffix}"].isin(
                            df_res.loc["Region" + str(region + 1), "no_cross_link"]
                        )
                    ]
                    cross_linked = table_of_pairs[
                        table_of_pairs[f"{iso_type}_{suffix}"].isin(
                            df_res.loc["Region" + str(region + 1), "cross_link"]
                        )
                    ]
                    non_cross_linked = table_of_pairs[
                        table_of_pairs[f"{iso_type}_{suffix}"].isin(
                            df_res.loc["Region" + str(region + 1), "no_cross_link"]
                        )
                    ]
                    PUD_diff = (
                        cross_linked["PUD_" + treatment].mean()
                        - non_cross_linked["PUD_" + treatment].mean()
                    )
                    is_pos = PUD_diff > 0

                else:
                    is_pos = (
                        df_res.loc[
                            "Region" + str(region + 1), "-log10(pval)*dPUD_" + treatment
                        ]
                        >= 0
                    )

                # Retrieve the p-values abd z-scores
                if df_res.loc["Region" + str(region + 1), "welch_" + treatment][1] == 0:
                    p_val = 300
                else:
                    p_val = -np.log10(
                        df_res.loc["Region" + str(region + 1), "welch_" + treatment][1]
                    )
                z_score = df_res.loc["Region" + str(region + 1), "welch_" + treatment][
                    0
                ]

                # Store the positive ones
                if is_pos:
                    p_values_pos[RBP_name] = p_val
                    if p_val >= -np.log10(p_val_sig):
                        z_scores_pos[RBP_name] = z_score
                # Store the negative ones
                else:
                    # print(f"Negative: {RBP_name}, region {region}")
                    p_values_neg[RBP_name] = p_val
                    if p_val >= -np.log10(p_val_sig):
                        z_scores_neg[RBP_name] = z_score

            if iso == "Ip":
                p_values_regions["Region" + str(region + 1)] = {
                    "p_val_pos": p_values_pos,
                    "p_val_neg": p_values_neg,
                }
                z_scores_regions["Region" + str(region + 1)] = {
                    "z_score_pos": z_scores_pos,
                    "z_score_neg": z_scores_neg,
                }
            else:
                p_values_regions["Region" + str(region + 1)] = {
                    "p_val_pos": p_values_neg,
                    "p_val_neg": p_values_pos,
                }
                z_scores_regions["Region" + str(region + 1)] = {
                    "z_score_pos": z_scores_neg,
                    "z_score_neg": z_scores_pos,
                }

        p_values_regions_treatment[treatment] = p_values_regions
        z_scores_regions_treatment[treatment] = z_scores_regions

    # return p_values_regions_treatment

    df_pvalues_regions_treatment = {}
    for treatment in p_values_regions_treatment.keys():
        df_pvalues_regions_treatment[treatment] = pd.concat(
            [
                pd.DataFrame(p_values_regions_treatment[treatment]).T,
                pd.DataFrame(z_scores_regions_treatment[treatment]).T,
            ],
            axis=1,
        )

    for treatment, df_pvalues_regions in df_pvalues_regions_treatment.items():
        df_pvalues_regions["stats_pos"] = df_pvalues_regions["p_val_pos"].apply(
            lambda dic: get_statistics_from_dict(dic)
        )
        df_pvalues_regions["Q1_pos"] = df_pvalues_regions["stats_pos"].apply(
            lambda x: x[0]
        )
        df_pvalues_regions["median_pos"] = df_pvalues_regions["stats_pos"].apply(
            lambda x: x[1]
        )
        df_pvalues_regions["Q3_pos"] = df_pvalues_regions["stats_pos"].apply(
            lambda x: x[2]
        )

        df_pvalues_regions["stats_neg"] = df_pvalues_regions["p_val_neg"].apply(
            lambda dic: get_statistics_from_dict(dic)
        )
        df_pvalues_regions["Q1_neg"] = df_pvalues_regions["stats_neg"].apply(
            lambda x: x[0]
        )
        df_pvalues_regions["median_neg"] = df_pvalues_regions["stats_neg"].apply(
            lambda x: x[1]
        )
        df_pvalues_regions["Q3_neg"] = df_pvalues_regions["stats_neg"].apply(
            lambda x: x[2]
        )

        df_pvalues_regions["region_range"] = [
            int(np.floor(nb_nucleotide_upstream + (region_length / 2)))
            + i * region_length
            for i in range(nb_regions)
        ]
        df_pvalues_regions.drop(["stats_neg", "stats_pos"], axis=1, inplace=True)
        df_pvalues_regions.fillna(0, inplace=True)

    for treatment in treatments:
        fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
        sns.lineplot(
            data=df_pvalues_regions_treatment[treatment],
            x="region_range",
            y="Q1_pos",
            linestyle="dashed",
            color="black",
            ax=ax[0],
        )
        sns.lineplot(
            data=df_pvalues_regions_treatment[treatment],
            x="region_range",
            y="median_pos",
            color="black",
            ax=ax[0],
        )
        sns.lineplot(
            data=df_pvalues_regions_treatment[treatment],
            x="region_range",
            y="Q3_pos",
            linestyle="dashed",
            color="black",
            ax=ax[0],
        )
        sns.lineplot(
            data=df_pvalues_regions_treatment[treatment],
            x="region_range",
            y="Q1_neg",
            linestyle="dashed",
            color="black",
            ax=ax[1],
        )
        sns.lineplot(
            data=df_pvalues_regions_treatment[treatment],
            x="region_range",
            y="median_neg",
            color="black",
            ax=ax[1],
        )
        sns.lineplot(
            data=df_pvalues_regions_treatment[treatment],
            x="region_range",
            y="Q3_neg",
            linestyle="dashed",
            color="black",
            ax=ax[1],
        )
        sns.despine()
        ax[0].set_ylabel("P-value [-log10]", fontsize=fs)
        ax[1].set_ylabel("P-value [-log10]", fontsize=fs)
        ax[0].set_xlabel("distance from " + iso + " 3' end", fontsize=fs)
        ax[1].set_xlabel("distance from " + iso + " 3' end", fontsize=fs)
        ax[0].yaxis.set_tick_params(labelsize=fs)
        ax[1].yaxis.set_tick_params(labelsize=fs)
        ax[0].xaxis.set_tick_params(labelsize=fs)
        ax[1].xaxis.set_tick_params(labelsize=fs)
        if annotate_region:
            if iso == "Ip":
                ax[0].axvspan(
                    xmin=-400, xmax=50, ymin=0.95, ymax=1, alpha=0.5, color="limegreen"
                )
                ax[1].axvspan(-150, 200, ymin=2, ymax=0.95, color="red", alpha=0.5)
                ax[0].text(
                    x=-300,
                    y=ax[0].get_ylim()[1] - 0.5,
                    s="Positive reg. zone",
                    weight="bold",
                    fontsize=12,
                )
                ax[1].text(
                    x=-100,
                    y=ax[1].get_ylim()[1] - 0.5,
                    s="Negative reg. zone",
                    weight="bold",
                    fontsize=12,
                )
            else:
                ax[0].axvspan(
                    xmin=-800, xmax=0, ymin=0.95, ymax=1, alpha=0.5, color="limegreen"
                )
                ax[0].text(
                    x=-500,
                    y=ax[0].get_ylim()[1] - 0.8,
                    s="Positive reg. zone",
                    weight="bold",
                    fontsize=12,
                )

        ax[0].fill_between(
            x=df_pvalues_regions_treatment[treatment]["region_range"],
            y1=df_pvalues_regions_treatment[treatment]["Q1_pos"],
            y2=df_pvalues_regions_treatment[treatment]["Q3_pos"],
            color="lightgray",
        )
        ax[1].fill_between(
            x=df_pvalues_regions_treatment[treatment]["region_range"],
            y1=df_pvalues_regions_treatment[treatment]["Q1_neg"],
            y2=df_pvalues_regions_treatment[treatment]["Q3_neg"],
            color="lightgray",
        )
        ax[0].set_title("Positive regulation; " + iso, weight="bold", fontsize=fs)
        ax[1].set_title("Negative regulation; " + iso, weight="bold", fontsize=fs)
        if ylim is not None:
            plt.ylim(0, ylim)
        fig.suptitle(treatment.capitalize(), fontsize=fs, weight="bold")
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(
                os.path.join(
                    save_path, treatment + "_welch_regulation" + "_" + iso + ".png"
                ),
                bbox_inches="tight",
            )

    return df_pvalues_regions_treatment


def plot_welch_singleRBP_along_proximal_and_distal(
    df_pvalues_regions_treatment_proximals,
    df_pvalues_regions_treatment_distals,
    RBP_name,
    treatment_name,
    path_save=None,
    ax=None,
    ylim=(-22, 22),
):
    """Plot the -log10(p-values) for a given RBP in each region of both the proximal and the distal isoform.

    Args:
        df_pvalues_regions_treatment_proximals (dict): dictionnary with treatments as keys and
        pd.DataFrame as values. The dataframes contains regions in rows, and
        welch statistics for te PROXIMAL isoform in columns, namely:
                                            -p_val_pos: dictionnary with the RBPs for which a higher PUD score
                                            is observed in the cross-linked population compared
                                            to the non cross-linked population and their associated -log10(p-values)
                                            -p_val_neg: dictionnary with the RBPs for which a lower PUD score
                                            is observed in the cross-linked population compared
                                            to the non cross-linked population and their associated -log10(p-values)
                                            -z_score_pos: dictionnary with the RBPs for which a higher PUD score
                                            is observed in the cross-linked population compared
                                            to the non cross-linked population and their associated z-scores
                                            -z_score_neg: dictionnary with the RBPs for which a lower PUD score
                                              is observed in the cross-linked population compared
                                            to the non cross-linked population and their associated z-scores
                                            -Q1_pos: first quartile of the -log10(pvalues) distribution across
                                            the RBPs for which a higher PUD score is observed in the cross-linked
                                             population compared
                                            to the non cross-linked population
                                            -median_pos: median of the -log10(pvalues) distribution across
                                            the RBPs for which a higher PUD score is observed in the cross-linked
                                            population compared
                                            to the non cross-linked population
                                            -Q3_pos: third quartile of the -log10(pvalues) distribution across
                                            the RBPs for which a higher PUD score is observed in the cross-linked
                                            population compared
                                            to the non cross-linked population
                                            -Q1_neg: first quartile of the -log10(pvalues) distribution across
                                            the RBPs for which a lower PUD score is observed in the cross-linked
                                            population compared
                                            to the non cross-linked population
                                            -median_neg: median of the -log10(pvalues) distribution across
                                            the RBPs for which a lower PUD score is observed in the cross-linked
                                            population compared
                                            to the non cross-linked population
                                            -Q3_neg: third quartile of the -log10(pvalues) distribution across
                                            the RBPs for which a lower PUD score is observed in the cross-linked
                                            population compared
                                            to the non cross-linked population
                                            -region_range: middle region nucleotide with respect to the
                                            polyadenylation site (PAS)
        This dictionnary is the output of the function *plot_welch_dist_per_region_per_association*.
        df_pvalues_regions_treatment_distals (_type_): Same than df_pvalues_regions_treatment_proximals but
        with welch statistics computed for the DISTAL isoform.
        RBP_name (_type_): Name of the RBP you want to plot.
        treatment_name (_type_): Name of the treatment condition you want to analyze.
        path_save (_type_, optional): Path to save the resulting figure. Defaults to None.
        ax (_type_, optional): ax to plot. Defaults to None.
    """

    def get_one_RBP_to_plot_per_region(row, RBP):
        """Signed significance of `RBP` in one region: -p_val_neg[RBP] if it is a negative regulator there,
        p_val_pos[RBP] otherwise.
        """
        if RBP in row["p_val_neg"].keys():
            return -row["p_val_neg"][RBP]
        else:
            return row["p_val_pos"][RBP]

    RBP_prox = pd.DataFrame(
        df_pvalues_regions_treatment_proximals[treatment_name].apply(
            lambda row: get_one_RBP_to_plot_per_region(row, RBP_name), axis=1
        )
    ).rename(columns={0: "score"})
    RBP_prox["distance to 3' end"] = df_pvalues_regions_treatment_proximals[
        treatment_name
    ]["region_range"].values
    RBP_dist = pd.DataFrame(
        -df_pvalues_regions_treatment_distals[treatment_name].apply(
            lambda row: get_one_RBP_to_plot_per_region(row, RBP_name), axis=1
        )
    ).rename(columns={0: "score"})
    RBP_dist["distance to 3' end"] = df_pvalues_regions_treatment_distals[
        treatment_name
    ]["region_range"].values

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    sns.lineplot(
        data=RBP_prox,
        x="distance to 3' end",
        y="score",
        linewidth=2,
        color="black",
        ax=ax,
        label="proximal",
    )
    sns.lineplot(
        data=RBP_dist,
        x="distance to 3' end",
        y="score",
        linewidth=2,
        color="gray",
        ax=ax,
        label="distal",
    )

    # ax.set_ylim(ylim[0], ylim[1])
    ax.axhline(y=0, linestyle="dashed", color="black")
    ax.set_title(RBP_name, weight="bold")
    ax.set_ylabel(" ")
    ax.legend(fontsize=15)
    if path_save is not None:
        fig.savefig(os.path.join(path_save, f"{RBP_name}_{treatment_name}.png"))


def plot_fisher_dist(
    df_results,
    proximal_pairs,
    distal_pairs,
    table_of_pairs,
    iso_type="proximal",
    sided="two-sided",
    save_file=None,
    proximal_pairs_all=[],
    distal_pairs_all=[],
    nb_nucleotide_upstream=-1000,
    region_length=50,
    fisher_name="",
):
    """Returns results of the Fisher's exact test testing for the association between RBP cross-link
    event and significant shifts.
       Plot the distribution across all regions.
    Args:
        df_results(dict): Dictionnary containing the results of the integration of CLIP data.
        Output of the function *load_and_combine_integrated_CLIP*, with RBPs in keys and pd.DataFrame in values.
        Should contain the results of the integration for the iso_type *iso_type*.
        proximal_pairs_Ip (pd.DataFrame): pairs exhibiting a significant shift towards proximal isoform usage.
        pd.DataFrame with pairs in rows. Columns should be ['proximal_id', 'distal_id'].
        distal_pairs_Ip (pd.DataFrame): pairs exhibiting a significant shift towards distal isoform usage.
        pd.DataFrame with pairs in rows. Columns should be ['proximal_id', 'distal_id'].
        iso_type (str, optional): Whether to analyze cross-linking events on the proximal isoform or
        on the distal isoform. Default to 'proximal'.
        sided (str, optional): Whether to perform a one-sided or two sided Fisher count test. Defaults to 'two-sided'.
        save_file (_type_, optional): Path to save the figure. Defaults to None.
        proximal_pairs_Ip_all (list, optional): all proximal pairs in case you provided only a
        subset in *proximal_pairs_Ip*. Defaults to [].
        distal_pairs_Ip_all (list, optional): all distal pairs in case you provided only a subset
        in *proximal_pairs_Ip*. Defaults to [].

    Returns:
        dict: df_fishers with two keys 'proximal' and 'distal', and pd.DataFrame as values.
        df_fishers['proximal'] contains results of the Fisher's test testing for the association between cross-link
        event on the *iso_type* isoform, and significant shift towards proximal isoform usage.
        df_fishers['distal'] contains results of the Fisher's test testing for the association between cross-link
        event on the *iso_type* isoform, and significant shift towards distal isoform usage.
        These two pd.DataFrames contains RBP in rows and regions in columns. Values corresponds
        to the -log10(p-values) multiplied by the sign of the association.

    """
    if fisher_name != "":
        fisher_name = f"_{fisher_name}"

    if iso_type == "proximal":
        proximal_pairs_iso = proximal_pairs["proximal_id"].values
        distal_pairs_iso = distal_pairs["proximal_id"].values
        if len(proximal_pairs_all) > 0:
            proximal_pairs_all = proximal_pairs_all["proximal_id"].values
            distal_pairs_all = distal_pairs_all["proximal_id"].values
    elif iso_type == "distal":
        proximal_pairs_iso = proximal_pairs["distal_id"].values
        distal_pairs_iso = distal_pairs["distal_id"].values
        if len(proximal_pairs_all) > 0:
            proximal_pairs_all = proximal_pairs_all["distal_id"].values
            distal_pairs_all = distal_pairs_all["distal_id"].values
    else:
        raise ValueError(
            "Please enter a valid isoform type. Either proximal, either distal."
        )

    for RBP_name, df_result in df_results.items():
        df_result.rename(
            columns={
                f"fisher_results{fisher_name}": "fisher_results_sig_prox_shift",
                f"fisher_table{fisher_name}": "fisher_table_sig_prox_shift",
            },
            inplace=True,
        )
        df_result["fisher_table_sig_prox_shift"] = df_result.apply(
            lambda row: get_fisher_table(
                row["cross_link"],
                proximal_pairs_iso,
                table_of_pairs[
                    ~(table_of_pairs[f"{iso_type}_id"].isin(list(proximal_pairs_iso)))
                ][f"{iso_type}_id"].values,
            ),
            axis=1,
        )
        df_result["fisher_results_sig_prox_shift"] = df_result[
            "fisher_table_sig_prox_shift"
        ].apply(lambda tab: fisher_exact(table=tab, alternative=sided))
        df_result["fisher_table_sig_dist_shift"] = df_result.apply(
            lambda row: get_fisher_table(
                row["cross_link"],
                distal_pairs_iso,
                table_of_pairs[
                    ~(table_of_pairs[f"{iso_type}_id"].isin(list(distal_pairs_iso)))
                ][f"{iso_type}_id"].values,
            ),
            axis=1,
        )
        df_result["fisher_results_sig_dist_shift"] = df_result[
            "fisher_table_sig_dist_shift"
        ].apply(lambda tab: fisher_exact(table=tab, alternative=sided))

    fisher_results_prox = {}
    fisher_results_dist = {}
    for region in range(len(df_result)):
        region_fisher_prox = []
        region_fisher_dist = []
        for RBP_name, df_res in df_results.items():
            a_prox = df_res.loc[
                "Region" + str(region + 1), "fisher_table_sig_prox_shift"
            ][0, 0]
            b_prox = df_res.loc[
                "Region" + str(region + 1), "fisher_table_sig_prox_shift"
            ][0, 1]
            c_prox = df_res.loc[
                "Region" + str(region + 1), "fisher_table_sig_prox_shift"
            ][1, 0]
            d_prox = df_res.loc[
                "Region" + str(region + 1), "fisher_table_sig_prox_shift"
            ][1, 1]
            region_fisher_prox.append(
                -np.log10(
                    df_res.loc[
                        "Region" + str(region + 1), "fisher_results_sig_prox_shift"
                    ][1]
                )
                * np.sign((a_prox / (a_prox + b_prox)) - (c_prox / (c_prox + d_prox)))
            )
            a_dist = df_res.loc[
                "Region" + str(region + 1), "fisher_table_sig_dist_shift"
            ][0, 0]
            b_dist = df_res.loc[
                "Region" + str(region + 1), "fisher_table_sig_dist_shift"
            ][0, 1]
            c_dist = df_res.loc[
                "Region" + str(region + 1), "fisher_table_sig_dist_shift"
            ][1, 0]
            d_dist = df_res.loc[
                "Region" + str(region + 1), "fisher_table_sig_dist_shift"
            ][1, 1]
            region_fisher_dist.append(
                -np.log10(
                    df_res.loc[
                        "Region" + str(region + 1), "fisher_results_sig_dist_shift"
                    ][1]
                )
                * np.sign((a_dist / (a_dist + b_dist)) - (c_dist / (c_dist + d_dist)))
            )
        fisher_results_prox[
            "["
            + str(nb_nucleotide_upstream + region_length * region)
            + ":"
            + str(nb_nucleotide_upstream + region_length * (region + 1))
            + "]"
        ] = region_fisher_prox
        fisher_results_dist[
            "["
            + str(nb_nucleotide_upstream + region_length * region)
            + ":"
            + str(nb_nucleotide_upstream + region_length * (region + 1))
            + "]"
        ] = region_fisher_dist

    df_fishers = {
        "proximal": pd.DataFrame(fisher_results_prox, index=df_results.keys()),
        "distal": pd.DataFrame(fisher_results_dist, index=df_results.keys()),
    }

    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    for i, type_shift in enumerate(["proximal", "distal"]):
        df_boxplot = pd.melt(
            df_fishers[type_shift][
                [
                    "["
                    + str(nb_nucleotide_upstream + region_length * region)
                    + ":"
                    + str(nb_nucleotide_upstream + region_length * (region + 1))
                    + "]"
                    for region in range(len(df_result))
                ]
            ],
            var_name="distance from 3' end",
            value_name="-log10(P-value)",
        )
        sns.boxplot(
            data=df_boxplot,
            x="distance from 3' end",
            y="-log10(P-value)",
            ax=ax[i],
            showfliers=False,
            color="plum",
        )
        ax[i].set_xticks(
            ticks=np.arange(df_fishers[type_shift].shape[1]),
            labels=df_fishers[type_shift].columns,
            rotation=60,
        )
        if type_shift == "proximal":
            n = len(proximal_pairs)
        else:
            n = len(distal_pairs)
        ax[i].set_title(
            "Promoter-"
            + type_shift
            + " shift in resistant cells"
            + " (n="
            + str(n)
            + ")",
            weight="bold",
            fontsize=17,
        )
        ax[i].set_xlabel("distance from 3' end", fontsize=15)
        ax[i].set_ylabel("-log10(P-value)", fontsize=15)

    if save_file is not None:
        fig.savefig(save_file, bbox_inches="tight")

    return df_fishers


def top_regulators_in_region(
    df_fishers,
    lower_nt_region,
    upper_nt_region,
    shift_type="proximal",
    length_region=50,
    type_regulators="positive",
    simple_names=False,
    nb_top=10,
):
    """_summary_

    Args:
        df_fishers (dict): df_fishers with two keys 'proximal' and 'distal', and pd.DataFrame as values.
        df_fishers['proximal'] contains results of the Fisher's test testing for the association between cross-link
        event on the *iso_type* isoform, and significant shift towards proximal isoform usage. Output of the
        function *plot_fisher_dist*.
        lower_nt_region (_type_): number of nucleotides downstream of the PAS to search for regulators
        upper_nt_region (_type_): number of nucleotides upstream of the PAS to search for regulators
        shift_type (str, optional): Whether you want to extract the regulators responsible for proximal
        shifts or distal shifts. Defaults to 'proximal'.
        length_region (int, optional): length of the regions that are anylzed. Defaults to 50.
        type_regulators (str, optional): Whether you want to extract the positive or negative regulators.
        Defaults to 'positive'.

    Returns:
        pd.DataFrame: The columns are:
                                        - RBP: the RBP name
                                        - log10(P-value) the result from the Fisher test
        The rows are sorted from the stronger regulator to the weakest.
    """
    if type_regulators == "positive":
        ascending = False
    else:
        ascending = True

    if shift_type == "proximal":
        df_fishers = df_fishers["proximal"]
    else:
        df_fishers = df_fishers["distal"]

    top_10_dict = {}
    nb_regions = int(np.floor((lower_nt_region + upper_nt_region) / length_region))
    for i in range(nb_regions):
        low_b = -lower_nt_region + i * length_region
        up_b = -lower_nt_region + (i + 1) * length_region
        top_10 = {
            k + "_region_[" + str(low_b) + ":" + str(up_b) + "]": v
            for k, v in df_fishers["[" + str(low_b) + ":" + str(up_b) + "]"]
            .sort_values(ascending=ascending)
            .head(nb_top)
            .to_dict()
            .items()
        }
        top_10_dict = top_10_dict | top_10

    df_top = (
        pd.DataFrame.from_dict(top_10_dict, orient="index", columns=["-log10(P-value)"])
        .reset_index()
        .rename(columns={"index": "RBP"})
    )
    # df_top['RBP'] = df_top['RBP'].apply(lambda rbp: rbp.split('_')[0])
    df_top.sort_values("-log10(P-value)", ascending=ascending, inplace=True)
    if simple_names:
        df_top["RBP"] = df_top["RBP"].apply(lambda x: simplify_clip_names(x))
    if ascending:
        df_top["-log10(P-value)"] = -df_top["-log10(P-value)"]
    return df_top


def simplify_clip_names(clip_name):
    """Short RBP name of a CLIP dataset: the part after "all-" for iCLIP names ("all-<RBP>-..."), the part before the
    first "_" otherwise (eCLIP names "<RBP>_<replicate>_<cell line>").
    """
    if "all" in clip_name:
        return clip_name.split("all-")[1].split("-")[0]
    else:
        return clip_name.split("_")[0]


def plot_individual_fisher_rbp(
    df_fishers: pd.DataFrame,
    RBP_name: str,
    pos_reg_zone: list = ["[-400:-350]", "[0:50]"],
    neg_reg_zone: list = ["[-150:-100]", "[150:200]"],
    ylim: tuple = (-5, 5),
    text_pos: str = "[-350:-300]",
    text_neg: str = "[-150:-100]",
    iso_type: str = "proximal",
    shift_type: str = "proximal",
    all_replicates: bool = False,
    ax=None,
    plot_zones: bool = False,
    path_save="./Figures",
):
    """Plot individual profile of Fisher results for a given RBP, across all regions.

    Args:
        df_fishers (pd.DataFrame): pd.DataFrame with RBPs in rows and regions in columns.
        Values are the fisher's results,
        testing for the association between cross-link event of the RBP on the *iso_type* isoform
        and significant shifts towards *shift_type* isoform usage.
        They represent the -log10(p-values) multiplied by the sign of the association.
        clips (list): list of clips you want to analyze (usually, replicated for a single RBP.)
        pos_reg_zone (list, optional): Name of the lower and upper regions defining the positive regulation zone.
        Only for plotting purposes. Defaults to ['[-400:-350]', '[0:50]'].
        neg_reg_zone (list, optional): Name of the lower and upper regions defining the negative regulation zone.
        Only for plotting purposes. Defaults to ['[-150:-100]', '[150:200]'].
        ylim (tuple, optional): ylim. Defaults to (-5, 5).
        text_pos (str, optional): x position to write "positive regulation zone". Defaults to '[-350:-300]'.
        text_neg (str, optional): x position to write "negative regulation zone". Defaults to '[-150:-100]'.
        iso_type (str, optional): type of isoform on which we study the cross-link events. Defaults to 'proximal'.
        shift_type (str, optional): Type of shift you want to analyze. Defaults to 'proximal'.
        all_replicates (bool, optional): If multiple replicates, whether to plot them independently
        of in distribution. Defaults to False.
        plot_zones: whether to draw the positive and negative regualtion zones.
    """

    if shift_type == "proximal":
        df_fishers = df_fishers["proximal"]
    elif shift_type == "distal":
        df_fishers = df_fishers["distal"]
    else:
        raise ValueError(
            "Please enter a valid type of shift. Either proximal, either distal."
        )

    if all_replicates or (len(RBP_name.split("_")) == 1):
        RBP_names = [rbp for rbp in df_fishers.index if rbp.startswith(RBP_name)]
    else:
        RBP_names = [RBP_name]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    if len(RBP_names) >= 2:
        replicates_proximal = df_fishers.loc[RBP_names].describe().T

        ax.fill_between(
            x=replicates_proximal.index,
            y1=replicates_proximal["min"],
            y2=replicates_proximal["max"],
            color="lightgray",
        )
        sns.lineplot(data=replicates_proximal["max"], label="max", ax=ax, color="black")
        sns.lineplot(
            data=replicates_proximal["50%"],
            label="median",
            ax=ax,
            color="black",
            linestyle="dashed",
        )
        sns.lineplot(data=replicates_proximal["min"], label="min", ax=ax, color="black")
        RBP_name = RBP_names[0]
    else:
        for RBP_name in RBP_names:
            sns.lineplot(
                df_fishers.loc[RBP_name],
                label=RBP_name,
                linewidth=2,
                color="black",
                ax=ax,
            )

    ax.axhline(
        y=-np.log10(0.05),
        color="firebrick",
        linestyle="dashed",
        label="signif. \n level",
    )
    ax.axhline(y=np.log10(0.05), color="firebrick", linestyle="dashed")
    ax.legend(loc="lower right")
    plt.xticks(rotation=90)

    if plot_zones:
        ax.axvspan(
            xmin=pos_reg_zone[0],
            xmax=pos_reg_zone[1],
            ymin=0.95,
            ymax=1,
            alpha=0.5,
            color="limegreen",
        )
        ax.text(x=text_pos, y=4.6, s="Positive reg. zone", weight="bold", fontsize=12)

        if iso_type == "proximal":
            ax.axvspan(
                xmin=neg_reg_zone[0],
                xmax=neg_reg_zone[1],
                ymin=0.9,
                ymax=0.95,
                alpha=0.5,
                color="red",
            )
            ax.text(
                x=text_neg, y=4.1, s="Negative reg. zone", weight="bold", fontsize=12
            )
    ax.set_ylabel(r"$-log_{10}(P-value)*$" + r"$\delta_{association}$")
    ax.set_title(
        RBP_names[0].split("_")[0]
        + " binding association with "
        + shift_type
        + " shift",
        weight="bold",
        fontsize=17,
    )
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.get_legend().remove()

    if iso_type == "proximal":
        ax.set_xlabel("distance from Ip 3' end")
    else:
        ax.set_xlabel("distance from Id 3' end")
    ax.set_xticks(
        ticks=np.arange(df_fishers.shape[1]), labels=df_fishers.columns, rotation=90
    )
    ax.set_ylim(ylim[0], ylim[1])
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    plt.savefig(
        os.path.join(
            path_save,
            f"{RBP_name.split('_')[0]}_{shift_type}_shift_along_{iso_type}_isoform.png",
        ),
        bbox_inches="tight",
    )


# TODO: document and review this function
def get_master_regulators_around_isoform(
    df_pvalues_regions_iso, positive_reg_regions=None, negative_reg_regions=None
):
    """Find the RBPs that are among the top regulators both upstream and downstream of the PAS, per treatment.

    In each region (row of the table), the top regulators are the 15 RBPs with the largest absolute Welch z-score, split
    into positive (z_score_pos) and negative (z_score_neg) regulators. They are pooled over the regions of
    `positive_reg_regions` (positive regulators) and of `negative_reg_regions` (negative regulators).

    Args:
        df_pvalues_regions_iso: {treatment: DataFrame with one row per region and the columns `region_range`,
            `z_score_pos` and `z_score_neg` ({RBP: z-score})}. Columns `positive_regulators` and `negative_regulators`
            are added to each DataFrame.
        positive_reg_regions: (start, end) range of `region_range` where positive regulators are collected.
        negative_reg_regions: (start, end) range of `region_range` where negative regulators are collected.

    Returns:
        dict: {treatment: {"pos", "neg", "both", "pos_only", "neg_only": lists of RBP names}}; "both" = positive
        regulators of the positive regions that are also negative regulators of the negative regions.
    """
    def get_top_regulators_of_the_region(z_score_pos, z_score_neg, nb_top=15):
        """Top `nb_top` RBPs of one region by absolute z-score, split into positive and negative regulators (names cut
        at the first "_").
        """
        dict_to_sort = {
            RBP: abs(z_score) for RBP, z_score in (z_score_pos | z_score_neg).items()
        }
        positive_regulators = list(z_score_pos.keys())
        negative_regulators = list(z_score_neg.keys())
        top_regs = [
            RBP_name[0]
            for RBP_name in sorted(dict_to_sort.items(), key=lambda x: x[1])[-nb_top:]
        ]
        top_regulators = {
            "pos": list(
                set(
                    [
                        RBP.split("_")[0]
                        for RBP in top_regs
                        if RBP in positive_regulators
                    ]
                )
            ),
            "neg": list(
                set(
                    [
                        RBP.split("_")[0]
                        for RBP in top_regs
                        if RBP in negative_regulators
                    ]
                )
            ),
        }
        return top_regulators

    master_regulators = {}
    for treatment, df in df_pvalues_regions_iso.items():
        df["positive_regulators"] = df.apply(
            lambda row: get_top_regulators_of_the_region(
                row["z_score_pos"], row["z_score_neg"]
            )["pos"],
            axis=1,
        )
        df["negative_regulators"] = df.apply(
            lambda row: get_top_regulators_of_the_region(
                row["z_score_pos"], row["z_score_neg"]
            )["neg"],
            axis=1,
        )

        if positive_reg_regions is None:
            positive_region_positive_regulators = []
        else:
            positive_region_positive_regulators = list(
                set(
                    df_pvalues_regions_iso[treatment][
                        (
                            df_pvalues_regions_iso[treatment]["region_range"]
                            >= positive_reg_regions[0]
                        )
                        & (
                            df_pvalues_regions_iso[treatment]["region_range"]
                            <= positive_reg_regions[1]
                        )
                    ]["positive_regulators"].sum()
                )
            )

        if negative_reg_regions is None:
            negative_region_negative_regulators = []
        else:
            negative_region_negative_regulators = list(
                set(
                    df_pvalues_regions_iso[treatment][
                        (
                            df_pvalues_regions_iso[treatment]["region_range"]
                            >= negative_reg_regions[0]
                        )
                        & (
                            df_pvalues_regions_iso[treatment]["region_range"]
                            <= negative_reg_regions[1]
                        )
                    ]["negative_regulators"].sum()
                )
            )

        regulators_both_regions = list(
            set(positive_region_positive_regulators)
            & set(negative_region_negative_regulators)
        )

        master_regulators[treatment] = {
            "pos": positive_region_positive_regulators,
            "neg": negative_region_negative_regulators,
            "both": regulators_both_regions,
            "pos_only": [
                RBP
                for RBP in positive_region_positive_regulators
                if RBP not in regulators_both_regions
            ],
            "neg_only": [
                RBP
                for RBP in negative_region_negative_regulators
                if RBP not in regulators_both_regions
            ],
        }
    return master_regulators


def plot_RBP_welch_scores_all_region(
    RBP_name,
    df_results,
    nt_before=-1000,
    region_length=50,
    nt_after=200,
    treatments=["parental", "resistant"],
    all_replicates=False,
    ax=None,
    fontsize=12,
):
    """Plot the obtained PUD Welch's scores for a given RBP along all defined regions.

    Args:
        RBP_name (str): Name of the RBP.
        df_results (dict): Dictionnary with RBPs in keys and DataFrame in values.
        The dataframes should contain informations about the binding of the RBPs across regions along
        the 3' UTR isoforms.
        nt_before (int, optional): Number of nucleotides upsteam of the PAS to look at. Defaults to -1000.
        region_length (int, optional): Length of the regions to look at. Defaults to 50.
        nt_after (int, optional): Number of nucleotides downstream of the PAS to look at. Defaults to 200.
        treatments (list, optional): list of treatments
        all_replicates (bool, optional): Whether to include all replicates in the statistics. Defaults to False.
        ax (matplotlib.axes.Axes, optional): ax on which to plot. Defaults to None.
        fontsize (int, optional): Fontsiwe for the plot. Defaults to 12.
    """

    if (all_replicates is True) or (len(RBP_name.split("_")) == 1):
        RBP_names = [rbp for rbp in df_results.keys() if rbp.startswith(RBP_name)]
    else:
        RBP_names = [RBP_name]

    nb_regions = len(df_results[RBP_names[0]])
    regions = [
        "["
        + str(nt_before + region * region_length)
        + ":"
        + str(nt_before + (region + 1) * region_length)
        + "]"
        for region in range(nb_regions)
    ]
    df_results[RBP_names[0]]["distance to PAS"] = regions
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    if len(RBP_names) == 1:
        for treatment in treatments:
            sns.lineplot(
                data=df_results[RBP_names[0]],
                x="distance to PAS",
                y="-log10(pval)*dPUD_" + treatment,
                ax=ax,
                label=treatment,
            )
        # sns.lineplot(data=df_results[RBP_names[0]], x='distance to PAS',
        #  y='-log10(pval)*dPUD_'+treatment2_name, ax=ax, label=treatment2_name)

        ax.set_title("Welch scores - " + RBP_names[0], weight="bold", fontsize=fontsize)
    else:
        for treatment in treatments:
            replicates = []
            for RBP_name in RBP_names:
                replicate = pd.DataFrame(
                    df_results[RBP_name]["-log10(pval)*dPUD_" + treatment]
                )
                replicate["RBP_replicate"] = RBP_name
                replicate["distance to PAS"] = regions
                replicates.append(replicate)
            df_replicates = pd.concat(replicates)
            sns.lineplot(
                data=df_replicates.reset_index(),
                x="distance to PAS",
                y="-log10(pval)*dPUD_" + treatment,
                ax=ax,
                label=treatment,
            )
        ax.set_title(
            "Welch scores - " + RBP_name.split("_")[0], weight="bold", fontsize=fontsize
        )
    ax.axhline(y=0, linestyle="dashed", color="black")
    ax.set_xticks(
        ticks=np.arange(nb_regions),
        labels=df_results[RBP_names[0]]["distance to PAS"],
        rotation=90,
        fontsize=fontsize,
    )
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.set_ylabel(r"$-log_{10}(P-value)*$" + r"$\delta_{PUD}$", fontsize=fontsize)
    ax.set_xlabel("distance to PAS", fontsize=fontsize)
    if ax.get_ylim()[0] < 0 and ax.get_ylim()[1] > 0:
        ylim = max(-ax.get_ylim()[0], ax.get_ylim()[1])
        ax.set_ylim(-ylim, ylim)


def plot_fisher_value_one_region(
    df_fishers, RBP_name, region, shift_type="proximal", ax=None
):
    """

    Plot the result of the Fisher count test for a given region, testing for the association
    between crosslink events in this region and significant APA shifts.


    Args:
        df_fishers (_type_): _description_
        RBP_name (_type_): _description_
        region (_type_): _description_
        shift_type (str, optional): _description_. Defaults to 'proximal'.
        ax (_type_, optional): _description_. Defaults to None.
    """
    if len(RBP_name.split("_")) == 1:
        RBP_names = [
            rbp for rbp in df_fishers[shift_type].index if rbp.startswith(RBP_name)
        ]
    else:
        RBP_names = [RBP_name]

    df = pd.DataFrame(df_fishers[shift_type].loc[RBP_names, region])
    df["shift_type"] = [shift_type] * len(RBP_names)
    df = df.reset_index().rename(columns={"index": "RBP replicate"})
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(1 * len(RBP_names), 4))

    df.sort_values("RBP replicate", inplace=True)
    sns.barplot(data=df, y=region, x="RBP replicate", color="dimgray", ax=ax)
    ax.set_title(
        "Fisher - " + RBP_names[0].split("_")[0] + " - region " + region,
        weight="bold",
        fontsize=15,
    )
    ax.set_xticks(
        ticks=np.arange(len(RBP_names)), labels=df["RBP replicate"].values, rotation=45
    )
    ax.set_ylabel(r"$-log_{10}(P-value)*$" + r"$\delta_{association}$")
    ax.axhline(
        y=-np.log10(0.05),
        linestyle="dashed",
        color="firebrick",
        label="signif. \n level",
    )
    ax.axhline(y=np.log10(0.05), linestyle="dashed", color="firebrick")
    ax.legend()
    if ax.get_ylim()[0] < 0 and ax.get_ylim()[1] > 0:
        ylim = max(-ax.get_ylim()[0], ax.get_ylim()[1])
        ax.set_ylim(-ylim, ylim)


def plot_RBP_welch_scores_one_region(
    RBP_name,
    df_results,
    region_name="Region1",
    nt_before=-1000,
    region_length=50,
    treatments=["parental", "resistant"],
    all_replicates=False,
    ax=None,
    fontsize=12,
):
    """Plot the obtained PUD Welch's scores for a given RBP in a given region. This is displayed as a barplot.

    Args:
        RBP_name (str): Name of the RBP.
        df_results (dict): Dictionnary with RBPs in keys and DataFrame in values.
        region_name (str, optional): Name of the region. Defaults to 'Region1'.
        nt_before (int, optional): Number of nucleotides upsteam of the PAS to look at. Defaults to -1000.
        region_length (int, optional): Length of the regions to look at. Defaults to 50.
        treatment1_name (str, optional): _description_. Defaults to 'parental'.
        treatment2_name (str, optional): _description_. Defaults to 'resistant'.
        all_replicates (bool, optional): _description_. Defaults to False.
        ax (_type_, optional): _description_. Defaults to None.
        fontsize (int, optional): _description_. Defaults to 12.
    """

    if all_replicates or (len(RBP_name.split("_")) == 1):
        RBP_names = [rbp for rbp in df_results.keys() if rbp.startswith(RBP_name)]
    else:
        RBP_names = [RBP_name]
    if region_name.startswith("["):
        low, up = region_name.split(":")
        low = int(low.split("[")[1])
        up = int(up.split("]")[0])
        region_numerical = region_name
        region_name = "Region" + str(
            int(np.floor((low - nt_before) / region_length)) + 1
        )
    else:
        region_nb = int(region_name.split("Region")[1])
        region_numerical = (
            "["
            + str(nt_before + (region_nb - 1) * region_length)
            + ":"
            + str(nt_before + (region_nb) * region_length)
            + "]"
        )

    # 1. Plot the Welch's scores
    welch_dict = {}
    for treatment in treatments:
        scores_welch_treatment = [
            df_results[RBP_name].loc[region_name, "-log10(pval)*dPUD_" + treatment]
            for RBP_name in RBP_names
        ]
        welch_dict[treatment] = scores_welch_treatment
    # scores_welch_treatment1 = [df_results[RBP_name].loc[region_name,
    # '-log10(pval)*dPUD_'+treatment1_name] for RBP_name in RBP_names]
    # scores_welch_treatment2 = [df_results[RBP_name].loc[region_name,
    # '-log10(pval)*dPUD_'+treatment2_name] for RBP_name in RBP_names]
    # df_welch_score = pd.melt(pd.DataFrame.from_dict({treatment1_name: scores_welch_treatment1,
    # treatment2_name: scores_welch_treatment2}, orient='columns'), var_name='treatment', value_name="welch score")
    df_welch_score = pd.melt(
        pd.DataFrame.from_dict(welch_dict, orient="columns"),
        var_name="treatment",
        value_name="welch score",
    )

    df_welch_score["RBP_replicate"] = RBP_names * len(treatments)
    df_welch_score.sort_values("RBP_replicate", inplace=True)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(1.5 * len(RBP_names), 4))
    # sns.barplot(data=df_welch_score, x='treatment', y='welch score', ax=ax)
    # if len(RBP_names) > 1:
    #     sns.scatterplot(data=df_welch_score, x='treatment', y='welch score',
    # style='RBP_replicate', s=100, palette=['black']*len(RBP_names), hue='RBP_replicate', ax=ax)
    #     sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    #     ax.set_title(''+str(RBP_names[0].split('_')[0])+" - Region "+region_numerical,
    # weight='bold', fontsize=15)
    # else:
    #     ax.set_title(""+str(RBP_name)+" - Region "+region_numerical, weight='bold', fontsize=15)
    sns.barplot(
        data=df_welch_score, y="welch score", x="RBP_replicate", hue="treatment", ax=ax
    )
    ax.set_title(
        "" + str(RBP_name) + " - Region " + region_numerical,
        weight="bold",
        fontsize=fontsize,
    )
    ax.set_xticks(
        ticks=np.arange(len(RBP_names)),
        labels=df_welch_score[df_welch_score["treatment"] == treatment][
            "RBP_replicate"
        ].values,
        rotation=45,
        fontsize=fontsize,
    )
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.legend()
    if ax.get_ylim()[0] < 0 and ax.get_ylim()[1] > 0:
        ylim = max(-ax.get_ylim()[0], ax.get_ylim()[1])
        ax.set_ylim(-ylim, ylim)


def plot_RBP_summary(
    RBP_name,
    region,
    df_results,
    df_fishers,
    cdata_utr_filtered,
    cdata_genes,
    cdata_pud,
    table_of_pairs,
    cdata_genes_treatment_col="treatment",
    iso_type="proximal",
    shift_type="proximal",
    RBP_name_cut=None,
    figsize=(12, 18),
    remove_zeroes=True,
    nt_before=-1000,
    region_length=50,
    nt_after=200,
    pos_reg_zone=["[-400:-200]", "[-200:0]"],
    neg_reg_zone=["[0:200]", "[200:400]"],
    ylim=(-5, 5),
    text_pos="[-400:-200]",
    text_neg="[0:200]",
):
    """Summary figure of one RBP (3 x 2 panels): Welch scores in one region and in all regions, Fisher results in one
    region and along the isoform, proportion of cells expressing the RBP's gene and its expression per treatment.

    Args:
        RBP_name: CLIP dataset name, key of `df_results` / `df_fishers`.
        region: region shown in the one-region panels.
        df_results: Welch results of the integrated CLIP data.
        df_fishers: Fisher results of the integrated CLIP data.
        cdata_utr_filtered, cdata_pud, table_of_pairs: not used by the current figure (kept for the earlier layout).
        cdata_genes: single-cell gene expression (AnnData) for the expression panels.
        cdata_genes_treatment_col: treatment column of `cdata_genes.obs`. Defaults to "treatment".
        iso_type, shift_type: isoform type and shift type of the Fisher profile panel. Default to "proximal".
        RBP_name_cut: gene name of the RBP. Defaults to the part of `RBP_name` before the first "_".
        figsize: figure size. Defaults to (12, 18).
        remove_zeroes: not used by the current figure.
        nt_before, region_length, nt_after: window of the integration (nt around the PAS, region length).
        pos_reg_zone, neg_reg_zone, ylim, text_pos, text_neg: regions highlighted, y-limits and annotation positions of
            the Fisher profile panel.
    """
    if RBP_name_cut is None:
        RBP_name_cut = RBP_name.split("_")[0]
    # if remove_zeroes:
    #     count_nans = False
    # else:
    #     count_nans = True
    # if len(table_of_pairs[table_of_pairs['gene'] == RBP_name_cut]) > 0:
    #     fig, ax = plt.subplots(4, 2, figsize=figsize)
    #     plot_RBP_welch_scores_one_region(RBP_name=RBP_name, region_name=region,
    # df_results=df_results, ax=ax[0][0], nt_before=nt_before, region_length=region_length)
    #     plot_RBP_welch_scores_all_region(RBP_name=RBP_name, df_results=df_results,
    # ax=ax[0][1], nt_before=nt_before, region_length=region_length)
    #     plot_fisher_value_one_region(df_fishers=df_fishers, RBP_name=RBP_name, region=region, ax=ax[1][0])
    #     plot_individual_fisher_rbp(df_fishers=df_fishers, RBP_name=RBP_name,
    # iso_type=iso_type, shift_type = shift_type, ax=ax[1][1],
    #                  pos_reg_zone=pos_reg_zone,
    #                  neg_reg_zone=neg_reg_zone,
    #                  ylim=ylim, text_pos=text_pos, text_neg=text_neg)
    #     plot_modality_pcts_table_of_pairs(cdata_utr_filtered, table_of_pairs,
    # gene_name=RBP_name_cut, ax=ax[2][0], count_nans=count_nans)
    #     # plot_PUD_one_gene(RBP_name_cut, table_of_pairs=table_of_pairs, ax=ax[2][1])
    #     plot_PUD_single_cell_one_gene_or_pair(cdata_pud, cdata_genes,
    # treatment_col='treatment', pair_name=None, gene_name=RBP_name_cut,
    # table_of_pairs=table_of_pairs, remove_zeroes=remove_zeroes, ax=ax[2][1])
    #     get_prop_cells_expressing_gene(cdata_genes, RBP_name_cut,
    # treatment_col=cdata_genes_treatment_col, ax=ax[3][0])
    #     get_gene_expression(cdata_genes, RBP_name_cut, treatment_col=cdata_genes_treatment_col, ax=ax[3][1])
    #     plt.tight_layout()
    # else:
    fig, ax = plt.subplots(3, 2, figsize=figsize)
    plot_RBP_welch_scores_one_region(
        RBP_name=RBP_name,
        region_name=region,
        df_results=df_results,
        ax=ax[0][0],
        nt_before=nt_before,
        region_length=region_length,
    )
    plot_RBP_welch_scores_all_region(
        RBP_name=RBP_name,
        df_results=df_results,
        ax=ax[0][1],
        nt_before=nt_before,
        region_length=region_length,
    )
    plot_fisher_value_one_region(
        df_fishers=df_fishers, RBP_name=RBP_name, region=region, ax=ax[1][0]
    )
    plot_individual_fisher_rbp(
        df_fishers=df_fishers,
        RBP_name=RBP_name,
        iso_type=iso_type,
        shift_type=shift_type,
        ax=ax[1][1],
        pos_reg_zone=pos_reg_zone,
        neg_reg_zone=neg_reg_zone,
        ylim=ylim,
        text_pos=text_pos,
        text_neg=text_neg,
    )
    get_prop_cells_expressing_gene(
        cdata_genes, RBP_name_cut, treatment_col=cdata_genes_treatment_col, ax=ax[2][0]
    )
    get_gene_expression(
        cdata_genes, RBP_name_cut, treatment_col=cdata_genes_treatment_col, ax=ax[2][1]
    )
    plt.tight_layout()
    sns.despine()


def get_percentage_cross_links_one_RBP_one_region(
    df_results: pd.DataFrame,
    RBP_name: str,
    region: str,
    label="",
    stat="fisher",
    cell_line=None,
    fisher_name="",
):
    """For a given df_results, it retrieves the number of pairs exhibiting a
    significant shift being bound or not by a given RBP in a given region; as well
    as the proportions of bound and unbound isoforms in the background.
    Returns a dataframe summarizing these statistics, taking into account the possible replicates.

    Args:
        df_results: DataFrame representing the results of the CLIP integration (output of clip_integration.py)
        RBP_name: Name of the RBP to look at.
        region: Name of the region to look at.
        label: Label for the study. Defaults to "".
        stat: Statistic to plot the p-value. Either the results from the fisher test, either from the
        test with null boostraped distribution.

    Returns:
       pd.DataFrame: With the columns "3' UTR shortening" and "no 3' UTR shortening", carying the percentages
       of isoforms bound by the RBP in this region together with their p-values.
    """

    if fisher_name != "":
        fisher_name = f"_{fisher_name}"
    rbp_replicates = [rbp for rbp in df_results.keys() if RBP_name in rbp]

    if cell_line is not None:
        rbp_replicates = [rbp for rbp in rbp_replicates if cell_line in rbp]

    pct_shifts = []
    pct_no_shifts = []
    p_values = []
    for rep in rbp_replicates:
        fisher_table1 = df_results[rep].loc[region][f"fisher_table{fisher_name}"]
        pct_shift1 = (
            fisher_table1[0][0] / (fisher_table1[0][0] + fisher_table1[1][0]) * 100
        )
        pct_no_shift1 = (
            fisher_table1[0][1] / (fisher_table1[0][1] + fisher_table1[1][1]) * 100
        )
        pct_shifts.append(pct_shift1)
        pct_no_shifts.append(pct_no_shift1)
        if stat == "fisher":
            p_values.append(
                df_results[rep].loc[region][f"fisher_results{fisher_name}"][1]
            )
        elif stat == "bootstrap":
            p_values.append(df_results[rep].loc[region]["bootstraped_pval"])
        else:
            raise NotImplementedError(
                "Please enter a valid statistic. Either fisher or boostrap."
            )

    df = pd.DataFrame(
        {
            "3' UTR shortening": pct_shifts,
            "no 3' UTR shortening": pct_no_shifts,
            "p-val": p_values,
            "pair": [label] * len(rbp_replicates),
        }
    )
    return df


def proportion_Ip_bound_sig_shift_compared_to_background_multiple_comparisons(
    df_results_list: list,
    RBP_name: str,
    region: str,
    labels_list: list,
    iso_type="proximal",
    stat="fisher",
    cell_line=None,
    fisher_name="",
):
    """For multiple comparisons, plot the percentage of isoforms bound by the RBP in the
    given region for 1) the pairs exhibiting a significant shift and 2) pairs in the background.
    Plot the associated p-value. Can be from the fisher test or from the bootsrap test.

    Args:
        df_results: DataFrame representing the results of the CLIP integration (output of clip_integration.py)
        RBP_name: Name of the RBP to look at.
        region: Name of the region to look at.
        label: Label for the study. Defaults to "".
        stat: Statistic to plot the p-value. Either the results from the fisher test, either from the
        test with null boostraped distribution.
    """
    from matplotlib.patches import Patch

    if iso_type == "proximal":
        iso = "Ip"
    else:
        iso = "Id"
    dfs_RBP = []
    for df_res, label in zip(df_results_list, labels_list):
        dfs_RBP.append(
            get_percentage_cross_links_one_RBP_one_region(
                df_res,
                RBP_name=RBP_name,
                region=region,
                label=label,
                stat=stat,
                cell_line=cell_line,
                fisher_name=fisher_name,
            )
        )
    df_RBP = pd.concat(dfs_RBP)

    df_plot = pd.melt(
        df_RBP[["no 3' UTR shortening", "3' UTR shortening"]],
        value_name=f"Percentage of Ip-Id pairs \n bound by {RBP_name} on {iso}",
        var_name="shift",
    )
    df_plot["pair"] = list(df_RBP["pair"].values) * 2
    p_vals = df_RBP.groupby("pair").agg("mean")["p-val"]

    ax = sns.barplot(
        data=df_plot,
        x="pair",
        y=f"Percentage of Ip-Id pairs \n bound by {RBP_name} on {iso}",
        hue="shift",
    )

    nb_pairs = len(df_plot.value_counts("pair"))

    hatch_patterns = [""] * nb_pairs + [".."] * nb_pairs

    for i, bar in enumerate(ax.patches):
        hatch = hatch_patterns[i % len(hatch_patterns)]
        bar.set_hatch(hatch)
        #         # bar.set_color(colors_patterns[i % len(hatch_patterns)])
        #         bar.set_color(colors_patterns[i])
        bar.set_color("white")

        bar.set_edgecolor("black")
        bar.set_linewidth(1.2)

    legend_elements = [
        Patch(
            facecolor="white", edgecolor="black", hatch="", label="no 3' UTR shortening"
        ),
        Patch(
            facecolor="white", edgecolor="black", hatch="..", label="3' UTR shortening"
        ),
    ]

    # ax.legend(handles=legend_elements, fontsize=fs, bbox_to_anchor=(1.1, 1.1));
    ax.legend(handles=legend_elements, fontsize=12, bbox_to_anchor=(1.5, 1.1))
    y = df_plot[f"Percentage of Ip-Id pairs \n bound by {RBP_name} on {iso}"].max() + 5

    for i, label in enumerate(ax.get_xticklabels()):
        ax.text(s=f"P={p_vals.loc[label.get_text()]:.2e}", x=i - 0.25, y=y, fontsize=13)
    sns.despine()


def PUD_distribution_according_to_RBP_cross_linking(
    df_results: dict,
    table_of_pairs: pd.DataFrame,
    RBP_name: str,
    region: str,
    treatments: list,
    iso_type: str = "proximal",
    region_name: str = None,
    score: str = "PUD",
):
    """Plot boxplot and histogram of the score (PUD/RUD/DUD) distributions of pairs that are
    cross-linked vs. not cross-linked by the RBP (RBP_name) in the given
    region of the iso_type, for each treatment.

    Args:
        df_results: dictionnary with RBPs in keys and their clip_integration results in values.
        table_of_pairs: Table of pairs output of the APA analysis.
        RBP_name: Name of the RBP to look at. If multiple replicates are found for the RBP,
        will plot them all.
        region: Name of the region you want to plot, as it appears in rows of the
        df_results values dataframes.
        treatments: List of treatments you want to plot. A score (PUD; RUD or DUD)
        for these treatments must be stored in table_of_pairs.
        iso_type: Type of isoform to look at the cross-links on. Defaults to 'proximal'.
        region_name: Another name for the region for legend purposes. Defaults to None.
        score: which score to plot. 'PUD', 'DUD' or 'RUD. Defaults to 'PUD'.
    """

    if iso_type == "proximal":
        iso = "Ip"
    else:
        iso = "Id"

    if region_name is None:
        region_name = region

    rbps = [rbp for rbp in df_results.keys() if RBP_name in rbp]
    for rbp in rbps:
        table_of_pairs["is_cross_linked"] = table_of_pairs[f"{iso_type}_id"].isin(
            df_results[rbp].loc[region, "cross_link"]
        )
        nb_cross_linked_pairs = len(table_of_pairs[table_of_pairs["is_cross_linked"]])
        nb_non_cross_linked_pairs = len(table_of_pairs) - nb_cross_linked_pairs
        df_plot = pd.melt(
            table_of_pairs[[score + "_" + treatment for treatment in treatments]],
            value_name=score,
        )
        df_plot[f"is {rbp} cross linked \n in {region_name} of {iso}"] = (
            list(table_of_pairs["is_cross_linked"].values) * 3
        )
        df_plot["condition"] = df_plot["variable"].apply(
            lambda x: x.split(f"{score}_")[1]
        )

        p_values = []
        for cond in df_plot.value_counts("condition").index:
            p_values.append(
                mannwhitneyu(
                    df_plot[
                        (df_plot["condition"] == cond)
                        & (
                            df_plot[
                                f"is {rbp} cross linked \n in {region_name} of {iso}"
                            ]
                        )
                    ][score],
                    df_plot[
                        (df_plot["condition"] == cond)
                        & (
                            ~df_plot[
                                f"is {rbp} cross linked \n in {region_name} of {iso}"
                            ]
                        )
                    ][score],
                    alternative="two-sided",
                )[1]
            )

        formatted_pvalues = [f"P={pvalue:.2e}" for pvalue in p_values]

        pairs = [
            [("Sensitive", False), ("Sensitive", True)],
            [("Acquired Resistance", False), ("Acquired Resistance", True)],
            [("Innate Resistance", False), ("Innate Resistance", True)],
        ]

        fig, ax = plt.subplots(
            1, 1, figsize=(1.5 * len(df_plot.value_counts("condition")), 5)
        )
        plotting_parameters = {
            "data": df_plot,
            "x": "condition",
            "y": score,
            "hue": f"is {rbp} cross linked \n in {region_name} of {iso}",
            "palette": {False: "gray", True: "orangered"},
            "linewidth": 2.3,
        }

        sns.boxplot(
            **plotting_parameters,
            ax=ax,
            boxprops={"edgecolor": "k"},
            medianprops={"color": "black"},
            whiskerprops={"color": "k"},
            capprops={"color": "k"},
        )

        ax.get_legend().remove()
        annotator = Annotator(ax, pairs, **plotting_parameters)
        annotator.configure(fontsize=12)
        annotator.set_custom_annotations(formatted_pvalues)
        annotator.annotate()
        ax.set_xticks(ticks=range(len(treatments)), labels=treatments, rotation=30)
        plt.text(
            s=r"$n_{\mathrm{pairs}}$" + f" = {nb_cross_linked_pairs}",
            color="orangered",
            x=0.5,
            y=-0.15,
            fontsize=12,
            weight="bold",
        )
        plt.text(
            s=r"$n_{\mathrm{pairs}}$" + f" = {nb_non_cross_linked_pairs}",
            color="gray",
            x=0.5,
            y=-0.07,
            fontsize=12,
            weight="bold",
        )
        plt.ylim(-0.2, 1.2)
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.title(rbp, weight="bold")

        fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
        for i, treatment in enumerate(treatments):
            sns.kdeplot(
                df_plot[df_plot["condition"] == treatment],
                x=score,
                hue=f"is {rbp} cross linked \n in {region_name} of {iso}",
                ax=ax[i],
                palette={False: "gray", True: "orangered"},
                common_norm=False,
                alpha=0.5,
                linewidth=2,
                fill=True,
            )
            if i + 1 < len(treatments):
                ax[i].get_legend().remove()
            else:
                sns.move_legend(ax[i], "upper left", bbox_to_anchor=(1, 1))
            ax[i].set_title(treatment, weight="bold")

        plt.tight_layout()


def PUD_distribution_cross_linked_isoforms_across_treatment(
    df_results: dict,
    table_of_pairs: pd.DataFrame,
    RBP_name: str,
    region: str,
    treatments: list,
    iso_type: str = "proximal",
    region_name: str = None,
    score: str = "PUD",
    palette=None,
    fontsize=12,
):
    """Plot boxplot and histogram of the score (PUD/RUD/DUD) distributions of pairs that are
    cross-linked by the RBP (RBP_name) in the given region of the iso_type, for each treatment.

    Args:
        df_results: dictionnary with RBPs in keys and their clip_integration results in values.
        table_of_pairs: Table of pairs output of the APA analysis.
        RBP_name: Name of the RBP to look at. If multiple replicates are found for the RBP,
        will plot them all.
        region: Name of the region you want to plot, as it appears in rows of the
        df_results values dataframes.
        treatments: List of treatments you want to plot. A score (PUD; RUD or DUD)
        for these treatments must be stored in table_of_pairs.
        iso_type: Type of isoform to look at the cross-links on. Defaults to 'proximal'.
        region_name: Another name for the region for legend purposes. Defaults to None.
        score: which score to plot. 'PUD', 'DUD' or 'RUD. Defaults to 'PUD'.
    """
    from itertools import combinations

    from scipy.stats import wilcoxon
    from statannotations.Annotator import Annotator

    if iso_type == "proximal":
        iso = "Ip"
    else:
        iso = "Id"

    if region_name is None:
        region_name = region

    rbps = [rbp for rbp in df_results.keys() if RBP_name in rbp]
    for rbp in rbps:
        table_of_pairs["is_cross_linked"] = table_of_pairs[f"{iso_type}_id"].isin(
            df_results[rbp].loc[region, "cross_link"]
        )

        table_linked = table_of_pairs[table_of_pairs["is_cross_linked"]]
        # nb_cross_linked_pairs = len(table_of_pairs[table_of_pairs["is_cross_linked"]])
        # nb_non_cross_linked_pairs = len(table_of_pairs) - nb_cross_linked_pairs
        df_plot = pd.melt(
            table_linked[[score + "_" + treatment for treatment in treatments]],
            value_name=score,
            var_name="condition",
        )

        df_plot["condition"] = df_plot["condition"].apply(
            lambda x: x.split(f"{score}_")[1]
        )
        df_plot["condition"] = df_plot["condition"].apply(
            lambda x: x.replace("_", "\n")
        )

        pairs = list(combinations(list(df_plot["condition"].value_counts().index), 2))
        p_values = []
        for p1, p2 in pairs:
            p_values.append(
                wilcoxon(
                    df_plot[(df_plot["condition"] == p1)][score],
                    df_plot[(df_plot["condition"] == p2)][score],
                    alternative="two-sided",
                )[1]
            )

        formatted_pvalues = [f"P={pvalue:.2e}" for pvalue in p_values]
        print(pairs)

        fig, ax = plt.subplots(
            1, 2, figsize=(4 * len(df_plot.value_counts("condition")), 5)
        )

        plotting_parameters = {
            "data": df_plot,
            "x": "condition",
            "y": score,
            "linewidth": 2.3,
        }

        sns.boxplot(
            **plotting_parameters,
            ax=ax[0],
            boxprops={"edgecolor": "k"},
            medianprops={"color": "black"},
            whiskerprops={"color": "k"},
            capprops={"color": "k"},
            palette=palette,
        )

        annotator = Annotator(ax[0], pairs, **plotting_parameters)
        annotator.configure(fontsize=fontsize)
        annotator.set_custom_annotations(formatted_pvalues)
        annotator.annotate()

        sns.kdeplot(
            data=df_plot,
            x=score,
            hue="condition",
            ax=ax[1],
            common_norm=False,
            alpha=0.5,
            linewidth=2,
            fill=True,
            palette=palette,
        )
        sns.move_legend(ax[1], "upper left", bbox_to_anchor=(0.9, 1))
        sns.despine(trim=True)
        plt.suptitle(
            f"PUD distributions of pairs bound by \n {rbp} on {iso} in {region_name}",
            weight="bold",
        )
        plt.tight_layout()


def get_positive_and_negative_regulators_per_region(
    df_results: dict,
    names: list = None,
    nb_top: int = 10,
    regions: list = None,
    bootstrap: bool = True,
    shift_type: str = "proximal",
    fisher_name="",
):
    """For each region, plot the top positive and negative regulators according
    to the boostraping test.

    Args:
        df_results: Dictionnary with RBP in keys and results of the clip integration in values.
        names: list of names if you want to rename the RBPs. Defaults to None.
        nb_top: Number of top regulators to plot. Defaults to 10.
        regions: Names of the regions if you want to rename them. Defaults to None.
        bootstrap: Whether to use the boostrap statistic or not. True will use bootstrap, otherwise
        Fisher will be used.
    """
    if fisher_name != "":
        fisher_name = f"_{fisher_name}"
    p_values = []
    regulation = []
    significance = []
    if names is None:
        new_names = []
    for RBP, df in df_results.items():
        if bootstrap:
            p_values.append(df["bootstraped_pval"])
            regulation.append(df["regulation_of_shifts"])
            significance.append(df["significance"])
        else:
            p_values.append(
                [res[1] for res in df[f"fisher_results{fisher_name}"].values]
            )
            regulation.append(
                [
                    "positive"
                    if (
                        table[0][0] / (table[1][0] + table[0][0])
                        > table[0][1] / (table[0][1] + table[1][1])
                    )
                    else "negative"
                    for table in df[f"fisher_table{fisher_name}"].values
                ]
            )
            significance.append(
                [
                    "sig" if res[1] < 0.05 else "unsig"
                    for res in df[f"fisher_results{fisher_name}"].values
                ]
            )
        if names is None:
            new_names.append(RBP)
    if names is None:
        names = new_names

    if regions is None:
        regions = df.index
    sig_reg = pd.DataFrame(
        np.where(pd.DataFrame(significance) == "sig", pd.DataFrame(regulation), np.nan)
    )
    sig_reg.index = names
    sig_reg.columns = regions

    neg_reg = pd.DataFrame(
        np.where(pd.DataFrame(sig_reg) == "negative", pd.DataFrame(p_values), np.nan)
    )
    neg_reg = neg_reg.apply(lambda x: -np.log10(x))
    neg_reg.index = names
    neg_reg.columns = regions

    pos_reg = pd.DataFrame(
        np.where(pd.DataFrame(sig_reg) == "positive", pd.DataFrame(p_values), np.nan)
    )
    pos_reg = pos_reg.apply(lambda x: -np.log10(x))
    pos_reg.index = names
    pos_reg.columns = regions

    fig, ax = plt.subplots(
        len(sig_reg.columns), 2, figsize=(10, 5 * len(sig_reg.columns)), sharex=True
    )
    for i, region in enumerate(sig_reg.columns):
        neg_reg.sort_values(region, ascending=False, inplace=True)
        df = neg_reg[[region]].dropna()
        if len(df) > 0:
            sns.barplot(
                data=df.head(nb_top),
                y=df.head(nb_top).index,
                x=region,
                orient="h",
                color="hotpink",
                ax=ax[i][0],
            )
        ax[i][0].axvline(x=-np.log10(0.05), linestyle="dashed", color="black")
        ax[i][0].set_title(
            f"Depleted in pairs with {shift_type} shifts on \n {region}", weight="bold"
        )
        ax[i][0].set_xlabel("-log10(P-Value)")

        pos_reg.sort_values(region, ascending=False, inplace=True)
        df = pos_reg[[region]].dropna()
        if len(df) > 0:
            sns.barplot(
                data=df.head(nb_top),
                y=df.head(nb_top).index,
                x=region,
                orient="h",
                color="mediumturquoise",
                ax=ax[i][1],
            )
        ax[i][1].axvline(x=-np.log10(0.05), linestyle="dashed", color="black")
        ax[i][1].set_title(
            f"Enriched in pairs with {shift_type} shifts on \n {region}", weight="bold"
        )
        ax[i][1].set_xlabel("-log10(P-Value)")

    plt.tight_layout()


def get_number_of_RBPs_difference_in_bounding_in_shifts_per_region(
    df_results: dict,
    names: list = None,
    regions: list = None,
    iso_type: str = "proximal",
    bootstrap=True,
    fisher_name="",
):
    """Plot the number of RBPs having a differential bounding
    per region.

    Args:
        df_results: Dictionnary with RBP in keys and results of the clip integration in values.
        names: list of names if you want to rename the RBPs. Defaults to None.
        regions: Names of the regions if you want to rename them. Defaults to None.
        iso_type: Type of isoform you look at the cross-links on. Defaults to "proximal".
        bootstrap: Whether to use the boostrap statistic or not. True will use bootstrap, otherwise
        Fisher will be used.
    """
    if fisher_name != "":
        fisher_name = f"_{fisher_name}"

    p_values = []
    regulation = []
    significance = []
    if names is None:
        new_names = []
    for RBP, df in df_results.items():
        if bootstrap:
            p_values.append(df["bootstraped_pval"])
            regulation.append(df["regulation_of_shifts"])
            significance.append(df["significance"])
        else:
            p_values.append(
                [res[1] for res in df[f"fisher_results{fisher_name}"].values]
            )
            regulation.append(
                [
                    "positive"
                    if (
                        table[0][0] / (table[1][0] + table[0][0])
                        > table[0][1] / (table[0][1] + table[1][1])
                    )
                    else "negative"
                    for table in df[f"fisher_table{fisher_name}"].values
                ]
            )
            significance.append(
                [
                    "sig" if res[1] < 0.05 else "unsig"
                    for res in df[f"fisher_results{fisher_name}"].values
                ]
            )
        if names is None:
            new_names.append(RBP)
    if names is None:
        names = new_names

    if regions is None:
        regions = df.index
    sig_reg = pd.DataFrame(
        np.where(pd.DataFrame(significance) == "sig", pd.DataFrame(regulation), np.nan)
    )
    sig_reg.index = names
    sig_reg.columns = regions

    neg_reg = pd.DataFrame(
        np.where(pd.DataFrame(sig_reg) == "negative", pd.DataFrame(p_values), np.nan)
    )
    neg_reg = neg_reg.apply(lambda x: -np.log10(x))
    neg_reg.index = names
    neg_reg.columns = regions

    pos_reg = pd.DataFrame(
        np.where(pd.DataFrame(sig_reg) == "positive", pd.DataFrame(p_values), np.nan)
    )
    pos_reg = pos_reg.apply(lambda x: -np.log10(x))
    pos_reg.index = names
    pos_reg.columns = regions

    if iso_type == "proximal":
        iso = "Ip"
    else:
        iso = "Id"
    sns.lineplot(
        (~pos_reg.isna()).sum(),
        label=f"more bound on {iso} in pairs with proximal shifts",
    )
    sns.lineplot(
        (~neg_reg.isna()).sum(),
        label=f"less bound on {iso} in pairs with proximal shifts",
    )
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.xticks(rotation=30)
    plt.ylabel(
        "Number of RBPs per region with \n a significant difference in bounding frequency"
    )
