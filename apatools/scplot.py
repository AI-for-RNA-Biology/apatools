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
from scanalysis.dimred import get_color_map
from scanalysis.plot import plot_pca
from scipy.stats import ttest_ind
from statannotations.Annotator import Annotator

from apatools.scapa import get_isoform_exp_across_cells
from apatools.scapa import get_pair_exp_single_cell
from apatools.scmodality import get_modality_pcts
from apatools.scmodality import pct_cells_per_modality


def get_PUD_histplot(adata, gene_name, palette=None):
    "Get PUD histogram for a given gene/pair."

    idx = list(adata.var.index).index(gene_name)
    df_plot = pd.DataFrame(adata.X[:, idx].todense(), index=adata.obs.index).rename(
        columns={0: "PUD"}
    )
    df_plot["treatment"] = adata.obs.treatment
    df_plot = df_plot.replace(0.5, np.NaN)
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    bin_edges = [-0.1, 0.2, 0.4, 0.6, 0.8, 1]
    df_plot["bin"] = pd.cut(
        df_plot["PUD"],
        bins=bin_edges,
        labels=["[0, 0.2]", "(0.2, 0.4]", "(0.4, 0.6]", "(0.6, 0.8]", "(0.8, 1]"],
    )

    sns.histplot(
        data=df_plot,
        x="bin",
        hue="treatment",
        ax=ax,
        binwidth=0.05,
        multiple="dodge",
        palette=palette,
        shrink=0.8,
    )
    sns.despine()
    ax.set_xlabel("PUD: Ip/(Ip+Id)", fontsize=15)
    ax.set_ylabel("Number of cells", fontsize=15)
    ax.set_title(gene_name.split(".")[0], weight="bold", fontsize=17)


def color_gene_pca_per_isoform_modality(
    genes_dimred,
    cat_pres,
    pair_name=None,
    modality_to_plot=["0", "1", "2", "3"],
    nb_PCs=5,
    PC_x=None,
    PC_y=None,
    alpha=1,
    color_pca=None,
    save_path=None,
    palette=None,
    kind="scatter",
    size=None,
    sizes=None,
    s=10,
    edgecolor="black",
):
    """Allows to plot any PCA representation by coloring cells by their modality for the given pair.
    Args:
        :param genes_dimred: AnnData object - containing the dimredthat will be plotted
        .param cat_pres: AnnData object - contaning the categorical modality representation (with 0, 1, 2 and 3).
        :param pair_name: string - for the name of the pair for which the modality representation is needed
        :param modality_to_plot: list - which modalities you want to explicitly plot.
        Default is [0, 1, 2, 3] meaning that all possible modalities, and therefore all cells are plotted.
    """
    if pair_name is None:
        raise ValueError("Please enter a valid pair name.")

    genes_dimred.obs[pair_name] = (
        get_color_map(cat_pres, cat_pres, gene_name=pair_name)[pair_name]
        .astype(int)
        .astype(str)
    )

    if kind == "jointplot":
        plot_pca(
            genes_dimred[genes_dimred.obs[pair_name].isin(modality_to_plot)],
            color_pca=color_pca,
            nb_PCs=nb_PCs,
            PC_x=PC_x,
            PC_y=PC_y,
            alpha=alpha,
            kind=kind,
            s=s,
            edgecolor=edgecolor,
        )
        plot_pca(
            genes_dimred[genes_dimred.obs[pair_name].isin(modality_to_plot)],
            color_pca=pair_name,
            nb_PCs=nb_PCs,
            PC_x=PC_x,
            PC_y=PC_y,
            palette=palette,
            alpha=alpha,
            kind=kind,
            s=s,
            edgecolor=edgecolor,
        )

    else:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        plot_pca(
            genes_dimred[genes_dimred.obs[pair_name].isin(modality_to_plot)],
            color_pca=color_pca,
            ax=ax[0],
            nb_PCs=nb_PCs,
            PC_x=PC_x,
            PC_y=PC_y,
            alpha=alpha,
            kind=kind,
            s=s,
            edgecolor=edgecolor,
        )
        ax[0].legend(fontsize=15, loc="lower right")
        plot_pca(
            genes_dimred[genes_dimred.obs[pair_name].isin(modality_to_plot)],
            color_pca=pair_name,
            ax=ax[1],
            nb_PCs=nb_PCs,
            PC_x=PC_x,
            PC_y=PC_y,
            palette=palette,
            alpha=alpha,
            kind=kind,
            s=s,
            edgecolor=edgecolor,
        )
        ax[1].legend(fontsize=15, loc="lower right")
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")


def get_histo_modality_one_pair(
    dimred, cat_pres, pair_name, PC, ax=None, replace_name=False, kind="boxplot"
):
    """Boxplots or violin plots representing PC value distribution according to
    the modality of the cells for a given pair.
    Args:
        dimred (AnnData): AnnData object to use for dimensionnality reduction
        cat_pres (AnnData): AnnData object containing the modality information
        pair_name (str): _Name of the pair
        PC (int): PC number
        ax (ax, optional): ax. Defaults to None.
        replace_name (bool, optional): Whether to replace the [0, 1, 2, 3] by
        ['no Ip/no Id', 'Ip only', 'Ip and Id', 'Id only']. Defaults to False.
        kind (str, optional): Boxplot or violin plot. Defaults to 'boxplot'.

    Raises:
        ValueError: In case of invalid pair name.
    """
    if pair_name is None:
        raise ValueError("Please enter a valid pair name.")

    dimred.obs[pair_name] = (
        get_color_map(cat_pres, cat_pres, gene_name=pair_name)[pair_name]
        .astype(int)
        .astype(str)
    )

    df_pca = pd.DataFrame(
        dimred.obsm["X_pca"][:, : PC + 1],
        index=dimred.obs.index,
        columns=["PC" + str(i + 1) for i in range(PC + 1)],
    )

    df_plot = df_pca.merge(dimred.obs, left_index=True, right_index=True)[
        ["PC" + str(PC), pair_name]
    ]
    df_plot[pair_name] = df_plot[pair_name].astype(int)
    df_plot.sort_values(pair_name, inplace=True)
    df_plot[pair_name] = df_plot[pair_name].astype(str)
    if replace_name:
        df_plot.replace(
            {"0": "no Ip/no Id", "1": "Ip only", "2": "Ip and Id", "3": "Id only"},
            inplace=True,
        )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    if kind == "boxplot":
        sns.boxplot(data=df_plot, x=pair_name, y="PC" + str(PC), ax=ax)
    else:
        sns.violinplot(data=df_plot, x=pair_name, y="PC" + str(PC), ax=ax)
    ax.set_title(pair_name, fontsize=17, weight="bold")
    ax.set_xlabel("Modality", fontsize=15)
    ax.set_ylabel("PC" + str(PC), fontsize=15)
    ax.axhline(y=0, linestyle="dashed", color="black")


def heatmap_modality_two_pairs(
    adata, pair1, pair2, remove_zeroes=True, ax=None, fmt="g"
):
    """Heatmap representing the modality of cells for two given pairs

    Args:
        adata (AnnData): AnnData object representing the modality of each pair
        pair1 (str): Name of the first pair.
        pair2 (str): Name of the second pair.
        remove_zeroes (bool, optional): Whether to remove the cells that do not express neither of the pairs.
        Defaults to True.
        ax (matplotlib.Axes, optional): ax to plot. Defaults to None.
        fmt (str, optional): String formatting code to use when adding annotation. Defaults to 'g'.
    """
    index_name = adata.var.index.name
    pair1_idx = adata.var.reset_index()[
        adata.var.reset_index()[index_name] == pair1
    ].index[0]
    pair2_idx = adata.var.reset_index()[
        adata.var.reset_index()[index_name] == pair2
    ].index[0]
    df = pd.DataFrame(
        pd.DataFrame(adata.X.todense())[[pair1_idx, pair2_idx]].value_counts()
    )
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    if remove_zeroes:
        df = df.unstack().to_numpy()[1:, 1:]
        sns.heatmap(
            df,
            annot=True,
            cmap="Purples",
            xticklabels=["Ip only", "Ip&Id", "Id only"],
            yticklabels=["Ip only", "Ip&Id", "Id only"],
            ax=ax,
            fmt=fmt,
        )
    else:
        df = df.unstack().to_numpy()
        sns.heatmap(
            df,
            annot=True,
            cmap="Purples",
            xticklabels=["no Ip&no Id", "Ip only", "Ip&Id", "Id only"],
            yticklabels=["no Ip&no Id", "Ip only", "Ip&Id", "Id only"],
            ax=ax,
            fmt=fmt,
        )

    ax.set_ylabel(pair1)
    ax.set_xlabel(pair2)


def plot_modality_pcts_table_of_pairs(
    adata_utrs_filtered,
    table_of_pairs,
    prox_id="proximal_id",
    dist_id="distal_id",
    treatment_col="treatment",
    gene_name=None,
    ax=None,
    count_nans=False,
):
    """Plot boxplots of the distributions of percentage of cells expressing proximal,
    distal or both isoforms in each condition.

    Args:
        adata_utrs_filtered (AnnData): Single-cell expression data of 3' UTR isoforms that has been previously filtered.
        table_of_pairs (pd.DataFrame): DataFrame containing pairs in rows and their proximal and distal
        isoforms names in columns
        treatment_col (str, optional): Name of the column of adata_utrs_filtered.obs containing the
        information about the treatment of the cells.
        prox_id (str, optional): Name of the column of *table_of_pairs* containing the proximal isoforms names.
        dist_id (str, optional):Name of the column of *table_of_pairs* containing the distal isoforms names.
        gene_name (str, optional): Gene name in case you want to plot the pairs corresponding to one single gene.
        Defaults to None.
        ax (matplolib.Axes, optional): ax to plot. Defaults to None.
    """

    if gene_name is not None:
        table_of_pairs = table_of_pairs[table_of_pairs["gene"] == gene_name]
    df_modality = pct_cells_per_modality(
        adata_utrs_filtered,
        table_of_pairs,
        treatment_col=treatment_col,
        count_nans=count_nans,
    )
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if len(table_of_pairs) > 1:
        sns.boxplot(
            data=df_modality,
            x="Modality",
            y="Percentage of cells [%]",
            hue="treatment",
            ax=ax,
        )
    else:
        sns.barplot(
            data=df_modality,
            x="Modality",
            y="Percentage of cells [%]",
            hue="treatment",
            ax=ax,
        )
    ax.legend(fontsize=15)
    if gene_name:
        ax.set_title(
            gene_name + " modality (number of pairs: " + str(len(table_of_pairs)) + ")",
            weight="bold",
        )


def get_plot_for_pair_single_cell(
    cdata,
    pair_name=None,
    prox_id=None,
    dist_id=None,
    treatment_col="treatment",
    treatment_1="parental",
    treatment_2="resistant",
    type_plot="violin",
    path_save=None,
    add_pcts=False,
    palette=None,
    annotate=True,
    fontsize=12,
    bbox_to_anchor=(1.6, 1.1),
    figsize=(4, 5),
):
    """Violin (or box) plots of the single-cell expression of the proximal and distal isoforms of a pair in two
    treatments, with the Welch's t-test p-value between treatments for each isoform.

    Args:
        cdata: AnnData of isoform counts.
        pair_name: pair as "<proximal_id>/<distal_id>"; or give `prox_id` and `dist_id` (column indices).
        prox_id, dist_id: column indices of the proximal and distal isoforms in `cdata`.
        treatment_col: treatment column of `cdata.obs`. Defaults to "treatment".
        treatment_1, treatment_2: the two treatments. Default to "parental" and "resistant".
        type_plot: "violin", "boxplot" or "histplot" (box plots). Defaults to "violin".
        path_save: if given, directory where `expression_<pair>[without_pcts].pdf` is saved.
        add_pcts: also write the percentage of detecting cells of each isoform and treatment. Defaults to False.
        palette: colours per treatment.
        annotate: add the p-values. Defaults to True.
        fontsize: font size. Defaults to 12.
        bbox_to_anchor: legend position. Defaults to (1.6, 1.1).
        figsize: figure size. Defaults to (4, 5).
    """
    df = get_pair_exp_single_cell(
        cdata,
        pair_name=pair_name,
        prox_id=prox_id,
        dist_id=dist_id,
        treatment_col=treatment_col,
        treatment_1=treatment_1,
        treatment_2=treatment_2,
    )

    plot_params = {
        "data": df,
        "x": "isoform",
        "y": "expression",
        "hue": "treatment",
        "palette": palette,
    }

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if type_plot == "violin":
        sns.violinplot(ax=ax, **plot_params)
    elif type_plot == "boxplot":
        sns.boxplot(showfliers=False, ax=ax, **plot_params)
    elif type_plot == "histplot":
        sns.boxplot(showfliers=False, ax=ax, **plot_params)

    # vertical_offset = df['expression'].mdeian * 0.05

    # pcts = get_isoform_exp_across_cells(cdata, pair_name=None, prox_id=None, dist_id=None,
    #  treatment_col='treatment', treatment_1 =treatment_1, treatment_2 = treatment_2)
    # for xtick in ploto.get_xticks():
    #    ploto.text(xtick, pcts[xtick])
    ax.set_title(pair_name.split(".")[0], weight="bold", fontsize=fontsize)
    ax.set_xlabel("Isoform", fontsize=fontsize)
    ax.set_ylabel(r"$log_{2}(readcounts)$", fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    # add_labels(ax, label_list=get_isoform_exp_across_cells(cdata, pair_name=pair_name,
    # prox_id=prox_id, dist_id=dist_id, treatment_col='treatment', treatment_1 =treatment_1, treatment_2 = treatment_2),
    #           offset=0.15)
    # TODO: should be modified: get_isoform_exp_across_cells has no treatment_1 / treatment_2 parameters (TypeError
    # when add_pcts=True); its result is per treatment, select treatment_1 and treatment_2 from it instead.
    if add_pcts:
        label_list = get_isoform_exp_across_cells(
            cdata,
            pair_name=pair_name,
            prox_id=prox_id,
            dist_id=dist_id,
            treatment_col="treatment",
            treatment_1=treatment_1,
            treatment_2=treatment_2,
        )
        ax.text(
            x=-0.6,
            y=ax.get_ylim()[1],
            s="Number of \n detecting \n cells:",
            fontsize=fontsize,
        )
        ax.text(
            x=-0.25,
            y=ax.get_ylim()[1],
            s="{:.1f}".format(label_list[0]) + "%",
            fontsize=fontsize,
        )
        ax.text(
            x=0.15,
            y=ax.get_ylim()[1],
            s="{:.1f}".format(label_list[1]) + "%",
            fontsize=fontsize,
        )
        ax.text(
            x=0.75,
            y=ax.get_ylim()[1],
            s="{:.1f}".format(label_list[2]) + "%",
            fontsize=fontsize,
        )
        ax.text(
            x=1.15,
            y=ax.get_ylim()[1],
            s="{:.1f}".format(label_list[3]) + "%",
            fontsize=fontsize,
        )
    plt.tight_layout()

    exp = get_pair_exp_single_cell(
        cdata,
        pair_name=pair_name,
        prox_id=prox_id,
        dist_id=dist_id,
        treatment_col="treatment",
        treatment_1=treatment_1,
        treatment_2=treatment_2,
    )
    stats_prox = ttest_ind(
        exp[(exp["treatment"] == treatment_1) & (exp["isoform"] == "Ip")][
            "expression"
        ].values,
        exp[(exp["treatment"] == treatment_2) & (exp["isoform"] == "Ip")][
            "expression"
        ].values,
        equal_var=False,
    )
    stats_dist = ttest_ind(
        exp[(exp["treatment"] == treatment_1) & (exp["isoform"] == "Id")][
            "expression"
        ].values,
        exp[(exp["treatment"] == treatment_2) & (exp["isoform"] == "Id")][
            "expression"
        ].values,
        equal_var=False,
    )
    sns.despine()

    if annotate:
        pairs = [
            [("Ip", treatment_1), ("Ip", treatment_2)],
            [("Id", treatment_1), ("Id", treatment_2)],
        ]
        formatted_pvalues = [
            f"P={pvalue:.2e}" for pvalue in [stats_prox[1], stats_dist[1]]
        ]
        annotator = Annotator(ax, pairs, **plot_params)
        annotator.configure(fontsize=15)
        annotator.set_custom_annotations(formatted_pvalues)
        annotator.annotate()

    ax.legend(loc="upper right", bbox_to_anchor=bbox_to_anchor, fontsize=fontsize)
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    # ax.text(x=-0.17, y = 5.5, s = "pvalue="+"{:.2e}".format(stats_prox[1]))
    # ax.text(x=0.82, y = 5.5, s = "pvalue="+"{:.2e}".format(stats_dist[1]))
    if path_save is not None:
        if add_pcts:
            fig.savefig(
                os.path.join(
                    path_save, "expression_" + pair_name.replace("/", "-") + ".pdf"
                ),
                bbox_inches="tight",
            )
        else:
            fig.savefig(
                os.path.join(
                    path_save,
                    "expression_" + pair_name.replace("/", "-") + "without_pcts.pdf",
                ),
                bbox_inches="tight",
            )


def get_fraction_barplot(
    cdata,
    pair_name="AGPAT5.1/AGPAT5.4",
    prox_id=None,
    dist_id=None,
    treatment_col="treatment",
    palette=None,
    bbox_to_anchor=(1.1, 1.1),
    figsize=(4, 6),
    fontsize=12,
    path_save=None,
):
    """Bar plot of the percentage of cells of each treatment in which the proximal and the distal isoform of a pair are
    detected.

    Args:
        cdata: AnnData of isoform counts.
        pair_name: pair as "<proximal_id>/<distal_id>"; or give `prox_id` and `dist_id` (column indices).
        prox_id, dist_id: column indices of the proximal and distal isoforms in `cdata`.
        treatment_col: treatment column of `cdata.obs`. Defaults to "treatment".
        palette: colours per treatment.
        bbox_to_anchor: legend position. Defaults to (1.1, 1.1).
        figsize: figure size. Defaults to (4, 6).
        fontsize: font size. Defaults to 12.
        path_save: if given, directory where `pct_cells_<pair>.pdf` is saved.
    """
    pcts = get_isoform_exp_across_cells(
        cdata,
        pair_name=pair_name,
        prox_id=prox_id,
        dist_id=dist_id,
        treatment_col=treatment_col,
    )
    df = pd.DataFrame(
        {
            "Percentage of cells [%]": pcts,
            "Treatment": list(cdata.obs[treatment_col].value_counts().keys()) * 2,
            "Isoform": ["Ip", "Ip", "Id", "Id"],
        }
    )
    # plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.barplot(
        data=df,
        x="Isoform",
        y="Percentage of cells [%]",
        hue="Treatment",
        ax=ax,
        palette=palette,
    )
    sns.despine()
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Isoform", fontsize=fontsize)
    plt.ylabel("Percentage of cells [%]", fontsize=fontsize)
    plt.legend(fontsize=fontsize, bbox_to_anchor=bbox_to_anchor)
    plt.title(pair_name.split(".")[0], fontsize=fontsize, weight="bold")

    if path_save is not None:
        fig.savefig(
            os.path.join(
                path_save, "pct_cells_" + pair_name.replace("/", "-") + ".pdf"
            ),
            bbox_inches="tight",
        )


def plot_PUD_single_cell_one_gene_or_pair(
    adata_pud,
    adata_treatment_data,
    treatment_col="treatment",
    pair_name=None,
    gene_name=None,
    table_of_pairs=None,
    remove_zeroes=True,
    ax=None,
):
    """Distribution of the single-cell PUD of one pair (histogram per treatment), or of all pairs of one gene (box plots
    of the percentage of cells per PUD bin of 0.1, per treatment).

    Args:
        adata_pud: AnnData of per-cell PUD values (one variable per pair).
        adata_treatment_data: AnnData with the treatment of the same cells in `obs`.
        treatment_col: treatment column. Defaults to "treatment".
        pair_name: pair to plot (name in `adata_pud.var`).
        gene_name: gene whose pairs are plotted (needs `table_of_pairs`); a gene with a single pair is plotted as a
            pair.
        table_of_pairs: table with `gene` and `pair_name` columns.
        remove_zeroes: ignore cells with PUD = 0.5 (neither isoform detected). Defaults to True.
        ax: matplotlib axes to draw on (a new figure if None).

    Raises:
        NotImplementedError: if neither a pair nor a gene with a table of pairs is given.
    """
    pud_exp = adata_pud.to_df()
    pud_exp["treatment"] = adata_treatment_data.obs[treatment_col].values

    if table_of_pairs is not None and gene_name is not None:
        if len(table_of_pairs[table_of_pairs["gene"] == gene_name]) == 1:
            pair_name = table_of_pairs[table_of_pairs["gene"] == gene_name][
                "pair_name"
            ].values[0]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    if pair_name is not None:
        idx = adata_pud.var.reset_index()[
            adata_pud.var.reset_index()[adata_pud.var.index.name] == pair_name
        ].index[0]

        if remove_zeroes:
            pud_exp = pud_exp[pud_exp[idx] != 0.5]

        sns.histplot(
            data=pud_exp,
            x=idx,
            hue="treatment",
            multiple="dodge",
            stat="percent",
            ax=ax,
        )
        ax.set_xlabel("PUD=Ip/(Ip+Id)")
        ax.set_ylabel("Percentage of cells [%]")
        ax.set_title(pair_name, weight="bold")
    elif table_of_pairs is not None and gene_name is not None:
        pairs = table_of_pairs[table_of_pairs["gene"] == gene_name]["pair_name"]
        treatments = list(pud_exp["treatment"].value_counts().index)
        df_boxplots = []
        for pair in pairs:
            idx = adata_pud.var.reset_index()[
                adata_pud.var.reset_index()[adata_pud.var.index.name] == pair
            ].index[0]
            for i, treatment in enumerate(treatments):
                pud_treatment = pud_exp[pud_exp["treatment"] == treatment]
                if i == 0:
                    if remove_zeroes:
                        df_treatments = (
                            pd.DataFrame(
                                pd.cut(
                                    pud_treatment[pud_treatment[idx] != 0.5][idx],
                                    bins=np.linspace(0, 1, 11),
                                ).value_counts()
                            )
                            * 100
                            / len(pud_treatment[pud_treatment[idx] != 0.5][idx])
                        )
                    else:
                        df_treatments = (
                            pd.DataFrame(
                                pd.cut(
                                    pud_treatment[idx], bins=np.linspace(0, 1, 11)
                                ).value_counts()
                            )
                            * 100
                            / len(pud_treatment)
                        )

                    df_treatments.columns = [treatment]
                else:
                    if remove_zeroes:
                        df_treatment = (
                            pd.DataFrame(
                                pd.cut(
                                    pud_treatment[pud_treatment[idx] != 0.5][idx],
                                    bins=np.linspace(0, 1, 11),
                                ).value_counts()
                            )
                            * 100
                            / len(pud_treatment[pud_treatment[idx] != 0.5][idx])
                        )
                    else:
                        df_treatment = (
                            pd.DataFrame(
                                pd.cut(
                                    pud_treatment[idx], bins=np.linspace(0, 1, 11)
                                ).value_counts()
                            )
                            * 100
                            / len(pud_treatment)
                        )
                    df_treatment.columns = [treatment]
                    df_treatments = df_treatments.merge(
                        df_treatment, left_index=True, right_index=True
                    )
            df_boxplot = pd.melt(
                df_treatments,
                var_name="treatment",
                value_name="Percentage of cells [%]",
            )
            df_boxplot["PUD=Ip/(Ip+Id)"] = list(df_treatments.index) * len(treatments)
            df_boxplot["pair_name"] = pair
            df_boxplots.append(df_boxplot)
        df_boxplots = pd.concat(df_boxplots)
        sns.boxplot(
            data=df_boxplots.sort_values("PUD=Ip/(Ip+Id)"),
            x="PUD=Ip/(Ip+Id)",
            y="Percentage of cells [%]",
            hue="treatment",
            showfliers=False,
            ax=ax,
        )
        ax.set_title(gene_name + " pairs (n = " + str(len(pairs)) + ")", weight="bold")
        ax.set_xticks(
            ticks=np.arange(len(df_treatments.index)),
            labels=df_treatments.sort_index().index,
            rotation=45,
        )

    else:
        raise NotImplementedError(
            "please enter a valid gene name together with a table of pairs, or a valid pair name."
        )


def plot_percentage_per_modality(
    cdata,
    pair_name,
    prox_id=None,
    dist_id=None,
    treatment_col="treatment",
    count_nans=False,
    hue_order=None,
    ax=None,
):
    """Plot the percentage of cells belonging to each modality for a given pair, in each treatment.
        Barplot for the given pair.
    Args:
        cdata (AnnData): Single-cell data representing 3' UTR isoforms expression.
        Cells in obs and 3' UTR isoforms in var. cdata.obs should contain a column *treatment_col*.
        pair_name (string): Name of the pair you want the modality percentages.
        prox_id (int, optional): _description_. Defaults to None.
        dist_id (int, optional): _description_. Defaults to None.
        treatment_col (str, optional): _description_. Defaults to 'treatment'.
        count_nans (bool, optional): Whether to give the percentage of cells in each modality regarding all cells (True)
        or only regarding cells expressing the pair (False).
        Defaults to False.
        hue_order (list of strings, optional): Order to plot the categorical levels in; otherwise
        the levels are inferred from the data object.
        Defaults to None.
        ax (matplotlib Axes, optional): Axes object to draw the plot onto, otherwise uses the current Axes.
          Defaults to None.
    """
    modality_dict = get_modality_pcts(
        cdata,
        pair_name=pair_name,
        prox_id=prox_id,
        dist_id=dist_id,
        treatment_col="treatment",
        count_nans=count_nans,
    )
    df_proportions = pd.melt(
        pd.DataFrame.from_dict(modality_dict).T,
        var_name="Modality",
        value_name="Percentage",
    )
    df_proportions["treatment"] = list(modality_dict.keys()) * 4
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    sns.barplot(
        data=df_proportions,
        x="Modality",
        y="Percentage",
        hue="treatment",
        hue_order=hue_order,
        ax=ax,
    )
    ax.set_title(pair_name, weight="bold", fontsize=17)
    ax.legend(fontsize=15)
    ax.set_ylabel("Percentage of cells [%]", fontsize=15)
    ax.set_xlabel("Modality", fontsize=15)
