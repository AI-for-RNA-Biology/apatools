#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileCopyrightText: Copyright © 2026 University of Bern
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@idiap.ch>
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@unibe.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
import numpy as np
import pandas as pd
from genomemanager.overlaps import get_overlap_file
from scanalysis.pseudobulk import aggregate_cells
from scanalysis.pseudobulk import pseudo_bulk_quantile
from scanalysis.utils import get_outliers

from apatools.apanalysis import get_categorical_one_pair
from apatools.apanalysis import get_PUD_one_pair
from apatools.apanalysis import get_RUD_one_pair


def get_iso_stats_per_cell(adata, cell):
    """Count, in one cell, the genes with 1, 2, 3, ... detected 3' UTR isoforms.

    Args:
        adata: AnnData of isoform counts (`var` has a `gene` column); an `obs["stats"]` column is reset to 0.
        cell: expression vector of the cell, in the order of `adata.var_names`.

    Returns:
        collections.Counter: {number of detected isoforms: number of genes}.
    """
    adata.obs["stats"] = 0
    df = pd.DataFrame({"expression": cell, "iso_ID": adata.var_names}).set_index(
        "iso_ID"
    )
    df["gene"] = adata.var["gene"]
    df = df[df["expression"] != 0]
    from collections import Counter

    stats = Counter(df.groupby("gene").count()["expression"].values)
    # adata.obs.loc[list(adata.obs.index)[0], 'stats'] = stats
    return stats


def remove_outliers_isoforms(adata, gene_to_remove, col_gene="gene_ids"):
    """From a list of genes, remove all transcripts in the adata object related to these genes.

    Args:
        adata (AnnData): Contains the single cell gene/isoform expression data
        gene_to_remove (list of strings): genes to remove
        col_gene (str, optional): Name of the column in adata.var which contains the gene name of the transcripts.
        Defaults to 'gene_ids'.

    Returns:
        AnnData: adata with the genes removed
    """
    genes_to_keep = adata.var[~adata.var[col_gene].str.startswith(gene_to_remove)][
        col_gene
    ].values
    filtered_adata = adata[:, adata.var[col_gene].isin(genes_to_keep)]
    return filtered_adata


def get_isoform_exp_across_cells(
    cdata, pair_name=None, prox_id=None, dist_id=None, treatment_col="treatment"
):
    """Percentage of cells of each treatment in which the proximal and the distal isoform of a pair are detected.

    Args:
        cdata: AnnData of isoform counts.
        pair_name: pair as "<proximal_id>/<distal_id>" (isoform names of `cdata.var`); or give `prox_id` and `dist_id`.
        prox_id, dist_id: column indices of the proximal and distal isoforms in `cdata`.
        treatment_col: treatment column of `cdata.obs`. Defaults to "treatment".

    Returns:
        list: percentages for the proximal isoform in each treatment, then for the distal isoform (treatments in the
        order of `value_counts()`).

    Raises:
        ValueError: if neither `pair_name` nor the isoform indices are given.
    """
    if pair_name is None and prox_id is None and dist_id is None:
        raise ValueError(
            "Please give a pair name or ids for proximal and distal isoforms."
        )

    if pair_name is not None:
        prox_id = cdata.var.reset_index()[
            cdata.var.reset_index()["index"] == pair_name.split("/")[0]
        ].index[0]
        dist_id = cdata.var.reset_index()[
            cdata.var.reset_index()["index"] == pair_name.split("/")[1]
        ].index[0]

    pcts = []
    for treatment in cdata.obs[treatment_col].value_counts().keys():
        cells_treatment = cdata[cdata.obs[treatment_col] == treatment, :]
        pcts.append(
            cells_treatment.X[:, prox_id].nnz * 100 / (cells_treatment.shape[0])
        )

    for treatment in cdata.obs[treatment_col].value_counts().keys():
        cells_treatment = cdata[cdata.obs[treatment_col] == treatment, :]
        pcts.append(
            cells_treatment.X[:, dist_id].nnz * 100 / (cells_treatment.shape[0])
        )

    return pcts


def one_cell_one_pair(cell, prox_id, dist_id, ratio="PUD"):
    """
    Args:
        ratio: string to describe the score you want to compute.
               -'PUD' will compute the PUD value for each pair and each cell
               -'RUD' will compute the RUD score for each pair and each cell
               -'categorical_modality' will return 0 if neither Ip neither Id is detected,
               1 if only Ip is detected, 2 if both are detected, 3 if only the distal isoform is detected.
    """

    Ip = cell[prox_id]
    Id = cell[dist_id]
    if ratio == "PUD":
        return get_PUD_one_pair(Ip, Id)
    elif ratio == "RUD":
        return get_RUD_one_pair(Ip, Id)
    elif ratio == "categorical_modality":
        return get_categorical_one_pair(Ip, Id)
    else:
        raise ValueError("Please enter a valid ratio method.")


def one_cell_all_pairs(i, cell, table_of_pairs, ratio="PUD"):
    """Score (`one_cell_one_pair`) of every pair of `table_of_pairs` in one cell.

    Args:
        i: index of the cell (progress is printed every 500 cells).
        cell: expression vector of the cell.
        table_of_pairs: table with `proximal_id` and `distal_id` columns (positions in `cell`).
        ratio: "PUD", "RUD" or "categorical_modality". Defaults to "PUD".

    Returns:
        np.ndarray: one score per pair.
    """
    if i % 500 == 0:
        print(i)
    return np.array(
        [
            one_cell_one_pair(cell, prox_id, dist_id, ratio)
            for prox_id, dist_id in zip(
                table_of_pairs["proximal_id"].values, table_of_pairs["distal_id"].values
            )
        ]
    )


def get_pair_exp_single_cell(
    cdata,
    pair_name=None,
    prox_id=None,
    dist_id=None,
    treatment_col="treatment",
    treatment_1="parental",
    treatment_2="resistant",
):
    """Expression of the proximal and distal isoforms of a pair in the cells where they are detected, for two
    treatments.

    Args:
        cdata: AnnData of isoform counts.
        pair_name: pair as "<proximal_id>/<distal_id>"; or give `prox_id` and `dist_id` (column indices).
        prox_id, dist_id: column indices of the proximal and distal isoforms in `cdata`.
        treatment_col: treatment column of `cdata.obs`. Defaults to "treatment".
        treatment_1, treatment_2: the two treatments. Default to "parental" and "resistant".

    Returns:
        pd.DataFrame: long table with `expression`, `treatment` and `isoform` ("Ip" / "Id") columns; an isoform detected
        in no cell of a treatment gets three 0 values.

    Raises:
        ValueError: if neither `pair_name` nor the isoform indices are given.
    """
    if pair_name is None and prox_id is None and dist_id is None:
        raise ValueError(
            "Please give a pair name or ids for proximal and distal isoforms."
        )

    if pair_name is not None:
        prox_id = cdata.var.reset_index()[
            cdata.var.reset_index()["index"] == pair_name.split("/")[0]
        ].index[0]
        dist_id = cdata.var.reset_index()[
            cdata.var.reset_index()["index"] == pair_name.split("/")[1]
        ].index[0]

    cells_treatment_1 = cdata[cdata.obs[treatment_col] == treatment_1, :]
    cells_treatment_2 = cdata[cdata.obs[treatment_col] == treatment_2, :]

    exp_treatment_1_Ip = cells_treatment_1.X[:, prox_id][
        cells_treatment_1.X[:, prox_id].nonzero()
    ]
    if exp_treatment_1_Ip.shape[1] == 0:
        df_exp_treatment_1_Ip = pd.DataFrame(
            {"expression": [0, 0, 0], "treatment": treatment_1, "isoform": "Ip"}
        )
    else:
        df_exp_treatment_1_Ip = pd.DataFrame(
            {
                "expression": np.squeeze(np.array(exp_treatment_1_Ip)),
                "treatment": treatment_1,
                "isoform": "Ip",
            }
        )

    exp_treatment_2_Ip = cells_treatment_2.X[:, prox_id][
        cells_treatment_2.X[:, prox_id].nonzero()
    ]
    if exp_treatment_2_Ip.shape[1] == 0:
        df_exp_treatment_2_Ip = pd.DataFrame(
            {"expression": [0, 0, 0], "treatment": treatment_2, "isoform": "Ip"}
        )
    else:
        df_exp_treatment_2_Ip = pd.DataFrame(
            {
                "expression": np.squeeze(np.asarray(exp_treatment_2_Ip)),
                "treatment": treatment_2,
                "isoform": "Ip",
            }
        )

    exp_treatment_1_Id = cells_treatment_1.X[:, dist_id][
        cells_treatment_1.X[:, dist_id].nonzero()
    ]
    if exp_treatment_1_Id.shape[1] == 0:
        df_exp_treatment_1_Id = pd.DataFrame(
            {"expression": [0, 0, 0], "treatment": treatment_1, "isoform": "Id"}
        )
    else:
        df_exp_treatment_1_Id = pd.DataFrame(
            {
                "expression": np.squeeze(np.asarray(exp_treatment_1_Id)),
                "treatment": treatment_1,
                "isoform": "Id",
            }
        )

    exp_treatment_2_Id = cells_treatment_2.X[:, dist_id][
        cells_treatment_2.X[:, dist_id].nonzero()
    ]
    if exp_treatment_2_Id.shape[1] == 0:
        df_exp_treatment_2_Id = pd.DataFrame(
            {"expression": [0, 0, 0], "treatment": treatment_2, "isoform": "Id"}
        )
    else:
        df_exp_treatment_2_Id = pd.DataFrame(
            {
                "expression": np.squeeze(np.asarray(exp_treatment_2_Id)),
                "treatment": treatment_2,
                "isoform": "Id",
            }
        )

    return pd.concat(
        [
            df_exp_treatment_1_Ip,
            df_exp_treatment_2_Ip,
            df_exp_treatment_1_Id,
            df_exp_treatment_2_Id,
        ]
    )


def compute_df_paired(
    utrs_filtered,
    transcript_id,
    gene_id,
    gtf_quantification,
    gtf_length,
    treatments=[],
    remove_overlaps=True,
    remove_length_outliers=False,
    treatment_col="treatment",
    to_keep=[],
    quantile_norm=False,
    expressed_in_all=True,
    offset_for_RUD=0.1,
):
    """Returns a DataFrame with pairs of reliably expressed isoforms,
    with the ids of the isoforms and their lengths. Are also reported their
    occurences and PUD score in each treatment.

    Args:
        utrs_filtered (AnnData): AnnData object representing the raw 3' UTR isoform expression data (filtered)
        transcript_id (str): name of the column in utrs_filtered where transcript ids are stored
        gene_id (str): name of the column in utrs_filtered where gene ids are stored
        gtf_quantification (_type_): path to the GTF file used for the quantification (i.e. focus in our case)
        gtf_length (str): path to the GTF file containing the length of the isoforms
        treatments (list, optional): list of treatments to analyze. Should be values contained
        in utrs_filtered[treatment_col].
        remove_overlaps (bool, optional): Whether to remove genes with overlapping annotations. Defaults to True.
        remove_length_outliers (bool, optional): Whether to remove outliers in length. Defaults to False.
        treatment_col (str, optional): Name of the column in the utrs_filtered.obs containing
        the treatment information. Defaults to 'treatment'.
        to_keep (list, optional): list of genes you want to keep for sure. Use it to avoid this genes to be filtered
        when removing the overlaps or the outliers. Defaults to [].
        quantile_norm (bool, optional): Whether to quantile normalize the counts. Defaults to False.
        expressed_in_all (bool, optional): Whether to keep only the pairs expressing at
        least one isoform of the pair in each of the treatments. Defaults to True.
        offset_for_RUD (float, optional): offset to add to the zero count to avoid
        NaN RUD values (RUD=log2(proximal_occurence/distal_occurence))

    Returns:
        pd.DataFrame: Each row represents a pair of proximal/distal isoforms. The columns are:
                        -'gene': name of the gene for this pair
                        -'proximal_id': name of the proximal isoform
                        -'proximal_length': length of the proximal isoform (in nt)
                        -'proximal_occurence_'+treatment_name: occurence of the proximal isoform in each treatment
                        -'distal_id': name of the distal isoform
                        -'distal_length': length of the distal isoform
                        -'distal_occurence_'+treatment_name: occurence of the distal isoform in each treatment
                        -'PUD_'+treatment_name: PUD for this pair in each treatment
                        (PUD: proximal_occurence/(proximal_occurence+distal_occurence))
    """
    if len(treatments) == 0:
        treatments = list(utrs_filtered.obs[treatment_col].value_counts().index)

    # drop unused columns created during the preprocessing
    for col in [
        "n_genes",
        "n_genes_by_counts",
        "total_counts",
        "total_counts_mt",
        "pct_counts_mt",
    ]:
        if col in utrs_filtered.obs.columns:
            utrs_filtered.obs.drop([col], axis=1, inplace=True)

    # Remove overlapping genes
    if remove_overlaps:
        overlaps = get_overlap_file(gtf_quantification)
        utrs_filtered = utrs_filtered[
            :,
            (
                (~utrs_filtered.var["gene"].isin(overlaps.index))
                | (utrs_filtered.var["gene"].isin(to_keep))
            ),
        ]

    # For each gene in the annotation file, get the proximal transcript and its length
    gtf_length["length"] = gtf_length["end"] - gtf_length["start"]
    if "length" not in utrs_filtered.var.columns:
        utrs_filtered.var = utrs_filtered.var.merge(
            gtf_length[["transcript_id", "length"]],
            left_index=True,
            right_on="transcript_id",
        ).set_index("transcript_id", drop=False)

    # Remove outliers isoforms (in terms of length)
    if remove_length_outliers:
        _, upper_length_outliers = get_outliers(utrs_filtered.var["length"])
        # outliers = list(upper_length_outliers.index)
        # outliers = ['ABCD3', 'UPK3B', 'ABCF3', 'PTTG2', 'ANKRD30A', 'AL583828.1', 'ZDHHC11', ]
        # outliers = utrs_filtered.var[utrs_filtered.var.index.isin(outliers)][gene_id]
        utrs_filtered = utrs_filtered[
            :,
            ~(utrs_filtered.var[transcript_id].isin(list(upper_length_outliers.index)))
            | (utrs_filtered.var[gene_id].isin(["KIFC"])),
        ]

    # Compute the pseudo-bulk based on the treatment column
    psbulk_treatment = aggregate_cells(
        utrs_filtered, group_column_name=treatment_col
    )  # compute the pseudo-bulk
    for ad in psbulk_treatment.values():
        ad.var["log_occurence"] = ad.var["occurence"].apply(
            lambda x: np.log2(x + 1)
        )  # log-transformation on the raw counts

    # Apply quantile normalization
    dict_to_seggregate = {}
    for treatment in treatments:
        dict_to_seggregate[treatment] = treatment
    if quantile_norm:
        psbulk_treatment = pseudo_bulk_quantile(
            psbulk_treatment, dict_to_seggregate=dict_to_seggregate
        )

    # Compute the table of pairs
    proximals = {}
    pairs = {}

    for condition in treatments:
        proximals[condition] = psbulk_treatment[condition].var.loc[
            psbulk_treatment[condition].var.groupby(gene_id).length.idxmin()
        ][
            [gene_id, transcript_id, "length", "occurence"]
        ]  # for each gene, find the proximal isoform based on the minimal length
        proximals[condition].index = proximals[condition][gene_id]
        psbulk_treatment[condition].var["is_proximal"] = (
            psbulk_treatment[condition]
            .var[transcript_id]
            .isin(proximals[condition][transcript_id])
        )  # for each isoform, label if it is a prox isoform or not
        psbulk_treatment[condition].var["proximal_occurence"] = (
            psbulk_treatment[condition]
            .var[gene_id]
            .apply(lambda gene: proximals[condition].loc[gene, "occurence"])
        )  # for each isoform, add the the proximal occurence
        psbulk_treatment[condition].var["proximal_length"] = (
            psbulk_treatment[condition]
            .var[gene_id]
            .apply(lambda gene: proximals[condition].loc[gene, "length"])
        )  # for each isoform, add the proximal length
        psbulk_treatment[condition].var["proximal_id"] = (
            psbulk_treatment[condition]
            .var[gene_id]
            .apply(lambda gene: proximals[condition].loc[gene, transcript_id])
        )  # for each isoform add the proximal id
        psbulk_treatment[condition].var["PUD"] = psbulk_treatment[condition].var[
            "proximal_occurence"
        ].astype(float) / (
            psbulk_treatment[condition].var["occurence"].astype(float)
            + psbulk_treatment[condition].var["proximal_occurence"].astype(float)
        )  # for each created pair, compute the PUD
        for occurence_col in ["proximal_occurence", "occurence"]:
            psbulk_treatment[condition].var[occurence_col + "_for_RUD"] = (
                psbulk_treatment[condition].var[occurence_col].replace(0, 0.1)
            )
        psbulk_treatment[condition].var["RUD"] = np.log2(
            psbulk_treatment[condition].var["proximal_occurence_for_RUD"].astype(float)
            / psbulk_treatment[condition].var["occurence_for_RUD"].astype(float)
        )
        pairs[condition] = (
            psbulk_treatment[condition]
            .var[~(psbulk_treatment[condition].var["is_proximal"])]
            .rename(
                columns={
                    transcript_id: "distal_id",
                    "length": "distal_length",
                    "occurence": "distal_occurence",
                }
            )
            .reset_index(drop=True)[
                [
                    gene_id,
                    "proximal_id",
                    "proximal_length",
                    "proximal_occurence",
                    "distal_id",
                    "distal_length",
                    "distal_occurence",
                    "PUD",
                    "RUD",
                ]
            ]
        )

    isoforms_paired = pairs[treatments[0]].merge(
        pairs[treatments[1]],
        right_on=[
            gene_id,
            "proximal_id",
            "proximal_length",
            "distal_id",
            "distal_length",
        ],
        left_on=[
            gene_id,
            "proximal_id",
            "proximal_length",
            "distal_id",
            "distal_length",
        ],
        suffixes=["_" + treatments[0], "_" + treatments[1]],
    )
    if len(treatments) > 2:
        for i in range(2, len(treatments)):
            isoforms_paired = isoforms_paired.merge(
                pairs[treatments[i]],
                right_on=[
                    gene_id,
                    "proximal_id",
                    "proximal_length",
                    "distal_id",
                    "distal_length",
                ],
                left_on=[
                    gene_id,
                    "proximal_id",
                    "proximal_length",
                    "distal_id",
                    "distal_length",
                ],
            ).rename(
                columns={
                    "proximal_occurence": "proximal_occurence_" + treatments[i],
                    "distal_occurence": "distal_occurence_" + treatments[i],
                    "PUD": "PUD_" + treatments[i],
                    "RUD": "RUD_" + treatments[i],
                }
            )

    if expressed_in_all:
        isoforms_paired = (
            isoforms_paired.dropna()
        )  # Remove pairs for which no isoform is detected in one of the conditions
    return isoforms_paired
