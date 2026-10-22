#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileCopyrightText: Copyright © 2026 University of Bern
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@idiap.ch>
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@unibe.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
import pandas as pd

from apatools.scapa import get_isoform_exp_across_cells


def get_modality_pcts(
    cdata,
    pair_name=None,
    prox_id=None,
    dist_id=None,
    treatment_col="treatment",
    count_nans=False,
):
    """For a given pair, get the percentage of cells in each modality for each treatment.

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

    Raises:
        ValueError: You need to enter either a valid pair name, or two valid prox_id and dist_id.

    Returns:
        dict: dictionnary with the percentage for each modality in each treatment.
        Keys of the dictionnary are the treatments.
        Values are dictionnary with the following keys: ['Ip & Id', 'Ip only', 'Id only', 'no Ip & no Id'].
    """
    if pair_name is None and (prox_id is None or dist_id is None):
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

    treatments = list(set(list(cdata.obs[treatment_col].values)))
    modality_pcts = {}
    for treatment in treatments:
        cdata_treatment = cdata[cdata.obs[treatment_col] == treatment, :]
        cells_Ip = cdata_treatment.X[:, prox_id].nonzero()[0]
        cells_Id = cdata_treatment.X[:, dist_id].nonzero()[0]
        nb_cells_expressing_both = len(list(set(cells_Ip) & set(cells_Id)))
        nb_cells_expressing_Ip_only = len(cells_Ip) - nb_cells_expressing_both
        nb_cells_expressing_Id_only = len(cells_Id) - nb_cells_expressing_both
        if count_nans:
            nb_cells_total = len(cdata_treatment)
            nb_cells_none = (
                nb_cells_total
                - nb_cells_expressing_both
                - nb_cells_expressing_Ip_only
                - nb_cells_expressing_Id_only
            )
            modality_pcts[treatment] = {
                "Ip & Id": nb_cells_expressing_both * 100 / nb_cells_total,
                "Ip only": nb_cells_expressing_Ip_only * 100 / nb_cells_total,
                "Id only": nb_cells_expressing_Id_only * 100 / nb_cells_total,
                "no Ip & no Id": nb_cells_none * 100 / nb_cells_total,
            }
        else:
            nb_cells_expressing_the_pair = (
                nb_cells_expressing_both
                + nb_cells_expressing_Ip_only
                + nb_cells_expressing_Id_only
            )
            modality_pcts[treatment] = {
                "Ip & Id": nb_cells_expressing_both
                * 100
                / nb_cells_expressing_the_pair,
                "Ip only": nb_cells_expressing_Ip_only
                * 100
                / nb_cells_expressing_the_pair,
                "Id only": nb_cells_expressing_Id_only
                * 100
                / nb_cells_expressing_the_pair,
                "no Ip & no Id": None,
            }
    return modality_pcts


def pct_cells_per_modality(
    adata_utrs_filtered,
    table_of_pairs,
    treatment_col="treatment",
    prox_id="proximal_id",
    dist_id="distal_id",
    count_nans=False,
):
    """From a table of pairs, compute the distributions of percentage of cells expressing proximal,
    distal or both isoforms in each condition.

    Args:
        adata_utrs_filtered (AnnData): Single-cell expression data of 3' UTR isoforms that has been previously filtered.
        table_of_pairs (pd.DataFrame): DataFrame containing pairs in rows and their proximal and distal isoforms
        names in columns
        treatment_col (str, optional): Name of the column of adata_utrs_filtered.obs containing the information
        about the treatment of the cells.
        prox_id (str, optional): Name of the column of *table_of_pairs* containing the proximal isoforms names.
        dist_id (str, optional):Name of the column of *table_of_pairs* containing the distal isoforms names.

    Returns:
        pd.DataFrame: DataFrame containing each pair of table_of_pairs, in each treatment in rows and
                    the percentage of cells, the modality and the treatment in columns.
    """
    table_of_pairs["modality_pcts"] = table_of_pairs.apply(
        lambda pair: get_modality_pcts(
            adata_utrs_filtered,
            prox_id=pair[prox_id],
            dist_id=pair[dist_id],
            count_nans=count_nans,
        ),
        axis=1,
    )

    dfs_shift = []
    for treatment in adata_utrs_filtered.obs[treatment_col].value_counts().keys():
        table_of_pairs["Ip only"] = table_of_pairs["modality_pcts"].apply(
            lambda x: x[treatment]["Ip only"]
        )
        table_of_pairs["Id only"] = table_of_pairs["modality_pcts"].apply(
            lambda x: x[treatment]["Id only"]
        )
        table_of_pairs["Ip & Id"] = table_of_pairs["modality_pcts"].apply(
            lambda x: x[treatment]["Ip & Id"]
        )

        df = pd.melt(
            table_of_pairs[["Ip only", "Id only", "Ip & Id"]],
            value_name="Percentage of cells [%]",
            var_name="Modality",
        )
        df["treatment"] = treatment
        dfs_shift.append(df)
    return pd.concat(dfs_shift)


def get_distributions_pct_per_modality(
    table_of_pairs, utr_filtered, prox_id_col="proximal_id", dist_id_col="distal_id"
):
    """Returns a dataframe representing the percentage of cells expressing a given isoform  in a given condition,
    for each
    pair of the table_of_pairs. Designed for plotting.

    Args:
        table_of_pairs (pd.DataFrame): table of pairs you want the distributions for.
        utr_filtered (AnnData): Single cell 3' UTR expression data filtered.
        prox_id_col (str, optional): Name of the column in *table_of_pairs* containing the proximal ids (int).
        Defaults to 'proximal_id'.
        dist_id_col (str, optional): Name of the column in *table_of_pairs* containing the distal ids (int).
        Defaults to 'distal_id'.

    Returns:
        pd.DataFrame: Each row represent an isoform of a gene in *table_of_pairs* in a given treatment condition.
        Total number of rows in the number of pairs in *table_of_pairs* x number of different isoforms
        (2: distal and proximal) x the number of different treatment.
        The columns are as follow:
            - Isoform: contains the isoform type
            - Percentage of cells [%]: contains the percentage of cells expressing this isoform in the given treatment
            - Treatment: the treatment condition
    """
    table_of_pairs["pcts"] = table_of_pairs.apply(
        lambda pair: get_isoform_exp_across_cells(
            cdata=utr_filtered, prox_id=pair[prox_id_col], dist_id=pair[dist_id_col]
        ),
        axis=1,
    )

    table_of_pairs["pct_par_Ip"] = table_of_pairs["pcts"].apply(lambda x: x[0])
    table_of_pairs["pct_res_Ip"] = table_of_pairs["pcts"].apply(lambda x: x[1])
    table_of_pairs["pct_par_Id"] = table_of_pairs["pcts"].apply(lambda x: x[2])
    table_of_pairs["pct_res_Id"] = table_of_pairs["pcts"].apply(lambda x: x[3])

    df_expression = pd.melt(
        table_of_pairs[["pct_par_Ip", "pct_res_Ip", "pct_par_Id", "pct_res_Id"]],
        value_name="Percentage of cells [%]",
        var_name="Isoform",
    )
    df_expression["treatment"] = df_expression["Isoform"].apply(
        lambda x: "parental" if "par" in x else "resistant"
    )

    df_expression.replace("pct_par_Ip", "Ip", inplace=True)
    df_expression.replace("pct_res_Ip", "Ip", inplace=True)
    df_expression.replace("pct_par_Id", "Id", inplace=True)
    df_expression.replace("pct_res_Id", "Id", inplace=True)
    return df_expression


def get_is_Ip_Id_pca(utrs_pairs, utrs, pair_name, treatment_col="treatment"):
    """First 3 principal components of every cell, with the detection status of the two isoforms of one pair.

    Args:
        utrs_pairs: AnnData with a PCA (`obsm["X_pca"]`) and the treatment column in `obs`.
        utrs: AnnData of isoform counts, same cells as `utrs_pairs`.
        pair_name: pair as "<proximal_id>/<distal_id>" (isoform names of `utrs.var`).
        treatment_col: treatment column of `utrs_pairs.obs`. Defaults to "treatment".

    Returns:
        pd.DataFrame: one row per cell with `PC1`-`PC3`, the treatment, `is Ip`, `is Id` and `mod` ("None", "Ip", "Id"
        or "Ip&Id").

    Raises:
        ValueError: if the two AnnData objects do not have the same number of cells.
    """
    if len(utrs_pairs) != len(utrs):
        raise ValueError(
            "Both AnnData object should be of the same length; meaning they should contain the same number of cells"
        )
    else:
        ip, id = pair_name.split("/")
        idx_p = list(utrs.var.index).index(ip)
        idx_d = list(utrs.var.index).index(id)
        df_pca = pd.DataFrame(
            utrs_pairs.obsm["X_pca"][:, :3],
            index=utrs_pairs.obs.index,
            columns=["PC" + str(i + 1) for i in range(3)],
        )
        df_pca[treatment_col] = utrs_pairs.obs[treatment_col].values
        df_pca["is Ip"] = (utrs.X[:, idx_p] != 0).toarray().ravel()
        df_pca["is Id"] = (utrs.X[:, idx_d] != 0).toarray().ravel()
        df_pca["mod"] = "None"
        df_pca["mod"] = df_pca.apply(
            lambda row: "Ip" if row["is Ip"] else row["mod"], axis=1
        )
        df_pca["mod"] = df_pca.apply(
            lambda row: "Id" if row["is Id"] else row["mod"], axis=1
        )
        df_pca["mod"] = df_pca.apply(
            lambda row: "Ip&Id" if (row["is Ip"] and row["is Id"]) else row["mod"],
            axis=1,
        )
        return df_pca
