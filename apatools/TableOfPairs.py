#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileCopyrightText: Copyright © 2026 University of Bern
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@idiap.ch>
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@unibe.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import textwrap
import numpy as np
import seaborn as sns
import pandas as pd

import scipy.stats as stats
from scipy.stats import ttest_ind
from scipy.stats import levene
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests

from omicstools.helper_plot import plot_boxplot_significance
from omicstools.LMM_helper import lmer_general
from scipy.stats import norm



class TableOfPairs():
    """Table of proximal/distal isoform pairs with per-sample scores (PUD, ...), its per-gene aggregate, and statistics
    across treatments: score differences and variances, linear mixed models per pair, GO-term enrichment of the scores,
    Levene tests on the variances.
    """
    
    
    def __init__(self, 
                 df, 
                 treatments=None, 
                 gene_col='gene', 
                 agg_method='max', 
                 treatment_name='treatment', 
                 distal_id='distal_id',
                 treatments_dict=None,
                 samples_metadata=None,
                 compute_dscores = False):
        """
        Initialize the TableOfPairs object.

        Parameters:
        - df (pandas.DataFrame): The input DataFrame containing the table of pairs-
        - treatments (list): A list of treatments.
        - gene_col (str): The column name for the gene identifier. Default is 'gene'.
        - agg_method (str): The aggregation method to use when grouping by gene. Default is 'max'.
        - treatment_name (str): The column name for the treatment identifier. Default is 'treatment'.
        - distal_id (str): The column name for the distal identifier. Default is 'distal_id'.
        - treatments_dict (dict): A dictionary of treatments. If provided, the treatments parameter will be ignored.
        - samples_metadata (pandas.DataFrame): The metadata associated with the samples. Sample names in index should match the sample names
        from the table of pairs (PUD_samplename).

        Raises:
        - ValueError: If neither treatments nor treatments_dict is provided.

        """
        
        self.df = df

        if treatments is None:
            if treatments_dict is None:
                raise ValueError("Please provide either a list of treatments or a dictionary of treatments.")
            else:
                self.treatments = list(treatments_dict.keys())
        else:
            self.treatments = treatments

        self.treatments_dict = treatments_dict
        self.gene_col = gene_col            
        self.df_per_gene = self.df.groupby(self.gene_col).agg(agg_method).reset_index()

        self.treatment_name = treatment_name
        self.distal_id = distal_id
        self.samples_metadata = samples_metadata

        # Compute the variance of the PUD for each treatment
        if (self.treatments_dict is not None) and (compute_dscores):
            self.df = self.compute_var_score(df=self.df, score='PUD')
            self.df = self.compute_dscore(df=self.df, score='varPUD')
            self.df_per_gene = self.compute_var_score(df=self.df_per_gene, score='PUD')
            self.df_per_gene = self.compute_dscore(df=self.df_per_gene, score='varPUD')
            self.df = self.compute_dscore(df=self.df, score='PUD')
            self.df_per_gene = self.compute_dscore(df=self.df_per_gene, score='PUD')

    
    def compute_dscore(self, df, score='PUD'):   
        """Add the difference of a score between every two treatments: column `d<score>_<treatment2>_<treatment1>` =
        `<score>_<treatment2>` - `<score>_<treatment1>`.

        Args:
            df: table with one `<score>_<treatment>` column per treatment.
            score: score name (e.g. "PUD", "varPUD"). Defaults to "PUD".

        Returns:
            pd.DataFrame: `df` with the difference columns.
        """
        for treatment1 in self.treatments:
            for treatment2 in self.treatments:
                if treatment1 != treatment2:
                    df[f"d{score}_{treatment2}_{treatment1}"] = df[f"{score}_{treatment2}"] - df[f"{score}_{treatment1}"]
        return df
    
    def compute_var_score(self, df, score='PUD'):
        """Add, per treatment, the variance of a score across the treatment's samples (`var<score>_<treatment>` column),
        from the columns whose name contains both a sample name of `treatments_dict` and the score name. Returns `df`
        unchanged if no `treatments_dict` was given.

        Args:
            df: table with per-sample score columns.
            score: score name. Defaults to "PUD".

        Returns:
            pd.DataFrame: `df` with the variance columns.
        """
        if self.treatments_dict is None:
            return df
        else:
            for treatment, samples in self.treatments_dict.items():
                if f'var{score}_{treatment}' not in df.columns:

                    # Look for the columns of interests
                    col_treatments_score = []
                    for sample in samples:
                        for col in df.columns:
                            if (sample in col) & (score in col):
                                col_treatments_score.append(col)

                    # compute the variance for each treatments
                    df[f'var{score}_{treatment}'] = df[col_treatments_score].var(axis=1)
            return df
    
    def compute_LMM_one_pair(self, pair, fixed_effect, random_effects, score='PUD'):
        """Linear mixed model of a pair's per-sample score: score ~ fixed effect + random effects (R lme4, via
        `omicstools.LMM_helper.lmer_general`).

        Args:
            pair: index (row) of the pair in the table.
            fixed_effect: column of `samples_metadata` used as fixed effect.
            random_effects: list of columns of `samples_metadata` used as random effects (`fixed_effect` is appended to
                it).
            score: score name; the per-sample columns are `<score>_<sample>`. Defaults to "PUD".

        Returns:
            tuple: the ANOVA result and the p-value of the fixed effect, as returned by `lmer_general`.

        Raises:
            ValueError: if no `samples_metadata` was given.
        """
        tables_per_treatment = []

        for treatment in self.treatments_dict.keys():
            tables_per_treatment.append(self.df[[f"{score}_{t}" for t in self.treatments_dict[treatment] if f"{score}_{t}" in self.df.columns]].loc[pair])

        df = pd.DataFrame(pd.concat(tables_per_treatment, axis=0))
        df.columns = [score]
        df.index = [idx.split(f'{score}_')[1] for idx in df.index]

        if self.samples_metadata is None:
            raise ValueError("Please provide the metadata of samples containing the fixed and random effects to take into account to perform the LMM.")

        meta_col = random_effects
        meta_col.append(fixed_effect)
        df = df.merge(self.samples_metadata[meta_col], left_index=True, right_index=True)
        anova_result, p_val = lmer_general(df, score, fixed_effect, random_effects)
        return anova_result, p_val


    
    def compute_dscore_and_statest(self, 
                                 gene_list, 
                                 treatment1, 
                                 treatment2, 
                                 per_pair=True, 
                                 per_max_diff=False, 
                                 score='PUD', 
                                 variance_test=False):
        
        """
        Compute the difference in PUD (PUD_{treatment2} - PUD_{treatment1}) for a given gene list,
        and perform a t-test to compare it with the rest of the genes.

        Args:
            gene_list (list): List of genes to compute the difference in PUD for.
            treatment1 (str): Name of the first treatment.
            treatment2 (str): Name of the second treatment.

        Returns:
            tuple: A tuple containing the following values:
                - len(gene_list_dPUD) (int): Number of pairs found from the gene list.
                - len(other_dPUD) (int): Number of pairs not in the gene list.
                - t_statistic (float): The calculated t-statistic from the t-test.
                - p_value (float): The calculated p-value from the t-test.
        """
        if per_pair:
            df = self.df
        else:
            if per_max_diff:
                df = self.df.loc[self.df.groupby('gene')[f"d{score}_{treatment2}_{treatment1}"].idxmax()]
            else:
                df = self.df_per_gene

            

        dPUD = df[f'{score}_{treatment2}'] - df[f'{score}_{treatment1}']

        nb_gene_in_gene_list = len(set(gene_list) & set(df[self.gene_col].values))
        nb_gene_not_in_gene_list = len(set(df[self.gene_col].values) - set(gene_list))

        gene_list_dPUD = dPUD[df[self.gene_col].isin(gene_list)]
        other_dPUD = dPUD[~df[self.gene_col].isin(gene_list)]

        if abs(gene_list_dPUD.mean()) > abs(other_dPUD.mean()):
            big_diff = True
        else:
            big_diff = False

        nb_pair_in_gene_list = len(gene_list_dPUD)
        nb_pair_not_in_gene_list = len(other_dPUD)

        if variance_test:
            statistic, p_value = levene(gene_list_dPUD, other_dPUD)
        else:
            statistic, p_value = kruskal(gene_list_dPUD, other_dPUD)
        return nb_gene_in_gene_list, nb_gene_not_in_gene_list, nb_pair_in_gene_list, nb_pair_not_in_gene_list, statistic, p_value, big_diff

    def compute_dscore_for_GO(self, df_GO, gene_list_colname, 
                            treatment1, treatment2, per_pair=True,
                            per_max_diff=False, 
                            score='PUD',
                            variance_test=False):
        """
        Compute the dPUD (difference in PUD) for a given gene list in the context of Gene Ontology (GO) analysis.

        Args:
            df_GO (DataFrame): The DataFrame containing the gene list and other relevant information for GO analysis.
            gene_list_colname (str): The column name in `df_GO` that contains the gene list.
            treatment1 (str): The name of the first treatment.
            treatment2 (str): The name of the second treatment.
            pair_stats (str, optional): The type of pair statistics to compute. Defaults to 'all'.
            suffix (str, optional): A suffix to append to the column names in `df_GO`. Defaults to ''.

        Returns:
            DataFrame: The updated `df_GO` DataFrame with additional columns for dPUD, t-test results, and p-values.

        Raises:
            None

        Examples:
            # Example usage
            df_GO = compute_dPUD_for_GO(df_GO, 'gene_list', 'treatment1', 'treatment2', pair_stats='max_diff', suffix='_1')
        """

        # df[f"dPUD_{treatment2}_{treatment1}"] = df[f"PUD_{treatment2}"] - df[f"PUD_{treatment1}"]

        df_GO[f'd{score}_{treatment2}_{treatment1}_ttest'] = df_GO[gene_list_colname].apply(lambda x: self.compute_dscore_and_statest(gene_list=x, 
                                                                                                                              treatment1=treatment1, 
                                                                                                                              treatment2=treatment2, 
                                                                                                                              per_pair=per_pair,
                                                                                                                              per_max_diff=per_max_diff, 
                                                                                                                              score=score,
                                                                                                                              variance_test=variance_test))
        df_GO[f"gene_in_GO"] = df_GO[f'd{score}_{treatment2}_{treatment1}_ttest'].apply(lambda x: x[0])
        df_GO[f"gene_not_in_GO"] = df_GO[f'd{score}_{treatment2}_{treatment1}_ttest'].apply(lambda x: x[1])
        df_GO[f"pair_in_GO"] = df_GO[f'd{score}_{treatment2}_{treatment1}_ttest'].apply(lambda x: x[2])
        df_GO[f"pair_not_in_GO"] = df_GO[f'd{score}_{treatment2}_{treatment1}_ttest'].apply(lambda x: x[3])
        df_GO[f"p_value"] = df_GO[f'd{score}_{treatment2}_{treatment1}_ttest'].apply(lambda x: x[5])
        df_GO[f"bigger_diff"] = df_GO[f'd{score}_{treatment2}_{treatment1}_ttest'].apply(lambda x: x[6])
        df_GO.dropna(inplace=True)
        df_GO[f"p_value_adj"] = multipletests(df_GO[f"p_value"].values, alpha=0.05, method='fdr_bh', maxiter=1, is_sorted=False, returnsorted=False)[1]
        return df_GO
    
    
    def plot_score_distributions_for_GO(self, 
                                      df_GO, 
                                      GO_term, 
                                      GO_term_colname, 
                                      gene_list_colname, 
                                      description_colname, 
                                      treatments=None, 
                                      per_pair=True,
                                      colors_in_GO=sns.color_palette("Set2"), 
                                      with_background=True,
                                      score='PUD',
                                      boxplot=True, 
                                      stripplot=False, 
                                      densityplot=False):
        """
        Plots the PUD distributions for a given Gene Ontology (GO) term.

        Parameters:
        - df_GO (DataFrame): The DataFrame containing the GO annotations.
        - GO_term (str): The GO term of interest.
        - GO_term_colname (str): The column name in df_GO that contains the GO terms.
        - gene_list_colname (str): The column name in df_GO that contains the gene list for the GO term.
        - description_colname (str): The column name in df_GO that contains the description of the GO term.
        - treatments (list, optional): The list of treatments to consider. If not provided, all treatments will be used.
        - pair_stats (str, optional): The type of pair statistics to use. Default is 'max_diff'.
        - adjust_p_val (bool, optional): Whether to adjust the p-values. Default is True.
        - colors_in_GO (list, optional): The colors to use for the GO term. Default is sns.color_palette("Set2").

        Returns:
        None
        """
        
        if treatments is None:
            treatments = self.treatments

        if per_pair:
            df = self.df.copy()
        else:
            df = self.df_per_gene.copy()

        gene_list = df_GO[df_GO[GO_term_colname] == GO_term][gene_list_colname].values[0]
        gene_list_PUD = df[df[self.gene_col].isin(gene_list)][[f"{score}_{treatment}" for treatment in treatments]]
        other_PUD = df[~df[self.gene_col].isin(gene_list)][[f"{score}_{treatment}" for treatment in treatments]]
        pathway_name = df_GO[df_GO[GO_term_colname] == GO_term][description_colname].values[0]

        if with_background:
            plt.figure(figsize=(3*len(treatments), 5))
            positions = range(1, len(treatments) + 1)
            fig, ax = plt.subplots(1, 1)

            for i, treatment in enumerate(treatments):
                if boxplot:
                    bplot = plt.boxplot(other_PUD[f"{score}_{treatment}"].values, widths=0.3, positions=[positions[i]], 
                                        labels=[f"not in GO \n {len(other_PUD)}"], showfliers=False, 
                    whiskerprops={'color': 'black', 'linestyle': 'dashed'}, 
                    medianprops={'color': 'black', 'linewidth': 2.3}, 
                    boxprops={'linewidth': 1.5}, capprops={'color': 'black', 'linestyle': 'dashed'}, patch_artist=True)
                    bplot['boxes'][0].set_facecolor('gray')
                    bplot_in_GO = plt.boxplot(gene_list_PUD[f"{score}_{treatment}"].values, labels=[f"in GO \n {len(gene_list_PUD)}"], widths=0.3, positions=[positions[i] + 0.5], showfliers=False, 
                    whiskerprops={'color': 'black', 'linestyle': 'dashed'}, medianprops={'color': 'black', 'linewidth': 2.3}, boxprops={'linewidth': 1.5}, capprops={'color': 'black', 'linestyle': 'dashed'}, patch_artist=True)
                    bplot_in_GO['boxes'][0].set_facecolor(colors_in_GO[i])
                else:
                    # raise NotImplementedError("Violin plot not implemented yet")
                    vplot = ax.violinplot(other_PUD[f"{score}_{treatment}"].values, widths=0.5, positions=[2*i],
                                          showmedians=True, showextrema=False)
                    vplot['bodies'][0].set_facecolor('gray')
                    vplot['cmedians'].set_color(['gray'])
                    vplot_in_GO = ax.violinplot(gene_list_PUD[f"{score}_{treatment}"].values, widths=0.5, 
                                                 positions=[2*i+1], showmedians=True, showextrema=False)
                    vplot_in_GO['bodies'][0].set_facecolor(colors_in_GO[i])
                    vplot_in_GO['cmedians'].set_colors([colors_in_GO[i]])
                    ax.set_xticks(range(len(treatments)*2), labels=[f"not in GO \n {len(other_PUD)}", f"in GO \n {len(gene_list_PUD)}"]*len(treatments))
            plt.ylabel(f"{score}")
            sns.despine()
        else:
            plt.figure(figsize=(1*len(treatments), 5))
            df_plot = pd.melt(gene_list_PUD, var_name='treatment', value_name=f'{score}')
            df_plot['treatment'] = df_plot['treatment'].apply(lambda x: x.split(f'{score}_')[1])   
            # if variance_test:
            #     self.plot_variance_with_levene_stats(subdf=gene_list_PUD, score=score.split('var')[1], showfliers=False, 
            #                                          palette=colors_in_GO, stripplot=stripplot, log_scale=False, 
            #                                          fontsize=12, boxplot=boxplot)
            # else:
            if densityplot:
                sns.kdeplot(df_plot, x=f'{score}', hue='treatment', palette=colors_in_GO, fill=True, alpha=0.3)
            else:
                plot_boxplot_significance(df=df_plot, x='treatment', y=f'{score}', log_scale=False, 
                                            showfliers=False, palette=colors_in_GO, stripplot=stripplot, 
                                            stat='wilcoxon', boxplot=boxplot)
            plt.legend().remove()
        plt.ylabel(f"{score}")
        sns.despine()
        plt.title(f"{pathway_name.split(' - ')[0]} {GO_term} \n {df_GO[df_GO[GO_term_colname] == GO_term]['gene_in_GO'].values[0]} genes in GO", weight='bold')
        plt.show()

    def plot_variance_with_levene_stats(self, subdf, score='PUD', showfliers=False, palette=None,
                                        stripplot=False, log_scale=False, fontsize=12, boxplot=True):
        """Box (or violin) plots of the per-treatment variance of a score (`var<score>_<treatment>` columns of the
        table), with the Levene test p-value of every pair of treatments, computed on the `<score>_<treatment>` columns
        of `subdf`.

        Args:
            subdf: table with one `<score>_<treatment>` column per treatment, used for the Levene tests.
            score: score name. Defaults to "PUD".
            showfliers: draw the outliers of the box plots. Defaults to False.
            palette: colours per treatment.
            stripplot: also draw the individual values. Defaults to False.
            log_scale: logarithmic y-axis. Defaults to False.
            fontsize: font size of the annotations. Defaults to 12.
            boxplot: box plots if True, violin plots otherwise. Defaults to True.
        """
        from itertools import combinations
        from statannotations.Annotator import Annotator

        pairs = list(combinations(self.treatments, 2))
        p_values = []
        for p1, p2 in pairs:
            p_values.append(levene(list(subdf[f'{score}_{p1}'].values), list(subdf[f'{score}_{p2}'].values))[1])

        df_plot = pd.melt(self.df[[f"var{score}_{treatment}" for treatment in self.treatments]], var_name=self.treatment_name, value_name=f'var{score}')
        df_plot[self.treatment_name] = df_plot[self.treatment_name].apply(lambda x: x.split(f'var{score}_')[1])   

        formatted_pvalues = [f"P={pvalue:.2e}" for pvalue in p_values]

        plotting_parameters = {"data": df_plot, "x": self.treatment_name, "y": f'var{score}', "linewidth": 2.3, 'palette': palette}

        if boxplot:
            ax = sns.boxplot(**plotting_parameters, showfliers=showfliers)
        else:
            ax = sns.violinplot(**plotting_parameters)
        if stripplot:
            sns.stripplot(**plotting_parameters, ax=ax, hue=self.treatment_name)

        if log_scale:
            ax.set_yscale('log')

        annotator = Annotator(ax, pairs, **plotting_parameters)
        annotator.configure(fontsize=fontsize)

        annotator.set_custom_annotations(formatted_pvalues)
        annotator.annotate()
        plt.ylabel(f"var{score}")
        sns.despine()
        plt.xticks(rotation=30)
        

        

    def plot_mean_score_one_gene_all_pairs(self, gene_name, score='PUD'):
        """Bar plot of the mean score of every pair (distal isoform) of one gene, per treatment.

        Args:
            gene_name: gene to plot.
            score: score name. Defaults to "PUD".
        """
        # TODO: should be modified: self.treatments is a list (see __init__), so .keys() fails; iterate over the list.
        df_pairs = self.df[self.df[self.gene_col].isin([gene_name])].set_index(self.distal_id)[[f"{score}_{treatment}" for treatment in self.treatments.keys()]]

        df_plot = pd.melt(df_pairs, var_name=self.treatment_name, value_name=score)
        df_plot[self.treatment_name] = df_plot[self.treatment_name].apply(lambda x: x.split(f'{score}_')[1])
        df_plot['distal_id'] = list(df_pairs.index.values)*df_pairs.shape[1]

        # for the sake of the plot; rename the distal ids
        plt.figure(figsize=(5, 5))
        sns.barplot(df_plot, x='distal_id', y=score, hue=self.treatment_name)
        plt.xticks(rotation=90)
        sns.despine()
        plt.title(f"Mean {score} across samples for {gene_name} pairs", weight='bold', fontsize=15)
        sns.move_legend(plt.gca(), "upper right", bbox_to_anchor=(1.5, 1), title=self.treatment_name)   


    def inGO_vs_NOTinGO(self, 
                        genes_in_GO,
                        sample,
                        per_pair=True,
                        score='PUD',
                        bootstrap=False,
                        bootstrap_iterations=100):
        """
        Compare the values of a given score between genes in GO and genes not in GO.

        Parameters:
        - genes_in_GO (list): List of genes a given GO pathway.
        - sample (str): Name of the sample to compare.
        - per_pair (bool): If True, compare the values per pair. If False, compare the values per gene (mean across the pairs).
        - score (str): Name of the score to compare.

        Returns:
        - in_GO (str): Indicates whether the score is increased or decreased in genes in GO compared to genes not in GO.
        - s (float): Mann-Whitney U statistic.
        - p (float): Two-sided p-value.

        Note:
        - If either the list of genes in GO or the list of genes not in GO is empty, the function returns 'NA' for all outputs.
        """
        if per_pair:
            df = self.df.copy()
        else:
            df = self.df_per_gene.copy()

        gene_list = df[df[self.gene_col].isin(genes_in_GO)][f"{score}_{sample}"]
        other = df[~df[self.gene_col].isin(genes_in_GO)][f"{score}_{sample}"]
        
        print(f"Number of genes in GO: {len(gene_list)}")
        print(f"Number of genes not in GO: {len(other)}")
        
        if len(gene_list) == 0 or len(other) == 0:
            return 'NA', 'NA', 'NA'
        
        if bootstrap:
            mean_gene_exp = []
            for i in range(bootstrap_iterations):
                if len(gene_list) > len(other):
                    other_boot = other.sample(len(other), replace=True)
                else:
                    other_boot = other.sample(len(gene_list), replace=True)
                mean_gene_exp.append(other_boot.mean())
                
            if gene_list.mean() > np.array(mean_gene_exp).mean():
                in_GO = 'increase'
            else:
                in_GO = 'decrease'    
            
        
            # Z-test
            z_score = (gene_list.mean() - np.array(mean_gene_exp).mean()) / np.array(mean_gene_exp).std()

            # Calculate the p-value
            p = 2 * (1 - norm.cdf(abs(z_score)))
            
            return in_GO, z_score, p
                

        else:

            s, p = mannwhitneyu(gene_list, other)

            if gene_list.mean() > other.mean():
                in_GO = 'increase'
            else:
                in_GO = 'decrease'

            return in_GO, s, p


    def inGO_vs_NOTinGO_one_sample_all_terms(self, 
                                            df_GO,
                                            GO_term_colname, 
                                            GO_description_colname,
                                            sample,
                                            gene_list_colname, 
                                            per_pair=True,
                                            score='PUD',
                                            bootstrap=False,
                                            bootstrap_iterations=100):
        """
        Compare the values of a given score between genes in GO and genes not in GO
         for a given sample and for each GO term in the GO dataframe.
        
        Args:
            df_GO (pandas.DataFrame): The dataframe containing the GO terms and gene lists.
            GO_term_colname (str): The column name for the GO terms in the dataframe.
            GO_description_colname (str): The column name for the GO term descriptions in the dataframe.
            sample (str): The name of the sample to be used in the calculations.
            gene_list_colname (str): The column name for the gene lists in the dataframe.
            per_pair (bool, optional): Whether to calculate the test and p-value per pair. Defaults to True.
            score (str, optional): The score to be used in the calculations. Defaults to 'PUD'.
        
        Returns:
            pandas.DataFrame: A dataframe containing the GO terms, descriptions, Mann-Whitney U test results,
            p-values, and score values for the given sample.
        """
        
        results = pd.DataFrame()
        results['Term'] = df_GO[GO_term_colname]
        results['Description'] = df_GO[GO_description_colname]

        results[f'mannwhitneyu_{sample}'] = df_GO.apply(lambda row: self.inGO_vs_NOTinGO(genes_in_GO=row[gene_list_colname],
                                                                                        sample=sample, 
                                                                                        per_pair=per_pair,
                                                                                        score=score,
                                                                                        bootstrap=bootstrap,
                                                                                        bootstrap_iterations=bootstrap_iterations), axis=1)
        results[f'pvalue_{sample}'] = results[f'mannwhitneyu_{sample}'].apply(lambda x: x[2])
        results[f'{score}_in_GO_{sample}']  = results[f'mannwhitneyu_{sample}'].apply(lambda x: x[0])

        return results

    def inGO_vs_NOTinGO_all_samples_all_terms(self,
                                              df_GO,
                                              samples,
                                              GO_term_colname,
                                              GO_description_colname,
                                              gene_list_colname,
                                              per_pair=True,
                                              score='PUD',
                                              bootstrap=False,
                                              bootstrap_iterations=100):
        """
        Compare the values of a given score between genes in GO and genes not in GO
         for a all samples and for each GO term in the GO dataframe.

        Parameters:
        - df_GO (DataFrame): The DataFrame containing the GO terms and their descriptions.
        - samples (list): The list of samples to analyze.
        - GO_term_colname (str): The column name in df_GO that contains the GO terms.
        - GO_description_colname (str): The column name in df_GO that contains the GO term descriptions.
        - gene_list_colname (str): The column name in the sample DataFrame that contains the gene lists.
        - per_pair (bool): Whether to calculate the comparison per gene pair or per gene.
        - score (str): The score to use for the comparison.

        Returns:
        - DataFrame: The concatenated DataFrame containing the results for all samples and all terms.
        """
        df_results = []

        for i, sample in enumerate(samples): 
            df_res = self.inGO_vs_NOTinGO_one_sample_all_terms(df_GO,
                                                        GO_term_colname,
                                                        GO_description_colname,
                                                        sample,
                                                        gene_list_colname,
                                                        per_pair=per_pair,
                                                        score=score,
                                                        bootstrap=bootstrap,
                                                        bootstrap_iterations=bootstrap_iterations)
            if i > 0:
                df_res.drop(['Term', 'Description'], axis=1, inplace=True)
            df_results.append(df_res)
        
        return pd.concat(df_results, axis=1)
        
