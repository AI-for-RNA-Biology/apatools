#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileCopyrightText: Copyright © 2026 University of Bern
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@idiap.ch>
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@unibe.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
import pickle
import sys
import os
import numpy as np
import scanpy as sc
import pandas as pd
import argparse
import yaml
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, pearsonr, fisher_exact
from joblib import Parallel, delayed
from gtfparse import read_gtf
import warnings
sys.path.insert(1, "/".join(os.getcwd().split('/')[:-1]))
from cliptools.processing import get_fisher_table, create_regions_around_PAS, is_cross_linked_per_region, get_bootstrap_distribution_one_region
warnings.filterwarnings('ignore')


def welch_between_PUD(df_iso, table_of_pairs, iso_type='proximal', treatments=['parental', 'resistant']):
    """Compute welch test between the PUD distribution of 1) the pairs of isoforms for which the *iso_type* isoform exhibit 
    cross-link event with the RBP  and 2) the pairs of isoforms for which the *iso_type* isoform exhibit no cross-link event 
    with the RBP. 

    Args:
        df_iso (pd.DataFrame): DataFrame containing isoforms in rows (either the proximals, either the distals) ['transcript_id', 'Region1', 'Region2'...]. 
        Should contain:
                        -one column per region containing information on wheter the RBP cross-link or not for each region 
                        surrounding the isoform. This is the output of is_cross_linked_per_region. 
                        -one column named transcript_id which should contain the isoform names
        table_of_pairs (pd.DataFrame): DataFrame containing all the analyzed pairs, along with their PUD statistics. Rows represent pairs 
        and columns are as follows:
                                -proximal_name: names of the proximal isoforms of the pairs
                                -distal_name: names of the distal isoforms of the pairs
                                -'PUD_*treatment1_name*': PUD values for each pair in the condition *treatment1_name*
                                -'PUD_*treatment2_name*':PUD values for each pair in the condition *treatment2_name*
                                
        The columns names required to fit with default arguments are: ['proximal_name', 'PUD_parental', 'PUD_resistant']
                                   -
                                    
        iso_type (str, optional): the isoforms you want to analyze. Either 'proximal', either 'distal'. Defaults to 'proximal'.
        treatment1_name (str, optional): Name of the first treatment. Defaults to 'parental'.
        treatment2_name (str, optional): Name of the second treatment. Defaults to 'resistant'.
    Returns:
        pd.DataFrame: returns a pandas DataFrame representing the RBP, with regions in rows and in columns:
                    -cross_link: list of *iso_type* isoforms exhibiting a cross link with the RBP in this region
                    -no_cross_link: list of *iso_type* isoforms exhibiting no cross link with the RBP in this region
                    -welch_*treatment1_name*: welch statistics when analyzing PUD differences in the condition *treatment1_name*
                    -welch_*treatment2_name*: welch statistics when analyzing PUD differences in the condition *treatment2_name*
                    -log10(pval)*dPUD_*treatment1_name: -log10(pvalue) of the welch test, multiplied by the sign of the difference of mean PUD 
                    of the cross-linked group and the mean PUD of the non-cross linked group in the condition *treatment1_name*
                    -log10(pval)*dPUD_*treatment2_name: -log10(pvalue) of the welch test, multiplied by the sign of the difference of mean PUD 
                    of the cross-linked group and the mean PUD of the non-cross linked group in the condition *treatment2_name*

    """

    nb_regions = len(df_iso.loc[0]['regions'])

    welch_cols = ['welch_'+treatment for treatment in treatments]
    log10_cols = ['-log10(pval)*dPUD_'+treatment for treatment in treatments]
    welch_cross_linked_btw_treatments_col = [f'welch_cross_linked_{treatments[t1]}_{treatments[t2]}' for t1 in range(len(treatments)) for t2 in range(t1+1, len(treatments))]
    log10_cross_linked_btw_treatments_col = [f'-log10(pval)*dPUD_cross_linked_{treatments[t1]}_{treatments[t2]}' for t1 in range(len(treatments)) for t2 in range(t1+1, len(treatments))]
    welch_non_cross_linked_btw_treatments_col = [f'welch_non_cross_linked_{treatments[t1]}_{treatments[t2]}' for t1 in range(len(treatments)) for t2 in range(t1+1, len(treatments))]
    log10_non_cross_linked_btw_treatments_col = [f'-log10(pval)*dPUD_non_cross_linked_{treatments[t1]}_{treatments[t2]}' for t1 in range(len(treatments)) for t2 in range(t1+1, len(treatments))]
    all_cols_for_df = ['cross_link', 'no_cross_link'] + welch_cols + log10_cols + welch_cross_linked_btw_treatments_col + log10_cross_linked_btw_treatments_col + welch_non_cross_linked_btw_treatments_col + log10_non_cross_linked_btw_treatments_col

    if len(treatments) >= 2:
        df_cross_linked = pd.DataFrame(columns=all_cols_for_df)
        # Check dtype of each column
    else:
        df_cross_linked = pd.DataFrame(columns=['cross_link', 'no_cross_link'])

    for i in range(nb_regions):
        df_cross_linked.loc['Region'+str(i+1), 'cross_link'] = df_iso[df_iso['Region'+str(i+1)] == True]['transcript_id'].values
        df_cross_linked.loc['Region'+str(i+1), 'no_cross_link'] = df_iso[df_iso['Region'+str(i+1)] == False]['transcript_id'].values

    if len(treatments) >= 2:
    
        if f"{iso_type}_name" in table_of_pairs.columns:
            suffix = 'name'
        else:
            suffix = 'id'
        for idx, region in df_cross_linked.iterrows():
            cross_linked = table_of_pairs[table_of_pairs[f"{iso_type}_{suffix}"].isin(region['cross_link'])]
            non_cross_linked = table_of_pairs[table_of_pairs[f"{iso_type}_{suffix}"].isin(region['no_cross_link'])]

            # dPUD between cross-linked and non cross-linked per treatment
            for treatment in treatments:
                df_cross_linked.at[region.name, 'welch_'+treatment] = ttest_ind(cross_linked['PUD_'+treatment], non_cross_linked['PUD_'+treatment], equal_var=False)
                PUD_diff = cross_linked['PUD_'+treatment].mean() - non_cross_linked['PUD_'+treatment].mean()
                df_cross_linked.at[region.name, '-log10(pval)*dPUD_'+treatment] = -np.log10(df_cross_linked.loc[region.name, 'welch_'+treatment][1])*np.sign(PUD_diff)

            for t1 in range(len(treatments)):
                for t2 in range(t1+1, len(treatments)):
                    if treatments[t1] != treatments[t2]:
                        #dPUD of cross-linked between the two treatments
            
                        
                        s_cr, p_cr = ttest_ind(cross_linked['PUD_'+treatments[t1]], 
                                         cross_linked['PUD_'+treatments[t2]], 
                                         equal_var=False)
                        df_cross_linked.loc[region.name, f'welch_cross_linked_{treatments[t1]}_{treatments[t2]}'] = (s_cr, p_cr)
                        PUD_diff_cross_linked_t1_t2 = cross_linked['PUD_'+treatments[t2]].mean() - cross_linked['PUD_'+treatments[t1]].mean()
                        df_cross_linked.loc[region.name, 
                                            f'-log10(pval)*dPUD_cross_linked_{treatments[t1]}_{treatments[t2]}'] = -np.log10(p_cr)*np.sign(PUD_diff_cross_linked_t1_t2) # if positive, PUD_cross_link_treatment2 > PUD_cross_link_treatment1
                        
                        #dPUD of non cross-linked between the two treatments
                        s_ncr, p_ncr = ttest_ind(non_cross_linked['PUD_'+treatments[t1]], 
                                                 non_cross_linked['PUD_'+treatments[t2]],
                                                 equal_var=False)
                        df_cross_linked.loc[region.name, f'welch_non_cross_linked_{treatments[t1]}_{treatments[t2]}'] = (s_ncr, p_ncr)
                        PUD_diff_non_cross_linked_t1_t2 = non_cross_linked['PUD_'+treatments[t2]].mean() - non_cross_linked['PUD_'+treatments[t1]].mean()
                        df_cross_linked.loc[region.name,
                                            f'-log10(pval)*dPUD_non_cross_linked_{treatments[t1]}_{treatments[t2]}'] = -np.log10(p_ncr)*np.sign(PUD_diff_non_cross_linked_t1_t2) # if positive, PUD_non_cross_link_treatment2 > PUD_non_cross_link_treatment1

    return df_cross_linked



def process_one_RBP(filepath_to_CLIP, filepath_to_gtf_isoforms, filepath_to_table_of_pairs, 
                    filepath_to_sig_pairs, path_to_results, nt_after, nt_before, length_region, 
                    pair_key=None, iso_type='proximal', treatments=['parental', 'resistant'],
                    split_name_before='', split_name_after='', split_number_before=1, split_number_after=1, bootstrap=True, 
                    only_long=False):

    """Process a .gff file containing the CLIP data for one RBP.
    Will save a .pkl file containing a pd.DataFrame with regions surrounding the PAS is rows and 
    in columns:
                -cross_link: list of *iso_type* isoforms exhibiting a cross link with the RBP in this region
                -no_cross_link: list of *iso_type* isoforms exhibiting no cross link with the RBP in this region
                -welch_*treatment1_name*: welch statistics when analyzing PUD differences in the condition *treatment1_name*
                -welch_*treatment2_name*: welch statistics when analyzing PUD differences in the condition *treatment2_name*
                -log10(pval)*dPUD_*treatment1_name: -log10(pvalue) of the welch test, multiplied by the sign of the difference of mean PUD 
                of the cross-linked group and the mean PUD of the non-cross linked group in the condition *treatment1_name*
                -log10(pval)*dPUD_*treatment2_name: -log10(pvalue) of the welch test, multiplied by the sign of the difference of mean PUD 
                of the cross-linked group and the mean PUD of the non-cross linked group in the condition *treatment2_name*
                -fisher_table: 2D contingency table containing to test for the association between cross-link event of the RBP and significant shift
                -fisher_results: results of the fisher exact test on the contingency table defined in the column *fisher_table*

    Args:
        filepath_to_CLIP (string): path to the .gff containing the CLIP data for the given RBP
        filepath_to_gtf_isoforms (string): path to the .gtf file describing the 3' UTR isoforms. Should contain the 3' UTR in rows and 
        the following columns:
                                -'transcript_id' (string): names of the transcripts (isoforms)
                                -'start' (int): start nucleotide for the transcript
                                -'end' (int): end nucleotide for the transcript 
                                -'strand' (string): '+' or '-'
        filepath_to_table_of_pairs (string): path to the .csv file containing the created proximal/distal pairs. Should contain the pairs in rows, 
        and should contain in columns:
                                -'proximal_id' (string): names of the proximal isoforms of the pairs
                                -'distal_id' (string): names of the distal isoforms of the pairs
                                -'PUD_*treatment1_name*' (float): PUD values for each pair in the condition *treatment1_name*
                                -'PUD_*treatment2_name*' (float): PUD values for each pair in the condition *treatment2_name*
                        
        filepath_to_sig_pairs (string): path to the .csv file containing the identified significant shifts. If you want to test the association
        between cross-link event on the *iso_type* isoform and proximal shift, please provide the list of significant pairs towrds proximal shift. 
        If you want to test for the association between cross-link event on the *iso_type* isoform and distal shift, please provide the pairs
        exhibiting a significant shift towards distal use. This .csv contains pairs in rows and 
        the columns should be:
                                -'proximal_id': names of the proximal isoforms of the significant pairs
                                -'distal_id': names of the distal isoforms of the significant pairs
        path_to_results (string): path where you want to store the results. 
        nt_after (int, optional): Number of nucleotides after (downstream) the PAS to analyze. 
        nt_before (int, optional): Number of nucleotides before (upstream) the PAS to analyze. 
        length_region (int, optional): Length of the regions to analyze. 
        iso_type (str, optional): _description_. Defaults to 'proximal'.
        treatment1_name (str, optional): _description_. Defaults to 'parental'.
        treatment2_name (str, optional): _description_. Defaults to 'resistant'.
    """
    if pair_key is None:
        pair_key = 'distal' if iso_type == 'stop_codon' else iso_type

    # Load the data
    print("I will process one RBP...", flush=True)
    isoforms = read_gtf(filepath_to_gtf_isoforms)
    print("GTF isoforms loaded", flush=True)
    table_of_pairs = pd.read_csv(filepath_to_table_of_pairs)
    print("table of pairs loaded", flush=True)
    if only_long: 
        table_of_pairs = table_of_pairs.loc[table_of_pairs.groupby('gene')['length'].idxmax()]

    #clip = read_gtf(filepath_to_CLIP)
    clip = pd.read_csv(filepath_to_CLIP, sep='\t', names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"])
    print("clip loaded", flush=True)

    if f"{pair_key}_name" in table_of_pairs.columns:
        suffix = 'name'
    else:
        suffix = 'id'

    # keep only the isoforms contained in the pairs, keep only the distals or proximals depending on the argument iso_type
    isoforms = isoforms[isoforms['transcript_id'].isin(table_of_pairs[f"{pair_key}_{suffix}"].values)].reset_index(drop=True)
    print("isoforms filtered", flush=True)
    print("creating the regions around the PAS....", flush=True)
    
    # Create regions around the PAS for each isoform
    if iso_type == 'stop_codon':
        # Anchor on the 3' UTR start (right after the stop codon): start on + strand, end on - strand.
        # create_regions_around_PAS uses `end` on + and `start` on -, so pass the anchor as both.
        isoforms['regions'] = isoforms.apply(lambda row: create_regions_around_PAS(
                             row['start'] if row['strand'] == '+' else row['end'],
                             row['start'] if row['strand'] == '+' else row['end'],
                             row['strand'],
                             nt_after=nt_after,
                             nt_before=nt_before,
                             length_region=length_region),
                             axis=1)
    else:
        isoforms['regions'] = isoforms.apply(lambda row: create_regions_around_PAS(row['start'],
                             row['end'],
                             row['strand'],
                             nt_after=nt_after,
                             nt_before=nt_before,
                             length_region=length_region),
                             axis=1)
   
    # Define the regions of where the RBP binds
    clip.dropna(inplace=True)


    # clip['link_regions'] = clip.apply(lambda row: pd.Interval(row['start'], row['end']), axis=1)

    RBP_regions = {}
    for strand in clip['strand'].value_counts().index:
        RBP_regions[strand] = {}
        for chrom in clip['seqname'].value_counts().index:
            if chrom.startswith('chr'):
                RBP_regions[strand][chrom.split('chr')[1]] = clip[(clip['strand'] == strand) & (clip['seqname'] == chrom)]['start'].values
            else:
                RBP_regions[strand][chrom] = clip[(clip['strand'] == strand) & (clip['seqname'] == chrom)]['start'].values
    
    
    # Guarantee every strand/chromosome combo present in the isoforms table has an entry,
    # even when the CLIP file has no sites there (e.g. m7G has none on chrY) -> no cross-link.
    for strand in isoforms['strand'].unique():
        RBP_regions.setdefault(strand, {})
        for chrom in isoforms['seqname'].unique():
            RBP_regions[strand].setdefault(chrom, np.array([]))
    
    
    # For each region of the isoform, determine if there is cross-link with the RBP or not
    print("Determine cross-links for each regions...", flush=True)
    isoforms = is_cross_linked_per_region(isoforms, RBP_regions=RBP_regions)

    print("Welch per region is computed on the PUD distributions...", flush=True)
    # For each region, Welch test between the PUD distributions of the group of isoform with cross-links and the group of isoforms without cross-links
    df_cross_linked = welch_between_PUD(isoforms, table_of_pairs, iso_type=pair_key, treatments=treatments)

    # Compute Fisher's exact test on each contingency table [[cross_link and sig shift, cross link and no sig shift], 
    #                                                       [no cross link and sig shift, no cross link and no sig shift]]
    #df_cross_linked['fisher_table'] = df_cross_linked.apply(lambda row: get_fisher_table_old(row['cross_link'], row['no_cross_link'], sig_pairs[iso_type+'_id'].values), axis=1)
    print("Compute the Fisher table...", flush=True)
    for sigpairs in filepath_to_sig_pairs:
            sig_pairs = pd.read_csv(sigpairs)
            print("sig pairs loaded", flush=True)
            ## Filter sig_pairs
            sig_pairs = sig_pairs[sig_pairs[f"distal_id"].isin(table_of_pairs[f"distal_id"].values)]
            
            ##
            sig_pairs_name = os.path.basename(sigpairs).split('.')[0]

            df_cross_linked[f'fisher_table_{sig_pairs_name}'] = df_cross_linked.apply(lambda row: get_fisher_table(cross_link_list=row['cross_link'], 
                                                                                                                   significant_isoforms_group=sig_pairs[pair_key+'_id'], 
                                                                                                                   not_significant_isoforms_group=table_of_pairs[~table_of_pairs['distal_id'].isin(sig_pairs['distal_id'])][pair_key+'_id']), axis=1)
            df_cross_linked[f'fisher_results_{sig_pairs_name}'] = df_cross_linked[f'fisher_table_{sig_pairs_name}'].apply(lambda tab: fisher_exact(table=tab, alternative='two-sided'))
    
            # Compute the bootstraped distributions
            if bootstrap:
                print("Compute the bootstrapped distributions...", flush=True)
                df_cross_linked[f'bootstraped_frequency_cross_link_{sig_pairs_name}'] = df_cross_linked['cross_link'].apply(lambda cl: get_bootstrap_distribution_one_region(table_of_pairs=table_of_pairs, 
                                                                                                    sig_pairs=sig_pairs,
                                                                                                    cross_link_isoforms=cl,
                                                                                                    iso_type=pair_key))
                df_cross_linked[f'bootstraped_pval_{sig_pairs_name}'] = df_cross_linked[f'bootstraped_frequency_cross_link_{sig_pairs_name}'].apply(lambda x: x[1])
                df_cross_linked[f'significance_{sig_pairs_name}'] = df_cross_linked[f'bootstraped_frequency_cross_link_{sig_pairs_name}'].apply(lambda x: x[2])                                
                df_cross_linked[f'regulation_of_shifts_{sig_pairs_name}'] = df_cross_linked[f'bootstraped_frequency_cross_link_{sig_pairs_name}'].apply(lambda x: x[3])       

    # Save the results
    # RBP_name = filepath_to_CLIP.split('/')[-1].split('_ENC')[0]
    print("Saving...", flush=True)
    if split_name_before != '':
        RBP_name = filepath_to_CLIP.split('/')[-1].split(split_name_before)[split_number_before]
    else:
        RBP_name = filepath_to_CLIP.split('/')[-1]
    if split_name_after != '':
        RBP_name = RBP_name.split(split_name_after)[split_number_after-1]
    if not os.path.exists(path_to_results):
        os.makedirs(path_to_results)
    with open(os.path.join(path_to_results, RBP_name+'_'+filepath_to_CLIP.split('/')[-2]+'_results_'+iso_type+'.pkl'), 'wb') as f:
        pickle.dump(df_cross_linked, f)
    print(f"RBP {RBP_name} processed!", flush=True)

def main():
    """Integrate one or several CLIP files (.gff of cross-link sites) with a bulkapa table of pairs, configured by a
    YAML file (see README and config/clip_integration_template.yaml); `--clip_file` overrides the CLIP files of the
    config, and `--iso_type` selects the isoforms (proximal, distal, or stop_codon = windows around the 3' UTR start).
    Writes one `<RBP>_results_<iso_type>.pkl` per CLIP file.
    """

    FLAGS = argparse.ArgumentParser(description='Process one or multiple CLIP data (.gff) and integrate them with RNAseq data.')
    FLAGS.add_argument('--config_file', help='Location of config file')
    FLAGS.add_argument('--iso_type', help='Type of isoform you want to analyze cross links on. Either proximal, distal or stop_codon.', default='proximal', choices=['proximal', 'distal', 'stop_codon'])
    FLAGS.add_argument('--n_jobs', help='number of threads', default=4)
    FLAGS.add_argument('--clip_file', default=None, help='Path to the CLIP data. This is optional and can also be provided in the config file.')

    print("Parsing arguments....", flush=True)
    args = FLAGS.parse_args()
    with open(args.config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(" Config file has been properly loaded!", flush=True)

    
    if args.clip_file is not None:
        all_files = [args.clip_file]
    else:
        if 'files' not in config.keys():
            all_files = []
            if os.path.isfile(config['filepath_to_CLIP']):
                all_files.append(config['filepath_to_CLIP'])
            else:
                for subdir, dirs, files in os.walk(config['filepath_to_CLIP']):
                    for filename in files:
                    # if np.isin(filename.split('_')[0], list(CLIP_to_analyze.index)):
                        if filename.endswith('.gff'):
                            all_files.append(os.path.join(subdir, filename))
        else:
            all_files = config['files']

    if 'bootstrap' in config.keys():
        bootstrap = config['bootstrap']
    else:
        bootstrap = True

    if 'treatments' not in config.keys():
        if 'treatment1_name' in config.keys() and 'treatment2_name' in config.keys():
            treatments = [config['treatment1_name'], config['treatment2_name']]
    else:
        treatments = config['treatments']

    if 'filepath_to_sig_pairs' not in config.keys():
        filepath_to_sig_pairs = []
    else:
        if type(config['filepath_to_sig_pairs']) == str:
            filepath_to_sig_pairs = [config['filepath_to_sig_pairs']]
        else:
            filepath_to_sig_pairs = config['filepath_to_sig_pairs']
            
    if "only_long" in config.keys():
        only_long = config['only_long']
    else:
        only_long = False
        
    pair_key = 'distal' if args.iso_type == 'stop_codon' else args.iso_type

    print(f"I will process the following number of files: {len(all_files)}", flush=True)

    if len(all_files) >= 4:    
        Parallel(n_jobs=int(args.n_jobs))(delayed(process_one_RBP)(filepath_to_CLIP=file, 
                                                                    filepath_to_gtf_isoforms=config['filepath_to_gtf_isoforms'],
                                                                    filepath_to_table_of_pairs=config['filepath_to_table_of_pairs'],
                                                                    filepath_to_sig_pairs=filepath_to_sig_pairs,
                                                                    path_to_results=config['path_to_results'],
                                                                    nt_after=config['nt_after'], 
                                                                    nt_before=config['nt_before'], 
                                                                    length_region=config['length_region'],
                                                                    pair_key=pair_key,
                                                                    iso_type=args.iso_type,
                                                                    split_name_before=config['split_name_before'], 
                                                                    split_name_after=config['split_name_after'], 
                                                                    split_number_before=config['split_number_before'],
                                                                    split_number_after=config['split_number_after'],
                                                                    bootstrap=bootstrap,
                                                                    treatments=treatments, 
                                                                    only_long=only_long) for file in all_files)
    else:
        for file in all_files:
            process_one_RBP(filepath_to_CLIP=file, 
                            filepath_to_gtf_isoforms=config['filepath_to_gtf_isoforms'],
                            filepath_to_table_of_pairs=config['filepath_to_table_of_pairs'],
                            filepath_to_sig_pairs=filepath_to_sig_pairs,
                            path_to_results=config['path_to_results'],
                            nt_after=config['nt_after'], 
                            nt_before=config['nt_before'], 
                            length_region=config['length_region'],
                            pair_key=pair_key,
                            iso_type=args.iso_type,
                            split_name_before=config['split_name_before'], 
                            split_name_after=config['split_name_after'], 
                            split_number_before=config['split_number_before'],
                            split_number_after=config['split_number_after'],
                            bootstrap=bootstrap,
                            treatments=treatments, 
                            only_long=only_long)



if __name__ == '__main__':
    main()
