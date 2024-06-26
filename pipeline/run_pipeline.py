import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import os
import logging
import json
import matplotlib.pyplot as plt

from dataset import ProteinDataset
from steps.clustering import apply_clustering, compute_score
from steps.hypothesis_testing import test_hypotheses
from steps.dim_reduction import apply_dimred
from steps.visualize import plot_factory
from utils import get_families, assign_membership, get_hierachy, average_structures, get_top_proteins

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./config', config_name='base')
def main(cfg: DictConfig):
    ''' 
    Parameters
    ----------
    cfg: DictConfig
        Configuration passed by hydra
    '''
    
    log.info(f'Saving results into {os.getcwd()}')

    np.random.seed(cfg.random_seed)

    dataset = ProteinDataset(cfg.model_id, cfg.data_path)

    if cfg.reload_families:
        # create the protein family assignment dictionary from scratch
        family_dict = get_families(cfg.similarity_file, dataset.proteins_table_df['entry_name (UniProt)'])
        with open(cfg.family_dict_path, 'w') as file:
            file.write(json.dumps(family_dict))
    
    else: 
        # read the protein family dictionary from a file
        with open(cfg.family_dict_path) as j_file:
            family_dict = json.load(j_file)
    
    # remove all proteins not in the dataset from the family dictionary
    reduced_dict = {}
    for fam in family_dict:
        if len(family_dict[fam]) > 0:
            reduced_dict[fam] = family_dict[fam]
    family_dict = reduced_dict

    # get the structure and target ids in the same order
    sids = dataset.structure_ids
    tids = [dataset._get_tid_from_structure_id(i) for i in sids]

    if cfg.subset.select_tids is not None:
        select_tids = pd.read_csv(cfg.subset.select_tids, index_col = 0).index

    else:
        select_tids = None

    # reduce the dataset to a subset
    sids, tids = get_top_proteins(sids, tids, cfg.subset.threshold, cfg.subset.rank, select_tids)

    X = dataset.get_structure_latent_features(sids)

    if cfg.avg_structures:
        # average all structures of the same protein
        X, tids = average_structures(X, tids)

    df = pd.DataFrame(columns = ['tid', 'family'])
    df['tid'] = tids
        

    # DIMENSIONALITY REDUCTION #############################
    # (assumes that one method is hardcoded in the config)

    X = apply_dimred(X, cfg)

    log.info(f'Applied {cfg.red_method} for dimensionality reduction to {len(X[0])} dimensions')


    # CLUSTERING #############################

    metrics = pd.DataFrame(columns=cfg.cluster_methods, index = cfg.metrics)

    for method in cfg.cluster_methods:
        df[method] = apply_clustering(method, cfg, X)

        # compute the score of the cluster assignment
        for m in cfg.metrics:
            metrics.loc[m, method] = compute_score(m, X, df[method], cfg.random_seed)

    # persist the cluster assignment, scores and reduced latent vectors
    metrics.to_csv('scores.csv')
    df.to_csv('cluster_labels.csv')
    pd.DataFrame(X).to_csv('latent_coords.csv')

    log.info(f'Finished Clustering')


    # PLOTTING #############################

    # get a mapping from composite family keys to super-, sub-, and family
    hierachy = get_hierachy(family_dict.keys())

    family_annotation = df.copy()
    

    if cfg.plotting.family is not None:

        df['family'] = np.nan

        # create a boolean column for each family
        for fam in cfg.plotting.family:

            df[fam] = assign_membership(df['tid'], fam, dataset, family_dict, hierachy)

        # create a string column with the family assignment of each protein
        for sample in df.index:
            for fam in cfg.plotting.family:

                if df.loc[sample, fam]:

                    if df.loc[sample, 'family'] == np.nan:
                        raise ValueError(f'Duplicate family membership in {df.loc[sample, "family"]} and {fam}')

                    df.loc[sample, 'family'] = fam
        
        family_annotation['family'] = df['family']
    
    if cfg.plotting.superfamily is not None:

        df['superfamily'] = np.nan

        # create a boolean column for each superfamily
        for spfam in cfg.plotting.superfamily:

            df[spfam] = assign_membership(df['tid'], spfam, dataset, family_dict, hierachy, 'super')
        
        # create a string column with the superfamily assignment of each protein
        for sample in df.index:
            for fam in cfg.plotting.superfamily:

                if df.loc[sample, fam]:

                    if not np.isnan(df.loc[sample, 'superfamily']):
                        raise ValueError(f'Duplicate family membership in {df.loc[sample, "superfamily"]} and {fam}')

                    df.loc[sample, 'superfamily'] = fam
        
        family_annotation['superfamily'] = df['superfamily']
    
    if cfg.plotting.subfamily is not None:

        df['subfamily'] = np.nan

        # create a boolean column for each subfamily
        for sbfam in cfg.plotting.subfamily:

            df[sbfam] = assign_membership(df['tid'], sbfam, dataset, family_dict, hierachy, 'sub')

        # create a string column with the subfamily assignment of each protein
        for sample in df.index:
            for fam in cfg.plotting.subfamily:

                if df.loc[sample, fam]:

                    if not np.isnan(df.loc[sample, 'subfamily']):
                        raise ValueError(f'Duplicate family membership in {df.loc[sample, "subfamily"]} and {fam}')

                    df.loc[sample, 'subfamily'] = fam

        family_annotation['subfamily'] = df['subfamily']
    
    
    if cfg.testing.annotation_path is not None:

        annotations = pd.read_csv(cfg.testing.annotation_path)

        if len(annotations.index) != len(df.index):

            annotations = annotations[annotations['tid'].isin(df['tid'])]
        
        annotations.drop('tid', inplace=True, axis=1)
        annotations.set_index(df.index, inplace = True)

        df = pd.concat([df, annotations], axis=1)

    # for each dim. reduction method, create all plots specified in the config
    for dimr in cfg.plotting.dimred:

        plotter, X_red = plot_factory(X, df, dimr, cfg.random_seed)

        for hue in cfg.plotting.colors:

            ax_obj = plotter.plot(hue)
            plt.close()


        if cfg.plotting.family is not None:
            for fam in cfg.plotting.family:
                    
                plotter.plot(fam)
                plt.close()
        
        if cfg.plotting.superfamily is not None:
            for spfam in cfg.plotting.superfamily:

                plotter.plot(spfam)
                plt.close()
        
        if cfg.plotting.subfamily is not None:
            for sbfam in cfg.plotting.subfamily:

                plotter.plot(sbfam)
                plt.close()
    

    # HYPOTHESIS TESTING #############################

    if cfg.hypotheses is not None:

        test_hypotheses(cfg, df if cfg.testing.filter_by is None else df[df[cfg.testing.filter_by].notnull().values])

        for ht in cfg.hypotheses:
            
            family_annotation[ht] = df[ht]

    family_annotation.to_csv('cluster_annotation_labels.csv')

    log.info('Finished running the pipeline')

 
if __name__ == '__main__':
    main()
