import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from . import Tester

logger = logging.getLogger(__name__)

def plot_hypothesis_distribution(cluster: int, cluster_method: str, freq_threshold: int, df: pd.DataFrame, hypotheses: list = None):
    '''
    Creates a histogram of the frequency of multiple hypotheses in a given cluster.
    Can be overlapping since hypotheses don't have to be mutually exclusive

    Parameters
    ----------
    cluster: int
        The index of the cluster

    cluster_method: str
        The column in df which contains the cluster assignment

    freq_threshold: int
        The number of occurrences below which a hypothesis is omitted from the histogram

    df: pd.DataFrame
        Holds cluster assignment and hypothesis assignment

    hypotheses: list[str]
        Subset of hypotheses to consider, if None use all in df. Default = None

<<<<<<< HEAD
def plot_hypothesis_distribution(cluster: int, cluster_method: str, freq_threshold: int, df: pd.DataFrame):
=======
    Returns
    --------
    bool
        Whether the plot contains any data, can be False if no annotation exceeded freq_threshold
    '''

    plt.rcParams["figure.figsize"] = [6.4,7]  #8] superfamily
>>>>>>> publication_clean

    sub_df = df.loc[df[cluster_method]==cluster, :]

    frequencies = []
    properties = []

    for col in sub_df.columns:
        if sub_df[col].dtype == 'bool':

            freq = sub_df[col].sum(axis=0)

            if freq >= freq_threshold:

                properties.append(col)
                frequencies.append(freq)

    if len(frequencies) == 0:
        return False
            
    frequencies = np.array(frequencies)/len(sub_df.index)

    dat = pd.DataFrame({'f': frequencies, 'p': properties})
    dat.sort_values(by='f', ascending=False, inplace = True)
    
    plt.bar(range(len(frequencies)), dat['f'])
    plt.xticks(range(len(frequencies)), dat['p'], rotation = 90, fontsize = 'xx-small')
    plt.tight_layout()
    plt.ylim(0,1)

    return True


def plot_cluster_distribution(hypothesis: str, cluster_method: str, df: pd.DataFrame):
    '''
    Creates a histogram of the frequency of the given hypothesis in each cluster

    Parameters
    ----------
    hypothesis: str
        Name of the boolean assignment column for this hypothesis

    cluster_method: str
        The name of the column which contains the cluster assignment

    df: pd.DataFrame
        The dataframe that holds the cluster assignments and hypothesis

    Returns
    --------
    None
    '''

    sub_df = df.loc[df[hypothesis]]

    counts=sub_df[cluster_method].value_counts(ascending=False)
    
    plt.bar(range(len(counts)), counts[cluster_method])
    plt.xticks(counts.index, counts.index, fontsize = 'xx-small')
    plt.xlabel('Cluster')
    plt.tight_layout()
    plt.ylim(0,1)


def to_boolean(property: str, df: pd.DataFrame):
    '''
    Converts a multi-categorical property into multiple single-category boolean assignments

    Parameters
    -----------
    property: str
        key of the multi-categorical property in df

    df: pd.DataFrame 
        Contains a column with the multi-categorical assignment

    Returns
    --------
    df: pd.DataFrame    
        df with a new added column for each category
    '''

    for category in df[property].unique():

        if isinstance(category, str) or isinstance(category, int):

            df[f'{property} {category}'] = df[property].apply(lambda x: True if x == category else False)
    
    return df


def test_column(ht: str, cfg: dict, df: pd.DataFrame, scores_global: pd.DataFrame, clusters_global: pd.DataFrame, control_global: pd.DataFrame):
    '''
    Hypothesis testing on a boolean column of df

    Parameters
    ----------
    ht: str
        Name of the hypothesis column

    cfg: dict
        Dict-style config with hyperparameters for the pipeline

    df: pd.DataFrame
        Contains the cluster assignment for each clustering method and a column for each
        hypothesis

    scores_global: pd.DataFrame
        DataFrame to write the score of the best cluster for this hypothesis to

    cluster_global: pd.DataFrame
        DataFrame to write the index of the best cluster for this hypothesis to

    control_global: pd.DataFrame
        DataFrame to write the score of the random control for this hypothesis to

    Returns
    --------
    scores_global, cluster_global, control_global
        The modified data frames


    '''

    for clustering in cfg.cluster_methods:

            tester = Tester(ht, cfg.testing.metric, df[clustering], df[ht].to_numpy(), cfg.testing.track)

            score, cluster, _ = tester.test()

            scores_global.loc[clustering, ht] = score
            clusters_global.loc[clustering, ht] = cluster

            logger.info(f'{ht} aligns best with cluster {cluster} in {clustering} clustering, score = {score} ({cfg.testing.metric})')

            if cfg.testing.random:

                p = np.count_nonzero(df[ht].to_numpy()) / len(df.index)

                control_labels = np.random.choice(a=[False, True], size=len(df.index), p = [1-p, p])
            
                tester = Tester('random', cfg.testing.metric, df[clustering], control_labels, False)

                score, cluster, _ = tester.test()

                control_global.loc[clustering, ht] = score

                logger.info(f'{ht} random control score = {score} ({cfg.testing.metric})')

    return scores_global, clusters_global, control_global


def test_single_hypothesis(ht: str, scores_global: pd.DataFrame, clusters_global: pd.DataFrame, control_global: pd.DataFrame, df: pd.DataFrame, cfg: dict):
    '''
    Calculate similarity score for a single hypothesis

    Parameters
    ----------
    ht: str
        Column name of the hypothesis in df
        
    scores_global: pd.DataFrame
        DataFrame to save the scores to

    clusters_global: pd.DataFrame
        DataFrame to save the best cluster to

    control_global: pd.DataFrame
        DataFrame to save the control scores to, can be None if cfg.random = False

    cfg: dict
        Dict-style config of the pipeline

    Returns
    -------
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
        Updated DataFrame with boolean columns for multicategorical hypotheses, updated scores_global, clusters_global, control_global
    '''

    if df[ht].dtype == 'float':
        raise ValueError(f'Hypothesis {ht} has continuous values, bin to categorical values before testing')

    if df[ht].dtype != 'bool':

        df = to_boolean(ht, df)
        
        for sub_ht in df[ht].unique():

            if isinstance(sub_ht, str):

                scores_global, clusters_global, control_global = test_column(f'{ht} {sub_ht}', cfg, df, scores_global, clusters_global, control_global)
        
    else:

        scores_global, clusters_global, control_global = test_column(ht, cfg, df, scores_global, clusters_global, control_global)
        
    return df, scores_global, clusters_global, control_global


def test_hypotheses(cfg: dict, df: pd.DataFrame):

    '''
    Calculates similarity scores for each hypothesis and cluster method given in cfg and saves them in csv files.
    Creates 3 DataFrames of shape (cluster methods x hypotheses): 
    Similarity scores, Label of the best cluster and Random control score
    
    Parameters
    ----------
    cfg: dict
        Dictionary-style config with hyperparameters for the pipeline

    df: pd.DataFrame 
        Must contain a column with cluster labels for every cluster method and boolean
        assignment for each hypothesis
    
    Returns
    --------
    None
    '''

    if 'tid' in df.columns:

        df['tid'] = df.tid.astype(str)
        print(df['tid'].dtype)

    scores_global = pd.DataFrame(index = cfg.cluster_methods, columns = cfg.hypotheses)
    clusters_global = pd.DataFrame(index = cfg.cluster_methods, columns = cfg.hypotheses)


    if cfg.testing.random:
        control_global = pd.DataFrame(index = cfg.cluster_methods, columns = cfg.hypotheses)
    else:
        control_global = None
    

    for ht in cfg.hypotheses:

        df, scores_global, clusters_global, control_global = test_single_hypothesis(ht, scores_global, clusters_global, control_global, df, cfg)


    scores_global.to_csv('hypothesis_scores.csv')
    clusters_global.to_csv('hypothesis_best_cluster.csv')

    if cfg.testing.random:

        control_global.to_csv('hypothesis_random_scores.csv')

    
    if cfg.testing.plot:
        for method in cfg.cluster_methods:
            for cluster in df[method].unique():

<<<<<<< HEAD
                plot_hypothesis_distribution(cluster, method, cfg.testing.plot_threshold, df)
                plt.title(f'Annotation frequency {method} (Cluster {cluster})')
                plt.savefig(f'ann_freq_{method}_{cluster}.png', dpi = 450)
                plt.close()
=======
                to_test = None

                if cfg.testing.plot_ht_only:

                    to_test = cfg.hypotheses

                has_data = plot_hypothesis_distribution(cluster, method, cfg.testing.plot_threshold, df, to_test)

                if has_data:
                    plt.title(f'Cluster {cluster}', fontsize='xx-large')
                    plt.tight_layout()
                    plt.savefig(f'ann_freq_{method}_{cluster}.png', dpi = 450)
                    plt.close()
>>>>>>> publication_clean





            

            



    
