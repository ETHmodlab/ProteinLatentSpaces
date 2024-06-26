from omegaconf import DictConfig
import logging
import numpy as np

from . import cluster_factory
from . import internal_factory

log = logging.getLogger(__name__)

def apply_clustering(method: str, cfg: DictConfig, X: np.ndarray):
    '''
    Applies clustering to the data matrix X and returns the cluster assignment

    Parameters
    ----------
    cfg: DictConfig 
        Contains the current hyperparameters and the clustering config
    
    X: np.ndarray 
        Latent vectors as an array of shape n_samples x n_features

    Returns
    --------
    labels: np.ndarray
        The cluster assignment of each data point in X
    
    '''

    clusterer = cluster_factory(method, cfg.clustering[method], cfg.random_seed, cfg.clustering.plot, cfg.clustering.graph_cfg)

    score, params, labels =  clusterer.cluster_optimally(X)

    if cfg.clustering.plot:

        clusterer.make_plot()

    if params is None:
        raise ValueError(f'No suitable parameter set for {method} in the input range')

    log.info(f'Used {method} clustering with parameters {params} and {cfg.clustering[method].metric} score = {score}')

    return labels

def compute_score(metric: str, X, labels, random_seed: int):
    '''
    Compute an internal evaluation metric for the given cluster assignment
    
    Parameters
    ----------
    metric: str
        Name of the evaluation metric

    X: np.ndarray
        Latent coordinates of the data points
    
    labels: np.ndarray
        Cluster assignment of the points in X

    Returns
    -------
    score: float
        The evaluation score of the given cluster assignment
    '''

    evaluator, _ = internal_factory(metric, random_seed, False)
    evaluator.evaluate(X, labels, None)

    score, _, __ = evaluator.get_current_best
    
    return score




