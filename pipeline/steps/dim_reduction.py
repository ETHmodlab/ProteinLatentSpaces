from omegaconf import DictConfig
import numpy as np

from . import reducer_factory

def apply_dimred(X: np.ndarray, cfg: DictConfig):
    '''
    Applies dimensionality reduction to X and returns the reduced data matrix
    
    Parameters
    -----------
    X: np.ndarray
        Latent space vectors of shape n_samples x n_features

    cfg: DictConfig 
        Holds all hyperparameters and the reduction config

    Returns
    -------
    np.ndarray
        Dimensionality-reduced latent space vectors of shape n_samples x reduction_dim
    '''

    reducer = reducer_factory(cfg.red_method, cfg.reduction, cfg.random_seed)
    return reducer.reduce(X)


