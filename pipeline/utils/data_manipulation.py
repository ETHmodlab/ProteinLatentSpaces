import numpy as np
import pandas as pd
import torch

def average_structures(X, tids):
    '''
    Averages the coordinates of all distinct structures of each protein

    Parameters
    ----------
    X: np.ndarray
        The latent coordinates of all structures

    tids: list
        The protein target ids of each structure, must be in the same order as X

    Returns
    -------
    average_lats: np.ndarray
        The averaged latent coordinates of each protein

    tids_unique:
        The protein target ids in the same order as the coordinates in average_lats
    '''

    dim = len(X[0])

    tids_unique = list(set(tids))
    occurrences = np.zeros(len(tids_unique))
    average_lats = np.zeros((len(tids_unique), dim))

    for i in range(len(X)):
        idx = tids_unique.index(tids[i])
        occurrences[idx] += 1
        for j in range(dim):
            average_lats[idx][j] += X[i][j]
    
    for i in range(len(occurrences)):
        average_lats[i] /= occurrences[i]
    
    return average_lats, tids_unique


def make_dataset(X: np.ndarray):
    '''
    Turn latent dictionary X into normalized TensorDataset

    Parameters
    ----------
    X: np.ndarray
        The latent coordinates

    Returns
    --------
    torch.utils.data.TensorDataset
    '''
    X = np.copy(X)

    mean = np.array([np.average(X[:,i]) for i in range(len(X[0]))])
    std = np.array([np.std(X[:,i]) for i in range(len(X[0]))])

    for i in range(len(X)):

        X[i] -= mean
        X[i] /= std
        
    
    return torch.utils.data.TensorDataset(torch.Tensor(X))


def get_top_proteins(sids: list, tids: list, threshold: int = None, rank: int = None, select_tids: list = None):
    '''
    Selects the most frequently occurring proteins (proteins with the most structures)

    Parameters
    -----------
    sids: list
        Contains the structure id of each structure as a string

    tids: list
        Contains the protein target id of each structure as a string

    threshold: int
        All proteins that occurr at least threshold times are selected, default = None.
        (Note: provide either threshold or rank, not both)

    rank: int
        Number of top proteins to include, default = None

    select_tids: list
        Target ids of proteins to consider, overrides rank and threshold
    
    Returns
    --------
    A list of tids (with multiple occurrences) of the selected proteins
    '''

    if threshold is not None and rank is not None:
        raise ValueError('Can only select by either rank or threshold')

    if select_tids is None:

        tid_ser = pd.Series(data = tids)
        sorted = tid_ser.value_counts(ascending=False)

        if rank is None and threshold is None:

            return sids, tids

        elif rank is not None:
            sorted = sorted.iloc[:rank]
        else:
            idx = 0
            while sorted.iloc[idx] >= threshold:
                idx += 1
            sorted = sorted.iloc[:idx]

        select_tids = sorted.index

    indices = [i for i in range(len(tids)) if tids[i] in select_tids]

    return [sids[i] for i in indices] , [tids[i] for i in indices]  
