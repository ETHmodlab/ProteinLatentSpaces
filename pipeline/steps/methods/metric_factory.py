import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_mutual_info_score, adjusted_rand_score, f1_score, calinski_harabasz_score
from networkx.algorithms import conductance, normalized_cut_size
from networkx.algorithms.community import modularity
import networkx as nx


def internal_factory(metric: str, random_seed: int, track: bool):
    '''
    Handler method for cluster validation metric. Returns an instance of the correct Evaluator for quantitatively evaluating a clustering

    Parameters
    ----------
    metric: str
        The evaluation metric to use. Must be one of silhouette, davies, calinski, modularity, conductance, ncut

    random_seed: int

    track: bool
        Whether to keep track of the value of the metric over the range of clustering hyperparameters

    Returns
    --------
    evaluator: Evaluator
        An instance of the correct metric class
    
    type: str
        Indicates whether the metric is euclidean (e) or graph-based (g)

    '''
    if metric == 'silhouette':
        return Silhouette(random_seed=random_seed, track=track), 'e'
    
    elif metric == 'davies':
        return Davies(track=track), 'e'
    
    elif metric == 'calinski':
        return Calinski(track=track), 'e'
    
    elif metric == 'modularity':
        return Modularity(track=track), 'g'
    
    elif metric == 'conductance':
        return Conductance(track=track), 'g'
    
    elif metric == 'ncut':
        return NormalizedCut(track=track), 'g'
    
    else:
        raise ValueError(f'{metric} is not a valid metric')


def external_factory(metric: str, random_seed: int, track: bool):
    '''Handler method for the hypothesis testing metric. Returns an instance of the correct Evaluator 
    for quantitatively evaluating the overlap of a clustering and a discrete annotation

    Parameters
    ----------
    metric: str
        The evaluation metric to use. Must be one of ari, ami, f1

    random_seed: int
        Not used

    track: bool
        Whether to keep track of the value of the metric over the range of clustering hyperparameters

    Returns
    --------
    evaluator: Evaluators
        An instance of the correct metric class
    '''

    if metric == 'ari':
        return AdjustedRand(track=track)
    
    elif metric == 'ami':
        return AdjustedMutualInfo(track=track)

    elif metric == 'f1':
        return F1(track=track)

    else:
        raise ValueError(f'{metric} is not a valid metric')



class Evaluator:
    
    def __init__(self, track:bool=False):
        
        self._score = None
        self._params = None
        self._labels = None
        self._y_scores = []
        self._x_params = []
        self._track = track

    @property
    def get_current_best(self):
        '''returns the current best score of the metric, 
        the corresponding cluster parameters and cluster assignments'''

        return self._score, self._params, self._labels


class Silhouette(Evaluator):

    def __init__(self, random_seed:int, track:bool=False):
        
        super().__init__(track)

        self._random_seed = random_seed
        self._score = -1
    
    def evaluate(self, X, labels, params):

        if len(set(labels)) == 1:
            raise ValueError('Cannot compute silhouette score, only one cluster found')
        
        X_noiseless, labels_noiseless = get_noiseless(X, labels)

        score = silhouette_score(X = X_noiseless, labels = labels_noiseless, random_state = self._random_seed)
        
        if score > self._score:
            self._score = score
            self._params = params
            self._labels = labels
        
        if self._track:

            self._y_scores.append(score)
            self._x_params.append(params)



class Davies(Evaluator):

    def __init__(self, track:bool=False):

        super().__init__(track)

        self._score = 99999

    def evaluate(self, X, labels, params):

        X_noiseless, labels_noiseless = get_noiseless(X, labels)

        score = davies_bouldin_score(X_noiseless, labels_noiseless)

        if score < self._score:
            self._score = score
            self._params = params
            self._labels = labels
        
        if self._track:

            self._y_scores.append(score)
            self._x_params.append(params)

class Calinski(Evaluator):

    def __init__(self, track:bool=False):

        super().__init__(track)

        self._score = 0

    def evaluate(self, X, labels, params):

        X_noiseless, labels_noiseless = get_noiseless(X, labels)

        score = calinski_harabasz_score(X_noiseless, labels_noiseless)

        if score > self._score:
            self._score = score
            self._params = params
            self._labels = labels
        
        if self._track:

            self._y_scores.append(score)
            self._x_params.append(params)





class Modularity(Evaluator):
    def __init__(self, track: bool):
        super().__init__(track)

        self._score = -10


    def evaluate(self, graph: nx.Graph, labels: np.array, params):

        communities = []

        for cluster in np.unique(labels):
            
            communities.append(set(np.where(labels==cluster)[0].tolist()))
        
        score = modularity(graph, communities)

        if score > self._score:
            self._score = score
            self._params = params
            self._labels = labels
        
        if self._track:

            self._y_scores.append(score)
            self._x_params.append(params)

            

class Conductance(Evaluator):
    
    def __init__(self, track: bool):
        super().__init__(track)

        self._score = 10


    def evaluate(self, graph: nx.Graph, labels: np.array, params):

        score = 0

        for cluster in np.unique(labels):
            
            score += conductance(graph, set(np.where(labels==cluster)[0].tolist()))

        
        score /= len(np.unique(labels))
        
        if score < self._score:

            self._score = score
            self._params = params
            self._labels = labels


        if self._track:

            self._y_scores.append(score)
            self._x_params.append(params)


class NormalizedCut(Evaluator):
    def __init__(self, track: bool):
        super().__init__(track)

        self._score = 99999


    def evaluate(self, graph: nx.Graph, labels: np.array, params):

        score = 0

        for cluster in np.unique(labels):
            
            score += normalized_cut_size(graph, set(np.where(labels == cluster)[0].tolist()))

        score /= len(np.unique(labels))

        if score < self._score:

            self._score = score
            self._params = params
            self._labels = labels


        if self._track:

            self._y_scores.append(score)
            self._x_params.append(params)





def get_noiseless(X, labels):

    #if -1 in labels:
        
    #    filt = [l != -1 for l in labels]
    #    return list(compress(X, filt)), list(compress(labels, filt))
        
    #else:
        return X, labels


class AdjustedRand(Evaluator):

    def __init__(self, track: bool):

        super().__init__(track)

        self._score = 0
    
    def evaluate(self, l_pred, l_true, cluster: int):

        score = adjusted_rand_score(labels_true=l_true, labels_pred=l_pred)

        if score > self._score:
            self._score = score
            self._labels = l_pred
            self._params = cluster
    

        if self._track:
            self._y_scores.append(score)
            self._x_params.append(cluster)


class AdjustedMutualInfo(Evaluator):

    def __init__(self, track: bool):

        super().__init__(track)

        self._score = 0
    
    def evaluate(self, l_pred, l_true, cluster: int):

        score = adjusted_mutual_info_score(labels_true=l_true, labels_pred=l_pred)

        if score > self._score:
            self._score = score
            self._labels = l_pred
            self._params = cluster
    

        if self._track:
            self._y_scores.append(score)
            self._x_params.append(cluster)
        

class F1(Evaluator):

    def __init__(self, track: bool):

        super().__init__(track)

        self._score = 0
    
    
    def evaluate(self, l_pred, l_true, cluster: int):

        score = f1_score(labels_true=l_true, labels_pred=l_pred)

        if score > self._score:
            self._score = score
            self._labels = l_pred
            self._params = cluster
    

        if self._track:
            self._y_scores.append(score)
            self._x_params.append(cluster)
        






