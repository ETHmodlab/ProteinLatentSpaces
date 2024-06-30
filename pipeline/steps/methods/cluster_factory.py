from sklearn.cluster import KMeans, AgglomerativeClustering, estimate_bandwidth, MeanShift, OPTICS, SpectralClustering
from sklearn.mixture import GaussianMixture
import logging
import matplotlib.pyplot as plt
import pandas as pd

from .metric_factory import internal_factory
from .graph import CKNNGraph

log = logging.getLogger(__name__)

def cluster_factory(cluster_method: str, config: dict, random_seed: int, track: bool, graph_cfg: dict = None):
    '''
    Handler method for clustering algorithm. Returns an instance of the correct Clusterer according to cluster_method

    Parameters
    -----------
    cluster_method: str
        The clustering algorithm to use. Must be one of kmeans, agglomerative, gaussianmixture,
        meanshift, optics, spectral

    config: dict
        The current hyperparameter configuration

    random_seed: int

    track: bool
        Whether to keep track of the value of the metric over the range of clustering hyperparameters
    
    graph_cfg: dict
        The configuration of the continuous k-nearest neighbors graph. Only needed when config['metric']
        is a graph-based metric, default = None

    Returns
    --------
    clusterer: Clusterer
        An instance of the correct clustering method class
        
    '''

    if cluster_method == 'kmeans':
        return KmeansClusterer(config['n_clusters'], config['metric'], random_seed, track, graph_cfg)
    
    elif cluster_method == 'agglomerative':
        return AgglomerativeClusterer(config['n_clusters'], config['metric'], random_seed, track, graph_cfg)
    
    elif cluster_method == 'gaussianmixture':
        return GaussianMixtureClusterer(config['n_clusters'], config['metric'], random_seed, track, graph_cfg)
    
    elif cluster_method == 'meanshift':
        return MeanShiftClusterer(config['bandwidth'], config['metric'], random_seed, track, graph_cfg)
    
    elif cluster_method == 'optics':
        return OPTICSClusterer(config['min_points'], config['metric'], random_seed, track, graph_cfg)
    
    elif cluster_method == 'spectral':
        return SpectralClusterer(config['n_clusters'], config['metric'], random_seed, track, graph_cfg, config['n_neighbors'])
    
    else:
        raise ValueError(f'{cluster_method} is not a valid clustering method')
    
    

class Clusterer:
    def __init__(self,
                evaluator: str,
                random_seed: int,
                track: bool,
                graph_cfg: dict):
        
        '''Interface for Clustering methods

        evaluator: metric to use for hyperparameter selection
        '''
        self.graph_based = False
        self._random_seed = random_seed
        self._evaluator = evaluator
        self.metric, typ = internal_factory(self._evaluator, self._random_seed, track)
        self.score_values = pd.DataFrame(columns=[evaluator])
        
        self._params = graph_cfg

        if typ == 'g':
            self.graph_based = True
            

    
    def cluster_optimally(self,
                X):

        '''Tries all clustering parameters and returns the score, 
        parameters and cluster assignment of the best combination'''

        if self.graph_based:
            
            self.graph = CKNNGraph(X, n_neighbors=self._params['n_neighbors'], delta=self._params['delta'], t=self._params['t'])
            return self.graph.get_graph
        
        else:
            
            return X
    
    def make_plot(self):

        plt.ylabel(f'{self._evaluator}', fontsize=25, fontname='Inter')
        plt.ylim(0,1)



class KmeansClusterer(Clusterer):
    def __init__(self,
                n_clusters: list,
                evaluator: str,
                random_seed: int,
                track: bool,
                graph_cfg: dict
                ):
        '''
        n_clusters: list of all values for k to try
        evaluator: name of metric to use for k selection
        '''
    
        super().__init__(evaluator, random_seed, track, graph_cfg)
        self._n_clusters = n_clusters
        
    
    def cluster_optimally(self, X):

        '''tries all clustering parameters and returns the score, parameters and cluster assignment of the best run'''

        X_eval = super().cluster_optimally(X)

        for k in self._n_clusters:

            self.metric.evaluate(X_eval, self.cluster(X, k), k)

            self.score_values.loc[self.metric._x_params[-1], self._evaluator] = self.metric._y_scores[-1]
        
        self.score_values.to_csv(f'kmeans_{self._evaluator}_scores_vs_params.csv')

        return self.metric.get_current_best           

    
    def cluster(self,
                X,
                k: int):
        
        clusterer = KMeans(n_clusters=k, random_state=self._random_seed)
        
        return clusterer.fit_predict(X)
    
    def make_plot(self):

        plt.plot(self.metric._x_params, self.metric._y_scores, c='black')

        super().make_plot()

        plt.xlabel('k', fontsize=15, fontname='Inter')
        plt.xticks(fontsize=15, fontname='Inter')
        plt.yticks(fontsize=15, fontname='Inter')

        plt.tight_layout()

        plt.savefig(f'kmeans_{self._evaluator}_score_plot.png', dpi=500)

        plt.close()


class AgglomerativeClusterer(Clusterer):

    def __init__(self, 
                n_clusters: list,
                evaluator: str, 
                random_seed: int,
                track: bool,
                graph_cfg: dict):

        super().__init__(evaluator, random_seed, track, graph_cfg)

        self._n_clusters = n_clusters
    
    def cluster_optimally(self, X):

        X_eval = super().cluster_optimally(X)

        for k in self._n_clusters:

            self.metric.evaluate(X_eval, self.cluster(X, k), k)

            self.score_values.loc[self.metric._x_params[-1], self._evaluator] = self.metric._y_scores[-1]
        
        self.score_values.to_csv(f'agglomerative_{self._evaluator}_scores_vs_params.csv')

        return self.metric.get_current_best           

    
    def cluster(self,
                X,
                k: int):
        
        clusterer = AgglomerativeClustering(n_clusters=k)
        
        return clusterer.fit_predict(X)
    

    def make_plot(self):

        plt.plot(self.metric._x_params, self.metric._y_scores, c='black')

        super().make_plot()

        plt.xlabel('k', fontsize=25, fontname='Inter')
        plt.xticks(fontsize=25, fontname='Inter')
        plt.yticks(fontsize=25, fontname='Inter')

        plt.tight_layout()

        plt.savefig(f'agglomerative_{self._evaluator}_score_plot.png', dpi=500)

        plt.close()



class GaussianMixtureClusterer(Clusterer):

    def __init__(self,
                n_clusters: int,
                evaluator: str,
                random_seed: int,
                track: bool,
                graph_cfg: dict
                ):

        super().__init__(evaluator, random_seed, track, graph_cfg)

        self._n_clusters = n_clusters
    
    def cluster_optimally(self, X):

        X_eval = super().cluster_optimally(X)

        for k in self._n_clusters:

            self.metric.evaluate(X_eval, self.cluster(X, k), k)

            self.score_values.loc[self.metric._x_params[-1], self._evaluator] = self.metric._y_scores[-1]
        
        self.score_values.to_csv(f'gaussianmixture_{self._evaluator}_scores_vs_params.csv')

        return self.metric.get_current_best

    
    def cluster(self, X, k: int):

        clusterer = GaussianMixture(n_components=k, init_params='k-means++', random_state=self._random_seed)

        return clusterer.fit_predict(X)

    
    def make_plot(self):

        plt.plot(self.metric._x_params, self.metric._y_scores, c='black')

        super().make_plot()

        plt.xlabel('k', fontsize=25, fontname='Inter')
        plt.xticks(fontsize=25, fontname='Inter')
        plt.yticks(fontsize=25, fontname='Inter')

        plt.tight_layout()

        plt.savefig(f'gaussianmixture_{self._evaluator}_score_plot.png', dpi=500)

        plt.close()


class MeanShiftClusterer(Clusterer):

    def __init__(self,
                bandwidth,
                evaluator: str,
                random_seed: int,
                track: bool,
                graph_cfg: dict):
        
        super().__init__(evaluator, random_seed, track, graph_cfg)

        self._bandwidth = bandwidth


    def cluster_optimally(self, X):

        X_eval = super().cluster_optimally(X)

        if self._bandwidth[0] == 'auto':
            self._bandwidth = [estimate_bandwidth(X, random_state=self._random_seed)]
        
        
        for b in self._bandwidth:

            self.metric.evaluate(X_eval, self.cluster(X, b), b)

            self.score_values.loc[self.metric._x_params[-1], self._evaluator] = self.metric._y_scores[-1]
        
        self.score_values.to_csv(f'meanshift_{self._evaluator}_scores_vs_params.csv')
        
        return self.metric.get_current_best
    
    def cluster(self, X, bandwidth: float):

        clusterer = MeanShift(bandwidth=bandwidth, cluster_all = False)

        return clusterer.fit_predict(X)
    

    def make_plot(self):

        plt.plot(self.metric._x_params, self.metric._y_scores, c='black')

        super().make_plot()

        plt.xlabel('bandwidth', fontsize=25)
        plt.xticks(fontsize=25, fontname='Inter')
        plt.yticks(fontsize=25, fontname='Inter')

        plt.tight_layout()

        plt.savefig('meanshift_{self._evaluator}_score_plot.png', dpi=500)

        plt.close()



class OPTICSClusterer(Clusterer):
    def __init__(self,
                min_points: list,
                evaluator: str,
                random_seed: int,
                track: bool,
                graph_cfg: dict):
        
        super().__init__(evaluator, random_seed, track, graph_cfg)
        
        self._min_points = min_points
    
    def cluster_optimally(self, X):

        X_eval = super().cluster_optimally(X)

        for mp in self._min_points:

            try:
                cluster_labs = self.cluster(X, mp)
                self.metric.evaluate(X_eval, cluster_labs, mp)

                self.score_values.loc[self.metric._x_params[-1], self._evaluator] = self.metric._y_scores[-1]
            
            except ValueError:
                    
                    log.info(f'Skipped parameter min_points={mp}')
            
            log.info(f'OPTICS yielded {len(set(cluster_labs))} clusters with min_points = {mp}')
        
        self.score_values.to_csv(f'optics_{self._evaluator}_scores_vs_params.csv')

        return self.metric.get_current_best

    
    def cluster(self, X, min_points: int):

        clusterer = OPTICS(min_samples=min_points)
        
        return clusterer.fit_predict(X)
    

    def make_plot(self):

        plt.plot(self.metric._x_params, self.metric._y_scores, c='black')

        super().make_plot()

        plt.xlabel('min # points in neighborhood', fontsize=25)
        plt.xticks(fontsize=25, fontname='Inter')
        plt.yticks(fontsize=25, fontname='Inter')

        plt.tight_layout()

        plt.savefig(f'optics_{self._evaluator}_score_plot.png', dpi=500)

        plt.close()


class SpectralClusterer(Clusterer):
    def __init__(self,
                n_clusters: list,
                evaluator: str,
                random_seed: int,
                track: bool,
                graph_cfg: dict,
                n_neighbors: list = None):


        super().__init__(evaluator, random_seed, track, graph_cfg)

        self._n_clusters=n_clusters
        self._n_neighbors=n_neighbors

    
    def cluster_optimally(self, X):

        X_eval = super().cluster_optimally(X)
    
        for k in self._n_clusters:

            if self._n_neighbors is not None:

                for nn in self._n_neighbors:

                    self.metric.evaluate(X_eval, self.cluster(X, k, nn), (k,nn))

                    #self.score_values.loc[self.metric._x_params[-1], self._evaluator] = self.metric._y_scores[-1]

            else:
                
                self.metric.evaluate(X_eval, self.cluster(X, k, None), k)

                #self.score_values.loc[self.metric._x_params[-1], self._evaluator] = self.metric._y_scores[-1]
        
        self.score_values.to_csv(f'spectral_{self._evaluator}_scores_vs_params.csv')

        return self.metric.get_current_best


    def cluster(self, X, k, nn):

        if nn is None:
            affinity = 'rbf'
        
        else:
            affinity = 'nearest_neighbors'

        clusterer = SpectralClustering(n_clusters=k, affinity=affinity, n_neighbors=nn, random_state=self._random_seed)

        return clusterer.fit_predict(X)
    

    def make_plot(self):

        params = self.metric._x_params

        clusters = list(set([a for a,b in params]))
        neighbors = list(set([b for a,b in params]))

        super().make_plot()


        if neighbors[0] == None:

            plt.plot(clusters, self.metric._y_scores, c='black')


        else: 

            Z = pd.DataFrame(index = clusters, columns = neighbors)

            for idx in range(len(params)):

                k, n = params[idx]

                Z.loc[k, n] = self.metric._y_scores[idx]

            for nn in neighbors:

                plt.plot(Z.index, Z.loc[:,nn], label = nn)
            
            plt.legend(title='# neighbors', title_fontsize='x-large')

        plt.xlabel('k', fontsize=25, fontname='Inter')
        plt.xticks(fontsize=25, fontname='Inter')
        plt.yticks(fontsize=25, fontname='Inter')

        plt.tight_layout()
              
        plt.savefig(f'spectral_{self._evaluator}_score_plot.png', dpi=500)
        plt.close()

            



    





    


