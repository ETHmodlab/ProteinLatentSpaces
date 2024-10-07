import warnings
from numba import errors
warnings.simplefilter('ignore', category=errors.NumbaDeprecationWarning)

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from umap import UMAP
from sklearn.manifold import TSNE
import torch

from autoencoder import Autoencoder, predict
from utils import make_dataset

def reducer_factory(method: str, config: dict, random_seed: int):
    '''Handler method for dimensionality reduction. Returns an instance of the correct Reducer class
    
    Parameters
    -----------
    method: str
        The dimensionality reduction method(s) to use. Must be one of pca, kernelpca, umap, autoencoder
        or a combination them separated by + (e.g. pca+umap)

    config: dict
        The current hyperparameter configuration

    random_seed: int

    Returns
    -------
    reducer: Reducer
    '''

    methods = method.split('+')

    if len(methods) > 1:
        return CombinedReducer(methods, config, random_seed)

    elif methods[0] == 'pca':
        return  PcaReducer(config['n_dimensions'], random_seed)
    
    elif methods[0] == 'kernelpca':
        return KernelPcaReducer(config['n_dimensions'], config['kernel'], random_seed)
    
    elif methods[0] == 'umap':
        return UmapReducer(config['n_dimensions'], config['n_neighbors'], config['min_distance'], random_seed)
    
    elif methods[0] == 'autoencoder':
        return AutoEncoderReducer(config['n_dimensions'], config['model_path'], random_seed)

    else:
        raise ValueError(f'{method} is not a valid dimensionality reduction method')


class Reducer:
    '''A template for all dimensionality reduction methods'''
    def __init__(self, 
                 n_dimensions: int,
                 random_seed: int = 0):
        
        self._n_dimensions = n_dimensions
        self._random_seed = random_seed
    
    ###PROPERTY METHODS###

    @property
    def dimension(self):
        return self._n_dimensions
    
    ###MEMBER METHODS###

    def reduce(self,
                X):
                '''Applies dimensionality reduction to X
                
                X: array-like of shape n_samples x n_features
                '''
                pass


class CombinedReducer(Reducer):
    def __init__(self,
                methods: list,
                cfg: dict,
                random_seed: int):
        super().__init__(cfg['n_dimensions'][-1], random_seed)

        self._methods = methods
        self._reducer_chain = []

        for i in range(len(methods)):
            self._reducer_chain.append(reducer_factory(methods[i], {'n_dimensions': cfg['n_dimensions'][i], 'n_neighbors': cfg['n_neighbors'], 'min_distance': cfg['min_distance']}, random_seed))
    
    def reduce(self,
                X):
        
        for reducer in self._reducer_chain:
            X = reducer.reduce(X)
        
        return X


class PcaReducer(Reducer):
    def __init__(self,
                n_dimensions: int,
                random_seed: int,
                scale: bool = True):

        super().__init__(n_dimensions, random_seed)
        
        if scale:
            self.reducer = make_pipeline(StandardScaler(), PCA(n_components=self._n_dimensions, random_state=self._random_seed))
        else:
            self.reducer = PCA(n_components=self._n_dimensions, random_state=self._random_seed)

    def reduce(self,
                X):
        
        return self.reducer.fit_transform(X)


class KernelPcaReducer(Reducer):
    def __init__(self,
                n_dimensions: int,
                kernel: str,
                random_seed: int):
        
        super().__init__(n_dimensions, random_seed)

        self._kernel = kernel
        self.reducer = KernelPCA(n_components=self._n_dimensions, kernel=self._kernel, random_state=self._random_seed)
    
    def reduce(self, X):
        return self.reducer.fit_transform(X)

class UmapReducer(Reducer):
    def __init__(self,
                n_dimensions: int,
                n_neighbors: int,
                min_distance: float,
                random_seed: int):
        
        super().__init__(n_dimensions, random_seed)

        self.reducer = UMAP(n_components=self._n_dimensions, random_state=self._random_seed, n_neighbors=n_neighbors, min_dist=min_distance)
    
    def reduce(self, X):
        return self.reducer.fit_transform(X)


class TsneReducer(Reducer):
    def __init__(self, n_dimensions: int, random_seed: int):
        super().__init__(n_dimensions, random_seed)
    
        self.reducer = TSNE(n_components=n_dimensions, random_state=random_seed)

    def reduce(self, X):
        return self.reducer.fit_transform(X)

class AutoEncoderReducer(Reducer):
    def __init__(self, n_dimensions: int, model_path: str, random_seed: int):
        super().__init__(n_dimensions)

        self.model = Autoencoder(encode_only=True)
        self.model.load_state_dict(torch.load(model_path, 
                                            map_location = None if torch.cuda.is_available() else torch.device('cpu')))
        self.model.eval()

    def reduce(self, X):

        dataset = make_dataset(X)
        return predict(self.model, dataset)



