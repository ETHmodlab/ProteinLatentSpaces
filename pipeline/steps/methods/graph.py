from cknn import cknneighbors_graph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class CKNNGraph:
    '''Wrapper for cknn.cknnneighbors_graph'''

    def __init__(self,
                X,
                n_neighbors: int,
                delta: float,
                t: float
                ):
        
        self._n_neighbors = n_neighbors
        self._delta = delta
        self._X=X
        self.adjacency_matrix = cknneighbors_graph(X, n_neighbors=n_neighbors, delta=delta, t=t, is_sparse=False, include_self=True)

        self._graph = nx.Graph(incoming_graph_data=self.adjacency_matrix)
    
    @property
    def get_graph(self):
        '''Return the underlying networkx graph'''

        return self._graph
    
    def plot_edges(self, X, ax: plt.Axes, color: str = 'lightgrey', linewidth: float = 0.6):
        ''' 
        Draw the edges of the graph onto the given Axis
        '''

        if len(X) <= 1:
            raise ValueError('Graph contains only 1 vertex, nothing to visualize.')
    
        for idx in range(len(X)-1):

            for j in range(idx+1, len(X)):

                if self.adjacency_matrix[idx, j] > 0:
            
                    dat = np.array([X[idx,:], X[j,:]]).T

                    ax.plot(dat[0], dat[1], color = color, linewidth=linewidth, zorder=-1)