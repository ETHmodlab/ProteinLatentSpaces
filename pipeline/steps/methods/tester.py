import pandas as pd
import numpy as np

from steps.methods.metric_factory import external_factory

class Tester:
    ''' 
    A class to test the alignment of a given hypothesis (i.e. a discrete annotation)
    with a given cluster assignment
    '''

    def __init__(self,
                hypothesis: str,
                metric: str,
                cluster_labels: pd.Series,
                hypothesis_labels: np.array,
                track: bool):
        
        self._hypothesis = hypothesis
        self._cluster_labels = cluster_labels
        self._true_labels = hypothesis_labels
        self._metric_name = metric

        self.metric = external_factory(metric, 1, track)
    

    def test(self):

        for cluster in self._cluster_labels.unique():

            bin_labels = [c == cluster for c in self._cluster_labels.to_numpy()]

            self.metric.evaluate(bin_labels, self._true_labels, cluster)

        return self.metric.get_current_best
    
    def get_scores(self):

        return pd.DataFrame({'cluster': self.metric._x_params, self._metric_name: self.metric._y_scores})

        

