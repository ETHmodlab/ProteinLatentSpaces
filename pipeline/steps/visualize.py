import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import colorcet as cc
import matplotlib.cm as cm

from steps.methods.reducer_factory import PcaReducer, UmapReducer, TsneReducer


def plot_factory(X: np.ndarray, df: pd.DataFrame, red: str, random_seed: int):
    '''
    Handler for the plotting dimensionality reduction, returns a Plotter object
    
    Parameters
    ----------
    X: np.ndarray
        The latent vectors as an array of n_samples x n_features

    df: pd.DataFrame
        Contains cluster assignments and hypothesis assignments for each sample

    red: str
        Dimensionality reduction method to use for projecting to 2D.
        Must be one of pca, umap, tsne.

    random_seed: int

    Returns
    -------
    Plotter
        An instance of the Plotter class with the correct hyperparameters

    X_red: np.ndarray
        The dimensionality-reduced latent vectors
    '''
    
    if red == 'pca':
        reducer = PcaReducer(2, random_seed)
        axname = 'PC'
    elif red == 'umap':
        reducer = UmapReducer(2, 15, 0.1, random_seed)
        axname = 'UMAP'
    elif red == 'tsne':
        reducer = TsneReducer(2, random_seed)
        axname = 't-SNE'
    else:
        raise ValueError(f'{red} is not a valid plotting method')

    X_red = np.array(reducer.reduce(X)).T
    
    df['x'] = X_red[0]
    df['y'] = X_red[1]

    return Plotter(df, axname), X_red


class Plotter:
    '''Can produce differently colored scatterplots for the given reduced Dataset'''
    def __init__(self,
                cluster_labels: pd.DataFrame,
                axname: str
                ):

        self._labels = cluster_labels
        self._axname = axname

    @property
    def get_reduction_method(self):
        return self._axname


    def plot(self, hue: str, save: bool = True, ax: plt.Axes = None):
        '''
        Visualize the data in a scatterplot colored according to hue

        Parameters
        ----------
        hue: str
            The column according to which the datapoints are colored

        save: bool
            Whether to save the plot in the current directory, default = True

        ax: plt.Axes
            An existing axis to plot onto, default = None

        Returns
        --------
        ax: plt.Axes
            The axis that was plotted onto
        '''

        fontsz = 15
        s = 10

        plt.rc('legend',fontsize=fontsz)

        if hue is not None and hue.split('+')[0] not in self._labels.columns:
            raise ValueError(f'The property {hue} does not exist in the data')
        
        c = None
        
        if hue == 'tid':
            
            palette = sns.color_palette(cc.glasbey, n_colors=445)
        
        elif hue is not None:
            legend = 'auto'
            hues = hue.split('+')
        
            if len(hues) == 2 and hues[1].split('_')[0] == 'ligand':

                dat = self._labels[~self._labels[hues[1]].isnull()]

                p = sns.color_palette(cc.glasbey, n_colors=len(dat[hues[0]].unique()))

                max = dat[hues[1]].max()
                min = dat[hues[1]].min()

                d=0.7


                cat_cols = [p[i] for i in range(len(dat[hues[0]].unique()))]
                cat_cols = [np.array([r,g,b]) for r,g,b in cat_cols]
                cat_cols = [rgb*(1-d)+d*rgb for rgb in cat_cols]

                c = [(1-(dat.loc[i,hues[1]]-min)/(max-min))*np.array([1,1,1]) + 
                (dat.loc[i,hues[1]]-min)/(max-min)*cat_cols[dat.loc[i,hues[0]]] for i in dat.index]

                palette = None
            
            elif len(self._labels[hue].unique()) == 2:
                palette = ['red', 'blue']

            elif self._labels[hue].dtype == 'float64':

                palette = cm.get_cmap('winter_r')

            else:
                palette = sns.color_palette(cc.glasbey, n_colors=len(self._labels[hue].unique()))
        
        else:
            legend=None
            palette=None

        fig, ax = plt.subplots()

        legend = None
        font = 'Inter'

        if palette is None:

            ax = sns.scatterplot(dat, x='x', y='y', palette=palette, legend=legend, zorder=1, ax=ax, c=c, s=s)
        
        else: 
            ax = sns.scatterplot(self._labels, x='x', y='y', hue=hue, palette=palette, legend=legend, s=s, zorder=1, ax=ax)

        if hue is None:
            hue = 'None'
        

        ax.set_xlabel(self._axname+'1', fontsize=fontsz, fontname=font)
        ax.set_ylabel(self._axname+'2', fontsize=fontsz, fontname=font)
        
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        for tick in ax.get_xticklabels():
            tick.set_fontname(font)
        for tick in ax.get_yticklabels():
            tick.set_fontname(font)

        plt.tight_layout()

        if save:
            contains_slash = hue.find('/')

            if contains_slash > -1:
                hue = hue[:contains_slash]+'_'+hue[contains_slash+1:]
            
            plt.savefig(f'scatter_red_{self._axname}_color_{hue}.png', dpi=450)
        
        return ax