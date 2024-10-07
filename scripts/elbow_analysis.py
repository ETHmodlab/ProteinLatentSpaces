import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import random

from pipeline import ProteinDataset
from pipeline.utils import average_structures

model_id = 170 #identifier of the model whose latent spaces should be used
latent_dim = 512  #dimensionality of the (unreduced) protein latent space
subset_frac = 1  #what portion of the original dataset should be used 
frac_range = np.linspace(0.1,1,5) #range of all portion sizes that to use in the 3D plot
random_seed = 1
step = 10  #step size for the number of components
resample = 20  #how many sampling runs to do, the results will be averaged
data_path = './data'  #path to the latent dicts and protein data 
method = 'pca'  #either pca or kernelpca
average = False

np.random.seed(random_seed)

def elbow_analysis(model_id: int, latent_dim: int, save: bool = False):
    ''' 
    Get the PCA variance plot for the given model

    Parameters
    ----------
    model_id: int
        identifier of the model whose latent spaces should be used

    latent_dim: int
        dimensionality of the (unreduced) protein latent space

    save: bool
        Whether to save the variance plot, default = False

    Returns
    -------
    None
    '''

    dataset = ProteinDataset(model_id, data_path)

    comp_range = np.arange(0, latent_dim, step=step)

    # variance plot
    plot_2D(comp_range, variance_per_component(subset_frac, dataset), save)
    

def reduce(X: np.ndarray):
    '''
    Apply PCA to X and get the variance per component
    
    Parameters
    ----------
    X: np.ndarray

    Returns
    -------
    np.ndarray
        Fraction of variance explained per component
    '''

    if method == 'pca':

        reducer = make_pipeline(StandardScaler(), PCA(random_state=random_seed))
        reducer.fit(X)

        return reducer.steps[1][1].explained_variance_ratio_

    elif method == 'kernelpca':

        reducer = KernelPCA(n_components=latent_dim, kernel='rbf', random_state=random_seed)
        reducer.fit(X)
        return reducer.eigenvalues_ / sum(reducer.eigenvalues_)

def plot_3D(comp_range, frac_range, Z):
    '''
    Make a 3D plot and save it

    Parameters
    ----------
    comp_range:
        The x data

    frac_range:
        The y data

    Z:
        The bound variable

    Returns
    -------
    None
    '''

    X, Y = np.meshgrid(comp_range, frac_range)

    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z,
                cmap='viridis', edgecolor='none')

    ax.set_ylabel('Dataset Fraction', fontsize='large')
    ax.set_xlabel('# Principal Components', fontsize='large')
    ax.set_zlabel('Variance explained', fontsize='large')
    plt.savefig(f'variance_plot_3D_{method}.png')

    plt.show()

def resolve_reduction():
    if average:
        return '_averaged'
    else:
        return ''
    

def plot_2D(comp_range, variance_mean, save: bool = False):
    ''' 
    Make a 2D-variance plot of variance_mean vs comp_range and save it

    Parameters
    ----------
    comp_range: Iterable
        The x data

    variance_mean: Iterable
        The y data

    save: bool
        Whether to save the plot, default = False

    Returns
    --------
    None
    '''

    fs='x-large'

    plt.plot(comp_range, variance_mean*100)
    plt.ylabel('Variance explained [%]', fontsize=fs)
    plt.xlabel('# Principal Components', fontsize=fs)

    plt.tight_layout()
    
    if save:
        plt.savefig(f'variance_plot_{method}{resolve_reduction()}.png')

    plt.show()



def variance_per_component(subset_frac, dataset):
        ''' 
        Determine the fraction of variance explained by each component

        Parameters
        ----------
        subset_frac: float
            Fraction of the dataset to use

        dataset: ProteinDataset
            Contains the latent coordinates

        Returns
        -------
        np.ndarray
            The fraction of variance explained per component
        '''

        if subset_frac < 1:

            sids = []

            for i in range(resample):

                sids.append(random.sample(dataset.structure_ids, int(dataset.num_structures*subset_frac)))
    
        else:
        
            sids = [dataset.structure_ids]

    
        print(f'Dataset size is {len(sids[0])}')

        y_mean = np.zeros((latent_dim,))


        for indices in sids:

            tids = [dataset._get_tid_from_structure_id(i) for i in indices]

            X = dataset.get_structure_latent_features(indices)

            if average:

                X, tids = average_structures(X, tids)

            y_mean += reduce(X)
    
        y_mean /= len(sids)

        
    
        for i in range(1, len(y_mean)):
            y_mean[i] = y_mean[i-1]+y_mean[i]
        
        return y_mean[::step]




if __name__ == '__main__':
    elbow_analysis(model_id, latent_dim)