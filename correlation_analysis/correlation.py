import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA
import seaborn as sns
import colorcet as cc

from correlation_analysis.regression import linreg

CLUSTER_METHOD = 'agglomerative'
FONTSZ = 25

def get_correlation(
    annotations: np.ndarray,
    coordinates: np.ndarray
    ):
    ''' 
    Calculate pearson correlation between annotations and coordinates

    Parameters
    ----------
    annotations: np.ndarray
        The datapoint annotations

    coordinates: np.ndarray
        The projected coordinate of the datapoints

    Returns
    -------
    float
        The Pearson correlation
    '''
    if len(annotations) != len(coordinates):
        raise ValueError('Arrays must have same length')

    coords = np.array([coordinates, annotations])

    corr = np.corrcoef(coords)

    return corr[0,1]


def cluster_regression(
    cluster: int,
    ann: str,
    coords_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    anns_df: pd.DataFrame,
    save: bool = False
    ):
    '''
    Regress annotation ann in the given cluster
    
    Parameters
    ----------
    cluster: int
        The index of the cluster

    ann: str
        The name of the annotation

    coords_df: pd.DataFrame
        Reduced latent space data (latent_coords.csv)

    labels_df: pd.DataFrame
        Cluster assignment (cluster_labels.csv)

    anns_df: pd.DataFrame
        Annotations (annotations.csv)

    save: bool
        Whether to save the plots
    
    Returns
    -------
    None
    '''
    labels_df[ann] = anns_df[ann].values

    ann_max = labels_df[ann].max()
    ann_min = labels_df[ann].min()

    # get the data points in the cluster
    coords_red = coords_df[labels_df[CLUSTER_METHOD].isin([cluster])]
    labels_red = labels_df[labels_df[CLUSTER_METHOD].isin([cluster])]

    # remove any points without annotation
    coords_red = coords_red[labels_red[ann].notnull()]
    labels_red = labels_red[labels_red[ann].notnull()]

    # get the gradient direction with linear regression
    coefs, intercept = linreg(coords_red, labels_red[ann])

    project_cluster(cluster, coords_red, labels_red[ann], coefs, True, ann_min, ann_max, save)



def project_cluster(
    cluster: int,
    cluster_coords: pd.DataFrame,
    annotation: pd.Series, 
    gradient: np.ndarray,
    normalize: bool,
    ann_min: float,
    ann_max: float,
    save: bool = False
    ):
    '''
    Project the datapoints from a cluster onto the gradient direction and plot them

    Parameters
    ----------
    cluster: int
        The cluster index

    cluster_coords: pd.DataFrame
        Coordinates of the cluster points

    annotation: pd.Series
        Annotation of the cluster points

    gradient: np.ndarray
        The gradient of the annotation in the given cluster

    normalize: bool
        Whether to normalize the gradient before projecting onto it

    ann_min: float
        Global minimum value of the annotation

    ann_max: float
        Global maximum value of the annotation

    save: bool
        Whether to save the plots, default = False

    Returns
    --------
    None
    '''

    if normalize:

        gradient /= np.sqrt(np.inner(gradient,gradient))

    X_projected = cluster_coords.to_numpy() @ gradient
    X_projected -= np.mean(X_projected)

    corr = get_correlation(annotation.to_numpy(), X_projected)

    # add dummy points for correct colormapping
    X_projected = np.append(X_projected, [-5, -5], axis=None)

    annotation = np.append(annotation, [ann_max, ann_min], axis=None)

    plt.scatter(X_projected, annotation, c='black', s=20)

    plt.xlabel(r'$c_{\vec{\nabla},}$'+r'$_{}$'.format(cluster),  fontname = 'Inter', fontsize=FONTSZ)
    plt.ylabel('# heavy atoms in ligand', fontname = 'Inter', fontsize=FONTSZ)

    name = f'cluster_{cluster}_proj_norm.png' if normalize else f'cluster_{cluster}_proj.png'


    plt.text(x = 0.75, y = 10, s = r'$r_{}$ = '.format(cluster)+str("%.2f" % corr), fontname = 'Inter', fontsize=FONTSZ, color = 'black')
    #plt.text(x = -1.5, y = 76, s = f'Cluster {cluster}', fontname = 'Inter', fontsize='xx-large', color = color)

    xticks = [-1, 0, 1]
    plt.xticks(xticks, labels= xticks, fontsize=FONTSZ, fontname='Inter')
    plt.yticks(fontsize=FONTSZ, fontname='Inter')

    plt.ylim(5,75)
    plt.xlim(-1.5,2)

    plt.tight_layout()

    if save:
        plt.savefig(name , dpi=500)

    plt.show()


def make_latent_space_plots(
        coords_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        save: bool = False):
    
    '''
    Create a plot of the full latent space and an excerpt thereof

    Parameters
    ----------
    coords_df: pd.DataFrame
        Contains the reduced latent coordinates (latent_coords.csv)

    labels_df: pd.DataFrame
        Contains the cluster assignment (cluster_labels.csv)

    save: bool
        Whether to save the plots, default = False

    Returns
    --------
    None
    '''

    # perform PCA on all datapoints
    pca = PCA(n_components=2, random_state=77)
    X_red = pca.fit_transform(coords_df)

    df = pd.DataFrame({'PC1': X_red[:,0], 'PC2': X_red[:,1], 
    CLUSTER_METHOD: labels_df[CLUSTER_METHOD].values})

    # plot the full dataset
    sns.scatterplot(df, x='PC1', y='PC2', hue=CLUSTER_METHOD, palette=sns.color_palette(cc.glasbey, n_colors=10), legend=None, s=20)

    plt.xlabel('PC1', fontsize=FONTSZ, fontname='Inter')
    plt.ylabel('PC2', fontsize=FONTSZ, fontname='Inter')
    
    xticks=[-20, -10, 0, 10, 20]

    plt.xticks(xticks, xticks, fontsize=FONTSZ, fontname='Inter')
    plt.yticks(fontsize=FONTSZ, fontname='Inter')

    plt.tight_layout()

    if save: 
        plt.savefig(f'top_10_clusters.png', dpi=500)

    yticks=[-10, -5, 0, 5, 10]
    xticks=[5, 10, 15, 20]

    plt.xticks(xticks, xticks, fontsize=FONTSZ, fontname='Inter')
    plt.yticks(yticks, yticks, fontsize=FONTSZ, fontname='Inter')

    # plot an excerpt of the dataset
    plt.xlim(1, 21)
    plt.ylim(-12, 10.7)

    plt.tight_layout()

    if save:
        plt.savefig(f'latent_space_excerpt.png', dpi=500)

    plt.show()


def perform_correlation_analysis(property: str,
                                 coords_df: pd.DataFrame,
                                 labels_df: pd.DataFrame,
                                 anns_df: pd.DataFrame,
                                 save: bool = False
                                 ):

    ''' 
    Regress a property in each individual cluster found during a run of the pipeline

    Parameters
    ----------
    property: str
        The annotation to regress

    coords_df: pd.DataFrame
        Contains the reduced latent coordinates

    labels_df: pd.DataFrame
        Contains the cluster assignment, i.e. a column with name CLUSTER_METHOD

    anns_df: pd.DataFrame
        Contains the annotations, i.e. a column with name property

    save: bool
        Whether to save the plots in the current directory, default = False

    Returns
    -------
    None
    '''

    if len(coords_df.index) != len(anns_df.index):

        anns_df = anns_df[anns_df['tid'].isin(labels_df['tid'])]

    for cluster in labels_df[CLUSTER_METHOD].unique():

        plt.figure(figsize=(6.4, 4.8))
        cluster_regression(cluster, property, coords_df, labels_df, anns_df, save)


    make_latent_space_plots(coords_df, labels_df, save)
