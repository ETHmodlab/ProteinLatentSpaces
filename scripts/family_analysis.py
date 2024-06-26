import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import colorcet as cc
from sklearn.decomposition import PCA
import matplotlib.patches as pa

PALETTE = sns.color_palette(cc.glasbey, n_colors=9)
FONTSZ = 35
FONT = 'Inter'


def analyze_family(
    family: str,
    df: pd.DataFrame,
    cluster_method: str,
    ticklabel: str,
    save: bool = False
    ):

    ''' 
    Get a barplot color-coded by cluster that indicates the number of structures in each 
    cluster from the given family

    Parameters
    ----------
    family: str
        The family to analyze

    df: pd.DataFrame
        Contains the cluster assignment and family assignment

    cluster_method: str
        Name of the column with the cluster assignment

    ticklabel: str
        The abbreviation of the family name displayed on the plot

    save: bool
        Whether to save the plot, default = False

    Returns
    -------
    None
    '''

    plot_data = pd.DataFrame(index=['in family', 'not in family', 'sum'], columns=df[cluster_method].unique())
    prev_df = pd.DataFrame(index=['in family', 'not in family'])
    prev_df[0] = [0,0]

    plt.rcParams["figure.figsize"] = (5.5, 6.5)

    for clus in range(len(plot_data.columns)):

        curr_df = df.loc[df[cluster_method]==clus, :]

        plot_data.loc['in family', clus] = len(curr_df.loc[curr_df['family']==family,:])
        plot_data.loc['not in family', clus] = len(curr_df.loc[curr_df['family']!=family, :])

        bottom = 0 if clus==0 else plot_data[clus-1]

        plot_data.loc['sum', clus] = plot_data.loc['in family', clus]+plot_data.loc['not in family', clus]
    
    clus_order = plot_data.sort_values(by='in family', axis=1, ascending=True, inplace=False).columns

    fig, ax = plt.subplots(1)

    for idx in range(len(clus_order)):

        clus = clus_order[idx]

        height = 1.25

        ax.add_patch(pa.Rectangle((0, idx*height), width=0.6, height=height, facecolor=PALETTE[clus], alpha=0.5))
        ax.add_patch(pa.Rectangle((0.8, idx*height), width=0.6, height=height, facecolor=PALETTE[clus], alpha=0.5))

        plt.text(0.3, (idx+0.5)*height, s= plot_data.loc['in family', clus], ha='center', va='center', fontname=FONT, fontsize=FONTSZ)
        plt.text(1.1, (idx+0.5)*height, s= plot_data.loc['not in family', clus], ha='center', va='center', fontname=FONT, fontsize=FONTSZ)
    
    ax.set_ylim(0, 8.75)
    ax.set_xlim(0, 1.4)

    ax.tick_params(top=False,
               bottom=False,
               left=False,
               right=False,
               labelleft=True,
               labelbottom=True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.set_xticks([0.3, 1.1], [ticklabel, 'others'], fontname=FONT, fontsize=FONTSZ)
    ax.set_yticks([])
    plt.tight_layout()

    names = family.split('/')

    if save:

        if len(names) == 1:

            plt.savefig(f'{family}.png', dpi=500)
        
        else:
            plt.savefig(f'{names[0]}_{names[1]}.png', dpi=500)

    plt.show()


def make_scatterplot(
    coords: pd.DataFrame,
    labs: pd.DataFrame,
    family: str,
    save: bool = False
    ):
    ''' 
    Create a scatterplot of the reduced latent space with
    the given family highlighted in black

    Parameters
    -----------
    coords: pd.DataFrame
        Contains the coordinates of the points in the reduced
        latent space

    labs: pd.DataFrame
        Contains the cluster labels and family assignment

    family: str
        The name fo the family

    save: bool
        Whether to save the plots, default = False

    Returns
    -------
    None
    '''

    plt.rcParams["figure.figsize"] = (6.4, 4.8)

    members = coords[labs['family'].isin([family])]
    other = coords[~labs['family'].isin([family])]

    reducer = PCA(n_components=2, random_state=71)
    reducer.fit(coords.values)
    
    members_X = reducer.transform(members.values)
    other_X = reducer.transform(other.values)

    plt.scatter(other_X[:,0], other_X[:,1], c='lightgrey', s=15, edgecolors='white', linewidths=0.2)
    plt.scatter(members_X[:,0], members_X[:,1], c='black', s=15, edgecolors='white', linewidths=0.2)

    xticks=[-0.5, 0, 0.5, 1]
    yticks= np.array(range(-8, 12, 4))*0.1

    plt.xticks(xticks, xticks, fontsize=FONTSZ, fontname=FONT)
    plt.yticks(yticks, yticks, fontsize=FONTSZ, fontname=FONT)

    plt.xlabel('PC1',fontsize=FONTSZ, fontname=FONT)
    plt.ylabel('PC2',fontsize=FONTSZ, fontname=FONT)

    plt.tight_layout()

    names = family.split('/')

    if save:

        if len(names) == 1:

            plt.savefig(f'scatter_{family}.png', dpi=500)
        
        else:
            plt.savefig(f'scatter_{names[0]}_{names[1]}.png', dpi=500)

    plt.show()
