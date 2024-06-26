import pandas as pd
import numpy as np
from tqdm import tqdm
import mmap
import matplotlib.pyplot as plt

from dataset import ProteinDataset

def get_num_lines(file_path):
        '''Count the number of lines in the file at file_path'''
        #https://blog.nelsonliu.me/2016/07/30/progress-bars-for-python-file-reading-with-tqdm/
        fp = open(file_path, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
                lines += 1
        return lines

def get_families(file_path: str, uniprot_entry_names: pd.Series):
    '''
    Reads the UniProt similarity text file and returns a dictionary which contains the members of each family
    
    Parameters
    ----------
    file_path: str
        Path to the similarity file from UniProt

    uniprot_entry_names: pd.Series
        UniProt entry names

    Returns
    --------
    families: dict
        A dictionary that contains a list of member proteins for each family
    '''

    families = {}

    with open(file_path, 'r') as f:
            current_fam = None

            for line in tqdm(f, total=get_num_lines(file_path), desc='Reading in family database'):

                    line = line.strip().strip(',\n').split(',')

                    if len(line) == 1 and line[0][-6:] == 'family':
                            current_fam=line[0]
                            families[current_fam] = []

                    else:
                            for protein in line:
                                    protein = protein.strip()
                                    protein = protein.split(' ')

                                    if uniprot_entry_names.str.contains(protein[0]).any():
                                        families[current_fam].append(protein[0])

    return families




def assign_membership(tids, 
                family: str,  
                dataset: ProteinDataset, 
                families: dict, 
                family_hierachy: pd.DataFrame,
                level: str = ''):
        
        '''Constructs a boolean array that reflects membership to family

        Parameters
        -----------
        tids: array-like 
            Contains the protein target ids of the structures in the dataset

        family: str
            name of the family for which membership should be assigned
        
        families: dict
           family dictionary as provided by get_families
        
        family hierachy: pd.DataFrame
            hierachy dataframe as provided in get_hierachy
        
        level: str
            one of super, sub or empty string, default = ''

        Returns
        --------
        membership: np.ndarray
            Boolean array of the same length as tids
        '''

        if level not in ['sub', 'super', '']:
                raise ValueError(f'{level} is not a valid hierachy level')
        
        membership = np.zeros(len(tids), dtype = bool) 

        for fam in tqdm(family_hierachy[family_hierachy[f'{level}family']==family].index, desc = f'Searching for {family} {level}family members'):

                for i in range(len(tids)):
        
                        membership[i] = membership[i] or is_member(fam, tids[i], dataset, families)
        
        return membership


def is_member(family: str, tid: int, dataset: ProteinDataset, families: dict):
        ''' 
        Check membership of a given protein to a given family

        Parameters
        ----------
        family: str
            The name of the family

        tid: int
            The target id of the protein

        dataset: ProteinDataset
            Dataset which contains the protein to get the UniProt entry name from

        families: dict
            The family membership dictionary as produced by get_families

        Returns
        -------
        bool
            Whether the protein is a member of the family
        '''

        df = dataset.proteins_table_df

        uniprot_id = df[df['tid (ChEMBL)'] == tid].loc[:,'entry_name (UniProt)'].values[0]

        return uniprot_id in families[family]


def _parse(name: str, idx: str, df: pd.DataFrame):
        '''
        Parameters
        ----------
        name: str
            family name

        idx: str
            composite family key that name is part of

        df: pd.DataFrame
            family hierarchy data frame as produced by get_hierachy

        Returns
        -------
        bool
        '''

        if name[-11:-6] == 'super':
                df.loc[idx, 'superfamily'] = name[:-11].strip()
                return True
    
        elif name[-9:-6] == 'sub':
                df.loc[idx, 'subfamily'] = name[:-9].strip()
                return True
    
        elif name[-6:] == 'family':
                df.loc[idx, 'family'] = name[:-6].strip()
                return True
    
        else: 
                return False

def get_hierachy(family_keys):
        '''Extracts the hierachy from the keys of the families dictionary obtained via get_families
        returns a dataframe that indicates family and superfamily and/or subfamily (if applicable) to each key
        
        Parameters
        ----------
        family_keys: list
            List of composite family keys separated by ., i.e. <superfamily>.<family>.<subfamily>

        Returns
        -------
        family_hierarchy: pd.DataFrame
             Contains 3 string columns indicating superfamily, family and subfamily for every
             composite key in family_keys 
                        
        '''


        family_hierachy = pd.DataFrame(columns=['superfamily', 'family', 'subfamily'], index=family_keys)

        for fam in family_keys:

                branches = fam.split('.')

                for i in range(len(branches)):
                        if not _parse(branches[i], fam, family_hierachy):
                                branches[i+1] = branches[i]+'.'+branches[i+1]
        
        return family_hierachy


def plot_frequency(level: str, reduced_dict: dict, hierachy: pd.DataFrame, threshold:int):
    '''
    Create a historgram of all families and their respective frequencies

    Parameters
    ----------
    level: str
        Must be one of family, superfamily, subfamily

    reduced_dict: dict
        Family dict which only contains the proteins from the dataset

    hierarchy: pd.DataFrame
        Hierarchy dataframe as produced by get_hierachy

    theshold: int
        Number of occurrences above which to show a family in the histogram

    Returns
    -------
    None
    '''

    names_full = hierachy[level].unique().tolist()

    if level == 'superfamily' or level == 'subfamily':
        names_full.pop(0)

    names = []
    y =[]



    for name in names_full:
    
        sum = 0
        for fam in hierachy[hierachy[level]==name].index:
            sum += len(reduced_dict[fam])

    
        if sum > threshold:
            names.append(name)
            y.append(sum)

    x = range(1, len(names)+1)

    plt.bar(x=x,height=y, tick_label=names)
    plt.xticks(fontsize = 3, rotation = 45)

    plt.title(f'{level} threshold {threshold}')
    plt.tight_layout()


    plt.savefig(f'frequency_barplot_{level}_threshold_{threshold}.png', dpi=450)
    plt.show()