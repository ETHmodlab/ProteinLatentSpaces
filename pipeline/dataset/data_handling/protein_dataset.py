# dataset/data_handling/protein_dataset.py

# Import custom modules
import os
import functools
import torch
import collections
import numpy as np
import pandas as pd
from pathlib import Path

class ProteinDataset(object):
    def __init__(self, gnn_model_id, data_dir='./data'):
        # Assign inputs to class attributes
        self.gnn_model_id = gnn_model_id
        self.data_dir     = data_dir

        ###########################################################################################################################################################
        ### Load the protein (information) table
        ###########################################################################################################################################################
        # Generate the path to the proteins table and check that the file exists
        file_path_proteins_table = str( Path(self.data_dir, 'tables', 'proteins_table.tsv') )
        if not os.path.isfile(file_path_proteins_table):
            err_msg = f"The file 'proteins_table.tsv' could not be found in the directory '{str( Path(self.data_dir, 'tables') )}'.\nPlease generate it first by running: 'python3 src/preparation/generate_proteins_table.py'."
            raise FileNotFoundError(err_msg)
        # Load this table as pandas.DataFrame
        self.proteins_table_df = pd.read_csv(file_path_proteins_table, sep='\t')
        
        ###########################################################################################################################################################
        ### Load the latent features
        ###########################################################################################################################################################
        # Check that the GNN model ID is available
        if self.gnn_model_id not in ProteinDataset.get_available_gnn_model_ids(data_dir=self.data_dir):
            err_msg = f"The GNN model ID '{self.gnn_model_id}' is not available.\nThe available GNN model IDs are: {self.available_gnn_model_ids}"
            raise ValueError(err_msg)

        # Generate the path to the dictionary containing the latent features of the GNN model
        file_path_latent_dict = str( Path(self.data_dir, 'latent', f"latent_dict_{gnn_model_id}.pt") )
        if not os.path.isfile(file_path_latent_dict):
            err_msg = f"The file 'latent_dict_{gnn_model_id}.pt' could not be found in the directory '{str( Path(self.data_dir, 'latent') )}'.\nPlease check that the file name has the form 'latent_dict_<gnn_model_id>.pt'."
            raise FileNotFoundError(err_msg)
        
        # Load the file as dictionary and make it an Ordered Dictionary
        # Remark: This will ensure that the order of the key-value pairs is always the same.
        self.latent_dict = collections.OrderedDict( torch.load(file_path_latent_dict) )

        ###########################################################################################################################################################
        ### Generate additional class attributes
        ###########################################################################################################################################################
        # Add the file paths to a class attribute dictionary
        self.file_paths = {'proteins_table': file_path_proteins_table, 'latent_dict': file_path_latent_dict}

        # Generate a list of all the unique TIDs of the proteins
        self.tids = list( set( map(lambda x: int(x.split('_')[0]), self.latent_dict.keys()) ) )

    ###########################################################################################################################################################
    ### Define property methods
    ###########################################################################################################################################################
    @property
    def available_protein_info_names(self):
        """ Return the information that is available for each protein. """
        return list( self.proteins_table_df.columns )

    @property
    def structure_ids(self):
        """ Return the protein structure IDs that have the form '<TID>_<PDB ID>' and correspond to the keys of the latent dictionary. """
        return list( self.latent_dict.keys() )

    @property
    def structure_latent_features(self):
        """ Return the protein structure latent features as numpy array and correspond to the values of the latent dictionary. """
        return np.vstack( list(self.latent_dict.values()) )

    @property
    def num_proteins(self):
        """ Return the number of proteins that corresponds to the number of target IDs (tids). """
        return len(self.tids)

    @property
    def num_structures(self):
        """ Return the number of protein structures that corresponds to the number of structure IDs. """
        return len(self.structure_ids)
        
    ###########################################################################################################################################################
    ### Define Getter Methods
    ###########################################################################################################################################################
    def get_latent_features(self, tids):
        """
        Return the latent features of all protein structures of proteins that are specified by their target IDs.
        
        Args:
            tids (str or iterable): A single target ID or an iterable of target IDs.

        Return:
            (numpy.array): The latent features of all the proteins as 2D array of shape (#protein_structures, #features).

        """
        # Get a list of all structure IDs of the proteins specified by their target IDs
        structure_ids = self.get_structure_ids(tids)
        
        # Get the latent features of all these structures as 2D numpy array and return it.
        return self.get_structure_latent_features(structure_ids)

    def get_structure_ids(self, tids):
        """
        Return the structure IDs of a protein specified by its target ID.
        
        Args:
            tids (str, int, or iterable): Single target ID (as str or int) or iterable (e.g. list) of target IDs.

        Return:
            (list): List of all structure IDs of the proteins specified by their target IDs.

        """
        # In case that the input 'tids' is a string or an interger, redefine it a list containing 
        # this single entry (casted to an integer).
        if isinstance(tids, (int, str)):
            tids = [int(tids)]

        # Check that 'tids' is an iterable (e.g. list)
        if not isinstance(tids, collections.abc.Iterable):
            err_msg = f"The input 'tids' must be either a string or an Iterable (e.g. list), got type '{type(tids)}' instead."
            raise TypeError(err_msg)

        # Map all tids to integers
        tids = [int(tid) for tid in tids]
        
        # Check that the target IDs are all valid by checking that all of them are in the set of target IDs
        set_diff = set(tids) - set(self.tids)
        if 0<len(set_diff):
            err_msg = f"The following input target IDs are not valid:\n{list(set_diff)}\nThe valid target IDs are:\n{self.tids}"
            raise ValueError(err_msg)

        # Generate a list of all structure IDs of all the proteins specified by their target IDs and return it
        return list( functools.reduce(lambda x, y: x+self._get_structure_ids_corresponding_to_tid(y), tids, []) )

    def get_structure_latent_features(self, structure_ids):
        """
        Return the latent features of protein structures that are specified by their structure IDs.
        
        Args:
            structure_ids (str or iterable): A single structure ID or an iterable of structure IDs.

        Return:
            (numpy.array): The protein structures' latent features as 2D array of shape (#protein_structures, #features).

        """
        # In case that the input 'structure_ids' is a string, make it a list containing this single entry.
        if isinstance(structure_ids, str):
            structure_ids = [structure_ids]

        # Check that 'structure_ids' is an iterable (e.g. list)
        if not isinstance(structure_ids, collections.abc.Iterable):
            err_msg = f"The input 'structure_ids' must be either a string or an Iterable (e.g. list), got type '{type(structure_ids)}' instead."
            raise TypeError(err_msg)
        
        # Check that the structure IDs are all valid by checking that all of them are in the set of structure IDs
        set_diff = set(structure_ids) - set(self.structure_ids)
        if 0<len(set_diff):
            err_msg = f"The following input structure IDs are not valid:\n{list(set_diff)}\nThe valid structure IDs are:\n{self.structure_ids}"
            raise ValueError(err_msg)

        # Use the latent dictionary to map all structure IDs to their corresponding latent features
        return np.vstack([self.latent_dict[structure_id] for structure_id in structure_ids])
    
    def get_protein_info(self, protein_info_name, ids):
        """
        Return a list of the protein information for a single ID as integer or string or an iterable (e.g. list) 
        containing ids.

        Args:
            protein_info_name (str): Protein information name (e.g. 'sequence (UniProt)').
            ids (int, str, or iterable): Single ID (either target ID or structure ID) as integer or string of iterable 
                (e.g. list) of IDs.

        Return:
            (obj or list of obj): The protein information for a single input ID or a list of the protein information
                for the IDs in the input iterable.

        Remark: Checking of the ids and the protein info name is done in 'get_protein_info_for_id'.

        """
        # Differ cases depending on the type of the input 'ids'
        if isinstance(ids, (int, str)):
            # In this case, the ids is actually a single ID, so call the class method '_get_protein_info_for_id'
            # for this ID
            return self._get_protein_info_for_id(protein_info_name, ids)
        elif isinstance(ids, collections.abc.Iterable):
            # In this case, map the class method '_get_protein_info_for_id' to all entries in the iterable ids.
            return list( map(lambda x: self._get_protein_info_for_id(protein_info_name, x), ids) )
        else:
            err_msg = f"The input 'ids' must be either a single ID (either target ID or structure ID) or an iterable (e.g. list) of IDs, got type '{type(ids)}' instead."
            raise TypeError(err_msg)

    ###########################################################################################################################################################
    ### Define auxiliary methods
    ###########################################################################################################################################################
    def _get_structure_ids_corresponding_to_tid(self, tid):
        """
        Return the structure IDs of a protein specified by its target ID.
        
        Args:
            tid (str or int): Target ID of the protein.

        Return:
            (list): List of all structure IDs of the protein.

        """
        # Generate a list of all structure IDs that start with the target ID.
        # Remark: Structure IDs have the form '<TID>_<PDB ID>'.
        tid_pattern = str(tid) + '_'
        return [structure_id for structure_id in self.structure_ids if structure_id.startswith(tid_pattern)]

    def _get_protein_info_for_id(self, protein_info_name, id):
        """
        Return the protein information (specified by 'protein_info_name') for an input ID (TID or structure ID). 
        
        Args:
            protein_info_name (str): Protein information name (e.g. 'sequence (UniProt)')
            id (int or str): Target ID or structure ID as integer or string.

        Return:
            (obj): Protein information for the input target ID.
        
        """
        # Differ cases for the different possible ID types and obtain the target ID (tid) from it
        if isinstance(id, int):
            # In case that the id is an integer, assume it corresponds to the TID
            tid = id
        else:
            # Otherwise, assume that the ID is a string and differ cases
            # Remark: Structure IDs have the form '<TID>_<PDB ID>', so check if the ID contains a '_' or not.
            if '_' in id:
                # Assume the ID is a structure ID and thus get the target ID (tid) from it
                structure_id = id
                tid          = self._get_tid_from_structure_id(structure_id)
            else:
                # Assume the ID is a target ID and thus directly assign it
                # Remark: Cast the tid to an integer (just in case it was a string)
                tid = int(id)

        # Check that the TID exists
        if tid not in self.tids:
            err_msg = f"The input TID '{tid}' is not one of the available TIDs, which are:\n{self.tids}"
            raise ValueError(err_msg)

        # Check that 'protein_info_name' is one of the available ones
        if protein_info_name not in self.available_protein_info_names:
            err_msg = f"Got '{protein_info_name}' as input but this is not one of the available 'protein_info_names':\n{self.available_protein_info_names}"
            raise ValueError(err_msg)

        # Get the protein information for the target ID and return it
        return self._get_protein_info_for_tid(protein_info_name, tid)

    def _get_tid_from_structure_id(self, structure_id):
        """
        Return the target ID (tid) for an input structure ID that has the form '<TID>_<PDB ID>'.
        
        Args:
            structure_id (str): Structure ID.

        Return:
            (int): Target ID as integer.

        """
        # Split on the structure ID on '_' and return the first element
        return int( structure_id.split('_')[0] )
    
    def _get_protein_info_for_tid(self, protein_info_name, tid):
        """
        Return the protein information (specified by 'protein_info_name') for an input target ID (tid). 
        
        Args:
            protein_info_name (str): Protein information name (e.g. 'sequence (UniProt)').
            tid (int): Target ID.

        Return:
            (obj): Protein information for the input target ID.
        
        Remark: This method does not check the correctness neither of the target ID nor of the protein info name, 
                so these checks have to be made before using this method.

        """
        # Filter the protein table DataFrame by the target ID
        filtered_df = self.proteins_table_df[self.proteins_table_df['tid (ChEMBL)']==tid]

        # Return the value in this filtered DataFrame corresponding to the protein information
        # Remark: Access the single value of the pandas.Series object 'filtered_df[protein_info_name]' using '.iloc[0]'
        return filtered_df[protein_info_name].iloc[0]
    
    def get_unique_protein_info_entries(self, protein_info_name):
        """ Return a list of the unique entries for a protein information (e.g. 'organism (ChEMBL)') available. """
        # Check that 'protein_info_name' is one of the available ones
        if protein_info_name not in self.available_protein_info_names:
            err_msg = f"Got '{protein_info_name}' as input but this is not one of the available 'protein_info_names':\n{self.available_protein_info_names}"
            raise ValueError(err_msg)

        # Get all the protein info entries and get a list of the unique ones (using 'set()')
        unique_protein_info_entries = list( set( self.proteins_table_df[protein_info_name] ) )

        # Sort and return them
        unique_protein_info_entries.sort()
        return unique_protein_info_entries

    ###########################################################################################################################################################
    ### Define static methods
    ###########################################################################################################################################################
    @staticmethod
    def get_available_gnn_model_ids(data_dir='./data'):
        """ Return a list of the available GNN model IDs. """
        # Get a list of the names of the files found in the latent dictionary
        file_names_latent_dir = os.listdir(Path(data_dir, 'latent'))

        # Filter all file names that have a '.pt' extension
        file_name_latent_dict_list = list( filter(lambda x: x.endswith('.pt'), file_names_latent_dir) )

        # Map each file name to their corresponding GNN model ID
        gnn_model_ids = list( map(ProteinDataset._map_file_name_to_gnn_model_id, file_name_latent_dict_list) )

        # Sort these IDs and return them
        gnn_model_ids.sort()

        return gnn_model_ids
    
    @staticmethod
    def _map_file_name_to_gnn_model_id(file_name_latent_dict):
        """ Map the file name of a 'latent_dict' file to the corresponding GNN model ID. """
        # Remark: These 'latent_dict' file names have the form 'latent_dict_<gnn_model_id>.pt'
        
        # First remove the file extension ('.pt' here) using 'os.path.splitext', which
        # returns a tuple ('<file_name>', '<file_extension>')
        file_name_latent_dict_no_ext = os.path.splitext(file_name_latent_dict)[0]

        # Split the file name by '_', which will return a tuple of all elements that were separated 
        # by '_' in the splitted string. Take the last entry, which will correspond to the the string 
        # representation of the GNN model ID. Cast this string to an integer and return it.
        return int( file_name_latent_dict_no_ext.split('_')[-1] )




