# src/preparation/generate_proteins_table.py

# Remark: Must be run from base directory (so that './data' is a direct subdirectory)

# Import public modules
import torch
import tqdm
import collections
import pandas as pd
from pathlib import Path

# Import custom module
import utils

# Define folder and file paths
# Define the base directory to the dictionaries containing the latent features
latent_dir = './data/latent/'

# Define the file path to the TID mapping table
file_path_tid_map  = './data/tables/tid_mapping_chembl30.csv'

# Define the file path in which the proteins table should be saved in
file_path_proteins_table = './data/tables/proteins_table.tsv'

# Throw an error if this file is not run as 'main' (i.e. from the command line)
if __name__=='__main__':
    # Define a list of the GNN model names
    # Remark: Different GNN models were trained (with the same architecture and hyperparameters)
    #         and the latent spaces of each protein were determined for each of these GNN models.
    gnn_model_names = ['model_170', 'model_171', 'model_172', 'model_173', 'model_174', 'model_175', 'model_176']

    # Loop over the GNN model names, load the latent feature dictionaries, and extract all their keys
    latent_dict_keys_list = list()
    print(f"Extracting the keys (protein structure identifiers) of the latent dictionaries...")
    for gnn_model_name in tqdm.tqdm( gnn_model_names ):
        # Determine the GNN model ID from the GNN model name
        # Remark: The GNN model names have the form 'model_<model ID>'
        gnn_model_id = gnn_model_name.split('_')[1]

        # Construct the file name of the file containing the dictionary of the latent features
        file_name = f"latent_dict_{gnn_model_id}.pt"

        # Construct path to the file containing the dictionary of the latent features
        file_path = Path(latent_dir, file_name)

        # Load the file
        # Remark: It will be loaded as dictionary
        latent_dict = torch.load(file_path)

        # Add the keys of the latent dictionary to the corresponding set by updating the 
        latent_dict_keys_list += list( latent_dict.keys() )

    print('Extraction done.')
    print()

    # Generate a list of all Target IDs (TIDs)
    # Remarks: (1) The latent dictionary keys (=protein structure identifiers) have the form '<TID>_<PDB ID>'. 
    #              Extract the TID and cast it to an integer.
    #          (2) As there are multiple structures (identified by the PDB ID) for 
    #              each protein (identified by TID), different keys hold the same TID.
    #              Remove duplicates using 'set' and make the set a list afterwards.
    tids = list( set( map(lambda x: int(x.split('_')[0]), set(latent_dict_keys_list)) ) )

    # Load the TID mapping table as pandas.DataFrame
    tid_map_df = pd.read_csv(file_path_tid_map, sep='\t')
    print(f"Loaded the TID mapping table from: {file_path_tid_map}")
    print()

    # Filter the dataframe
    filtered_tid_map_df = tid_map_df[tid_map_df['TID'].isin(tids)]

    # Check that the number of entries in the filtered DataFrame is equal to the number of TIDs
    if len(tids)!=len(filtered_tid_map_df):
        err_msg = f"The number of entries of the filtered TID-mapping DataFrame (={len(filtered_tid_map_df)}) does not match the number of TIDs (={len(tids)})."
        raise ValueError(err_msg)

    # Loop over the rows, which correspond to the protein entries, of the filtered TID mapping DataFrame
    # Remark: The rows will be of type pandas.Series.
    proteins_dict = collections.defaultdict(list)
    print('Scrape information about the proteins from UniProt...')
    for _, protein_series in tqdm.tqdm( filtered_tid_map_df.iterrows(), total=len(filtered_tid_map_df) ):
        # Access certain quantities of the current row/protein
        p_chembl_id               = protein_series['chembl_id']
        uniprot_accession_ids_str = protein_series['uniprot_accession_ids']

        # Append these quantities and others of the protein to their corresponding lists in proteins dictionary
        # Remark: Explicitly state for all quantities that the information was taken from ChEMBL,
        proteins_dict['tid (ChEMBL)'].append(protein_series['TID'])
        proteins_dict['p_chembl_id (ChEMBL)'].append(p_chembl_id)
        proteins_dict['organism (ChEMBL)'].append(protein_series['organism'])
        proteins_dict['target_prefname (ChEMBL)'].append(protein_series['target_prefname'])
        proteins_dict['uniprot_accession_ids_str (ChEMBL)'].append(uniprot_accession_ids_str)

        # Scrape protein information from UniProt
        scraped_protein_info_dict = utils.scrape_protein_info_from_uniprot(p_chembl_id, uniprot_accession_ids_str)

        # Loop over the scraped protein info dictionary
        for key, value in scraped_protein_info_dict.items():
            # Update the key to contain the information that is was taken from UniProt
            key = f"{key} (UniProt)"

            # Append the quantity value to the corresponding list in the proteins dictionary
            proteins_dict[key].append(value)


    print('Scraping done.')
    print()

    # Make the quantities dictionary a pandas.DataFrame
    proteins_df = pd.DataFrame(proteins_dict)

    # Save the DataFrame as .tsv file
    proteins_df.to_csv(file_path_proteins_table, sep='\t', index=False)
    print(f"Saved the table containing the protein information in: {file_path_proteins_table}")