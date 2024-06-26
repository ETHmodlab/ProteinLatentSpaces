import requests
import pandas as pd
from tqdm import tqdm
from rdkit.Chem import rdMolDescriptors, Crippen, rdchem, SDMolSupplier
import rdkit
import math

from dataset import ProteinDataset

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

def annotate_uniprot(df: pd.DataFrame):
    '''
    Extracts molecular weight, sequence length and keyword annotations for a UniProt 
    id and appends them to the provided DataFrame

    Parameters
    ---------
    df: pd.DataFrame 
        Has to store the UniProt accession ids in df['acc_id']

    Returns
    --------
    df: pd.DataFrame
        Input data frame with new column for each UniProt property
    '''

    for id in tqdm(df['acc_id'], desc='Loading UniProt annotations'):

        URL = ("http://www.uniprot.org/uniprot/" + id + ".json")
        result = requests.get(URL)
        data = result.json()

        df.loc[df['acc_id']==id,'mol_weight'] = data['sequence']['molWeight']
        df.loc[df['acc_id']==id,'length'] = data['sequence']['length']

        for keyword in data['keywords']:

            if df.loc[df['acc_id']==id, keyword['category']].iloc[0] == 'NaN':

                df.loc[df['acc_id']==id, keyword['category']] = keyword['name']
        
            else:
        
                df.loc[df['acc_id']==id, keyword['category']] = df.loc[df['acc_id']==id, keyword['category']].iloc[0] + f', {keyword["name"]}'
        
    
    return df
        

def extract_pdb_batch(pdb_ids: list):
    '''
    Extracts the annotation data for a single batch of PDB ids and returns a list of dictionaries

    Parameters
    ----------
    pdb_ids: list
        A list of PDB ids as strings for which to extract the annotations, case insensitive

    Returns
    --------
    result: list
    Contains a dictionary with annotation data for each pdb id
    '''

    id_string = '['
    for id in pdb_ids:

        id_string += f'"{id}" '
  
    id_string += ']'

    URL = ('https://data.rcsb.org/graphql?query='
    '{entries'
    f'(entry_ids: {id_string.upper()})'
    '{rcsb_id cell {Z_PDB} diffrn {ambient_temp} diffrn_detector {detector} diffrn_source {source type} em_3d_reconstruction {resolution symmetry_type} '
    'em_diffraction_stats {high_resolution} em_experiment {aggregation_state reconstruction_method} em_imaging {accelerating_voltage microscope_model} '
    'em_image_recording {film_or_detector_model} em_single_particle_entity {point_symmetry} em_specimen {embedding_applied staining_applied vitrification_applied} exptl {method} '
    'exptl_crystal {density_Matthews density_percent_sol} exptl_crystal_grow {method pH temp} pdbx_nmr_ensemble {conformer_selection_criteria conformers_calculated_total_number conformers_submitted_total_number} '
    'pdbx_nmr_exptl {type} pdbx_nmr_exptl_sample_conditions {ionic_strength pH pressure pressure_units temperature} pdbx_nmr_refine {method} pdbx_nmr_spectrometer {field_strength} '
    'rcsb_binding_affinity {comp_id} rcsb_entry_container_identifiers {entry_id} rcsb_entry_info {deposited_atom_count deposited_model_count deposited_nonpolymer_entity_instance_count '
    'deposited_polymer_entity_instance_count deposited_polymer_monomer_count deposited_solvent_atom_count disulfide_bond_count entity_count molecular_weight nonpolymer_entity_count polymer_entity_count_DNA '
    'polymer_entity_count_nucleic_acid_hybrid polymer_entity_count_protein polymer_entity_count_RNA selected_polymer_entity_types diffrn_resolution_high {value}} '
    'struct_keywords {pdbx_keywords} reflns {B_iso_Wilson_estimate} polymer_entities {entity_poly {rcsb_entity_polymer_type rcsb_sample_sequence_length type} rcsb_polymer_entity {pdbx_number_of_molecules '
    'formula_weight rcsb_enzyme_class_combined {ec} rcsb_macromolecular_names_combined {name}} entity_src_gen {plasmid_name} rcsb_entity_host_organism {ncbi_scientific_name} '
    'rcsb_entity_source_organism {ncbi_taxonomy_id} rcsb_polymer_entity_container_identifiers {entity_id}} assemblies {rcsb_assembly_info {polymer_monomer_count polymer_entity_instance_count} '
    'pdbx_struct_assembly {oligomeric_count} rcsb_struct_symmetry {kind oligomeric_state stoichiometry type}} nonpolymer_entities {nonpolymer_comp {chem_comp {formula_weight name} rcsb_chem_comp_descriptor { '
    'InChI}} rcsb_nonpolymer_entity_container_identifiers {entity_id}} branched_entities {rcsb_branched_entity_container_identifiers {rcsb_id chem_comp_monomers} pdbx_entity_branch {rcsb_branched_component_count} '
    'rcsb_branched_entity {pdbx_description} branched_entity_instances {rcsb_branched_struct_conn {role}}}}}'
    )

    result = requests.get(URL)
    return result.json()['data']['entries']


def annotate_pdb(name: str, levels: list, ann_data: list, df: pd.DataFrame):

    for idx in range(len(df.index)):

        elem = ann_data[idx]

        for lev in levels:
            elem = elem[lev]
        
        df.loc[idx, name] = elem

    return df


def annotate_ligands(pdb_ids: list, df: pd.DataFrame, path_to_sdf: str):
    '''
    Calculates molecular properties of the ligand given in the sdf files and adds them to df

    Parameters
    -----------
    pdb_ids: list
        list of pdb ids in string format
    
    df: pd.DataFrame
        DataFrame to write annotations into

    path_to_sdf: str
        Path to directory which contains a file <pdb_id>_ligand.sdf for each ligand

    Returns
    --------
    df: DataFrame
        Input data frame with a new column for each molecular property
    '''

    mols = []

    for pdb_id in pdb_ids: #tqdm(pdb_ids, desc = 'Loading ligand annotations'):

        supp = SDMolSupplier(f'{path_to_sdf}/{pdb_id}_ligand.sdf', removeHs=False)
        if len(supp) > 1:
            print(f'{len(supp)} molecules detected')

        mols.append(supp[0])

    f = lambda x: rdMolDescriptors.CalcExactMolWt(x) if x is not None else x
    df['ligand_weight'] = [f(m) for m in mols]

    f = lambda x: rdchem.Mol.GetNumHeavyAtoms(x) if x is not None else x
    df['ligand_n_ha'] = [f(m) for m in mols]

    f = lambda x: rdMolDescriptors.CalcNumRings(x) if x is not None else x
    df['ligand_n_rings'] = [f(m) for m in mols]

    f = lambda x: rdMolDescriptors.CalcNumHBD(x) if x is not None else x
    df['ligand_hbd'] = [f(m) for m in mols]

    f = lambda x: rdMolDescriptors.CalcNumHBA(x) if x is not None else x
    df['ligand_hba'] = [f(m) for m in mols]

    f = lambda x: rdMolDescriptors.CalcTPSA(x) if x is not None else x
    df['ligand_tpsa'] = [f(m) for m in mols]

    f = lambda x: Crippen.MolLogP(x) if x is not None else x
    df['ligand_logP'] = [f(m) for m in mols]

    print(f'{df["ligand_weight"].isna().sum()} molecules could not be read')

    return df


def categorize_uniprot_keywords(category: str, threshold: int, df: pd.DataFrame):
    '''
    Transform each keyword from a given category into a binary column in df

    Parameters
    ----------
    category: str
        The category for which to binarize all member keywords

    threshold: int
        Minimum number of occurrences of a keyword for it to be binarized

    df: pd.DataFrame
        The dataframe that contains category as a column

    Returns
    --------
    df: pd.DataFrame
        The input dataframe with the added binarized keywords
    '''

    word_count = {}

    for i in df[category].index:

        words = df.loc[i, category].split(', ')

        for word in words:

            if word in word_count.keys():

                word_count[word] += 1
            
            else:
                word_count[word] = 1

    selected_w = []
    
    for word, count in word_count.items():

        if count >= threshold and word != 'NaN':

            selected_w.append(word)

            df[f'{category} - {word}'] = df[category].apply(lambda x: True if word in x.split(', ') else False)
    
    
    return df, selected_w


def annotate(dataset: ProteinDataset, sids: list, tids: list, uniprot: bool, ligands: bool):
    ''' 
    Get a dataframe of annotations for the proteins in sids

    Parameters
    ----------
    dataset: ProteinDataset
        The dataset of proteins which contains all structures in sids
    
    sids: list
        The structure ids of the protein structures, used to determine structure-level annotations

    tids: list
        The protein target ids of the protein structures, used to determine protein-level annotations

    uniprot: bool
        Whether to load UniProt annotations

    ligands: bool
        Whether to calculate ligand annotations

    Returns
    --------
    df: pd.DataFrame
        A dataframe containing the annotations where each row corresponds to a protein structure
    '''

    # create df with row for each protein
    sub_table=dataset.proteins_table_df[dataset.proteins_table_df['tid (ChEMBL)'].isin(tids)]

    prot_df = pd.DataFrame({'tid': sub_table.loc[:, 'tid (ChEMBL)'],
    'acc_id': [elem.split('|')[0] for elem in sub_table.loc[:, 'uniprot_accession_ids_str (ChEMBL)']], 
    'organism': sub_table.loc[:, 'organism (ChEMBL)']})

    if uniprot:

        for category in ['Biological process', 'Cellular component', 'Coding sequence diversity', 
                        'Developmental stage', 'Disease', 'Domain', 'Ligand', 'Molecular function', 
                        'PTM', 'Technical term']:

            prot_df[category] = 'NaN'

        prot_df = annotate_uniprot(prot_df)

    #Transfer data into df with row for each structure

    structure_df = pd.DataFrame(columns = prot_df.columns)
    
    for tid in tids:

        structure_df = structure_df.append(prot_df.loc[prot_df['tid']==tid, :], ignore_index=True)
    

    pdb_ids = [sid.split('_')[1] for sid in sids]

    # if pdb:
    #     batch_size = 100
    #     data_list = []

    #     for batch_idx in tqdm(range(1, len(pdb_ids)//batch_size), desc = 'Loading PDB data in batches'):
    #         data_list += extract_pdb_batch(pdb_ids[(batch_idx-1)*batch_size:batch_idx*batch_size])

    #     if batch_idx*batch_size < len(pdb_ids):
    #         data_list += extract_pdb_batch(pdb_ids[batch_idx*batch_size:])
    

    #     with open(pdb_ann_keys_file, 'w') as file:

    #         lines = file.readlines()

    #     for levels in tqdm(lines, desc='Loading PDB annotations'):
        
    #         levels = levels.split('|')
    #         structure_df = annotate_pdb(levels[0], levels[1:], data_list, structure_df)
    
    if ligands: 
        structure_df = annotate_ligands(pdb_ids, structure_df, 'data/sdf/')

    return structure_df
        

def find_bin_idx(x, lower_cut, upper_cut, bin_sz):

    if x < lower_cut:
        return 0
    elif x >= upper_cut:
        return -1
    else:
        return int((x-lower_cut)//bin_sz+1)


def bin_continuous_data(property: str, df: pd.DataFrame, n_bins: int, lower_cut: float, upper_cut: float):
    '''
    Bin a continuous annotation into n_bins binary annotations

    Parameters
    ----------
    property: str
        The column name of the continuous annotation

    df: pd.DataFrame
        The dataframe which contains the continuous annotation
    
    n_bins: int
        Number of bins to create for the annotation

    lower_cut: float
        Left cutoff for the first bin

    upper_cut: float
        Right cutoff for the last bin

    Returns
    -------
    df: pd.DataFrame
        The input dataframe with n_bins additional boolean columns
    '''

    bin_sz = (upper_cut-lower_cut)/(n_bins-2)

    bins = []

    for bin in range(n_bins):

        if bin == 0:
            name = f'< {lower_cut}'
        elif bin == n_bins-1:
            name = f'>= {upper_cut}'
        else:
            name = f'{lower_cut+(bin-1)*bin_sz} - {lower_cut+bin*bin_sz}'

        bins.append(name)
        
    key = property + ' (bins)'

    df[key] = df[property].apply(lambda x: None if math.isnan(x) else bins[find_bin_idx(x, lower_cut, upper_cut, bin_sz)])

    return df


def create_anns_dataframe(model_id: int):
    '''
    Create a dataframe with UniProt and ligand annotations for each protein in the given model

    Parameters
    -----------
    model_id: int
        The id of the model

    Returns
    -------
    df: pd.DataFrame
        The annotations dataframe
    '''

    dataset = ProteinDataset(model_id)

    sids = dataset.structure_ids
    tids = [dataset._get_tid_from_structure_id(sid) for sid in sids]

    # get UniProt and ligand annotations
    df = annotate(dataset, sids, tids, True, True)

    # binarize the UniProt keywords
    categories = ['Biological process','Molecular function', 'Disease','PTM', 'Domain','Ligand']

    for category in categories:

        df, kws = categorize_uniprot_keywords(category, 500, df)

    # bin ligand annotations
    df = bin_continuous_data('ligand_logP', df, 8, -2, 7)
    df = bin_continuous_data('mol_weight', df, 6, 20000, 170000)
    df = bin_continuous_data('ligand_weight', df, 8, 150, 700)
    df = bin_continuous_data('length', df, 6, 200, 1500)
    df = bin_continuous_data('ligand_n_ha', df, 6, 10, 50)
    df = bin_continuous_data('ligand_tpsa', df, 8, 10, 200)
    df = bin_continuous_data('ligand_hba', df, 6, 1, 12)
    df = bin_continuous_data('ligand_hbd', df, 6, 1, 6)
    df = bin_continuous_data('ligand_n_rings', df, 8, 1, 7)

    return df