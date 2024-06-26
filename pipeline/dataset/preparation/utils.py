# src/preparation/utils.py

# Import public modules
import requests
import re
import tqdm
import collections

class NoProteinQuantityFoundException(Exception):
    """
    Custom exception for a specific error case during protein scraping 
    from UniProt that can be catched and treated explicitly. 
    """
    pass

def extract_protein_quantity(quantity_pattern, string):
    """
    Extract protein quantities from a string by passing a regex pattern of the quantity.
    
    Args:
        quantity_pattern (str): Regex pattern corresponding to the to be extracted quantity.
        string (str): The string from which the sought for quantity should be extracted from.

    Return:
        (str): The to be extracted quantity.

    """
    # Find all matches of the pattern in the string
    findall = re.findall(quantity_pattern, string)

    # Differ the cases where no, multiple or a single match(es) were found.
    # Remark: There should be only a single match
    if 0==len(findall):
        err_msg = f"Found no match for '{quantity_pattern}' in '{string}'."
        raise NoProteinQuantityFoundException(err_msg)
    elif 1<len(findall):
        err_msg = f"Found multiple occurances (={len(findall)}) for '{quantity_pattern}' in '{string}'."
        raise ValueError(err_msg)
    else:
        # Return the single found quantity value
        return findall[0]

def scrape_protein_info_from_uniprot(p_chembl_id, uniprot_accession_ids_str, display_details=False):
    """
    Scrape informations about a certain protein from UniProt.

    Args:
        p_chembl_id (str): ChEMBL ID of the protein.
        uniprot_accession_ids_str (str): List of the UniProt Accession IDs as string of the form 
            '<uniprot_accession_id_1>|<uniprot_accession_id_2>|...'
        display_details (bool): Boolean flag to display more details about the scraping.
            (Default: False)

    Return:
        (dict): Dictionary containing the protein information.
    
    """
    # Define a dictionary containing the quantities and their regex pattern used to extract
    # them from .xml files extracted for the different UniProt Accession IDs
    quantity_pattern_dict = {
        'entry_name': r'</accession>\s*?<name>(.*?)</name>',
        'sequence': r'<sequence length=.*?>(.*?)</sequence>',
    }

    # The uniprot_acccession_ids_str is a string of the form '<uniprot_accession_id_1>|<uniprot_accession_id_2>|...',
    # thus split it into the uniprot accesssion ids
    uniprot_accession_ids_list = uniprot_accession_ids_str.split('|')

    # Initialize the quantities dictionary as empty defaults dictionary (containing empty lists)
    quantities_dict = collections.defaultdict(list)

    # Loop over the UniProt Accession IDs
    for uniprot_accession_id in uniprot_accession_ids_list:
        # Construct the URL corresponding to the UniProt Accession ID 
        page_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_accession_id}?format=xml"

        # Get the page content
        page_response = requests.get(page_url)
        page_content  = page_response.text

        # Try to extract the quantities from the page content
        for quantity_name, quantity_pattern in quantity_pattern_dict.items():
            try:
                # Extract the quantity
                quantity_value = extract_protein_quantity(quantity_pattern, page_content)

                # Append the quantity value to the corresponding list in the quantities dictionary
                quantities_dict[quantity_name].append(quantity_value)
            except NoProteinQuantityFoundException:
                # Only display that no match was found if details should be displayed
                if display_details:
                    print(f" Could not extract the quantity '{quantity_name}' for UniProt Accession ID: {uniprot_accession_id}.")

    # Loop over the extracted quantities and construct the protein information dictionary
    protein_info_dict = dict()
    for quantity_name, quantity_values in quantities_dict.items():
        # Get the set of the extracted quantity values (to access only the unique values)
        quantity_values_set = set(quantity_values)

        # Check that there is only one value in the set and throw errors in case that there is none or multiple
        if 0==len(quantity_values_set):
            err_msg = f"Protein uantity '{quantity_name}' not found for the protein '{p_chembl_id}'."
            raise ValueError(err_msg)

        if 1<len(quantity_values_set):
            err_msg = f"Multiple values found for protein quantity '{quantity_name}' for the protein '{p_chembl_id}', got values: {list(quantity_values_set)}."
            raise ValueError(err_msg)

        # Access the (unique) quantity of the protein and assign it as value to the protein information dictionary
        protein_info_dict[quantity_name] = list(quantity_values_set)[0]

    return protein_info_dict