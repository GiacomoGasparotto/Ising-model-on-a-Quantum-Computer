import os
import pickle

### Handle data stream
def load_results(filename="simulation_results.pkl"):
    """
    Loads simulation results from a pickle file.

    Args:
        filename (str, optional): The path to the pickle file. Defaults to "simulation_results.pkl".

    Returns:
        dict or None: The loaded data dictionary if the file exists, None otherwise.
    """

    if not os.path.exists(filename):
        print(f"ERROR: File '{filename}' doesn't exist!\n")
        return None
    
    with open(filename, "rb") as f: 
        data = pickle.load(f)
    print(f"Data correctly loaded from '{filename}'\n\n")
    return data

def append_results(original_db, new_data_chunk):
    """
    Appends new results to the existing database, overwriting entries if they already exist.

    Args:
        original_db (dict): The dictionary containing the aggregated results so far.
        new_data_chunk (dict): The dictionary containing the new results to be added.

    Returns:
        dict: The updated database dictionary.
    """

    for g, n_dict in new_data_chunk.items():
        if g not in original_db:
            original_db[g] = {}
        # Overwrite
        for nq, data in n_dict.items():
            original_db[g][nq] = data
    return original_db

def save_results(data, filename="simulation_results.pkl"):
    """
    Saves the simulation results dictionary to a pickle file.

    Args:
        data (dict): The data structure to be saved.
        filename (str, optional): The target filename. Defaults to "simulation_results.pkl".
    """

    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Data correctly saved in '{filename}'")

def update_results(original_db, new_data):
    """
    Updates the existing results database with new data using dictionary updates.

    Args:
        original_db (dict): The original dictionary of results.
        new_data (dict): The dictionary containing new entries to update or add.

    Returns:
        dict: The updated results dictionary.
    """
    
    for g, n_dict in new_data.items():
        if g not in original_db:
            original_db[g] = {}
        # Add new keys to existing dict
        original_db[g].update(n_dict) 
    return original_db