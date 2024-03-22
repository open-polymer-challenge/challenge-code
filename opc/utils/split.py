from joblib import Parallel, delayed
from collections import defaultdict
from copy import deepcopy
import numpy as np

from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

def get_fingerprints(smiles_list):
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:  # Ensure the molecule could be parsed
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fingerprints.append(fp)
        else:
            fingerprints.append(None)
    return fingerprints

def get_scaffold(mol):
    """Extracts the Murcko Scaffold from a molecule."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def parallel_scaffold_computation(molecule, molecule_id):
    """Computes scaffold for a single molecule in parallel."""
    scaffold = get_scaffold(molecule)
    return scaffold, molecule, molecule_id


def cluster_molecules_by_scaffold(molecules, all_data_id, n_jobs=-1):
    """Clusters molecules based on their scaffolds using parallel processing."""
    # Ensure molecules and IDs are paired correctly
    paired_results = Parallel(n_jobs=n_jobs)(
        delayed(parallel_scaffold_computation)(mol, molecule_id)
        for mol, molecule_id in zip(molecules, all_data_id)
    )

    # Initialize dictionaries for batches and IDs
    batched_mol = defaultdict(list)
    batched_id = defaultdict(list)

    # Process results to fill batched_mol and batched_id dictionaries
    for scaffold, mol, molecule_id in paired_results:
        batched_mol[scaffold].append(mol)
        batched_id[scaffold].append(molecule_id)

    # Convert dictionaries to lists for output
    scaffolds = list(batched_mol.keys())
    batched_id = list(batched_id.values())

    return scaffolds, batched_id


def similarity_split(train_df, test_dev, train_ratio=0.8):        
    test_dev_smiles = test_dev['SMILES'].tolist()
    data_smiles = train_df['SMILES'].tolist()
    data_smiles_fps = get_fingerprints(data_smiles)
    test_dev_smiles_fps = get_fingerprints(test_dev_smiles)
    average_similarities = []
    for data_fp in data_smiles_fps:
        if data_fp is not None:
            similarities = [DataStructs.TanimotoSimilarity(data_fp, test_fp) for test_fp in test_dev_smiles_fps if test_fp is not None]
            average_similarity = np.mean(similarities)
            average_similarities.append(average_similarity)
        else:
            average_similarities.append(0)
    sorted_indices = np.argsort(-np.array(average_similarities))\
    
    valid_ratio = 1 - train_ratio
    num_valid = int(len(sorted_indices) * valid_ratio)
    valid_idx = sorted_indices[:num_valid]
    train_idx = sorted_indices[num_valid:]

    return train_idx.tolist(), valid_idx.tolist(), []

def scaffold_split(train_df, train_ratio=0.8, valid_ratio=None, test_ratio=None):
    """Splits a dataframe of molecules into scaffold-based clusters."""
    # Get smiles from the dataframe
    train_smiles_list = train_df["SMILES"]
    indinces = list(range(len(train_smiles_list)))
    train_mol_list = [Chem.MolFromSmiles(smiles) for smiles in train_smiles_list]
    scaffold_names, batched_id = cluster_molecules_by_scaffold(train_mol_list, indinces)

    train_cutoff = int(train_ratio * len(train_df))
    if valid_ratio is None:
        valid_cutoff = len(train_df)
    else:
        valid_cutoff = int(valid_ratio * len(train_df)) + train_cutoff
    train_inds, valid_inds, test_inds = [], [], []
    inds_all = deepcopy(batched_id)
    np.random.seed(3)
    np.random.shuffle(inds_all)
    idx_count = 0
    for inds_list in inds_all:
        for ind in inds_list:
            if idx_count < train_cutoff:
                train_inds.append(ind)
            elif idx_count < valid_cutoff:
                valid_inds.append(ind)
            else:
                test_inds.append(ind)
            idx_count += 1

    return train_inds, valid_inds, test_inds

