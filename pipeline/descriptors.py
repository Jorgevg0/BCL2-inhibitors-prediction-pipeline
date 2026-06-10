import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def calculate_descriptors(df, mol_col="Mol", verbose=True):
    """
    Calculate RDKit molecular descriptors from Mol objects.

    Parameters:
        df (pd.DataFrame): DataFrame with a column of RDKit Mol objects.
        mol_col (str): Name of the column containing Mol objects (default "Mol").
        verbose (bool): Print warnings for None Mol objects.

    Returns:
        pd.DataFrame: DataFrame of descriptor values, indexed as the input.
    """
    descriptor_names = [name[0] for name in Descriptors._descList]
    n_desc = len(descriptor_names)

    def get_descriptors(mol):
        if mol:
            return list(Descriptors.CalcMolDescriptors(mol).values())
        if verbose:
            print("Warning: encountered None Mol object")
        return [None] * n_desc

    values = df[mol_col].apply(get_descriptors).tolist()
    return pd.DataFrame(values, columns=descriptor_names, index=df.index)


def calculate_morgan_fingerprints(df, smiles_column="Smiles", n_bits=2048, radius=2,
                                   return_invalid=False):
    """
    Generate Morgan (ECFP) fingerprints for molecules given as SMILES strings.

    Parameters:
        df (pd.DataFrame): DataFrame with a SMILES column.
        smiles_column (str): Name of the SMILES column (default "Smiles").
        n_bits (int): Fingerprint bit vector size (default 2048).
        radius (int): Morgan radius / ECFP diameter (default 2 → ECFP4).
        return_invalid (bool): If True, also returns indices of invalid SMILES.

    Returns:
        pd.DataFrame: Binary fingerprint matrix (rows = molecules, cols = bits).
        list (optional): Indices of invalid SMILES, returned when return_invalid=True.
    """
    fingerprints = []
    invalid_indices = []
    morgan_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)

    for ind, smiles in enumerate(df[smiles_column].tolist()):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_indices.append(ind)
                continue
            fp = morgan_gen.GetFingerprint(mol)
            fp_array = np.zeros((n_bits,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, fp_array)
            fingerprints.append(fp_array)
        except Exception as e:
            print(f"Error processing SMILES at index {ind}: {e}")
            invalid_indices.append(ind)

    fp_df = pd.DataFrame(fingerprints)
    if return_invalid:
        return fp_df, invalid_indices
    return fp_df


def clean_na_and_duplicates(df, log_df):
    """
    Remove rows with missing values and duplicate descriptor profiles.

    Parameters:
        df (pd.DataFrame): Input descriptor DataFrame.
        log_df (pd.DataFrame): Log with columns ["Step", "Removed", "Remaining"].

    Returns:
        tuple: (Filtered DataFrame, updated log DataFrame)
    """
    before = df.shape[0]

    df_filtered = df.dropna().reset_index(drop=True)
    removed_na = before - df_filtered.shape[0]
    log_df.loc[len(log_df)] = ["NA in descriptors filter", removed_na, df_filtered.shape[0]]
    print(f"NA filter: removed {removed_na} rows. Remaining: {df_filtered.shape[0]}")

    before_dup = df_filtered.shape[0]
    df_filtered = df_filtered.loc[~df_filtered.duplicated(keep="first")]
    removed_dup = before_dup - df_filtered.shape[0]
    log_df.loc[len(log_df)] = ["Duplicate descriptors filter", removed_dup, df_filtered.shape[0]]
    print(f"Duplicate filter: removed {removed_dup} rows. Remaining: {df_filtered.shape[0]}")

    return df_filtered, log_df


def remove_highly_correlated_columns(descriptors, threshold=0.8):
    """
    Remove features with pairwise Pearson correlation above the threshold.

    Parameters:
        descriptors (pd.DataFrame): Descriptor matrix (no ID or target columns).
        threshold (float): Correlation cutoff (default 0.8).

    Returns:
        tuple: (Filtered DataFrame, log DataFrame)
    """
    corr_matrix = descriptors.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    descriptors_filtered = descriptors.drop(columns=high_corr_cols)

    log_df = pd.DataFrame({
        "Step": ["High correlation filter"],
        "n_columns_removed": [len(high_corr_cols)],
        "Columns_removed": [high_corr_cols],
    })

    print(f"Removed {len(high_corr_cols)} highly correlated columns. "
          f"Remaining: {descriptors_filtered.shape[1]}")
    return descriptors_filtered, log_df


def remove_low_variance_columns(descriptors, log_df=None, threshold=0.1):
    """
    Remove features with variance below the threshold.

    Parameters:
        descriptors (pd.DataFrame): Descriptor matrix.
        log_df (pd.DataFrame, optional): Existing log table; created if None.
        threshold (float): Variance cutoff (default 0.1).

    Returns:
        tuple: (Filtered DataFrame, updated log DataFrame)
    """
    selector = VarianceThreshold(threshold)
    selector.fit(descriptors)
    retained = descriptors.columns[selector.get_support(indices=True)]
    removed = [col for col in descriptors.columns if col not in retained]

    if removed:
        print(f"Removed {len(removed)} low-variance columns (< {threshold})")
    else:
        print(f"No columns removed. All variance >= {threshold}")

    filtered = descriptors[retained]
    log_entry = pd.DataFrame({
        "Step": ["Low variance filter"],
        "n_columns_removed": [len(removed)],
        "Columns_removed": [removed],
    })

    log_df = log_entry if log_df is None else pd.concat([log_df, log_entry], ignore_index=True)
    return filtered, log_df


def standardize_data(df):
    """
    Z-score standardize all columns (mean=0, std=1).

    Parameters:
        df (pd.DataFrame): Descriptor matrix (numeric columns only).

    Returns:
        pd.DataFrame: Standardized DataFrame.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index)


def normalize_data(df):
    """
    Min-max normalize all columns to [0, 1].

    Parameters:
        df (pd.DataFrame): Descriptor matrix (numeric columns only).

    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(df)
    return pd.DataFrame(normalized, columns=df.columns, index=df.index)


def merge_ID_response_variable(filtered_descriptors, original_df,
                                id_col="Molecule ChEMBL ID", value_col="Standard Value"):
    """
    Re-attach molecule ID and Standard Value columns to a processed descriptor DataFrame,
    aligning by original index.

    Parameters:
        filtered_descriptors (pd.DataFrame): Descriptor matrix after filtering.
        original_df (pd.DataFrame): Original DataFrame with ID and Standard Value.
        id_col (str): Name of the molecule ID column.
        value_col (str): Name of the activity value column.

    Returns:
        pd.DataFrame: [ID | descriptors | Standard Value]
    """
    ids_and_values = original_df.loc[
        filtered_descriptors.index, [id_col, value_col]
    ].reset_index(drop=True)
    descriptors_reset = filtered_descriptors.reset_index(drop=True)
    return pd.concat([ids_and_values, descriptors_reset], axis=1)
