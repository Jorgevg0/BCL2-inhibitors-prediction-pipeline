import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import SaltRemover
from rdkit.Chem.inchi import MolToInchiKey


def first_filter_molecules(df, target_name, unit_col="Standard Units", expected_unit="nM"):
    """
    Apply sequential filters to keep only valid molecules:
    1. Standard Type is IC50
    2. Target Name matches the target of interest
    3. Assay Type is "B" (Binding Assays)
    4. Target Organism is Homo Sapiens
    5. Correct units (default: nM)
    6. Non-null Standard Value (IC50)
    7. Non-null Smiles

    Parameters:
        df (pd.DataFrame): Input dataset.
        target_name (str): Target name to filter.
        unit_col (str): Column name for units (default "Standard Units").
        expected_unit (str): Expected unit (default "nM").

    Returns:
        tuple: (Filtered DataFrame, Filter log DataFrame)
    """
    log = pd.DataFrame(columns=["Step", "Removed", "Remaining"])
    df_filtered = df.copy()

    filters = [
        ("Standard Type IC50", lambda d: d[d["Standard Type"] == "IC50"]),
        ("Target Name filter", lambda d: d[d["Target Name"] == target_name]),
        ("Assay Type filter (Binding)", lambda d: d[d["Assay Type"] == "B"]),
        ("Target Organism filter", lambda d: d[d["Target Organism"] == "Homo sapiens"]),
        ("Unit filter", lambda d: d[d[unit_col] == expected_unit]),
        ("Missing Standard Value (IC50)", lambda d: d[d["Standard Value"].notna()]),
        ("Missing Smiles", lambda d: d[d["Smiles"].notna()]),
    ]

    for step_name, filter_fn in filters:
        before = df_filtered.shape[0]
        df_filtered = filter_fn(df_filtered)
        after = df_filtered.shape[0]
        log.loc[len(log)] = [step_name, before - after, after]

    df_filtered = df_filtered.reset_index(drop=True)
    return df_filtered, log


def remove_duplicate_ids(df, log=None, id_col="Molecule ChEMBL ID", value_col="Standard Value"):
    """
    Remove duplicate molecules by keeping the entry with the lowest Standard Value.

    Parameters:
        df (pd.DataFrame): Input dataset.
        log (pd.DataFrame, optional): DataFrame tracking removed/remaining molecules.
        id_col (str): Column containing molecule IDs.
        value_col (str): Column to sort for selecting the lowest value.

    Returns:
        tuple: (Filtered DataFrame, updated log DataFrame)
    """
    before = df.shape[0]
    df_sorted = df.sort_values(by=value_col)
    df_no_duplicates = df_sorted.drop_duplicates(subset=id_col, keep="first").reset_index(drop=True)
    after = df_no_duplicates.shape[0]
    removed = before - after

    print(f"Removed {removed} duplicate molecules. Remaining: {after}")

    if log is not None:
        log.loc[len(log)] = ["Duplicate molecules filter", removed, after]

    return df_no_duplicates, log


def remove_high_ic50_percentage(df, percentage_to_remove, log=None):
    """
    Remove the top percentage of molecules with the highest IC50 values.

    Parameters:
        df (pd.DataFrame): Input dataset.
        percentage_to_remove (float): Fraction of molecules to remove (e.g., 0.05 for 5%).
        log (pd.DataFrame, optional): DataFrame tracking removed/remaining molecules.

    Returns:
        tuple: (Filtered DataFrame, updated log DataFrame)
    """
    percentile = 1 - percentage_to_remove
    cutoff_point = df["Standard Value"].quantile(percentile)

    df_filtered = df[df["Standard Value"] <= cutoff_point].copy()
    removed = df.shape[0] - df_filtered.shape[0]
    remaining = df_filtered.shape[0]

    print(f"Cutoff IC50: {cutoff_point:.2f} nM | Removed: {removed} | Remaining: {remaining}")

    if log is not None:
        log.loc[len(log)] = [f"{percentage_to_remove * 100}% high IC50 filter", removed, remaining]

    return df_filtered, log


# --- QSAR structure curation helpers ---

def _valid_smiles(smiles):
    if smiles is None:
        return None
    return Chem.MolFromSmiles(smiles)


def _desalting(mol):
    if mol is None:
        return None
    remover = SaltRemover.SaltRemover()
    return remover.StripMol(mol)


def _neutralize(mol):
    if mol is None:
        return None
    uncharger = rdMolStandardize.Uncharger()
    return uncharger.uncharge(mol)


def _normalize(mol):
    if mol is None:
        return None
    normalizer = rdMolStandardize.Normalizer()
    try:
        mol = normalizer.normalize(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def _tautomer_canonicalize(mol):
    if mol is None:
        return None
    try:
        return rdMolStandardize.CanonicalTautomer(mol)
    except Exception:
        return None


def _remove_stereochemistry(mol):
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return mol


def _generate_identifiers(mol):
    if mol is None:
        return None, None
    try:
        inchikey = MolToInchiKey(mol)
    except Exception:
        inchikey = None
    try:
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        canonical_smiles = None
    return inchikey, canonical_smiles


def standardize_molecules(df, log=None, smiles_col="Smiles"):
    """
    Apply QSAR-ready structure curation pipeline:
    1. Validate SMILES
    2. Desalt (remove counterions/fragments)
    3. Neutralize charges
    4. Normalize valences and aromaticity
    5. Canonicalize tautomers
    6. Remove stereochemistry
    7. Generate InChIKey and canonical SMILES, deduplicate

    Parameters:
        df (pd.DataFrame): Input dataset with a SMILES column.
        log (pd.DataFrame, optional): DataFrame tracking removed/remaining molecules.
        smiles_col (str): Name of the SMILES column (default "Smiles").

    Returns:
        tuple: (Curated DataFrame with 'Mol', 'InChIKey', 'Canonical_SMILES' columns, updated log)
    """
    df_filtered = df.copy()

    df_filtered["Mol"] = df_filtered[smiles_col].apply(_valid_smiles)
    df_filtered = df_filtered[df_filtered["Mol"].notna()]

    df_filtered["Mol"] = df_filtered["Mol"].apply(_desalting)
    df_filtered["Mol"] = df_filtered["Mol"].apply(_neutralize)
    df_filtered["Mol"] = df_filtered["Mol"].apply(_normalize)
    df_filtered["Mol"] = df_filtered["Mol"].apply(_tautomer_canonicalize)
    df_filtered["Mol"] = df_filtered["Mol"].apply(_remove_stereochemistry)

    results = df_filtered["Mol"].apply(_generate_identifiers)
    df_filtered[["InChIKey", "Canonical_SMILES"]] = pd.DataFrame(
        results.tolist(), index=df_filtered.index
    )
    df_filtered = df_filtered[df_filtered["InChIKey"].notna()]

    before = df_filtered.shape[0]
    df_filtered = df_filtered.sort_values(by="Standard Value", ascending=True)
    df_filtered = df_filtered.drop_duplicates(subset="InChIKey", keep="first")
    after = df_filtered.shape[0]

    if log is not None:
        log.loc[len(log)] = ["InChIKey duplicates", before - after, after]

    df_filtered = df_filtered.reset_index(drop=True)
    return df_filtered, log
