import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont


def load_dataset(path, sep=";"):
    """
    Load a CSV dataset and print basic info.

    Parameters:
        path (str): Path to the CSV file.
        sep (str): Separator used in the CSV file (default ";").

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(path, sep=sep)
    print(f"Dataset loaded from: {path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def check_required_columns(df, required_columns):
    """
    Check if all required columns are present in the dataframe.

    Parameters:
        df (pd.DataFrame): Dataset to check.
        required_columns (list): List of required column names.

    Returns:
        None
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print("Missing required columns:", missing)
    else:
        print("All required columns are present.")


def draw_molecule(df, molecule_id, id_col="Molecule ChEMBL ID", smiles_col="Smiles",
                  img_size=(600, 300), text_height=50):
    """
    Draw a molecule from a DataFrame based on its ChEMBL ID, including its SMILES string.

    Parameters:
        df (pd.DataFrame): Dataset containing molecule information.
        molecule_id (str): ChEMBL ID of the molecule to draw.
        id_col (str): Column containing molecule IDs.
        smiles_col (str): Column containing SMILES strings.
        img_size (tuple): Size of the molecule image (width, height).
        text_height (int): Height reserved for SMILES text below the image.

    Returns:
        PIL.Image: Image object of the molecule with SMILES annotation.
    """
    smiles = df.loc[df[id_col] == molecule_id, smiles_col].values
    if len(smiles) == 0:
        raise ValueError(f"Molecule ID '{molecule_id}' not found in the dataset.")
    smiles = smiles[0]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for molecule '{molecule_id}': {smiles}")

    img = Draw.MolToImage(mol, size=img_size)

    new_img = Image.new("RGB", (img_size[0], img_size[1] + text_height), "white")
    new_img.paste(img, (0, 0))

    draw = ImageDraw.Draw(new_img)
    draw.text((10, img_size[1] + 5), smiles, fill="black")

    return new_img
