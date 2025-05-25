import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)  # Convertimos el valor de SMILES en un objeto mol
    if mol:
        descriptor_values = Descriptors.CalcMolDescriptors(mol)  # Extrae todos los descriptores disponibles
        return list(descriptor_values.values())  # Retorna solo los valores de los descriptores
    else:
        # Si la molécula no puede ser creada (SMILES inválido), retornar una lista vacía
        # en lugar de None.
        print(f"Advertencia: No se pudo procesar el SMILES inválido: {smiles}")
        return []

# Aplicamos la función a cada valor de Smile en nuestro set y creamos un nuevo DataFrame
descriptor_names = list(Descriptors._descList)  # Obtiene los nombres de los descriptores
descriptores_rdkit = pd.DataFrame(set_ic50_bcl2["Smiles"].apply(get_descriptors).tolist(),
                                  columns=[name[0] for name in descriptor_names],  # Nombres de los descriptores
                                  index=set_ic50_bcl2.index)

# Agregar identificador de molécula
descriptores_rdkit = pd.concat([set_ic50_bcl2[["Molecule ChEMBL ID"]], descriptores_rdkit], axis=1)# pd.concat necesita recibir varios df en una tupla
descriptores_rdkit