import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

def calcular_morgan_fingerprints(smiles_list,n_bits=2048,radius=2,return_invalid=False):
    fingerprints=[] # almacena los arrays de fingerprints
    indices_invalidos=[] # almacena los indices de smiles que no se puedan procesar

    for ind,smiles in enumerate(smiles_list):
        try:
            mol=Chem.MolFromSmiles(smiles)
            if mol is None:
                indices_invalidos.append(ind)
                continue  # pasa al siguiente mol si la molecula es invalida
            morgan_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)# Configuramos un generador de fingerprints
            fp = morgan_gen.GetFingerprint(mol)  # Genera fingerprint como ExplicitBitVect
            fp_array=np.zeros((n_bits,),dtype=np.int8)# creamos un array vacío para optimizar memoria del tamaño del fingerprint
            Chem.DataStructs.ConvertToNumpyArray(fp,fp_array)# copiamos los bits del objeto fp en fp_array para obtener los resultados en un array de numpy
            fingerprints.append(fp_array) # vamos añadiendo cada matriz a una lista

        except Exception as e: # manejamos errores durante el proceso e indicamos el indice de los smiles que esten fallando
                print(f"Error en SMILES {ind}: {smiles} - {str(e)}")
                indices_invalidos.append(ind)

    if return_invalid:
        return fingerprints,indices_invalidos
    else:
        return fingerprints