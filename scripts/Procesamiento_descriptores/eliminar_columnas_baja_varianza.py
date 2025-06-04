import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def eliminar_baja_varianza(datos, threshold=0.1):
    datos_sin_id_respuesta=datos.iloc[:,1:-1]
    selection = VarianceThreshold(threshold)  # Se crea una instancia de VarianceThreshold con el umbral
    selection.fit(datos_sin_id_respuesta)  # Calculamos la varianza de cada columna en datos

    # Obtener las columnas que se conservan
    columnas_conservadas = datos_sin_id_respuesta.columns[selection.get_support(indices=True)]

    # Identificar las columnas eliminadas (baja varianza)
    columnas_eliminadas = [col for col in datos_sin_id_respuesta.columns if col not in columnas_conservadas]

    # Mostrar las columnas eliminadas
    if columnas_eliminadas:
        print(f"Se eliminaron {len(columnas_eliminadas)} columnas por baja varianza (< {threshold}): {columnas_eliminadas}")
    else:
        print(f"No se eliminaron columnas con baja varianza < {threshold}")

    datos_baja_varianza=pd.concat([datos[['Molecule ChEMBL ID']], datos[columnas_conservadas], datos[['Standard Value']]], axis=1)

    return datos_baja_varianza

#Código basado en:https://github.com/herrynguyen1706/Selecting-the-best-molecular-descriptor/blob/main/Cleaning_Molecular_descriptors.ipynb
