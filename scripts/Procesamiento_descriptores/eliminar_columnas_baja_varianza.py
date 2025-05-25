import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def eliminar_baja_varianza(datos, threshold=0.1):
    datos_sin_id_respuesta = datos.iloc[:,1:-1]
    selection = VarianceThreshold(threshold)
    selection.fit(datos_sin_id_respuesta)

    columnas_conservadas = datos_sin_id_respuesta.columns[selection.get_support(indices=True)]
    columnas_eliminadas = [col for col in datos_sin_id_respuesta.columns if col not in columnas_conservadas]

    if columnas_eliminadas:
        print(f"Se eliminaron {len(columnas_eliminadas)} columnas por baja varianza (< {threshold}): {columnas_eliminadas}")
    else:
        print(f"No se eliminaron columnas con baja varianza < {threshold}")

    datos_baja_varianza = pd.concat([datos[['Molecule ChEMBL ID']], datos[columnas_conservadas], datos[['Standard Value']]], axis=1)

    return datos_baja_varianza
