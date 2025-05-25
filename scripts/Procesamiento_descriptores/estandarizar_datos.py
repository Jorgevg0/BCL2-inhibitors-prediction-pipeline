import pandas as pd
from sklearn.preprocessing import StandardScaler

def estandarizar_datos(datos):
    datos_a_escalar = datos.iloc[:,1:-1]
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(datos_a_escalar)
    datos_escalados = pd.DataFrame(datos_escalados, columns=datos_a_escalar.columns, index=datos.index)
    datos_escalados = pd.concat([datos[['Molecule ChEMBL ID']], datos_escalados, datos[['Standard Value']]], axis=1)
    return datos_escalados
