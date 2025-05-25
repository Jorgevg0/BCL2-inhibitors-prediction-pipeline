import numpy as np
import pandas as pd

def eliminar_columnas_alta_correlacion(descriptores, threshold):
    matriz_cor = descriptores.iloc[:,1:].corr().abs()
    matriz_cor_superior = matriz_cor.where(np.triu(np.ones(matriz_cor.shape), k=1).astype(bool))
    columnas_alta_correlacion = [column for column in matriz_cor_superior.columns if any(matriz_cor_superior[column] > threshold)]
    print(columnas_alta_correlacion)
    print("Numero de columnas con alta correlación:", len(columnas_alta_correlacion))
    descriptores_correlacionados_eliminados = descriptores.drop(columns=columnas_alta_correlacion)
    return descriptores_correlacionados_eliminados
