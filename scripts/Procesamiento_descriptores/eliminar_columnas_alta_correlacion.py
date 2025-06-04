import numpy as np
import pandas as pd

def eliminar_columnas_alta_correlacion (descriptores,threshold):
    matriz_cor= descriptores.iloc[:,1:].corr().abs() # Obtenemos la matriz de correlación en términos absolutos
    matriz_cor_superior=matriz_cor.where(np.triu(np.ones(matriz_cor.shape),k=1).astype(bool)) # Necesitamos solo la parte superior de la matriz
    #np.ones(matriz_cor.shape): Crea una matriz del mismo tamaño llena de 1s.
    #np.triu(..., k=1): Extrae la parte superior de la matriz (por encima de la diagonal principal).
    #.astype(bool): Convierte la matriz en una de True/False, donde la parte superior es True y el resto False.
    #matriz_cor.where(...): Sustituye los valores de matriz_cor donde True, y pone NaN donde False.
    #Fuente:https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
    columnas_alta_correlacion=[column for column in matriz_cor_superior.columns if any(matriz_cor_superior[column]>threshold)]
    print(columnas_alta_correlacion)
    print("Numero de columnas con alta correlación:",len(columnas_alta_correlacion))
    descriptores_correlacionados_eliminados=descriptores.drop(columns=columnas_alta_correlacion)
    return descriptores_correlacionados_eliminados