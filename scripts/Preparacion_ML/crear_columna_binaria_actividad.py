import numpy as np
import pandas as pd

def crear_columna_binaria_actividad(datos):
    mediana = datos["Standard Value"].median()
    print("El umbral de corte de Standard Value (mediana ) es:", mediana)
    datos["Actividad"] = np.where(datos["Standard Value"] <= mediana, 1, 0)
    return datos
