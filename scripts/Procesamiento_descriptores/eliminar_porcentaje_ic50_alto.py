import pandas as pd

def eliminar_porcentaje_ic50_alto(df, porcentaje_eliminar):

    percentil = 1 - porcentaje_eliminar
    punto_de_corte = df['Standard Value'].quantile(percentil)

    # Filtrar moléculas con IC50 menor o igual al umbral
    df_filtrado = df[df['Standard Value'] <= punto_de_corte].copy()
    mediana = df_filtrado['Standard Value'].median()
    cuartil_25 = df_filtrado['Standard Value'].quantile(0.25)

    # Calcular estadísticas
    moleculas_eliminadas = df.shape[0] - df_filtrado.shape[0]
    moleculas_restantes = df_filtrado.shape[0]

    print(f"Porcentaje eliminado: {porcentaje_eliminar*100}%")
    print(f"El punto de corte para la eliminación de un IC50 superior ha sido: {punto_de_corte:.2f} nM")
    print(f"Moléculas eliminadas: {moleculas_eliminadas}")
    print(f"Moléculas restantes: {moleculas_restantes}")
    print(f"Mediana de las moleculas restantes es: {mediana:.2f} nM")
    print(f"Cuartil 25 de las moleculas restantes es: {cuartil_25:.2f} nM")

    return df_filtrado