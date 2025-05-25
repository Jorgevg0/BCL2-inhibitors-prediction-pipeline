import pandas as pd
from sklearn.model_selection import train_test_split

X = datos_BCL2_curados.iloc[:,1:-1]
y = datos_BCL2_curados["Actividad"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999, stratify=y)

print("Entrenamiento:", X_train.shape)
print("Prueba:", X_test.shape)
