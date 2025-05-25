from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def evaluar_naive_bayes(X_train, y_train, X_test, y_test):
    # Modelo base
    nb = GaussianNB()

    # Validación cruzada con el modelo optimizado
    metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    resultados_cv = cross_validate(nb, X_train, y_train, cv=5, scoring=metricas)
    resultados = {
        'Exactitud (CV)': f"{resultados_cv['test_accuracy'].mean():.3f} (±{resultados_cv['test_accuracy'].std():.3f})",
        'Precisión (CV)': f"{resultados_cv['test_precision'].mean():.3f} (±{resultados_cv['test_precision'].std():.3f})",
        'Sensibilidad (CV)': f"{resultados_cv['test_recall'].mean():.3f} (±{resultados_cv['test_recall'].std():.3f})",
        'F1-score (CV)': f"{resultados_cv['test_f1'].mean():.3f} (±{resultados_cv['test_f1'].std():.3f})",
        'AUC (CV)': f"{resultados_cv['test_roc_auc'].mean():.3f} (±{resultados_cv['test_roc_auc'].std():.3f})"
    }

    # Evaluación en el conjunto de prueba
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    y_pred_proba = nb.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva
    y_pred_proba = nb.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva
    resultados['Exactitud (Prueba)'] = f"{accuracy_score(y_test, y_pred):.3f}"
    resultados['Precisión (Prueba)'] = f"{precision_score(y_test, y_pred):.3f}"
    resultados['Sensibilidad (Prueba)'] = f"{recall_score(y_test, y_pred):.3f}"
    resultados['F1-score (Prueba)'] = f"{f1_score(y_test, y_pred):.3f}"
    resultados['AUC (Prueba)'] = f"{roc_auc_score(y_test, y_pred_proba):.3f}"

    # Cálculo de la curva ROC para el conjunto de prueba
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    # Visualización de la curva ROC
    fig = plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # Línea diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos',fontsize=16)
    plt.ylabel('Tasa de Verdaderos Positivos',fontsize=16)
    plt.title('Curva ROC - Conjunto de Prueba (Naive Bayes)',fontsize=16)
    plt.legend(loc="lower right",fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)

    # Tabla de resultados
    tabla_resultados = pd.DataFrame.from_dict(resultados, orient='index', columns=['Valor'])
    tabla_resultados.index.name = 'Métrica'

    return tabla_resultados,fig,nb