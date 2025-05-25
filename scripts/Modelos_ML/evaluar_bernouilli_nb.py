from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def evaluar_bernoulli_nb(X_train, y_train, X_test, y_test):
    # Modelo base
    nb = BernoulliNB()

    # Ajuste de hiperparámetros
    param_grid = {
        'alpha': [0, 1, 10]
    }

    metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    grid_nb = GridSearchCV(nb, param_grid, cv=5, scoring=metricas, refit='roc_auc', n_jobs=-1)
    grid_nb.fit(X_train, y_train)

    # Mostrar mejores parámetros y la exactitud (accuracy)
    print("Mejores parámetros:", grid_nb.best_params_)
    print("Mejor precisión en CV (GridSearch):", grid_nb.best_score_)

    # Extraer métricas de validación cruzada
    resultados = {
        'Mejores hiperparámetros': ', '.join([f'{k}={v}' for k, v in grid_nb.best_params_.items()]),
        'Exactitud (CV)': f"{grid_nb.cv_results_['mean_test_accuracy'][grid_nb.best_index_]:.3f} (±{grid_nb.cv_results_['std_test_accuracy'][grid_nb.best_index_]:.3f})",
        'Precisión (CV)': f"{grid_nb.cv_results_['mean_test_precision'][grid_nb.best_index_]:.3f} (±{grid_nb.cv_results_['std_test_precision'][grid_nb.best_index_]:.3f})",
        'Sensibilidad (CV)': f"{grid_nb.cv_results_['mean_test_recall'][grid_nb.best_index_]:.3f} (±{grid_nb.cv_results_['std_test_recall'][grid_nb.best_index_]:.3f})",
        'F1-score (CV)': f"{grid_nb.cv_results_['mean_test_f1'][grid_nb.best_index_]:.3f} (±{grid_nb.cv_results_['std_test_f1'][grid_nb.best_index_]:.3f})",
        'AUC (CV)': f"{grid_nb.cv_results_['mean_test_roc_auc'][grid_nb.best_index_]:.3f} (±{grid_nb.cv_results_['std_test_roc_auc'][grid_nb.best_index_]:.3f})"
    }

    # Evaluación en test
    modelo_optimizado = grid_nb.best_estimator_
    y_pred = modelo_optimizado.predict(X_test)
    y_pred_proba = modelo_optimizado.predict_proba(X_test)[:, 1]

    resultados['Exactitud (Prueba)'] = f"{accuracy_score(y_test, y_pred):.3f}"
    resultados['Precisión (Prueba)'] = f"{precision_score(y_test, y_pred):.3f}"
    resultados['Sensibilidad (Prueba)'] = f"{recall_score(y_test, y_pred):.3f}"
    resultados['F1-score (Prueba)'] = f"{f1_score(y_test, y_pred):.3f}"
    resultados['AUC (Prueba)'] = f"{roc_auc_score(y_test, y_pred_proba):.3f}"

    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    fig = plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos', fontsize=16)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=16)
    plt.title('Curva ROC - Conjunto de Prueba (BernoulliNB)', fontsize=16)
    plt.legend(loc="lower right", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)

    # Tabla de resultados
    tabla_resultados = pd.DataFrame.from_dict(resultados, orient='index', columns=['Valor'])
    tabla_resultados.index.name = 'Métrica'

    return tabla_resultados, fig, modelo_optimizado