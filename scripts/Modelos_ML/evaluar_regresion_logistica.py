from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def evaluar_lr(X_train, y_train, X_test, y_test):
    lr = LogisticRegression(max_iter=1000)

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    grid_lr = GridSearchCV(lr, param_grid, cv=5, scoring=metricas, refit='roc_auc', n_jobs=-1)
    grid_lr.fit(X_train, y_train)

    print("Mejores parámetros:", grid_lr.best_params_)
    print("Mejor Exactitud (accuracy) en CV (GridSearch):", grid_lr.best_score_)

    resultados = {
        'Mejores hiperparámetros': ', '.join([f'{k}={v}' for k, v in grid_lr.best_params_.items()]),
        'Exactitud (CV)': f"{grid_lr.cv_results_['mean_test_accuracy'][grid_lr.best_index_]:.3f} (±{grid_lr.cv_results_['std_test_accuracy'][grid_lr.best_index_]:.3f})",
        'Precisión (CV)': f"{grid_lr.cv_results_['mean_test_precision'][grid_lr.best_index_]:.3f} (±{grid_lr.cv_results_['std_test_precision'][grid_lr.best_index_]:.3f})",
        'Sensibilidad (CV)': f"{grid_lr.cv_results_['mean_test_recall'][grid_lr.best_index_]:.3f} (±{grid_lr.cv_results_['std_test_recall'][grid_lr.best_index_]:.3f})",
        'F1-score (CV)': f"{grid_lr.cv_results_['mean_test_f1'][grid_lr.best_index_]:.3f} (±{grid_lr.cv_results_['std_test_f1'][grid_lr.best_index_]:.3f})",
        'AUC (CV)': f"{grid_lr.cv_results_['mean_test_roc_auc'][grid_lr.best_index_]:.3f} (±{grid_lr.cv_results_['std_test_roc_auc'][grid_lr.best_index_]:.3f})"
    }

    modelo_optimizado = grid_lr.best_estimator_
    y_pred = modelo_optimizado.predict(X_test)
    y_pred_proba = modelo_optimizado.predict_proba(X_test)[:, 1]
    resultados['Exactitud (Prueba)'] = f"{accuracy_score(y_test, y_pred):.3f}"
    resultados['Precisión (Prueba)'] = f"{precision_score(y_test, y_pred):.3f}"
    resultados['Sensibilidad (Prueba)'] = f"{recall_score(y_test, y_pred):.3f}"
    resultados['F1-score (Prueba)'] = f"{f1_score(y_test, y_pred):.3f}"
    resultados['AUC (Prueba)'] = f"{roc_auc_score(y_test, y_pred_proba):.3f}"

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    fig = plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos', fontsize=16)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=16)
    plt.title('Curva ROC - Conjunto de Prueba (Regresión Logística)', fontsize=16)
    plt.legend(loc="lower right", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)

    tabla_resultados = pd.DataFrame.from_dict(resultados, orient='index', columns=['Valor'])
    tabla_resultados.index.name = 'Métrica'

    return tabla_resultados, fig, modelo_optimizado
