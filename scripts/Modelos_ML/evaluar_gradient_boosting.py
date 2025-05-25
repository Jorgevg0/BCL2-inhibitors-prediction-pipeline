from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def evaluar_gradient_boosting(X_train, y_train, X_test, y_test):
    # Modelo base
    gb = GradientBoostingClassifier(random_state=999)

    # Ajuste de hiperparámetros con GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5]
    }
    metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    grid_gb = GridSearchCV(gb, param_grid, cv=5, scoring=metricas, refit='roc_auc', n_jobs=-1)
    grid_gb.fit(X_train, y_train)

    # Mostrar mejores parámetros y la exactitud (accuracy)
    print("Mejores parámetros:", grid_gb.best_params_)
    print("Mejor precisión en CV (GridSearch):", grid_gb.best_score_)

    # Extraer métricas de validación cruzada para el mejor modelo
    resultados = {
        'Mejores hiperparámetros': ', '.join([f'{k}={v}' for k, v in grid_gb.best_params_.items()]), # Por cada par (k, v) (clave, valor) en el diccionario, se crea una cadena con formato clave=valor.
        'Exactitud (CV)': f"{grid_gb.cv_results_['mean_test_accuracy'][grid_gb.best_index_]:.3f} (±{grid_gb.cv_results_['std_test_accuracy'][grid_gb.best_index_]:.3f})",
        'Precisión (CV)': f"{grid_gb.cv_results_['mean_test_precision'][grid_gb.best_index_]:.3f} (±{grid_gb.cv_results_['std_test_precision'][grid_gb.best_index_]:.3f})",
        'Sensibilidad (CV)': f"{grid_gb.cv_results_['mean_test_recall'][grid_gb.best_index_]:.3f} (±{grid_gb.cv_results_['std_test_recall'][grid_gb.best_index_]:.3f})",
        'F1-score (CV)': f"{grid_gb.cv_results_['mean_test_f1'][grid_gb.best_index_]:.3f} (±{grid_gb.cv_results_['std_test_f1'][grid_gb.best_index_]:.3f})",
        'AUC (CV)': f"{grid_gb.cv_results_['mean_test_roc_auc'][grid_gb.best_index_]:.3f} (±{grid_gb.cv_results_['std_test_roc_auc'][grid_gb.best_index_]:.3f})"
    }

    # Evaluación en el conjunto de prueba
    modelo_optimizado = grid_gb.best_estimator_
    y_pred = modelo_optimizado.predict(X_test)
    y_pred_proba = modelo_optimizado.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva
    resultados['Exactitud (Prueba)'] = f"{accuracy_score(y_test, y_pred):.3f}"
    resultados['Precisión (Prueba)'] = f"{precision_score(y_test, y_pred):.3f}"
    resultados['Sensibilidad (Prueba)'] = f"{recall_score(y_test, y_pred):.3f}"
    resultados['F1-score (Prueba)'] = f"{f1_score(y_test, y_pred):.3f}"
    resultados['AUC (Prueba)'] = f"{roc_auc_score(y_test, y_pred_proba):.3f}"

    # Cálculo de la curva ROC para el conjunto de prueba
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # Visualización de la curva ROC
    fig = plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # Línea diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos',fontsize=16)
    plt.ylabel('Tasa de Verdaderos Positivos',fontsize=16)
    plt.title('Curva ROC - Conjunto de Prueba (Gradient Boosting)',fontsize=16)
    plt.legend(loc="lower right",fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)

    # Tabla de resultados
    tabla_resultados = pd.DataFrame.from_dict(resultados, orient='index', columns=['Valor'])
    tabla_resultados.index.name = 'Métrica'

    return tabla_resultados, fig, modelo_optimizado