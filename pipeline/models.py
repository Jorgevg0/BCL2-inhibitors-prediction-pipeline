import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier


METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _build_results_table(grid, y_test, y_pred, y_pred_proba):
    """Build a results DataFrame from GridSearchCV results + test set evaluation."""
    idx = grid.best_index_
    cv = grid.cv_results_

    results = {
        "Best hyperparameters": ", ".join(
            f"{k}={v}" for k, v in grid.best_params_.items()
        ) or "Default parameters",
        "Accuracy (CV)": f"{cv['mean_test_accuracy'][idx]:.3f} (±{cv['std_test_accuracy'][idx]:.3f})",
        "Precision (CV)": f"{cv['mean_test_precision'][idx]:.3f} (±{cv['std_test_precision'][idx]:.3f})",
        "Recall (CV)": f"{cv['mean_test_recall'][idx]:.3f} (±{cv['std_test_recall'][idx]:.3f})",
        "F1-score (CV)": f"{cv['mean_test_f1'][idx]:.3f} (±{cv['std_test_f1'][idx]:.3f})",
        "ROC AUC (CV)": f"{cv['mean_test_roc_auc'][idx]:.3f} (±{cv['std_test_roc_auc'][idx]:.3f})",
        "Accuracy (Test)": f"{accuracy_score(y_test, y_pred):.3f}",
        "Precision (Test)": f"{precision_score(y_test, y_pred):.3f}",
        "Recall (Test)": f"{recall_score(y_test, y_pred):.3f}",
        "F1-score (Test)": f"{f1_score(y_test, y_pred):.3f}",
        "ROC AUC (Test)": f"{roc_auc_score(y_test, y_pred_proba):.3f}",
    }
    table = pd.DataFrame.from_dict(results, orient="index", columns=["Value"])
    table.index.name = "Metric"
    return table


def _roc_plot(y_test, y_pred_proba, model_name):
    """Return a matplotlib Figure with the ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    fig = plt.figure()
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title(f"ROC Curve - Test Set ({model_name})", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    return fig


def _fit_and_evaluate(estimator, param_grid, X_train, y_train, X_test, y_test,
                      folds, model_name):
    grid = GridSearchCV(
        estimator, param_grid, cv=folds, scoring=METRICS, refit="roc_auc", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"Best params ({model_name}):", grid.best_params_)
    print(f"Best ROC AUC (CV): {grid.best_score_:.3f}")

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    table = _build_results_table(grid, y_test, y_pred, y_pred_proba)
    fig = _roc_plot(y_test, y_pred_proba, model_name)
    return table, fig, best_model


def evaluate_logistic_regression(X_train, y_train, X_test, y_test, param_grid=None, folds=5):
    """Train and evaluate Logistic Regression with GridSearchCV."""
    if param_grid is None:
        param_grid = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
        }
    return _fit_and_evaluate(
        LogisticRegression(max_iter=1000), param_grid,
        X_train, y_train, X_test, y_test, folds, "Logistic Regression"
    )


def evaluate_knn(X_train, y_train, X_test, y_test, param_grid=None, folds=5):
    """Train and evaluate k-Nearest Neighbors with GridSearchCV."""
    if param_grid is None:
        param_grid = {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        }
    return _fit_and_evaluate(
        KNeighborsClassifier(), param_grid,
        X_train, y_train, X_test, y_test, folds, "k-NN"
    )


def evaluate_gaussian_nb(X_train, y_train, X_test, y_test, param_grid=None, folds=5):
    """Train and evaluate Gaussian Naive Bayes with GridSearchCV."""
    if param_grid is None:
        param_grid = {}
    return _fit_and_evaluate(
        GaussianNB(), param_grid,
        X_train, y_train, X_test, y_test, folds, "GaussianNB"
    )


def evaluate_bernoulli_nb(X_train, y_train, X_test, y_test, folds=5):
    """Train and evaluate Bernoulli Naive Bayes with GridSearchCV."""
    param_grid = {"alpha": [0, 1, 10]}
    return _fit_and_evaluate(
        BernoulliNB(), param_grid,
        X_train, y_train, X_test, y_test, folds, "BernoulliNB"
    )


def evaluate_svm(X_train, y_train, X_test, y_test, param_grid=None, folds=5):
    """Train and evaluate Support Vector Machine with GridSearchCV."""
    if param_grid is None:
        param_grid = {
            "C": [0.1, 1.0, 10.0, 100],
            "kernel": ["linear", "rbf"],
        }
    return _fit_and_evaluate(
        SVC(probability=True, random_state=999), param_grid,
        X_train, y_train, X_test, y_test, folds, "SVM"
    )


def evaluate_gradient_boosting(X_train, y_train, X_test, y_test, param_grid=None, folds=5):
    """Train and evaluate Gradient Boosting with GridSearchCV."""
    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5],
        }
    return _fit_and_evaluate(
        GradientBoostingClassifier(random_state=999), param_grid,
        X_train, y_train, X_test, y_test, folds, "Gradient Boosting"
    )


def evaluate_random_forest(X_train, y_train, X_test, y_test, param_grid=None, folds=5):
    """Train and evaluate Random Forest with GridSearchCV."""
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": [0.5, "sqrt", "log2"],
        }
    return _fit_and_evaluate(
        RandomForestClassifier(random_state=999), param_grid,
        X_train, y_train, X_test, y_test, folds, "Random Forest"
    )


def evaluate_mlp(X_train, y_train, X_test, y_test, param_grid=None, folds=5):
    """Train and evaluate MLP (Artificial Neural Network) with GridSearchCV."""
    if param_grid is None:
        param_grid = {
            "hidden_layer_sizes": [(64,), (64, 32)],
            "activation": ["logistic", "relu", "tanh"],
            "max_iter": [500, 1000],
            "alpha": [0.001, 0.01],
            "learning_rate_init": [0.001, 0.01, 0.1],
            "solver": ["adam"],
        }
    return _fit_and_evaluate(
        MLPClassifier(random_state=999, max_iter=500), param_grid,
        X_train, y_train, X_test, y_test, folds, "Artificial Neural Network"
    )
