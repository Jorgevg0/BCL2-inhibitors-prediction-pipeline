import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV


def create_binary_activity_median(data, value_col="Standard Value", activity_col="Activity"):
    """
    Create a binary activity label using the median IC50 as threshold.
    Molecules at or below the median are labelled active (1), others inactive (0).

    Parameters:
        data (pd.DataFrame): DataFrame containing value_col.
        value_col (str): Column with continuous activity values (default "Standard Value").
        activity_col (str): Name of the new binary column (default "Activity").

    Returns:
        pd.DataFrame: DataFrame with value_col replaced by binary activity_col.
    """
    median_value = data[value_col].median()
    print(f"Activity threshold ({value_col} median): {median_value:.2f}")

    data = data.copy()
    data[activity_col] = np.where(data[value_col] <= median_value, 1, 0)
    data = data.drop(columns=[value_col]).reset_index(drop=True)
    return data


def split_train_test(df, target_col, test_size=0.2, random_state=999, stratify=True):
    """
    Split a DataFrame into train/test sets, separating features from the target.
    Assumes the first column is a molecule ID (excluded from X).

    Parameters:
        df (pd.DataFrame): DataFrame with [ID | features | target].
        target_col (str): Name of the target column.
        test_size (float): Fraction reserved for testing (default 0.2).
        random_state (int): Random seed for reproducibility (default 999).
        stratify (bool): Stratify split by target distribution (default True).

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = df.iloc[:, 1:].drop(columns=[target_col])
    y = df[target_col]
    stratify_val = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_val
    )

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print("Class distribution (train):")
    print(y_train.value_counts())

    return X_train, X_test, y_train, y_test


def feature_selection_lasso(X_train, y_train, X_test, log=None,
                             alphas=None, cv=10, random_state=42):
    """
    Select features via LassoCV, fitting only on training data to prevent leakage.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features (filtered to selected features).
        log (pd.DataFrame, optional): Log DataFrame to append the selection step.
        alphas (array-like, optional): Alpha values for LassoCV grid.
        cv (int): Number of cross-validation folds (default 10).
        random_state (int): Random seed (default 42).

    Returns:
        tuple: (X_train_sel, X_test_sel, log, selected_features, best_alpha)
    """
    lasso = LassoCV(alphas=alphas, cv=cv, random_state=random_state, n_jobs=-1)
    lasso.fit(X_train, y_train)

    coef = pd.Series(lasso.coef_, index=X_train.columns)
    selected_features = coef[coef != 0].index.tolist()
    removed_features = [col for col in X_train.columns if col not in selected_features]

    print(f"LASSO: {len(selected_features)} / {X_train.shape[1]} features retained "
          f"(alpha={lasso.alpha_:.6f})")

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    if log is not None:
        log.loc[len(log)] = [
            "LASSO feature selection",
            len(removed_features),
            removed_features,
        ]

    # Plot non-zero coefficients
    coef_nonzero = coef[coef != 0].sort_values()
    fig, ax = plt.subplots(figsize=(12, max(6, len(coef_nonzero) * 0.3)))
    coef_nonzero.plot(kind="barh", color="skyblue", ax=ax)
    ax.set_title("Non-zero LASSO coefficients", fontsize=14)
    ax.set_xlabel("Coefficient weight", fontsize=12)
    for bar in ax.patches:
        w = bar.get_width()
        offset = 0.001 * (max(coef_nonzero) - min(coef_nonzero))
        ha = "left" if w >= 0 else "right"
        ax.text(w + (offset if w >= 0 else -offset),
                bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", ha=ha, fontsize=9)
    plt.tight_layout()
    plt.show()

    return X_train_sel, X_test_sel, log, selected_features, lasso.alpha_
