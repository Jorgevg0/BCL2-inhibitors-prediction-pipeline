import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project root = two levels up from this file (pipeline/visualization.py → project/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")

# Model name → subfolder name mapping
_MODEL_FOLDERS = {
    "LR":  "Logistic_Regression",
    "KNN": "k-NN",
    "GNB": "Naive_Bayes",
    "BNB": "Naive_Bayes",
    "SVM": "SVM",
    "GB":  "Gradient_Boosting",
    "RF":  "Random_Forest",
    "ANN": "Neural_Network",
}


def plot_feature_importance(model, feature_names=None, top_n=None, figsize=(10, 8), title=None):
    """
    Plot horizontal bar chart of feature importances for tree-based models.

    Parameters:
        model: Fitted model with a `feature_importances_` attribute.
        feature_names (list, optional): Feature names. Inferred from model if None.
        top_n (int, optional): Show only the top N features. Shows all if None.
        figsize (tuple): Figure dimensions.
        title (str, optional): Plot title.
    """
    importances = model.feature_importances_
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(len(importances))]

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    if top_n is not None:
        feat_imp = feat_imp[-top_n:]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(feat_imp.index, feat_imp.values, color="skyblue")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(title or "Feature Importance")

    max_val = max(feat_imp.values) if len(feat_imp) > 0 else 1
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01 * max_val, bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.show()
    return fig


def save_combined_table_as_image(dfs, column_names, filepath=None, fig_size=(570, 320)):
    """
    Combine multiple DataFrames side by side and render them as a table image.

    Requires df2img and plotly installed.

    Parameters:
        dfs (list of pd.DataFrame): DataFrames to concatenate horizontally.
        column_names (list of str): Column group labels for the combined table.
        filepath (str, optional): Path to save the PNG (including filename).
        fig_size (tuple): Figure size in pixels (width, height).

    Returns:
        plotly Figure of the table.
    """
    import df2img
    import plotly.io as pio

    combined_df = pd.concat(dfs, axis=1)
    combined_df.columns = column_names

    fig = df2img.plot_dataframe(combined_df, fig_size=fig_size)

    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            pio.write_image(fig, filepath, format="png")
        except Exception as e:
            print(f"Error saving image: {e}")

    fig.show()
    return fig


def visualize_three_roc(fig1, fig2, fig3, file_path=None):
    """
    Arrange three ROC curve figures in a 2×2 grid (third figure spans the bottom row).

    Parameters:
        fig1 (matplotlib.Figure): ROC for target 1.
        fig2 (matplotlib.Figure): ROC for target 2.
        fig3 (matplotlib.Figure): ROC for target 3.
        file_path (str, optional): Path to save the combined figure.

    Returns:
        matplotlib.Figure: Combined figure.
    """
    def fig_to_rgb(fig):
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return img[:, :, :3]

    img1, img2, img3 = fig_to_rgb(fig1), fig_to_rgb(fig2), fig_to_rgb(fig3)

    combined = plt.figure(figsize=(25, 16), dpi=150)
    gs = combined.add_gridspec(2, 2, hspace=0.05, wspace=0.001)

    ax1 = combined.add_subplot(gs[0, 0])
    ax1.imshow(img1)
    ax1.axis("off")

    ax2 = combined.add_subplot(gs[0, 1])
    ax2.imshow(img2)
    ax2.axis("off")

    ax3 = combined.add_subplot(gs[1, :])
    ax3.imshow(img3)
    ax3.axis("off")

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        combined.savefig(file_path, bbox_inches="tight", dpi=150)

    return combined


def save_results(results, figures, target, feature_type):
    """
    Save ROC curves and metrics tables for all models, organised by target and model.

    Directory layout produced (always relative to the project root):
        results/
        └── BCL-2/
            ├── Random_Forest/
            │   ├── roc_descriptors.png
            │   └── metrics_descriptors.csv
            ├── Gradient_Boosting/
            │   └── ...
            └── summary_descriptors.csv   ← all models side by side

    Parameters:
        results (dict): {model_key: pd.DataFrame} — metrics table per model.
            Valid keys: "LR", "KNN", "GNB", "BNB", "SVM", "GB", "RF", "ANN".
        figures (dict): {model_key: matplotlib.Figure} — ROC curve per model.
        target (str): Target name, used as subfolder (e.g. "BCL-2").
        feature_type (str): "descriptors" or "fingerprints".
    """
    target_dir = os.path.join(_DEFAULT_RESULTS_DIR, target)

    for key, table in results.items():
        folder = _MODEL_FOLDERS.get(key, key)
        model_dir = os.path.join(target_dir, folder)
        os.makedirs(model_dir, exist_ok=True)

        table.to_csv(os.path.join(model_dir, f"metrics_{feature_type}.csv"))

        if key in figures:
            figures[key].savefig(
                os.path.join(model_dir, f"roc_{feature_type}.png"),
                bbox_inches="tight", dpi=150,
            )

        print(f"  {key} → {os.path.relpath(model_dir, _PROJECT_ROOT)}/")

    summary = pd.concat(
        [df.rename(columns={"Value": key}) for key, df in results.items()], axis=1
    )
    summary_path = os.path.join(target_dir, f"summary_{feature_type}.csv")
    summary.to_csv(summary_path)
    print(f"Summary → {os.path.relpath(summary_path, _PROJECT_ROOT)}")


def save_descriptor_list(descriptor_names, target, feature_type):
    """
    Save the feature names retained after LASSO selection.

    Produces:
        results/<target>/descriptors_list/descriptors_<feature_type>.csv

    Parameters:
        descriptor_names (list): Feature names retained after selection.
        target (str): Target name (e.g. "BCL-2").
        feature_type (str): "descriptors" or "fingerprints".
    """
    out_dir = os.path.join(_DEFAULT_RESULTS_DIR, target, "descriptors_list")
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"descriptors_{feature_type}.csv")
    pd.DataFrame({"descriptor": descriptor_names}).to_csv(path, index=False)
    print(f"Descriptor list ({len(descriptor_names)}) → {os.path.relpath(path, _PROJECT_ROOT)}")
