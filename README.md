# BCL-2 Family Inhibitors Prediction Pipeline

Machine learning pipeline to predict the inhibitory activity of small molecules against antiapoptotic proteins of the BCL-2 family (BCL-2, MCL-1 and BCL-xL).

## Background

The overexpression of BCL-2 family proteins is a key mechanism by which cancer cells evade apoptosis, particularly in leukemia. While inhibitors such as venetoclax and navitoclax have been developed, they present important limitations: venetoclax can induce resistance through MCL-1 compensatory overexpression, and navitoclax causes thrombocytopenia due to BCL-xL inhibition.

This project builds a computational pipeline to identify small molecules with selective inhibitory activity against each of the three main targets, using molecular descriptors and fingerprints as features for seven different ML classifiers.

## Pipeline overview

```
ChEMBL data → Data cleaning → Molecular descriptors / Fingerprints
    → Feature processing → Train/test split → ML model training → Evaluation
```

1. **Data collection**: bioactivity data retrieved from ChEMBL for BCL-2, MCL-1 and BCL-xL
2. **Preprocessing**: removal of duplicates, molecules with missing values, and high IC50 percentage outliers
3. **Descriptor generation**: molecular descriptors via RDKit; Morgan fingerprints (ECFP4)
4. **Feature processing**: removal of low-variance and highly correlated features, standardization
5. **Activity labeling**: binary classification based on IC50 threshold (active / inactive)
6. **Model training**: 8 classifiers evaluated for each target and feature type
7. **Evaluation**: ROC-AUC, accuracy, precision, recall, F1-score; confusion matrices

## Models evaluated

| Model | Abbreviation |
|---|---|
| Logistic Regression | LR |
| k-Nearest Neighbors | k-NN |
| Gaussian Naive Bayes | GNB |
| Bernoulli Naive Bayes | BNB |
| Gradient Boosting | GB |
| Support Vector Machine | SVM |
| Random Forest | RF |
| Artificial Neural Network | ANN |

Random Forest and Gradient Boosting consistently achieved the best performance across targets.

## Repository structure

```
├── pipeline/                         # Reusable Python package
│   ├── data.py                       # Dataset loading and molecule visualization
│   ├── preprocessing.py              # ChEMBL filters, IC50 cleaning, QSAR structure curation
│   ├── descriptors.py                # RDKit descriptors, Morgan fingerprints, feature cleaning
│   ├── features.py                   # Activity labelling, train/test split, LASSO selection
│   ├── models.py                     # 8 ML classifiers with GridSearchCV + ROC curves
│   └── visualization.py              # Feature importance, combined ROC plots, table images
├── notebooks/
│   ├── BCL-2_analysis.ipynb            # Full pipeline applied to BCL-2
│   ├── MCL-1_analysis.ipynb            # Full pipeline applied to MCL-1
│   └── BCL-xL_analysis.ipynb           # Full pipeline applied to BCL-xL
├── data/
│   ├── raw/                          # Original ChEMBL bioactivity downloads
│   │   ├── set_BCL-2.csv
│   │   ├── set_BCL-XL.csv
│   │   └── set_MCL-1.csv
│   └── processed/                    # Pipeline outputs ready for ML training
│       ├── datos_BCL2_curados.csv    # Curated descriptors per target
│       ├── datos_BCLXL_curados.csv
│       ├── datos_MCL1_curados.csv
│       ├── fingerprints_BCL2.csv     # Morgan fingerprints per target
│       ├── fingerprints_BCLXL.csv
│       └── fingerprints_MCL1.csv
├── results/
│   ├── BCL-2/                          # Auto-generated when running BCL-2 notebook
│   │   ├── Random_Forest/
│   │   │   ├── roc_descriptors.png
│   │   │   ├── roc_fingerprints.png
│   │   │   ├── metrics_descriptors.csv
│   │   │   └── metrics_fingerprints.csv
│   │   ├── Gradient_Boosting/ ...
│   │   ├── descriptors_list/
│   │   │   └── descriptors_descriptors.csv
│   │   └── summary_descriptors.csv
│   ├── MCL-1/
│   └── BCL-xL/
└── requirements.txt
```

## Requirements

- Python 3.8+
- RDKit
- scikit-learn
- pandas, numpy
- matplotlib, seaborn

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Run the analysis

Each notebook is independent. Open the one for the target of interest and run all cells:

- [`notebooks/BCL-2_analysis.ipynb`](notebooks/BCL-2_analysis.ipynb)
- [`notebooks/MCL-1_analysis.ipynb`](notebooks/MCL-1_analysis.ipynb)
- [`notebooks/BCL-xL_analysis.ipynb`](notebooks/BCL-xL_analysis.ipynb)

Results are saved automatically to `results/<target>/` at the project root.

### Apply the pipeline to your own dataset

The `pipeline/` package is target-agnostic. To use it on any ChEMBL IC50 dataset:

```python
import sys
sys.path.insert(0, ".")          # run from repo root

from pipeline import (
    load_dataset, first_filter_molecules, standardize_molecules,
    calculate_descriptors, calculate_morgan_fingerprints,
    clean_na_and_duplicates, remove_highly_correlated_columns,
    remove_low_variance_columns, standardize_data, merge_ID_response_variable,
    create_binary_activity_median, split_train_test, feature_selection_lasso,
    evaluate_random_forest,
)

df = load_dataset("my_target_data.csv")
df, log = first_filter_molecules(df, target_name="My Target Name")
df, log = standardize_molecules(df, log=log)

descriptors = calculate_descriptors(df)
descriptors, log = clean_na_and_duplicates(descriptors, log)
descriptors, _   = remove_highly_correlated_columns(descriptors)
descriptors, _   = remove_low_variance_columns(descriptors)
descriptors      = standardize_data(descriptors)
descriptors      = merge_ID_response_variable(descriptors, df)

clean = create_binary_activity_median(descriptors)
Xtr, Xte, ytr, yte = split_train_test(clean, "Activity")
Xtr, Xte, _, _, _  = feature_selection_lasso(Xtr, ytr, Xte)
results, roc_fig, model = evaluate_random_forest(Xtr, ytr, Xte, yte)
```

## Results

ROC AUC on the test set for each target, feature type, and model:

| Target | Feature type | LR | KNN | NB | GB | SVM | RF | ANN |
|--------|-------------|-----|-----|-----|-----|-----|-----|-----|
| BCL-2 | Descriptors | 0.890 | 0.923 | 0.817 | 0.914 | 0.898 | 0.921 | **0.925** |
| BCL-2 | Fingerprints | 0.908 | 0.912 | 0.857 | 0.920 | 0.907 | **0.921** | 0.920 |
| MCL-1 | Descriptors | 0.817 | 0.865 | 0.807 | 0.876 | 0.864 | **0.886** | 0.841 |
| MCL-1 | Fingerprints | 0.887 | 0.896 | 0.875 | 0.871 | 0.885 | **0.902** | 0.857 |
| BCL-xL | Descriptors | 0.944 | 0.955 | 0.931 | 0.952 | 0.955 | **0.962** | 0.956 |
| BCL-xL | Fingerprints | 0.981 | 0.978 | 0.931 | **0.984** | 0.972 | 0.974 | 0.981 |

Key observations:

- **Random Forest** is the most consistent top performer across targets and feature types.
- **BCL-xL** achieves the highest AUC values (up to 0.984), suggesting its actives and inactives are more structurally separable, despite having the smallest dataset (908 compounds).
- **MCL-1** yields the lowest AUC (max 0.902), likely reflecting greater chemical diversity in its dataset (1887 compounds).
- **Morgan fingerprints** match or outperform molecular descriptors for most target–model combinations.
- **Naive Bayes** is consistently the weakest classifier across all targets.

## Data source

Bioactivity data was retrieved from [ChEMBL](https://www.ebi.ac.uk/chembl/) (BCL-2: CHEMBL4523, MCL-1: CHEMBL4171, BCL-xL: CHEMBL4822).

## Academic context

This project was developed as part of the Master's thesis for the **MU Bioinformática y Bioestadística** programme at the [Universitat Oberta de Catalunya (UOC)](https://www.uoc.edu), within the Drug Design and Structural Biology track.

**Author**: Jorge Velázquez Gómez  
**Supervisor**: Jorge Valencia Delgadillo
