# MLOps Assignment – Data & Modeling Tasks (Parts 1–2)

This repo contains runnable scripts for:
1) Data acquisition + EDA (histograms, correlation heatmap, class balance)
2) Feature engineering + model development (Logistic Regression, Random Forest with CV)

## Quickstart
```bash
# 1) Install deps
pip install -r requirements.txt

# 2) Download data
python data/download_data.py --output data/raw/heart.csv

# 3) Run EDA (outputs plots to reports/figures)
python -m src.eda --data data/raw/heart.csv --out reports/figures

# 4) Train models (saves best pipeline + metrics)
python -m src.model_train --data data/raw/heart.csv --models-dir models --reports-dir reports
```

## Notes
- Dataset source: Heart Disease UCI (processed Cleveland CSV: http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data).
- Cleaning handles common missing markers, imputes numerics by median and categoricals with a placeholder.
- Feature engineering uses `StandardScaler` for numeric features and `OneHotEncoder` for categoricals.
- Cross-validation uses stratified 5-fold; metrics logged: accuracy, precision, recall, ROC-AUC. Best model (by ROC-AUC) is persisted to `models/`.
