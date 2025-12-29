import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from src.model_train import build_preprocessor


def test_pipeline_predict_shape(tmp_path):
    # Minimal synthetic dataset with expected columns
    df = pd.DataFrame(
        {
            "age": [60, 55],
            "sex": [1, 0],
            "cp": [2, 1],
            "trestbps": [120, 130],
            "chol": [230, 240],
            "fbs": [0, 1],
            "restecg": [0, 1],
            "thalach": [150, 160],
            "exang": [0, 1],
            "oldpeak": [2.0, 1.0],
            "slope": [1, 2],
            "ca": [0, 0],
            "thal": [2, 3],
            "target": [0, 1],
        }
    )
    X = df.drop(columns=["target"])
    y = df["target"]
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = build_preprocessor(cat_cols, num_cols)
    pipe = Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=200))])
    pipe.fit(X, y)

    preds = pipe.predict(X)
    assert preds.shape[0] == X.shape[0]

    # Ensure model can be serialized
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipe, model_path)
    loaded = joblib.load(model_path)
    assert loaded.predict(X).shape[0] == X.shape[0]



