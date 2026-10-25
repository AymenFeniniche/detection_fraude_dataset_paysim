"""Tests rapides des métriques et pipelines du module 3."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import load_config
from src.metrics import seuil_sous_contrainte
from src.models import build_pipeline


MODULE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config():
    return load_config(MODULE_DIR / "config.yaml")


@pytest.fixture(scope="module")
def sample_data(config):
    parquet_path = (
        Path(config["_config_dir"]) / config["data"]["path"]
    ).resolve()
    data = pd.read_parquet(
        parquet_path,
        filters=[(config["data"]["time_col"], "<=", config["split"]["train_max_step"])],
    )
    sample = data.sample(n=3000, random_state=42)
    features = config["features"]["numeric"] + config["features"]["categorical"]
    return sample.loc[:, features], sample[config["data"]["target"]]


@pytest.mark.parametrize(
    "family,params",
    [
        ("dummy", {}),
        ("logreg", {"C": 1.0}),
        (
            "random_forest",
            {
                "n_estimators": 5,
                "max_depth": 3,
                "min_samples_leaf": 10,
                "max_features": "sqrt",
            },
        ),
        (
            "xgboost",
            {
                "n_estimators": 5,
                "max_depth": 2,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        ),
    ],
)
def test_pipeline_fit_predict_proba(family, params, config, sample_data):
    X, y = sample_data
    pipeline = build_pipeline(
        family,
        params,
        config,
        scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)),
    )
    pipeline.fit(X, y)
    predictions = pipeline.predict_proba(X.iloc[:20])

    assert predictions.shape == (20, 2)
    assert np.isfinite(predictions).all()

    transformer = pipeline.named_steps["pretraitement"]
    selected_columns = [
        column
        for _, _, columns in transformer.transformers
        for column in columns
    ]
    assert not set(config["features"]["excluded"]).intersection(selected_columns)


def test_threshold_obeys_maximum_false_positive_rate():
    y = np.array([0] * 1000 + [1] * 10)
    scores = np.linspace(0.0, 1.0, len(y))
    max_rate = 0.01
    threshold = seuil_sous_contrainte(y, scores, max_rate)
    false_positive_rate = ((scores > threshold) & (y == 0)).sum() / (y == 0).sum()

    assert false_positive_rate <= max_rate
