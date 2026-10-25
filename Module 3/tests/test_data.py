"""Tests de non-régression du découpage et des features PaySim."""

from pathlib import Path

import pandas as pd

from src.data import get_xy, load_config, split_temporal


MODULE_DIR = Path(__file__).resolve().parents[1]


def _data_and_config():
    config = load_config(MODULE_DIR / "config.yaml")
    parquet_path = (
        Path(config["_config_dir"]) / config["data"]["path"]
    ).resolve()
    # Les tests de découpage ne chargent que les périodes train et validation.
    data = pd.read_parquet(
        parquet_path,
        filters=[
            (config["data"]["time_col"], "<=", config["split"]["val_max_step"])
        ],
    )
    config["split"]["expected_sizes"] = {
        **config["split"]["expected_sizes"],
        "test": 0,
    }
    return config, data


def test_temporal_split_sizes_and_order():
    config, data = _data_and_config()
    train, validation, test = split_temporal(data, config)
    time_col = config["data"]["time_col"]

    assert len(train) == config["split"]["expected_sizes"]["train"]
    assert len(validation) == config["split"]["expected_sizes"]["val"]
    assert test.empty
    assert train[time_col].max() < validation[time_col].min()
    assert validation[time_col].max() == config["split"]["val_max_step"]


def test_features_exclude_forbidden_columns_and_target():
    config, data = _data_and_config()
    train, _, _ = split_temporal(data, config)
    X, y = get_xy(train, config)

    assert not set(config["features"]["excluded"]).intersection(X.columns)
    assert config["data"]["target"] not in X.columns
    assert y.name == config["data"]["target"]
