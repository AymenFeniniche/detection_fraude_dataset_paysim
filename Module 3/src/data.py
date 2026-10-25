"""Chargement, découpage temporel et préparation des données PaySim."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


MODULE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = MODULE_DIR / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Charge la configuration et mémorise le dossier servant de base aux chemins."""
    config_path = Path(path).expanduser() if path is not None else DEFAULT_CONFIG_PATH
    config_path = config_path.resolve()

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"La configuration doit être un objet YAML : {config_path}")

    config["_config_dir"] = str(config_path.parent)
    return config


def _config_directory(cfg: dict[str, Any]) -> Path:
    """Retourne le dossier de configuration, ou celui du Module 3 par défaut."""
    return Path(cfg.get("_config_dir", MODULE_DIR)).resolve()


def load_data(cfg: dict[str, Any]) -> pd.DataFrame:
    """Charge le fichier Parquet indiqué, relativement au fichier de configuration."""
    parquet_path = Path(cfg["data"]["path"]).expanduser()
    if not parquet_path.is_absolute():
        parquet_path = _config_directory(cfg) / parquet_path
    parquet_path = parquet_path.resolve()

    if not parquet_path.is_file():
        raise FileNotFoundError(f"Fichier Parquet introuvable : {parquet_path}")

    return pd.read_parquet(parquet_path)


def split_temporal(
    df: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpe les transactions par périodes successives de `step`."""
    time_col = cfg["data"]["time_col"]
    required_sizes = cfg["split"]["expected_sizes"]
    train_max_step = cfg["split"]["train_max_step"]
    val_max_step = cfg["split"]["val_max_step"]

    if time_col not in df.columns:
        raise ValueError(f"Colonne temporelle absente des données : {time_col}")

    train = df.loc[df[time_col] <= train_max_step].copy()
    validation = df.loc[
        (df[time_col] > train_max_step) & (df[time_col] <= val_max_step)
    ].copy()
    test = df.loc[df[time_col] > val_max_step].copy()

    actual_sizes = {
        "train": len(train),
        "val": len(validation),
        "test": len(test),
    }
    mismatches = {
        name: (actual_sizes[name], expected)
        for name, expected in required_sizes.items()
        if name in actual_sizes and actual_sizes[name] != expected
    }
    if mismatches:
        details = ", ".join(
            f"{name}: obtenu {actual}, attendu {expected}"
            for name, (actual, expected) in mismatches.items()
        )
        raise ValueError(
            "Les tailles du découpage temporel ne correspondent pas à "
            f"expected_sizes ({details})."
        )

    return train, validation, test


def get_xy(
    df: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare les features autorisées de la cible après contrôles anti-fuite."""
    feature_config = cfg["features"]
    feature_columns = feature_config["numeric"] + feature_config["categorical"]
    target = cfg["data"]["target"]
    missing_columns = [column for column in feature_columns + [target] if column not in df]
    if missing_columns:
        raise ValueError(
            "Colonnes requises absentes des données : "
            + ", ".join(missing_columns)
        )

    X = df.loc[:, feature_columns].copy()
    excluded_columns = set(feature_config["excluded"]).intersection(X.columns)
    if excluded_columns:
        raise ValueError(
            "Des colonnes exclues figurent dans les features : "
            + ", ".join(sorted(excluded_columns))
        )
    if target in X.columns:
        raise ValueError(f"La cible {target!r} ne doit pas être une feature.")

    return X, df[target].copy()


def main() -> None:
    """Affiche un contrôle synthétique des jeux de données et des features."""
    config = load_config()
    data = load_data(config)
    train, validation, test = split_temporal(data, config)

    for name, subset in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        fraud_count = int(subset[config["data"]["target"]].sum())
        fraud_rate = float(subset[config["data"]["target"]].mean())
        print(
            f"{name}: lignes={len(subset)}, fraudes={fraud_count}, "
            f"taux_fraude={fraud_rate:.4%}"
        )

    X_train, _ = get_xy(train, config)
    print(f"Colonnes de X: {list(X_train.columns)}")


if __name__ == "__main__":
    main()
