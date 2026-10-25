"""Importance par permutation calculée exclusivement sur la validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from mlflow.tracking import MlflowClient

from src.data import load_config


MODULE_DIR = Path(__file__).resolve().parents[1]
FEATURE_LABELS = {
    "type": "Type de transaction",
    "amount": "Montant",
    "oldbalanceOrg": "Solde initial de l'émetteur",
    "oldbalanceDest": "Solde initial du destinataire",
    "nb_tx_orig_avant": "Transactions antérieures de l'émetteur",
    "nb_recu_dest_avant": "Réceptions antérieures du destinataire",
}


def _load_validation(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """Lit uniquement les lignes de validation, sans charger le test."""
    base = Path(config["_config_dir"]).resolve()
    parquet_path = Path(config["data"]["path"]).expanduser()
    if not parquet_path.is_absolute():
        parquet_path = base / parquet_path
    parquet_path = parquet_path.resolve()
    val_min = int(config["split"]["train_max_step"])
    val_max = int(config["split"]["val_max_step"])
    time_col = config["data"]["time_col"]
    validation = pd.read_parquet(
        parquet_path,
        filters=[
            (time_col, ">", val_min),
            (time_col, "<=", val_max),
        ],
    )
    expected_size = int(config["split"]["expected_sizes"]["val"])
    if len(validation) != expected_size:
        raise ValueError(
            f"Taille validation incorrecte : {len(validation)}, attendu {expected_size}."
        )
    columns = config["features"]["numeric"] + config["features"]["categorical"]
    target = config["data"]["target"]
    return validation.loc[:, columns], validation[target]


def _log_artifacts_to_final_run(
    config: dict[str, Any], csv_path: Path, png_path: Path
) -> str:
    """Ajoute les deux graphiques au run final_xgboost déjà terminé."""
    base = Path(config["_config_dir"]).resolve()
    tracking_uri = f"sqlite:///{(base / 'mlflow.db').as_posix()}"
    mlflow_client = MlflowClient(tracking_uri=tracking_uri)
    experiment = mlflow_client.get_experiment_by_name(
        config["mlflow"]["experiment_name"]
    )
    if experiment is None:
        raise ValueError("Expérience paysim-module3 introuvable.")
    runs = mlflow_client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'final_xgboost'",
        order_by=["attributes.start_time DESC"],
        max_results=10,
    )
    finished = [run for run in runs if run.info.status == "FINISHED"]
    if not finished:
        raise ValueError("Aucun run final_xgboost FINISHED trouvé dans MLflow.")
    run = finished[0]
    mlflow_client.log_artifact(run.info.run_id, str(csv_path), artifact_path="reports")
    mlflow_client.log_artifact(run.info.run_id, str(png_path), artifact_path="reports")
    return run.info.run_id


def main() -> None:
    config = load_config()
    base = Path(config["_config_dir"]).resolve()
    model_path = base / "models" / "model_final.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Modèle final introuvable : {model_path}")
    model = joblib.load(model_path)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Le modèle doit être un Pipeline, reçu : {type(model)}")
    if list(model.named_steps) != ["pretraitement", "modele"]:
        raise ValueError(f"Étapes de Pipeline inattendues : {list(model.named_steps)}")

    X_val, y_val = _load_validation(config)
    print(
        f"Permutation importance sur validation seulement : {len(X_val)} lignes, "
        f"{X_val.shape[1]} features",
        flush=True,
    )
    results = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=2,
    )
    frame = pd.DataFrame(
        {
            "variable": X_val.columns,
            "importance_moyenne": results.importances_mean,
            "ecart_type": results.importances_std,
        }
    ).sort_values("importance_moyenne", ascending=False, kind="stable")
    reports = base / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "permutation_importance.csv"
    png_path = reports / "permutation_importance.png"
    frame.to_csv(csv_path, index=False)

    plot_frame = frame.sort_values("importance_moyenne", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    labels = [
        FEATURE_LABELS.get(name, name)
        for name in plot_frame["variable"].tolist()
    ]
    ax.barh(
        labels,
        plot_frame["importance_moyenne"],
        xerr=plot_frame["ecart_type"],
        color="#3274a1",
        ecolor="#333333",
        capsize=5,
    )
    ax.set_title(
        "Importance des variables par permutation — validation",
        fontsize=17,
        pad=16,
    )
    ax.set_xlabel("Baisse moyenne de la PR-AUC après permutation", fontsize=13)
    ax.set_ylabel("Variable", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    run_id = _log_artifacts_to_final_run(config, csv_path, png_path)
    print(f"Artefacts ajoutés au run final_xgboost : {run_id}")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
