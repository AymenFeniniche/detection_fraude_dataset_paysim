"""Rejoue dans MLflow les artefacts finaux déjà sauvegardés, sans prédiction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data import load_config
from src.mlflow_utils import log_sklearn_model


MODULE_DIR = Path(__file__).resolve().parents[1]
METRIC_NAMES = {
    "PR-AUC": "pr_auc",
    "Rappel": "recall",
    "Précision": "precision",
    "Taux de faux positifs": "false_positive_rate",
    "Montant fraudé intercepté": "fraud_amount_captured",
}


def _read_json(path: Path) -> dict[str, Any]:
    """Charge un objet JSON et signale explicitement un format inattendu."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Le JSON doit contenir un objet : {path}")
    return value


def _metrics_for_run(final_metrics: dict[str, Any]) -> dict[str, float]:
    """Convertit uniquement les métriques persistées en noms MLflow stables."""
    result: dict[str, float] = {}
    for prefix, section in (
        ("selected_test", final_metrics["selected_model"]["test_metrics"]),
        ("ref_test", final_metrics["reference_module2"]["test_metrics"]),
    ):
        for label, value in section.items():
            result[f"{prefix}_{METRIC_NAMES[label]}"] = float(value)
    for prefix, section in (
        ("selected_test", final_metrics["selected_model"]["bootstrap_ci_95"]),
        ("ref_test", final_metrics["reference_module2"]["bootstrap_ci_95"]),
    ):
        for label, bounds in section.items():
            metric_name = METRIC_NAMES[label]
            result[f"{prefix}_{metric_name}_ci95_low"] = float(bounds[0])
            result[f"{prefix}_{metric_name}_ci95_high"] = float(bounds[1])

    for prefix, section in (
        ("selected_val", final_metrics["selected_model"].get("validation_metrics", {})),
        ("ref_val", final_metrics["reference_module2"].get("validation_metrics", {})),
    ):
        for label, value in section.items():
            metric_name = METRIC_NAMES.get(label)
            if metric_name is not None:
                result[f"{prefix}_{metric_name}"] = float(value)
    return result


def _remove_failed_final_runs(client: MlflowClient, experiment_id: str) -> list[str]:
    """Supprime les anciens runs final_xgboost non terminés avec succès."""
    deleted: list[str] = []
    runs = client.search_runs([experiment_id], max_results=1000)
    for run in runs:
        if run.info.run_name != "final_xgboost":
            continue
        if run.info.status == "RUNNING":
            client.set_terminated(run.info.run_id, status="FAILED")
        if run.info.status != "FINISHED":
            client.delete_run(run.info.run_id)
            deleted.append(run.info.run_id)
    return deleted


def replay_final_run() -> str:
    """Crée le run MLflow à partir des artefacts, sans accéder aux données."""
    config = load_config()
    base = Path(config["_config_dir"]).resolve()
    database = base / "mlflow.db"
    reports = base / "reports"
    models = base / "models"
    final_path = reports / "final_metrics.json"
    selection_path = reports / "selection.json"
    cv_path = reports / "cv_results.csv"
    meta_path = models / "model_meta.json"
    model_path = models / "model_final.joblib"

    for path in (final_path, selection_path, cv_path, meta_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Artefact requis introuvable : {path}")

    final_metrics = _read_json(final_path)
    metadata = _read_json(meta_path)
    selection = _read_json(selection_path)
    model = joblib.load(model_path)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Le modèle doit être un Pipeline, reçu : {type(model)}")
    if list(model.named_steps) != ["pretraitement", "modele"]:
        raise ValueError(f"Étapes de Pipeline inattendues : {list(model.named_steps)}")
    if not isinstance(model.named_steps["modele"], XGBClassifier):
        raise TypeError("L'étape modele ne contient pas un XGBClassifier.")

    tracking_uri = f"sqlite:///{database.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = config["mlflow"]["experiment_name"]
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Expérience MLflow introuvable : {experiment_name}")

    removed = _remove_failed_final_runs(client, experiment.experiment_id)
    for run_id in removed:
        print(f"Ancien run final_xgboost supprimé : {run_id}")

    params: dict[str, Any] = {
        "famille": metadata["famille"],
        "hyperparametres": json.dumps(metadata["params"], sort_keys=True),
        "seuil_validation": metadata["seuil"],
        "reference_seuil_validation": final_metrics["reference_module2"]["threshold"],
        "features_numeric": json.dumps(metadata["features_numeric"]),
        "features_categorical": json.dumps(metadata["features_categorical"]),
    }
    metrics = _metrics_for_run(final_metrics)
    # Le rapport historique ne contient pas les métriques de validation, seulement ses seuils.
    metrics["selected_val_threshold"] = float(final_metrics["selected_model"]["threshold"])
    metrics["ref_val_threshold"] = float(final_metrics["reference_module2"]["threshold"])

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name="final_xgboost",
    ) as active_run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("famille", "xgboost")
        mlflow.set_tag("evaluation_test", "une_seule_fois")
        serialization = log_sklearn_model(model, name="model")
        mlflow.set_tag("serialization_format", serialization)
        for artifact_path in (selection_path, cv_path, final_path, meta_path):
            artifact_group = "models" if artifact_path == meta_path else "reports"
            mlflow.log_artifact(str(artifact_path), artifact_path=artifact_group)

    print(f"Run créé : {active_run.info.run_id} (serialization={serialization})")
    print("Métriques MLflow enregistrées :")
    print(pd.Series(metrics).to_string())
    return active_run.info.run_id


def main() -> None:
    replay_final_run()


if __name__ == "__main__":
    main()
