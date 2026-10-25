"""Helpers MLflow partagés par les commandes du module 3."""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.exceptions import MlflowException


TRUSTED_SKOPS_TYPES = [
    "numpy.dtype",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]


def log_sklearn_model(model: Any, name: str) -> str:
    """Journalise un modèle sklearn, puis tente cloudpickle si skops le refuse."""
    try:
        mlflow.sklearn.log_model(
            model,
            name=name,
            skops_trusted_types=TRUSTED_SKOPS_TYPES,
        )
        return "skops"
    except TypeError:
        try:
            mlflow.sklearn.log_model(
                model,
                artifact_path=name,
                skops_trusted_types=TRUSTED_SKOPS_TYPES,
            )
            return "skops"
        except MlflowException as error:
            print(f"MLflow skops a échoué, repli cloudpickle : {error}")
    except MlflowException as error:
        print(f"MLflow skops a échoué, repli cloudpickle : {error}")

    try:
        mlflow.sklearn.log_model(
            model,
            name=name,
            serialization_format="cloudpickle",
        )
    except TypeError:
        mlflow.sklearn.log_model(
            model,
            artifact_path=name,
            serialization_format="cloudpickle",
        )
    return "cloudpickle"
