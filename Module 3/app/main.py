"""API de prédiction de fraude PaySim."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


LOGGER = logging.getLogger(__name__)
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR.parent / "models"
MODEL_PATH = MODEL_DIR / "model_final.joblib"
META_PATH = MODEL_DIR / "model_meta.json"
EXAMPLES_PATH = APP_DIR / "exemples.json"
_EXAMPLE_DATA = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))

DESCRIPTION_SCORE = (
    "Score du modèle, non calibré (classes pondérées) ; le seuil a été choisi "
    "pour limiter à 1 % les fausses alertes sur la validation."
)


class Transaction(BaseModel):
    """Caractéristiques d'une transaction disponibles au moment de la décision."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["CASH_OUT", "TRANSFER"] = Field(
        description="Type de transaction mobile."
    )
    amount: float = Field(
        ge=0,
        description="Montant de la transaction.",
    )
    oldbalanceOrg: float = Field(
        ge=0,
        description="Solde du compte émetteur avant la transaction.",
    )
    oldbalanceDest: float = Field(
        ge=0,
        description="Solde du compte destinataire avant la transaction.",
    )
    nb_tx_orig_avant: int = Field(
        ge=0,
        description=(
            "Nombre de transactions précédentes du compte émetteur, "
            "calculé par le système en amont."
        ),
    )
    nb_recu_dest_avant: int = Field(
        ge=0,
        description=(
            "Nombre de transactions reçues précédemment par le compte "
            "destinataire, calculé par le système en amont."
        ),
    )


class PredictRequest(BaseModel):
    """Requête unitaire pour la prédiction d'une transaction."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "features": _EXAMPLE_DATA["fraude"]["features"]
                },
                {
                    "features": _EXAMPLE_DATA["legitime"]["features"]
                },
            ]
        }
    )

    features: Transaction


class BatchPredictRequest(BaseModel):
    """Requête de prédiction par lots, limitée à mille transactions."""

    model_config = ConfigDict(extra="forbid")

    transactions: Annotated[
        list[Transaction],
        Field(min_length=1, max_length=1000),
    ]


class Prediction(BaseModel):
    """Résultat de prédiction pour une transaction."""

    probabilite: float
    seuil: float
    alerte: bool
    modele: str
    version_modele: str


@dataclass
class ApplicationState:
    """État chargé au démarrage du service."""

    model: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


state = ApplicationState()


def _runtime_versions() -> dict[str, str]:
    """Retourne les versions des bibliothèques liées à la sérialisation."""
    return {
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def _version_model(metadata: dict[str, Any]) -> str:
    """Construit un identifiant lisible à partir des métadonnées persistées."""
    return str(metadata.get("date_utc") or metadata.get("run_name") or "inconnue")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Charge le modèle et ses métadonnées sans empêcher le démarrage s'ils manquent."""
    state.model = None
    state.metadata = {}

    try:
        state.metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        LOGGER.warning("Métadonnées du modèle introuvables : %s", META_PATH)
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.warning("Impossible de lire les métadonnées %s : %s", META_PATH, error)

    try:
        state.model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        LOGGER.warning("Modèle introuvable : %s", MODEL_PATH)

    saved_versions = state.metadata.get("versions", {})
    runtime_versions = _runtime_versions()
    for name, runtime_version in runtime_versions.items():
        saved_version = saved_versions.get(name)
        if saved_version and saved_version != runtime_version:
            LOGGER.warning(
                "Version différente pour %s : modèle=%s, runtime=%s",
                name,
                saved_version,
                runtime_version,
            )

    yield


app = FastAPI(
    title="API de détection de fraude PaySim",
    description=(
        "Service de scoring de transactions mobiles à partir des variables "
        "disponibles avant exécution."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _predict_one(transaction: Transaction) -> Prediction:
    """Construit une ligne avec l'ordre des variables attendu par le pipeline."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")

    features = state.metadata.get("features_numeric", []) + state.metadata.get(
        "features_categorical", []
    )
    if not features:
        raise HTTPException(
            status_code=503,
            detail="Les features du modèle ne sont pas disponibles.",
        )
    payload = transaction.model_dump()
    frame = pd.DataFrame([{name: payload[name] for name in features}], columns=features)
    probability = float(state.model.predict_proba(frame)[0, 1])
    threshold = float(state.metadata["seuil"])
    return Prediction(
        probabilite=probability,
        seuil=threshold,
        alerte=probability > threshold,
        modele=str(state.metadata.get("famille", "xgboost")),
        version_modele=_version_model(state.metadata),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    """Expose l'état de disponibilité du modèle et ses versions."""
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "modele": state.metadata.get("famille"),
        "seuil": state.metadata.get("seuil"),
        "versions": _runtime_versions(),
    }


@app.post(
    "/predict",
    response_model=Prediction,
    description=DESCRIPTION_SCORE,
)
def predict(request: PredictRequest) -> Prediction:
    """Retourne le score et l'alerte pour une transaction unique."""
    return _predict_one(request.features)


@app.post(
    "/predict_batch",
    response_model=list[Prediction],
    description=DESCRIPTION_SCORE,
)
def predict_batch(request: BatchPredictRequest) -> list[Prediction]:
    """Retourne les prédictions dans le même ordre que les transactions reçues."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")

    features = state.metadata.get("features_numeric", []) + state.metadata.get(
        "features_categorical", []
    )
    if not features:
        raise HTTPException(
            status_code=503,
            detail="Les features du modèle ne sont pas disponibles.",
        )
    rows = [
        {name: transaction.model_dump()[name] for name in features}
        for transaction in request.transactions
    ]
    frame = pd.DataFrame(rows, columns=features)
    probabilities = state.model.predict_proba(frame)[:, 1]
    threshold = float(state.metadata["seuil"])
    model_name = str(state.metadata.get("famille", "xgboost"))
    version = _version_model(state.metadata)
    return [
        Prediction(
            probabilite=float(probability),
            seuil=threshold,
            alerte=float(probability) > threshold,
            modele=model_name,
            version_modele=version,
        )
        for probability in probabilities
    ]
