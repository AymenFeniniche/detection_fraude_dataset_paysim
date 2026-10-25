"""Construction des pipelines de classification PaySim."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


COMPLEXITE = {"dummy": 0, "logreg": 1, "random_forest": 2, "xgboost": 3}


def build_pipeline(
    famille: str,
    params: dict[str, Any],
    cfg: dict[str, Any],
    scale_pos_weight: float | None = None,
) -> Pipeline:
    """Construit un pipeline complet avec prétraitement sans fuite de données."""
    numeriques = cfg["features"]["numeric"]
    categorielles = cfg["features"]["categorical"]

    pipeline_numerique = Pipeline(
        steps=[
            ("imputation", SimpleImputer(strategy="median")),
            ("standardisation", StandardScaler()),
        ]
    )
    pipeline_categoriel = Pipeline(
        steps=[
            ("imputation", SimpleImputer(strategy="most_frequent")),
            ("encodage", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocesseur = ColumnTransformer(
        transformers=[
            ("numerique", pipeline_numerique, numeriques),
            ("categoriel", pipeline_categoriel, categorielles),
        ]
    )

    graine = cfg["project"]["random_state"]
    if famille == "dummy":
        modele = DummyClassifier(strategy="prior", **params)
    elif famille == "logreg":
        modele = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=graine,
            **params,
        )
    elif famille == "random_forest":
        modele = RandomForestClassifier(
            class_weight="balanced_subsample",
            n_jobs=4,
            random_state=graine,
            **params,
        )
    elif famille == "xgboost":
        modele = XGBClassifier(
            tree_method="hist",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            n_jobs=4,
            random_state=graine,
            **params,
        )
    else:
        raise ValueError(
            f"Famille inconnue : {famille!r}. "
            f"Valeurs autorisées : {', '.join(COMPLEXITE)}"
        )

    return Pipeline(
        steps=[
            ("pretraitement", preprocesseur),
            ("modele", modele),
        ]
    )
