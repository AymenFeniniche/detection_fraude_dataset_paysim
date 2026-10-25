"""Métriques de détection de fraude et intervalles bootstrap."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score


def mesures(
    y: Any, score: Any, montant: Any, seuil: float
) -> dict[str, float]:
    """Calcule les métriques avec une alerte lorsque le score dépasse le seuil."""
    y_array = np.asarray(y)
    score_array = np.asarray(score)
    montant_array = np.asarray(montant)
    fraude = y_array == 1
    alerte = score_array > seuil
    vrais_positifs = int((alerte & fraude).sum())
    faux_positifs = int((alerte & ~fraude).sum())

    return {
        "PR-AUC": float(average_precision_score(y_array, score_array)),
        "Rappel": float(vrais_positifs / fraude.sum()),
        "Précision": float(
            vrais_positifs / max(vrais_positifs + faux_positifs, 1)
        ),
        "Taux de faux positifs": float(faux_positifs / (~fraude).sum()),
        "Montant fraudé intercepté": float(
            montant_array[alerte & fraude].sum() / montant_array[fraude].sum()
        ),
    }


def seuil_sous_contrainte(y: Any, score: Any, taux_max: float) -> float:
    """Retourne le quantile des scores légitimes respectant le taux maximal."""
    y_array = np.asarray(y)
    score_array = np.asarray(score)
    scores_legitimes = score_array[y_array == 0]
    if scores_legitimes.size == 0:
        raise ValueError("Impossible de calculer le seuil sans transactions légitimes.")
    if not 0 <= taux_max < 1:
        raise ValueError("taux_max doit être compris entre 0 inclus et 1 exclu.")
    return float(
        np.quantile(scores_legitimes, 1 - taux_max, method="higher")
    )


def bootstrap_ci(
    y: Any,
    score: Any,
    montant: Any,
    seuil: float,
    n: int = 300,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Calcule des IC percentile à 95 % par bootstrap des transactions."""
    if n < 1:
        raise ValueError("n doit être strictement positif.")

    y_array = np.asarray(y)
    score_array = np.asarray(score)
    montant_array = np.asarray(montant)
    if not (len(y_array) == len(score_array) == len(montant_array)):
        raise ValueError("y, score et montant doivent avoir la même longueur.")
    if len(y_array) == 0:
        raise ValueError("Le bootstrap nécessite au moins une observation.")

    rng = np.random.default_rng(seed)
    noms = ("PR-AUC", "Rappel", "Taux de faux positifs")
    tirages: dict[str, list[float]] = {nom: [] for nom in noms}
    for _ in range(n):
        indices = rng.integers(0, len(y_array), len(y_array))
        scores = mesures(
            y_array[indices],
            score_array[indices],
            montant_array[indices],
            seuil,
        )
        for nom in noms:
            tirages[nom].append(scores[nom])

    return {
        nom: [
            float(np.quantile(valeurs, 0.025)),
            float(np.quantile(valeurs, 0.975)),
        ]
        for nom, valeurs in tirages.items()
    }
