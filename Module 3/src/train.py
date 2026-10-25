"""Comparaison temporelle des modèles et évaluation finale contrôlée.

Sélection CV : le meilleur modèle non Dummy maximise la PR-AUC moyenne.
L'erreur standard du meilleur vaut son écart-type divisé par la racine du
nombre de plis. Les configurations à moins d'une erreur standard du meilleur
sont éligibles ; parmi elles, on retient la famille la moins complexe puis la
plus grande PR-AUC moyenne.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import __version__ as xgboost_version

from src.data import load_config, load_data, split_temporal
from src.metrics import bootstrap_ci, mesures, seuil_sous_contrainte
from src.mlflow_utils import log_sklearn_model
from src.models import COMPLEXITE, build_pipeline


MODULE_DIR = Path(__file__).resolve().parents[1]


def _paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Résout les chemins de sortie relativement au fichier config.yaml."""
    base = Path(cfg["_config_dir"]).resolve()
    return {
        "base": base,
        "database": base / "mlflow.db",
        "mlruns": base / "mlruns",
        "reports": base / "reports",
        "models": base / "models",
    }


def _tracking_setup(cfg: dict[str, Any]) -> str:
    """Initialise MLflow avec une base et un emplacement d'artefacts locaux."""
    paths = _paths(cfg)
    paths["reports"].mkdir(parents=True, exist_ok=True)
    paths["models"].mkdir(parents=True, exist_ok=True)
    paths["mlruns"].mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{paths['database'].as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = cfg["mlflow"]["experiment_name"]
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=paths["mlruns"].as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return experiment_id


def _read_train_only(cfg: dict[str, Any]) -> pd.DataFrame:
    """Lit uniquement les lignes d'entraînement du Parquet, sans charger le test."""
    base = Path(cfg["_config_dir"]).resolve()
    parquet_path = Path(cfg["data"]["path"]).expanduser()
    if not parquet_path.is_absolute():
        parquet_path = base / parquet_path
    parquet_path = parquet_path.resolve()
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Fichier Parquet introuvable : {parquet_path}")

    time_col = cfg["data"]["time_col"]
    train_max_step = cfg["split"]["train_max_step"]
    train = pd.read_parquet(
        parquet_path,
        filters=[(time_col, "<=", train_max_step)],
    )
    if not (train[time_col] <= train_max_step).all():
        raise ValueError("Le filtre Parquet a inclus des lignes hors entraînement.")
    return train


def _configurations(cfg: dict[str, Any], quick: bool) -> list[tuple[str, dict[str, Any], str]]:
    """Retourne Dummy puis les configurations déclarées dans le YAML."""
    result: list[tuple[str, dict[str, Any], str]] = [
        ("dummy", {}, "dummy")
    ]
    for family in ("logreg", "random_forest", "xgboost"):
        configured = cfg["models"][family]
        if quick:
            configured = configured[:1]
        for index, params in enumerate(configured, start=1):
            if family == "logreg":
                run_name = f"logreg_C{params['C']}"
            else:
                run_name = f"{family}_cfg{index}"
            result.append((family, dict(params), run_name))
    return result


def _model_params(
    family: str, params: dict[str, Any], scale_pos_weight: float
) -> dict[str, Any]:
    """Renvoie les paramètres utiles à la construction/sérialisation du modèle."""
    model_params = dict(params)
    if family == "xgboost":
        model_params["scale_pos_weight"] = scale_pos_weight
    return model_params


def _cv_phase(cfg: dict[str, Any], quick: bool, forced_run: str | None) -> None:
    """Exécute la validation croisée uniquement sur les lignes d'entraînement."""
    experiment_id = _tracking_setup(cfg)
    data = _read_train_only(cfg)
    time_col = cfg["data"]["time_col"]
    tx_id = "tx_id"
    if tx_id not in data.columns:
        raise ValueError(f"Colonne requise pour le tri temporel absente : {tx_id}")
    data = data.sort_values([time_col, tx_id], kind="stable").reset_index(drop=True)

    X_columns = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    target = cfg["data"]["target"]
    missing = [col for col in X_columns + [target] if col not in data]
    if missing:
        raise ValueError("Colonnes requises absentes : " + ", ".join(missing))
    X = data.loc[:, X_columns]
    y = data[target]

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0:
        raise ValueError("Le train ne contient aucune fraude.")
    scale_pos_weight = negatives / positives

    n_splits = 2 if quick else int(cfg["evaluation"]["cv_splits"])
    splitter = TimeSeriesSplit(n_splits=n_splits)
    configurations = _configurations(cfg, quick)
    resultats: list[dict[str, Any]] = []

    print(
        f"CV: lignes train={len(data)}, fraudes={positives}, "
        f"plis={n_splits}, configurations={len(configurations)}",
        flush=True,
    )
    for family, params, run_name in configurations:
        run_start = time.perf_counter()
        print(f"[START] {run_name}", flush=True)
        fold_pr: list[float] = []
        fold_train_pr: list[float] = []
        fold_roc: list[float] = []
        fold_recall: list[float] = []
        fit_time_total = 0.0
        pipeline = build_pipeline(
            family,
            params,
            cfg,
            scale_pos_weight=scale_pos_weight,
        )

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
        ) as run:
            log_params: dict[str, Any] = {
                "famille": family,
                "hyperparametres": json.dumps(params, sort_keys=True),
                "features_numeric": json.dumps(cfg["features"]["numeric"]),
                "features_categorical": json.dumps(
                    cfg["features"]["categorical"]
                ),
                "cv_splits": n_splits,
                "random_state": cfg["project"]["random_state"],
            }
            if family == "xgboost":
                log_params["scale_pos_weight"] = scale_pos_weight
            mlflow.log_params(log_params)
            mlflow.set_tag("famille", family)
            if family == "logreg" and params.get("C") == 1.0:
                mlflow.set_tag("reference_module2", "true")

            for fold_index, (train_index, val_index) in enumerate(
                splitter.split(X), start=1
            ):
                X_train, y_train = X.iloc[train_index], y.iloc[train_index]
                X_val, y_val = X.iloc[val_index], y.iloc[val_index]
                model = clone(pipeline)
                fit_start = time.perf_counter()
                model.fit(X_train, y_train)
                fit_time = time.perf_counter() - fit_start
                fit_time_total += fit_time

                train_scores = model.predict_proba(X_train)[:, 1]
                val_scores = model.predict_proba(X_val)[:, 1]
                train_pr = float(average_precision_score(y_train, train_scores))
                val_pr = float(average_precision_score(y_val, val_scores))
                val_roc = float(roc_auc_score(y_val, val_scores))
                threshold = seuil_sous_contrainte(
                    y_val,
                    val_scores,
                    float(cfg["evaluation"]["max_false_positive_rate"]),
                )
                val_labels = y_val.to_numpy() == 1
                val_alerts = val_scores > threshold
                val_recall = float(
                    (val_alerts & val_labels).sum() / val_labels.sum()
                )
                fold_train_pr.append(train_pr)
                fold_pr.append(val_pr)
                fold_roc.append(val_roc)
                fold_recall.append(val_recall)
                print(
                    f"  pli {fold_index}/{n_splits}: "
                    f"train_PR-AUC={train_pr:.4f}, "
                    f"val_PR-AUC={val_pr:.4f}, ROC-AUC={val_roc:.4f}, "
                    f"rappel@1%FPR={val_recall:.4f}, fit={fit_time:.1f}s",
                    flush=True,
                )

            pr_mean = float(np.mean(fold_pr))
            pr_std = float(np.std(fold_pr, ddof=1)) if len(fold_pr) > 1 else 0.0
            train_pr_mean = float(np.mean(fold_train_pr))
            metrics = {
                "cv_pr_auc_mean": pr_mean,
                "cv_pr_auc_std": pr_std,
                "train_pr_auc_mean": train_pr_mean,
                "ecart_train_cv": train_pr_mean - pr_mean,
                "cv_roc_auc_mean": float(np.mean(fold_roc)),
                "cv_recall_at_1pct_fpr_mean": float(np.mean(fold_recall)),
                "fit_time_s": fit_time_total,
            }
            mlflow.log_metrics(metrics)
            resultats.append(
                {
                    "run_name": run_name,
                    "run_id": run.info.run_id,
                    "famille": family,
                    "params": json.dumps(params, sort_keys=True),
                    **metrics,
                    "duration_s": time.perf_counter() - run_start,
                }
            )
        print(
            f"[DONE] {run_name}: PR-AUC={pr_mean:.4f} ± {pr_std:.4f}, "
            f"durée={time.perf_counter() - run_start:.1f}s",
            flush=True,
        )

    results_frame = pd.DataFrame(resultats)
    paths = _paths(cfg)
    results_path = paths["reports"] / "cv_results.csv"
    results_frame.to_csv(results_path, index=False)
    selection = _select_model(results_frame, n_splits, forced_run)
    selection_path = paths["reports"] / "selection.json"
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Résultats CV ===")
    columns = [
        "famille",
        "params",
        "cv_pr_auc_mean",
        "cv_pr_auc_std",
        "ecart_train_cv",
        "cv_recall_at_1pct_fpr_mean",
        "duration_s",
    ]
    print(results_frame[columns].to_string(index=False))
    print(f"\nRésultats enregistrés : {results_path}")
    print(f"Sélection enregistrée : {selection_path}")
    print(json.dumps(selection, ensure_ascii=False, indent=2))


def _select_model(
    results: pd.DataFrame, n_splits: int, forced_run: str | None
) -> dict[str, Any]:
    """Applique la règle à une erreur standard, ou respecte un run explicitement forcé."""
    candidates = results.loc[results["famille"] != "dummy"].copy()
    if candidates.empty:
        raise ValueError("Aucune configuration de modèle à sélectionner.")

    best_index = candidates["cv_pr_auc_mean"].idxmax()
    best = candidates.loc[best_index]
    standard_error = float(best["cv_pr_auc_std"]) / math.sqrt(n_splits)
    threshold = float(best["cv_pr_auc_mean"]) - standard_error
    eligible = candidates.loc[candidates["cv_pr_auc_mean"] >= threshold].copy()

    if forced_run is not None:
        selected_rows = candidates.loc[candidates["run_name"] == forced_run]
        if selected_rows.empty:
            raise ValueError(
                f"Run demandé introuvable parmi les résultats CV : {forced_run}"
            )
        selected = selected_rows.iloc[0]
        forced = True
        justification = (
            f"Choix forcé par --select {forced_run}. Sa PR-AUC moyenne est "
            f"{selected['cv_pr_auc_mean']:.4f}, l'écart train/CV vaut "
            f"{selected['ecart_train_cv']:.4f} et sa complexité est "
            f"{COMPLEXITE[selected['famille']]}/3."
        )
    else:
        eligible = eligible.assign(
            _complexite=eligible["famille"].map(COMPLEXITE)
        ).sort_values(
            ["_complexite", "cv_pr_auc_mean"],
            ascending=[True, False],
            kind="stable",
        )
        selected = eligible.iloc[0]
        forced = False
        justification = (
            f"PR-AUC CV moyenne de {selected['cv_pr_auc_mean']:.4f} "
            f"(meilleur={best['cv_pr_auc_mean']:.4f}, seuil d'éligibilité "
            f"à une erreur standard={threshold:.4f}), écart train/CV de "
            f"{selected['ecart_train_cv']:.4f}, famille {selected['famille']} "
            f"retenue pour sa complexité {COMPLEXITE[selected['famille']]}/3 "
            "la plus faible parmi les configurations éligibles."
        )

    selected_params = json.loads(str(selected["params"]))
    eligible_payload = [
        {
            "run_name": row["run_name"],
            "famille": row["famille"],
            "params": json.loads(str(row["params"])),
            "cv_pr_auc_mean": float(row["cv_pr_auc_mean"]),
            "cv_pr_auc_std": float(row["cv_pr_auc_std"]),
            "ecart_train_cv": float(row["ecart_train_cv"]),
            "complexite": COMPLEXITE[row["famille"]],
        }
        for _, row in eligible.iterrows()
    ]
    return {
        "run_name": str(selected["run_name"]),
        "run_id": str(selected["run_id"]),
        "famille": str(selected["famille"]),
        "params": selected_params,
        "cv_pr_auc_mean": float(selected["cv_pr_auc_mean"]),
        "cv_pr_auc_std": float(selected["cv_pr_auc_std"]),
        "ecart_train_cv": float(selected["ecart_train_cv"]),
        "eligible_configs": eligible_payload,
        "selection_forcee": forced,
        "justification": justification,
    }


def _final_phase(cfg: dict[str, Any], force: bool) -> None:
    """Réentraîne les pipelines choisis puis réserve le test à une évaluation finale."""
    paths = _paths(cfg)
    final_path = paths["reports"] / "final_metrics.json"
    if final_path.exists() and not force:
        raise FileExistsError(
            f"{final_path} existe déjà. Utilisez --force pour autoriser "
            "une nouvelle consultation du test."
        )
    selection_path = paths["reports"] / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Sélection CV introuvable : {selection_path}. Lancez --cv d'abord."
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    experiment_id = _tracking_setup(cfg)

    all_data = load_data(cfg)
    train, validation, test = split_temporal(all_data, cfg)
    numeric = cfg["features"]["numeric"]
    categorical = cfg["features"]["categorical"]
    features = numeric + categorical
    target = cfg["data"]["target"]
    train = train.sort_values(
        [cfg["data"]["time_col"], "tx_id"], kind="stable"
    )
    y_train = train[target]
    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())
    scale_pos_weight = negative / positive

    chosen_family = selection["famille"]
    chosen_params = selection["params"]
    reference_params = {"C": 1.0}
    chosen_pipeline = build_pipeline(
        chosen_family, chosen_params, cfg, scale_pos_weight
    )
    reference_pipeline = build_pipeline("logreg", reference_params, cfg)

    chosen_pipeline.fit(train[features], y_train)
    reference_pipeline.fit(train[features], y_train)
    chosen_val_scores = chosen_pipeline.predict_proba(validation[features])[:, 1]
    reference_val_scores = reference_pipeline.predict_proba(
        validation[features]
    )[:, 1]
    max_fpr = float(cfg["evaluation"]["max_false_positive_rate"])
    chosen_threshold = seuil_sous_contrainte(
        validation[target], chosen_val_scores, max_fpr
    )
    reference_threshold = seuil_sous_contrainte(
        validation[target], reference_val_scores, max_fpr
    )

    y_test = test[target].to_numpy()
    amount_test = test["amount"].to_numpy()
    chosen_test_scores = chosen_pipeline.predict_proba(test[features])[:, 1]
    reference_test_scores = reference_pipeline.predict_proba(test[features])[:, 1]
    chosen_metrics = mesures(
        y_test, chosen_test_scores, amount_test, chosen_threshold
    )
    reference_metrics = mesures(
        y_test, reference_test_scores, amount_test, reference_threshold
    )
    chosen_ci = bootstrap_ci(
        y_test, chosen_test_scores, amount_test, chosen_threshold
    )
    reference_ci = bootstrap_ci(
        y_test, reference_test_scores, amount_test, reference_threshold
    )

    selected_alerts = chosen_test_scores > chosen_threshold
    confusion = {
        "true_negative": int(((y_test == 0) & ~selected_alerts).sum()),
        "false_positive": int(((y_test == 0) & selected_alerts).sum()),
        "false_negative": int(((y_test == 1) & ~selected_alerts).sum()),
        "true_positive": int(((y_test == 1) & selected_alerts).sum()),
    }
    final_results = {
        "selected_model": {
            "famille": chosen_family,
            "params": chosen_params,
            "run_name": selection["run_name"],
            "threshold": chosen_threshold,
            "test_metrics": chosen_metrics,
            "bootstrap_ci_95": chosen_ci,
        },
        "reference_module2": {
            "famille": "logreg",
            "params": reference_params,
            "threshold": reference_threshold,
            "test_metrics": reference_metrics,
            "bootstrap_ci_95": reference_ci,
            "module2_reference": {
                "PR-AUC": 0.347,
                "Rappel": 0.386,
                "Taux de faux positifs": 0.0106,
                "Montant fraudé intercepté": 0.906,
            },
            "ecarts_absolus": {
                "PR-AUC": abs(reference_metrics["PR-AUC"] - 0.347),
                "Rappel": abs(reference_metrics["Rappel"] - 0.386),
                "Taux de faux positifs": abs(
                    reference_metrics["Taux de faux positifs"] - 0.0106
                ),
                "Montant fraudé intercepté": abs(
                    reference_metrics["Montant fraudé intercepté"] - 0.906
                ),
            },
        },
        "sizes": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "test_confusion_matrix_selected": confusion,
    }
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(
        json.dumps(final_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model_path = paths["models"] / "model_final.joblib"
    joblib.dump(chosen_pipeline, model_path)
    meta_path = paths["models"] / "model_meta.json"
    metadata = {
        "famille": chosen_family,
        "params": chosen_params,
        "seuil": chosen_threshold,
        "features_numeric": numeric,
        "features_categorical": categorical,
        "type_values": sorted(str(value) for value in train["type"].dropna().unique()),
        "versions": {
            "python": sys.version,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost_version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": selection["run_name"],
    }
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    val_metrics = {
        "selected_val_pr_auc": float(
            average_precision_score(validation[target], chosen_val_scores)
        ),
        "selected_val_recall": mesures(
            validation[target],
            chosen_val_scores,
            validation["amount"],
            chosen_threshold,
        )["Rappel"],
        "reference_val_pr_auc": float(
            average_precision_score(validation[target], reference_val_scores)
        ),
        "reference_val_recall": mesures(
            validation[target],
            reference_val_scores,
            validation["amount"],
            reference_threshold,
        )["Rappel"],
    }
    test_metrics = {
        f"selected_{key}": value for key, value in chosen_metrics.items()
    }
    test_metrics.update(
        {f"reference_{key}": value for key, value in reference_metrics.items()}
    )
    try:
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"final_{chosen_family}",
        ):
            mlflow.log_params(
                {
                    "famille": chosen_family,
                    "hyperparametres": json.dumps(chosen_params, sort_keys=True),
                    "seuil_validation": chosen_threshold,
                    "reference_seuil_validation": reference_threshold,
                    "features_numeric": json.dumps(numeric),
                    "features_categorical": json.dumps(categorical),
                }
            )
            mlflow.log_metrics(val_metrics)
            mlflow.log_metrics(test_metrics)
            serialization = log_sklearn_model(
                chosen_pipeline,
                name="model",
            )
            mlflow.set_tag("serialization_format", serialization)
            mlflow.log_artifact(str(final_path), artifact_path="reports")
            mlflow.log_artifact(str(selection_path), artifact_path="reports")
            cv_results_path = paths["reports"] / "cv_results.csv"
            if cv_results_path.is_file():
                mlflow.log_artifact(str(cv_results_path), artifact_path="reports")
            mlflow.log_artifact(str(meta_path), artifact_path="models")
    except MlflowException as error:
        print(
            "Les fichiers finaux (JSON et joblib) ont été sauvegardés avant "
            f"l'enregistrement MLflow, qui a échoué : {error}",
            file=sys.stderr,
        )
        raise

    print(json.dumps(final_results, ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_mutually_exclusive_group(required=True)
    phases.add_argument("--cv", action="store_true", help="Validation croisée.")
    phases.add_argument("--final", action="store_true", help="Évaluation finale.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="CV fumée : deux plis et première configuration par famille.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Autorise --final si final_metrics.json existe déjà.",
    )
    parser.add_argument(
        "--select",
        metavar="NOM_DU_RUN",
        help="Force le choix d'un run CV par son nom.",
    )
    args = parser.parse_args()
    if args.quick and not args.cv:
        parser.error("--quick ne s'utilise qu'avec --cv.")
    if args.force and not args.final:
        parser.error("--force ne s'utilise qu'avec --final.")
    if args.select and not args.cv:
        parser.error("--select ne s'utilise qu'avec --cv.")
    return args


def main() -> int:
    args = _parse_args()
    try:
        cfg = load_config()
        if args.cv:
            _cv_phase(cfg, args.quick, args.select)
        else:
            _final_phase(cfg, args.force)
    except MemoryError:
        print(
            "MemoryError : la mémoire disponible est insuffisante pour cette "
            "étape. Réduisez la charge ou libérez de la mémoire avant de relancer.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
