"""Exporte le tableau comparatif des expériences de validation croisée MLflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = MODULE_DIR / "reports"
TRACKING_DB = MODULE_DIR / "mlflow.db"
EXPERIMENT_NAME = "paysim-module3"


def _number(run: Any, key: str) -> float:
    """Retourne une métrique de run, avec NaN si elle n'a pas été journalisée."""
    value = run.data.metrics.get(key)
    return float(value) if value is not None else float("nan")


def _hyperparameters(run: Any) -> str:
    """Formate les principaux hyperparamètres conservés dans les paramètres MLflow."""
    try:
        params = json.loads(run.data.params.get("hyperparametres", "{}"))
    except json.JSONDecodeError:
        params = {}
    if not params:
        params = {
            key: value
            for key, value in run.data.params.items()
            if key not in {
                "famille",
                "features_numeric",
                "features_categorical",
                "cv_splits",
                "random_state",
                "scale_pos_weight",
            }
        }
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items())) or "—"


def _load_comparison() -> pd.DataFrame:
    """Lit exclusivement les paramètres et métriques déjà enregistrés dans MLflow."""
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Expérience MLflow introuvable : {EXPERIMENT_NAME}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=1000,
    )
    selected_path = REPORTS_DIR / "selection.json"
    selection = json.loads(selected_path.read_text(encoding="utf-8"))
    selected_run = selection["run_name"]

    rows: list[dict[str, Any]] = []
    for run in runs:
        metrics = run.data.metrics
        if "cv_pr_auc_mean" not in metrics:
            continue
        duration_s = (
            (run.info.end_time - run.info.start_time) / 1000
            if run.info.end_time is not None and run.info.start_time is not None
            else float("nan")
        )
        rows.append(
            {
                "famille": run.data.params.get("famille", run.data.tags.get("famille", "—")),
                "run": run.info.run_name,
                "hyperparamètres": _hyperparameters(run),
                "PR-AUC CV moyenne": _number(run, "cv_pr_auc_mean"),
                "Écart-type CV": _number(run, "cv_pr_auc_std"),
                "Écart train/CV": _number(run, "ecart_train_cv"),
                "Rappel à 1 % FPR": _number(run, "cv_recall_at_1pct_fpr_mean"),
                "Durée (s)": duration_s,
                "retenu": run.info.run_name == selected_run,
            }
        )

    if not rows:
        raise RuntimeError("Aucun run FINISHED de validation croisée trouvé dans MLflow.")
    return pd.DataFrame(rows).sort_values(
        "PR-AUC CV moyenne",
        ascending=False,
        kind="stable",
    )


def _save_chart(results: pd.DataFrame, output_path: Path) -> None:
    """Crée un tableau graphique clair destiné à une projection."""
    display = results.copy()
    display["PR-AUC CV moyenne ± écart-type"] = display.apply(
        lambda row: f"{row['PR-AUC CV moyenne']:.4f} ± {row['Écart-type CV']:.4f}",
        axis=1,
    )
    display["Écart train/CV"] = display["Écart train/CV"].map(lambda value: f"{value:.4f}")
    display["Rappel à 1 % FPR"] = display["Rappel à 1 % FPR"].map(lambda value: f"{value:.4f}")
    display["Durée (s)"] = display["Durée (s)"].map(lambda value: f"{value:.1f}")
    display["Modèle retenu"] = display["retenu"].map(lambda value: "RETENU" if value else "")
    columns = [
        "famille",
        "run",
        "hyperparamètres",
        "PR-AUC CV moyenne ± écart-type",
        "Écart train/CV",
        "Rappel à 1 % FPR",
        "Durée (s)",
        "Modèle retenu",
    ]
    table = display[columns]
    fig, axis = plt.subplots(figsize=(22, max(6, 0.72 * len(table) + 2)))
    fig.patch.set_facecolor("#f4f7fb")
    axis.set_facecolor("#f4f7fb")
    axis.axis("off")
    axis.set_title(
        "Comparaison des modèles PaySim — validation croisée",
        fontsize=22,
        fontweight="bold",
        pad=24,
    )
    mpl_table = axis.table(
        cellText=table.values,
        colLabels=table.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.09, 0.13, 0.25, 0.17, 0.11, 0.12, 0.08, 0.12],
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(11)
    mpl_table.scale(1, 2.0)

    selected_index = table.index[display["retenu"]].tolist()
    for (row, _), cell in mpl_table.get_celld().items():
        cell.set_edgecolor("#d5dce5")
        if row == 0:
            cell.set_facecolor("#183153")
            cell.set_text_props(color="white", weight="bold")
        else:
            source_index = table.index[row - 1]
            if source_index in selected_index:
                cell.set_facecolor("#d8f3dc")
                cell.set_text_props(weight="bold", color="#174d2a")
            else:
                cell.set_facecolor("#ffffff" if row % 2 else "#eaf0f6")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Produit le CSV et la capture du tableau sans recalculer les métriques."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = _load_comparison()
    csv_path = REPORTS_DIR / "mlflow_comparaison.csv"
    png_path = REPORTS_DIR / "mlflow_comparaison.png"
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _save_chart(results, png_path)
    print(
        results.to_string(
            index=False,
            formatters={
                "PR-AUC CV moyenne": "{:.4f}".format,
                "Écart-type CV": "{:.4f}".format,
                "Écart train/CV": "{:.4f}".format,
                "Rappel à 1 % FPR": "{:.4f}".format,
                "Durée (s)": "{:.1f}".format,
            },
        )
    )
    print(f"\nCSV : {csv_path}")
    print(f"Tableau graphique : {png_path}")


if __name__ == "__main__":
    main()
