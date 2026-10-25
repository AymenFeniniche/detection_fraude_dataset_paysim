# Module 3 — Comparaison de modèles avancés et API de fraude PaySim

## Objectif

Comparer une régression logistique de référence, une forêt aléatoire et XGBoost par validation croisée temporelle, tracer les expériences avec MLflow, sélectionner un modèle, l'évaluer une seule fois sur le test, puis exposer le pipeline complet dans une API FastAPI et une image Docker.

## Structure du dossier

- `config.yaml` : chemins, découpage, features, modèles et paramètres d'évaluation.
- `src/data.py` : chargement de la configuration et découpage temporel.
- `src/models.py`, `src/metrics.py`, `src/train.py` : pipelines, métriques et phases CV/finale.
- `src/importance.py` : importance par permutation calculée sur la validation.
- `src/export_mlflow.py` : export du tableau comparatif enregistré dans MLflow.
- `src/log_final.py` : réenregistrement des artefacts finaux à partir des fichiers sauvegardés.
- `app/` : API, schémas et exemples d'entrée.
- `demo/` : client HTTP de démonstration.
- `models/` : pipeline complet final et métadonnées.
- `reports/` : résultats CV, sélection, métriques finales et graphiques.
- `tests/` : tests des données, des pipelines et de l'API.
- `Dockerfile`, `.dockerignore` : conteneur de service minimal.

## Installation

Depuis le dossier `Module 3` :

```powershell
python -m pip install -r requirements.txt
```

## Commandes

Depuis `Module 3` :

```powershell
python -m src.data
python -m src.train --cv
python -m src.train --final
python -m src.importance
python -m src.export_mlflow
python -m src.log_final
python -m pytest tests -q
mlflow ui --backend-store-uri sqlite:///mlflow.db
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
python demo/appeler_api.py http://127.0.0.1:8000
```

`python -m src.data` est un diagnostic initial qui lit le Parquet complet, y compris les lignes de test. Ne pas l'exécuter lorsque le jeu de test doit rester non consulté.

Ne relancer `python -m src.train --final` qu'en sachant que cette phase consulte le test. L'option de forçage ne doit pas servir à répéter une évaluation déjà effectuée.

Construire et démarrer l'image depuis `Module 3` :

```powershell
docker build -t paysim-fraude-api .
docker run --rm -p 8000:8000 paysim-fraude-api
```

Exemple de requête :

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"features":{"type":"CASH_OUT","amount":10000000,"oldbalanceOrg":10000000,"oldbalanceDest":0,"nb_tx_orig_avant":0,"nb_recu_dest_avant":0}}'
```

La documentation interactive est disponible sur `http://127.0.0.1:8000/docs`.

## Méthode

- Le découpage est temporel, selon `step` : train jusqu'à 281 inclus, validation de 282 à 354 inclus, puis test au-delà de 354.
- La comparaison CV utilise `TimeSeriesSplit` sur le seul train ; la validation croisée n'accède pas au test.
- La sélection privilégie la PR-AUC moyenne, puis retient les modèles dans la limite d'une erreur standard du meilleur ; parmi eux, la complexité la plus faible est favorisée.
- Le choix final enregistré est `xgboost_cfg1` : cfg1 a été retenu plutôt que cfg2, car l'écart train/CV est plus faible (0,129 contre 0,253), à performance CV comparable.
- Le seuil est déterminé sur la validation avec une contrainte de 1 % de faux positifs.
- Le jeu de test a été évalué une seule fois. Les commandes de préparation de l'API, d'export et de démonstration ne le chargent pas.

## Résultats finaux

Valeurs lues dans `reports/final_metrics.json`. Les intervalles entre crochets sont les IC bootstrap à 95 %.

| Modèle | PR-AUC (IC 95 %) | Rappel (IC 95 %) | Précision | Taux de faux positifs (IC 95 %) | Montant fraudé intercepté |
|---|---:|---:|---:|---:|---:|
| Référence logreg C=1 | 0,3472 [0,3133 ; 0,3771] | 0,3855 [0,3535 ; 0,4167] | 0,2181 | 0,01062 [0,01003 ; 0,01119] | 0,9057 |
| XGBoost cfg1 | 0,8343 [0,8140 ; 0,8559] | 0,8861 [0,8648 ; 0,9062] | 0,4095 | 0,00982 [0,00934 ; 0,01035] | 0,9895 |

Le taux de faux positifs mesuré pour XGBoost est inférieur à 1 %, mais son intervalle bootstrap dépasse légèrement cette limite à sa borne supérieure.

## Importance par permutation

`python -m src.importance` permute les six features brutes du pipeline sur la validation uniquement. Les résultats sont écrits dans `reports/permutation_importance.csv` et `reports/permutation_importance.png`. Cette mesure compare la baisse de PR-AUC associée à la permutation ; ce n'est pas une mesure de causalité.

## Limites

- PaySim est un jeu synthétique ; les résultats ne garantissent pas une performance identique sur des opérations réelles.
- L'échantillon est issu d'un échantillonnage Bernoulli de 20 % des transactions `TRANSFER` et `CASH_OUT`.
- Le taux de fraude varie entre les partitions : train 0,18 %, validation 0,12 %, test 0,76 %. La précision dépend du taux de base et peut changer si celui-ci évolue.
- Les scores sont non calibrés et produits par des modèles entraînés avec pondération de classes.
- Le taux de faux positifs mesuré est 0,98 %, avec un IC 95 % de 0,93 à 1,03 % : l'objectif ponctuel est atteint, mais pas avec une marge statistique large.
- Les variables d'historique (`nb_tx_orig_avant`, `nb_recu_dest_avant`) doivent être calculées et fournies par le système en amont.
- cfg1 a été choisi à la place de cfg2 sur le critère de généralisation/complexité : écart train/CV 0,129 contre 0,253.
- Il n'y a pas encore de supervision de production ni de suivi de dérive ; ces sujets relèvent du Module 4.

## Pistes d'amélioration

- Étudier de nouvelles variables disponibles avant l'exécution d'une transaction.
- Évaluer une calibration des scores sur une validation adaptée aux taux de base observés en production.
- Choisir et documenter un seuil selon les coûts métier des faux positifs et des fraudes manquées.
