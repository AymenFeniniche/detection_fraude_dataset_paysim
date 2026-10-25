# Annexe — Usage de l'intelligence artificielle (Module 3)

**Projet** : Groupe E — Détection de fraude aux paiements mobiles (PaySim)
**Module** : 3 — Modélisation avancée et MLOps
**Auteur** : Glenn Tchamagam Wouagam
**Date** : 25 octobre 2026

## 1. Outils utilisés

| Outil | Rôle |
|---|---|
| Claude (Anthropic) | Assistant de conversation : planification du travail, rédaction des consignes envoyées à Copilot, relecture et explication des résultats, préparation du document de démonstration et de cette annexe. |
| GitHub Copilot (VS Code, mode agent) | Écriture et exécution du code dans le dépôt, à partir des consignes rédigées avec Claude. |

## 2. Pour quoi faire

- Mise en place du dossier `Module 3/` : configuration (`config.yaml`), chargement et découpage des données, tests.
- Comparaison de trois familles de modèles (régression logistique, forêt aléatoire, XGBoost) avec trois réglages chacune, validation croisée temporelle et suivi des expériences dans MLflow.
- Choix du seuil, évaluation finale, intervalles de confiance (bootstrap) et importance des variables par permutation.
- API FastAPI, Dockerfile, intégration continue (GitHub Actions) et README.
- Document de démonstration (PDF) et rédaction de cette annexe.

## 3. Ce que j'ai fait et décidé moi-même

- Reprise du dépôt du groupe (Modules 1 et 2 réalisés par l'équipe) et décision de travailler uniquement dans un dossier `Module 3/` et sur ma propre branche (`glenn/m3-modeles-api`), sans modifier les Modules 1 et 2 ni la branche `main`.
- Validation du choix du modèle retenu (XGBoost, réglage cfg1) après comparaison des résultats de validation croisée.
- Pilotage du travail : envoi des consignes à Copilot, relecture de ses réponses, tests manuels de l'API, vérification de la CI sur GitHub, captures d'écran de MLflow et de l'API.

## 4. Ce qui a été vérifié

- **Pipeline de référence du Module 2** : le fichier sauvegardé n'est plus utilisable avec la version actuelle de scikit-learn (1.7.2 à l'enregistrement, 1.9.0 installée). La régression logistique a donc été réentraînée avec le même code et le même découpage ; elle retrouve les résultats du Module 2 à moins de 0,05 point près.
- **Fuites de données** : seules 6 variables connues avant l'exécution de la transaction sont utilisées ; `newbalanceOrig`, `newbalanceDest` et `isFlaggedFraud` sont exclues. Un test automatique vérifie qu'aucune colonne exclue n'entre dans le modèle, et l'API refuse un champ supplémentaire comme `newbalanceOrig` (erreur 422 vérifiée à la main).
- **Découpage temporel** identique à celui du Module 2 (entraînement jusqu'au step 281, validation jusqu'au step 354, test au-delà), tailles contrôlées par un test.
- **Sélection du modèle** sur la validation croisée de l'entraînement uniquement ; seuil fixé sur la validation ; importance par permutation calculée sur la validation.
- **Jeu de test évalué une seule fois.** L'enregistrement du modèle dans MLflow a échoué juste après ce calcul (types non approuvés par le format de sérialisation) : l'évaluation n'a pas été relancée ; les résultats sauvegardés ont été relus puis enregistrés dans MLflow.
- Les chiffres de ce rendu proviennent des fichiers produits par l'exécution (`reports/final_metrics.json`, MLflow) et non de texte généré par l'IA.
- **Tests et mise en service** : 14 tests pytest passent ; la CI GitHub Actions est validée ; l'API a été testée dans Swagger (fraude : alerte ; transaction légitime : pas d'alerte ; type invalide : 422 ; champ de fuite : 422) et dans un conteneur Docker (statut healthy). Les exemples de l'API viennent de la validation, jamais du test.

## 5. Limites et points de vigilance

- Les outils d'IA peuvent se tromper : aucun résultat n'a été repris sans exécution ni contrôle (tests, métriques, captures).
- Les données sont synthétiques (PaySim) : aucune donnée personnelle n'a été confiée aux outils d'IA, et les performances obtenues ne garantissent rien en production.
- Le choix de cfg1 plutôt que cfg2 (que la règle automatique de sélection proposait) est un arbitrage sur le surapprentissage, expliqué dans le README du Module 3.
