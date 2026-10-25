# Annexe — Usage de l'intelligence artificielle pour le Module 3

Les outils utilisés sont Claude (planification, rédaction des prompts et relecture des résultats) et GitHub Copilot dans VS Code (écriture et exécution du code).

L'intelligence artificielle a été utilisée pour contribuer à la structure du projet, au code de comparaison des modèles, à l'intégration de MLflow, à l'API FastAPI, au Dockerfile, à la CI et à la documentation.

Les éléments suivants ont été vérifiés :

- la régression logistique réentraînée retrouve les chiffres du Module 2 à moins de 0,05 point ;
- les colonnes de fuite `newbalanceOrig`, `newbalanceDest` et `isFlaggedFraud` ne figurent pas dans les features ;
- le découpage temporel respecte celui du Module 2 ;
- le jeu de test a été consulté une seule fois ;
- la sélection du modèle a reposé uniquement sur la validation croisée ;
- les tests pytest ont été exécutés ;
- l'API et le conteneur ont été testés avec des exemples provenant de la validation.

À relire et adapter par l'étudiant avant dépôt
