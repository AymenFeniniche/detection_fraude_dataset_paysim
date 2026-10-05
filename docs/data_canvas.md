# Data Canvas — Responsable des données

> **Problème métier :** détecter les transactions frauduleuses avant leur exécution.  
> **Cible :** `isFraud` · **Unité d’analyse :** une transaction.  
> **Moment de prédiction :** à l’initiation, avant exécution.  
> **Critère de succès :** détecter une part importante des fraudes en limitant les faux positifs. Aucun seuil chiffré fixé à ce stade.

| Sources & qualité | Variables & risques | Gouvernance & transmission |
| :--- | :--- | :--- |
| **PaySim** — Source principale imposée. Données synthétiques. **1 ligne = 1 transaction.** | **Variables exclues — fuite de données :** `newbalanceOrig` et `newbalanceDest`, soldes après transaction. | **RGPD / conformité :** PaySim est synthétique. D’après la documentation disponible, ses identifiants ne correspondent pas à des personnes physiques réelles. |
| **OSMP 2025 — Banque de France** — Source complémentaire retenue. Statistiques réelles agrégées pour contextualiser PaySim. **Aucune jointure transactionnelle.** | **Variable exclue — sortie de règle :** `isFlaggedFraud`, résultat d’une règle existante. | **Contexte réel :** identifiants de comptes et historiques de transactions nécessiteraient des protections adaptées. |
| **Contrôles qualité :** structure · types · valeurs manquantes · doublons · distribution de `isFraud` · cohérence générale. | **Variables non recommandées brutes :** `nameOrig`, `nameDest`. Identifiants à forte cardinalité ; risque de surapprentissage. | **Accès :** transactions bancaires individuelles rarement publiques ; l’OSMP apporte principalement un contexte statistique agrégé. |
| — | **Risques :** data leakage · classes très déséquilibrées · faux positifs · données synthétiques · généralisation limitée au réel. | **Lineage :** source · URL · date de récupération · version lorsque connue · SHA-256 · transformations · statut. Détail : [journal_lineage.csv](journal_lineage.csv). |
| — | **Limites :** PaySim entièrement synthétique ; aucune performance en production réelle démontrée. OSMP contextualise, sans entraîner directement le modèle transactionnel. | **Transmission M2 :** cible `isFraud` · unité transaction · prédiction avant exécution · exclusions pour fuite de données · identifiants non recommandés bruts · fort déséquilibre · données synthétiques · protocole reproductible. |

**Responsable des données :** sourcing · qualité · dictionnaire · disponibilité des variables · data leakage · conformité / RGPD · lineage · limites · transmission à l’équipe.

*Documentation détaillée : [02_documentation_data.ipynb](02_documentation_data.ipynb).*
