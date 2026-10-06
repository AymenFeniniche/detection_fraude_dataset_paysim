# Data Canvas

**Sujet :** Détecter la fraude aux paiements mobiles (Sujet E)

**Membres :** FENNINICHE Aymen · FERSAOUI Selena · GHEDDAB Yasmine · MEDFAI Kalil · TCHAMAGAN Glenn · SYLLA Mariam

## 1. Besoin métier

Un opérateur de paiement mobile veut réduire les pertes liées aux transactions frauduleuses en les détectant avant leur exécution.

L’enjeu est de limiter le montant fraudé sans bloquer inutilement des transactions légitimes. Un excès de faux positifs pénaliserait les clients en dégradant leur expérience de paiement et augmenterait la charge des équipes internes, qui devraient traiter davantage d’alertes et de contestations.

PaySim sert de jeu transactionnel d’étude ; l’OSMP 2025 sert uniquement à replacer le projet dans un contexte réel de fraude en France.

## 2. Décision à améliorer

Au moment où une transaction est initiée, le système antifraude doit aider l’opérateur à décider de l’autoriser ou de la bloquer. La décision doit être prise avant exécution et uniquement avec les informations disponibles à cet instant.

## 3. Question data

- **Question prédictive :** cette transaction est-elle frauduleuse au moment où elle est initiée ?
- **Question explicative :** quelles caractéristiques disponibles avant exécution distinguent les transactions frauduleuses des transactions légitimes ?
- **Type de problème :** classification supervisée binaire.

## 4. Variable cible

**Cible :** `isFraud`. **Valeurs :** 1 = frauduleuse ; 0 = légitime.

Dans le fichier contrôlé : **8 213 fraudes** sur **6 362 620 transactions** (**0,1291 %**) et **6 354 407 transactions légitimes**. Les classes sont donc très déséquilibrées.

## 5. Unité d’analyse et granularité

**Une ligne = une transaction.**

Volume : **6 362 620 lignes et 11 colonnes brutes**.

`nameOrig` et `nameDest` sont des identifiants de comptes et non des identifiants uniques de transaction.

## 6. Moment de la prédiction

**Prédiction à l’initiation, avant exécution.**

Variables disponibles dans PaySim avant exécution : `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `nameDest`, `oldbalanceDest`. Leur disponibilité exacte dans un système réel reste à confirmer.

Variables non disponibles avant exécution : `newbalanceOrig` et `newbalanceDest`. `isFlaggedFraud` est la sortie d’une règle existante et ne doit pas être utilisée comme variable explicative. Ces trois variables sont exclues.

## 7. Variables explicatives candidates

| Variable | Type | Connue au moment de la décision ? | Hypothèse sur le lien avec la cible |
| --- | --- | --- | --- |
| `step` | Quantitative discrète | Oui dans PaySim ; à valider en réel | Effet temporel possible sur le risque de fraude. |
| `type` | Qualitative nominale | Oui | Dans PaySim, les fraudes observées concernent `TRANSFER` et `CASH_OUT`. |
| `amount` | Quantitative continue | Oui | Distribution et valeurs extrêmes à analyser entre les classes. |
| `oldbalanceOrg` | Quantitative continue | A priori oui ; à vérifier | La relation entre montant et solde avant transaction peut être informative. |
| `oldbalanceDest` | Quantitative continue | A priori oui ; à vérifier | Le solde du destinataire avant transaction peut apporter un signal. |
| `nameOrig` | Qualitative nominale / identifiant | Oui | Forte cardinalité : risque de surapprentissage. |
| `nameDest` | Qualitative nominale / identifiant | Oui | Forte cardinalité : risque de surapprentissage. |

## 8. Sources de données

| Source | Interne / externe | Mode d’accès | Granularité | Période couverte | Licence / réutilisation |
| --- | --- | --- | --- | --- | --- |
| PaySim — source principale | Externe | Kaggle ; CSV via KaggleHub | 1 ligne = 1 transaction | `step` 1 à 743, soit environ 31 jours simulés | CC BY-SA 4.0 |
| OSMP 2025 — Banque de France | Externe institutionnelle | Site Banque de France ; PDF + XLSX | Statistiques agrégées nationales | Activité 2025 ; publication 09/09/2026 | Conditions de réutilisation Banque de France, sous réserve des mentions spécifiques et droits de tiers |

## 9. Évaluation des sources (check-list)

### PaySim

- **Pertinence :** élevée, car le jeu contient les transactions et la cible `isFraud`.
- **Couverture :** 6,36 M de transactions, 11 variables et 5 types de transactions.
- **Fraîcheur :** limitée, car il s’agit d’un jeu de simulation statique.
- **Granularité :** une transaction par ligne.
- **Qualité :** aucune valeur manquante ni ligne dupliquée détectée dans les contrôles du projet.
- **Biais / limites :** données entièrement synthétiques, donc les comportements réels peuvent être imparfaitement représentés.
- **Licence :** CC BY-SA 4.0.

### OSMP 2025 — Banque de France

- **Pertinence :** forte pour contextualiser la fraude réelle en France.
- **Couverture :** statistiques nationales portant sur l’activité 2025.
- **Fraîcheur :** rapport publié le 09/09/2026.
- **Granularité :** agrégats par indicateur, moyen de paiement et période ; aucune jointure ligne à ligne avec PaySim.
- **Qualité :** source institutionnelle documentée.
- **Biais / limites :** périmètre français et définitions différentes de celles du simulateur.
- **Licence / réutilisation :** conditions générales de réutilisation de la Banque de France, sous réserve des mentions spécifiques et droits de tiers.

## 10. Critère de succès

**Indicateur métier :** détecter une part importante des fraudes tout en limitant les faux positifs.

Seuil à valider avec le métier ; objectif provisoire : réduire fortement le montant fraudé tout en maintenant les faux positifs sous 1 %.

- **Faux positif :** transaction légitime bloquée à tort, avec risque de friction client et de perte commerciale.
- **Faux négatif :** transaction frauduleuse autorisée, avec perte financière potentielle et risque opérationnel.

## 11. Parties prenantes

**Utilisateurs :** équipe de détection de fraude et opérateur de paiement mobile.

**Parties affectées :** clients dont les transactions peuvent être autorisées ou bloquées, ainsi que l’opérateur exposé aux pertes de fraude et aux faux positifs.

## 12. Contraintes éthiques, juridiques et techniques

**Données personnelles / RGPD :** PaySim est synthétique et, d’après la documentation du projet, ses identifiants ne correspondent pas à des personnes physiques réelles. L’OSMP est agrégé. En production réelle, les identifiants de comptes et historiques transactionnels seraient des données personnelles à minimiser, sécuriser et, si nécessaire, pseudonymiser. La base légale exacte n’est pas documentée dans le projet.

**Risques de discrimination :** aucune variable sensible explicite n’est documentée dans PaySim. Cependant, un risque de discrimination peut apparaître si certains groupes d’utilisateurs sont traités moins favorablement par le modèle. Ce risque reste limité dans le jeu synthétique actuel mais devra être évalué sur des données réelles.

**Contraintes techniques :** 6,36 M de lignes et classes très déséquilibrées ; utiliser un échantillon stratifié pendant l’exploration. Le téléchargement Kaggle doit rester reproductible.

**Contraintes éthiques :** le modèle doit limiter les faux positifs afin de ne pas bloquer injustement des transactions légitimes et de ne pas dégrader l’expérience des clients. Les décisions du modèle doivent rester compréhensibles et contrôlables par les équipes internes. Le modèle devra être validé sur des données réelles avant tout usage en production.

## 13. Risques de fuite de données et limites

**Fuites de données :** `newbalanceOrig` et `newbalanceDest` sont des informations post-transaction ; `isFlaggedFraud` est le résultat d’une règle de détection existante.

**Limites :** PaySim est entièrement synthétique, la fraude est très rare (0,1291 %), `nameOrig` et `nameDest` ont une forte cardinalité, et de bonnes performances sur PaySim ne garantissent pas les mêmes performances en production. L’OSMP sert uniquement à contextualiser et ne peut pas être joint transaction par transaction avec PaySim.

## 14. Démarche prévue

Extraction reproductible de PaySim → contrôles qualité (structure, types, valeurs manquantes, doublons, cible) → création d’un jeu de modélisation sans variables de fuite → EDA avec échantillon stratifié → modèle de référence (régression logistique) → modèles avancés et suivi MLflow → API de prédiction dockerisée.

Le lineage, les empreintes SHA-256 et la documentation des transformations sont conservés tout au long du projet.

Les détails complémentaires et les analyses associées sont disponibles dans le dossier du projet (notebooks, dictionnaire des variables, sources et lineage).
