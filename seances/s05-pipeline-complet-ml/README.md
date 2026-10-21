# Pipeline complet de Machine Learning

> **Séance 5 · samedi 17 octobre 2026** — Statut : dispensée

Séance de synthèse : on assemble tout ce qui a été vu en S1-S4 (cadrage, données, validation, prétraitement sans fuite) dans une démarche de bout en bout, de la question métier à un modèle évalué honnêtement.

## Objectifs

- Cadrer un problème ML : unité observée, cible, variables disponibles au moment de la décision
- Auditer la qualité des données (schéma, doublons, valeurs manquantes, modalités rares) et documenter leur provenance
- Identifier et prévenir les fuites de données dès la définition du scénario de prédiction
- Conduire une EDA utile à la modélisation plutôt qu'une collection de graphiques
- Choisir une séparation train / test adaptée à la structure temporelle des données
- Construire un prétraitement robuste avec `ColumnTransformer` et `Pipeline`
- Comparer une baseline, un modèle linéaire et des modèles à base d'arbres par validation croisée
- Régler les hyperparamètres (`GridSearchCV`, `RandomizedSearchCV`) sans toucher au jeu de test
- Choisir ses métriques avant de voir les résultats, et comprendre pourquoi l'accuracy seule est insuffisante face à une classe minoritaire
- Évaluer une seule fois sur le test, régler le seuil de décision, analyser les erreurs, interpréter le modèle avec un regard critique

## Déroulé

| Phase | Contenu | Support |
|---|---|---|
| Cadrage | Question métier, formulation ML, moment de prédiction, portée des données | Slides 1-4, notebook §1-2 |
| Données | Acquisition traçable, audit qualité, compréhension des variables | Slide 5, notebook §2-4 |
| Scénario & fuite | Variables indisponibles avant l'appel, séparation train / test | Slides 7 et 13, notebook §5 et 7 |
| EDA | Distributions, catégories rares, taux de souscription, corrélations | Slide 6, notebook §6 |
| Prétraitement | Imputation, encodage, standardisation, `ColumnTransformer` | Slide 8, notebook §8 |
| Modèles | Baseline, régression logistique, arbre, forêt aléatoire, gradient boosting | Slide 9, notebook §9-11 |
| Réglage | Validation croisée, recherche d'hyperparamètres | Slide 10, notebook §12 |
| Évaluation | Métriques, évaluation finale, seuil, analyse d'erreurs, interprétation | Slides 11-12, notebook §10 et 13-17 |
| Mise en pratique | TP de classification de survie sur Titanic | Slide 14 |
| Synthèse | Erreurs méthodologiques à éviter, points à retenir | Slides 13 et 15, notebook §18-19 |

## Jeux de données

- **Bank Marketing** (UCI) — fil conducteur du notebook de référence : prédire la souscription à un dépôt à terme. Téléchargé automatiquement par le notebook, voir [`data/bank-marketing/`](../../data/bank-marketing/README.md)
- **Titanic** — travail pratique de fin de séance (slide 14)

## Supports

- [`notebook_pipeline_complet.ipynb`](notebook_pipeline_complet.ipynb) — notebook de référence et support pratique (19 sections, exercices d'approfondissement en §18). Il télécharge les données depuis UCI : une connexion internet est nécessaire à la première exécution
- [`slides/s05-slides.pptx`](slides/s05-slides.pptx) — 15 slides

[![Ouvrir dans Colab](https://img.shields.io/badge/Colab-Ouvrir%20le%20notebook-1a5f3f?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/modestesavadogo/Formation-ML/blob/main/seances/s05-pipeline-complet-ml/notebook_pipeline_complet.ipynb)

## Séance précédente et suivante

- Précédente : [S4 — Généralisation, validation & prétraitement](../s04-generalisation-validation-pretraitement/)
- Suivante : [S6 — Modèles à base d'arbres & réduction de dimension](../s06-arbres-reduction-de-dimension/)
