<div align="center">

# Formation Machine Learning

**De la régression linéaire au pipeline complet, en 9 séances pratiques avec Python et scikit-learn**

[![Python](https://img.shields.io/badge/Python-3.12%2B-1a5f3f?logo=python&logoColor=white)](requirements.txt)
[![Licence MIT](https://img.shields.io/badge/Licence-MIT-1a5f3f)](LICENSE)
[![Séances](https://img.shields.io/badge/Séances%20dispensées-5%2F9-1a5f3f)](#calendrier-et-supports)
[![Colab](https://img.shields.io/badge/Colab-Ouvrir%20un%20notebook-1a5f3f?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/modestesavadogo/Formation-ML/blob/main/seances/s01-introduction-au-machine-learning/notebook_pratique_introduction.ipynb)

[Objectifs](#objectifs-de-la-formation) · [Calendrier](#calendrier-et-supports) · [Fonctionnement](#fonctionnement-dune-séance) · [Démarrer](#démarrer) · [Données](#jeux-de-données) · [Projet final](#projet-final-et-évaluation) · [Structure](#structure-du-dépôt)

</div>

---

## À propos

Formation de Machine Learning en 9 séances de 2h30, du 26 septembre au 14 novembre 2026. Elle prolonge la Formation Python 360° : on part des fondations (qu'est-ce qu'un modèle, comment il apprend) pour arriver à une démarche complète de projet, de la question métier à l'évaluation honnête d'un modèle.

Ce dépôt réunit tout le matériel : slides, notebooks à trous, corrigés publiés après chaque séance, jeux de données et consignes du projet final.

**Prérequis :** bases de Python (listes, dictionnaires, fonctions) et une première exposition à NumPy et pandas.

## Objectifs de la formation

À l'issue des 9 séances, vous saurez :

- poser un problème de Machine Learning : cible, variables, moment de la prédiction ;
- entraîner et interpréter des modèles de régression et de classification ;
- comprendre comment un modèle apprend (fonction de coût, descente de gradient) plutôt que l'utiliser comme une boîte noire ;
- estimer honnêtement la performance : séparation train / test, validation croisée, métriques adaptées ;
- détecter et éviter la fuite de données grâce aux `Pipeline` scikit-learn ;
- comparer plusieurs modèles, régler leurs hyperparamètres et analyser leurs erreurs ;
- défendre vos choix techniques dans un projet de bout en bout.

## Calendrier et supports

Les séances ont lieu le samedi, à l'exception de la séance 2 (mercredi). Les supports sont publiés au fil de la formation.

| # | Date | Séance | Statut | Supports |
|---|------|--------|--------|----------|
| 1 | sam. 26 sept. | [Introduction au Machine Learning](seances/s01-introduction-au-machine-learning/) | Dispensée | [Slides](seances/s01-introduction-au-machine-learning/slides/s01-slides.pdf) · [Notebook](seances/s01-introduction-au-machine-learning/notebook_pratique_introduction.ipynb) · [Corrigé](seances/s01-introduction-au-machine-learning/corrige_introduction.ipynb) |
| 2 | mer. 30 sept. | [Régression linéaire & descente de gradient](seances/s02-regression-lineaire-descente-de-gradient/) | Dispensée | [Slides](seances/s02-regression-lineaire-descente-de-gradient/slides/s02-slides.pdf) · [Notebook](seances/s02-regression-lineaire-descente-de-gradient/notebook_pratique_regression.ipynb) · [Corrigé](seances/s02-regression-lineaire-descente-de-gradient/corrige_regression.ipynb) |
| 3 | sam. 3 oct. | [Classification & régression logistique](seances/s03-classification-regression-logistique/) | Dispensée | [Slides](seances/s03-classification-regression-logistique/slides/s03-slides.pdf) · [Notebook](seances/s03-classification-regression-logistique/notebook_pratique_classification.ipynb) |
| 4 | sam. 10 oct. | [Généralisation, validation & prétraitement](seances/s04-generalisation-validation-pretraitement/) | Dispensée | [Slides](seances/s04-generalisation-validation-pretraitement/slides/s04-slides.pdf) · [Notebook](seances/s04-generalisation-validation-pretraitement/notebook_pratique_generalisation.ipynb) · [Corrigé](seances/s04-generalisation-validation-pretraitement/corrige_generalisation.ipynb) |
| 5 | sam. 17 oct. | [Pipeline complet de Machine Learning](seances/s05-pipeline-complet-ml/) | Dispensée | [Slides](seances/s05-pipeline-complet-ml/slides/s05-slides.pptx) · [Notebook](seances/s05-pipeline-complet-ml/notebook_pipeline_complet.ipynb) |
| 6 | sam. 24 oct. | [Modèles à base d'arbres & réduction de dimension](seances/s06-arbres-reduction-de-dimension/) | Prochaine séance | À venir |
| 7 | sam. 31 oct. | [Projet guidé (1/2)](seances/s07-projet-guide-1/) | À venir | À venir |
| 8 | sam. 7 nov. | [Projet guidé (2/2)](seances/s08-projet-guide-2/) | À venir | À venir |
| 9 | sam. 14 nov. | [Évaluation & restitution des projets](seances/s09-evaluation-restitution/) | À venir | [Grille d'évaluation](projet-final/grille-evaluation.md) |

### Parcours pédagogique

| Séance | Ce que l'on apprend |
|---|---|
| S1 | Vocabulaire (feature, cible, paramètres), programmation classique vs apprentissage, premier contact avec les données |
| S2 | Le modèle linéaire, la fonction de coût (MSE), les équations normales, la descente de gradient, la corrélation |
| S3 | Sigmoïde, log-loss, frontière de décision, métriques de classification |
| S4 | Sur-apprentissage et biais-variance, validation croisée, régularisation Ridge/Lasso, prétraitement, fuite de données, `Pipeline` |
| S5 | Démarche de bout en bout : cadrage, audit, EDA, prévention des fuites, comparaison de modèles, réglage, évaluation finale, seuil de décision, analyse d'erreurs |
| S6 | Arbres de décision, forêts aléatoires, PCA |
| S7-S8 | Projet de groupe encadré sur California Housing |
| S9 | Soutenance des projets |

## Fonctionnement d'une séance

Chaque dossier de `seances/` suit la même organisation :

- un `README.md` avec les objectifs, le déroulé et les supports ;
- des **slides** pour la partie théorique ;
- un **notebook pratique** à compléter pendant la séance ;
- un **corrigé**, publié après la séance.

Conseil de travail : rejouez le notebook de la séance précédente avant de commencer la suivante, chaque séance s'appuie sur la précédente.

## Démarrer

### Sans rien installer (Google Colab)

Ouvrez un notebook directement depuis GitHub avec ce modèle de lien, en remplaçant le chemin :

```
https://colab.research.google.com/github/modestesavadogo/Formation-ML/blob/main/<chemin du notebook>
```

Exemple pour la séance 5 :

[![Ouvrir dans Colab](https://img.shields.io/badge/Colab-Séance%205-1a5f3f?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/modestesavadogo/Formation-ML/blob/main/seances/s05-pipeline-complet-ml/notebook_pipeline_complet.ipynb)

Premier réflexe une fois le notebook ouvert : **Fichier ▸ Enregistrer une copie dans Drive**, sans quoi vos modifications seront perdues.

### En local

```bash
git clone https://github.com/modestesavadogo/Formation-ML.git
cd Formation-ML
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## Jeux de données

| Dataset | Tâche | Séances |
|---|---|---|
| **California Housing** | Régression (prix médian de logement) | S1, S2, S4, S6, projet final |
| **Titanic** | Classification (survie) | S3, S4, S5, S6 |
| **Bank Marketing** (UCI) | Classification (souscription à un dépôt à terme) | S5 |

California Housing et Titanic servent de fil rouge : on les retrouve d'une séance à l'autre pour se concentrer sur les concepts plutôt que sur un nouveau contexte. Sources, licences et colonnes : [`data/`](data/README.md).

## Projet final et évaluation

Le projet final est un travail de groupe sur California Housing, préparé en séances 7 et 8 et présenté en séance 9. Il impose du feature engineering, la comparaison d'au moins trois modèles, une justification écrite des choix et une analyse d'erreur.

- Consignes : [`projet-final/README.md`](projet-final/README.md)
- Barème : [`projet-final/grille-evaluation.md`](projet-final/grille-evaluation.md)

## Structure du dépôt

```
Formation-ML/
├── seances/          un dossier par séance : README, slides, notebooks
├── data/             jeux de données et leur documentation
├── projet-final/     consignes et grille d'évaluation
├── portail/          page d'accueil publique (GitHub Pages)
├── requirements.txt  dépendances Python
└── LICENSE
```

Arborescence détaillée : [`ARBORESCENCE.txt`](ARBORESCENCE.txt).

## Stack

| Outil | Version |
|---|---|
| Python | 3.12+ |
| pandas | 3.0+ |
| NumPy | 2.5+ |
| scikit-learn | 1.5+ |
| matplotlib | 3.10+ |
| seaborn | 0.13+ |

## Licence

Code et supports distribués sous licence MIT — voir [`LICENSE`](LICENSE). Les jeux de données conservent la licence de leur source, indiquée dans leur README.
