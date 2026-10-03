<div align="center">

# formation-ml

**Formation Machine Learning — suite logique de la Formation Python 360°**

[![Python](https://img.shields.io/badge/Python-3.12%2B-1a5f3f?logo=python&logoColor=white)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/Licence-MIT-1a5f3f)](LICENSE)
[![Status](https://img.shields.io/badge/Statut-en%20cours-e67e22)](#progression)
[![Séances](https://img.shields.io/badge/Séances-2%2F8-1a5f3f)](#progression)
[![Open in Colab](https://img.shields.io/badge/Colab-Ouvrir%20un%20notebook-1a5f3f?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/modestesavadogo/Formation-ML/blob/main/seances/s01-introduction-au-machine-learning/notebook_pratique_introduction.ipynb)

8 séances, du 26 septembre au 21 octobre 2026 (mercredi/samedi, 2h30 chacune)

[Programme](#le-programme) · [Progression](#progression) · [Démarrer](#démarrer) · [Structure](#structure-du-dépôt) · [Données](#jeux-de-données) · [Stack](#stack)

</div>

<br>

<p align="center">
  <a href="https://www.python.org" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="Python" width="40" height="40"/></a>&nbsp;&nbsp;
  <a href="https://pandas.pydata.org" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/></a>&nbsp;&nbsp;
  <a href="https://numpy.org" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" alt="NumPy" width="40" height="40"/></a>&nbsp;&nbsp;
  <a href="https://scikit-learn.org" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/scikitlearn/scikitlearn-original.svg" alt="scikit-learn" width="40" height="40"/></a>&nbsp;&nbsp;
  <a href="https://matplotlib.org" target="_blank" rel="noreferrer"><img src="https://upload.wikimedia.org/wikipedia/commons/8/84/Matplotlib_icon.svg" alt="Matplotlib" width="40" height="40"/></a>&nbsp;&nbsp;
  <a href="https://jupyter.org" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/jupyter/jupyter-original.svg" alt="Jupyter" width="40" height="40"/></a>
</p>

---

## À propos

Ce dépôt contient l'intégralité du matériel d'une formation Machine Learning en 8 séances : slides, notebooks à trous, corrigés publiés après chaque séance, et le projet final guidé. **California Housing** (régression) et **Titanic** (classification) forment l'ossature du module ; d'autres jeux de données s'y ajoutent au fil des séances selon les besoins de chaque sujet — détails dans [Jeux de données](#jeux-de-données).

Prérequis : la Formation Python 360° (ou niveau équivalent) — bases de Python, listes/dicts, fonctions, une première exposition à NumPy/pandas.

**Portail :** https://modestesavadogo.github.io/Formation-ML/

---

## Progression

Le dépôt se remplit séance après séance, au fil de la formation — un dossier vide (hors README) signifie simplement que la séance n'a pas encore eu lieu.

| # | Séance | Date | Statut |
|---|--------|------|--------|
| 1 | Introduction au Machine Learning | sam. 26 sept. | ✅ Slides + notebooks |
| 2 | Régression Linéaire & Descente de Gradient | mer. 30 sept. | ✅ Slides + notebooks |
| 3 | Classification & Régression Logistique | sam. 3 oct. | ⬜ À venir |
| 4 | Généralisation, Validation & Prétraitement | mer. 7 oct. | ⬜ À venir |
| 5 | Modèles à Base d'Arbres & Réduction de Dimension | sam. 10 oct. | ⬜ À venir |
| 6 | Projet Guidé (1/2) | mer. 14 oct. | ⬜ À venir |
| 7 | Projet Guidé (2/2) | sam. 17 oct. | ⬜ À venir |
| 8 | Évaluation & Restitution des Projets | mer. 21 oct. | ⬜ À venir |

## Le programme

Le détail de chaque séance (objectifs, plan minute par minute) est dans le README de son dossier, sous [`seances/`](seances/).

| # | Séance | Contenu clé |
|---|--------|-------------|
| 1 | Introduction au ML | Vocabulaire (feature/target/paramètres), programmation classique vs. apprentissage, premier contact avec les données |
| 2 | Régression Linéaire & Descente de Gradient | Modèle, fonction de coût (MSE), équations normales, descente de gradient, corrélation |
| 3 | Classification & Régression Logistique | Sigmoïde, log-loss, frontière de décision, métriques de classification |
| 4 | Généralisation, Validation & Prétraitement | Sur/sous-apprentissage, validation croisée, standardisation, fuite de données |
| 5 | Modèles à Base d'Arbres & Réduction de Dimension | Arbres de décision, forêts aléatoires, PCA |
| 6-7 | Projet Guidé | Application sur un problème complet, en 2 temps |
| 8 | Évaluation & Restitution | Soutenance des projets |

---

## Démarrer

### Sans rien installer (Colab)

```
https://colab.research.google.com/github/modestesavadogo/Formation-ML/blob/main/seances/s01-introduction-au-machine-learning/notebook_pratique_introduction.ipynb
```

Premier réflexe une fois le notebook ouvert : **Fichier ▸ Enregistrer une copie dans Drive**.

### En local

```bash
git clone https://github.com/modestesavadogo/Formation-ML.git
cd Formation-ML
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Structure du dépôt

```
formation-ml/
├── portail/              site public, déployé sur GitHub Pages
├── seances/              un dossier par séance : slides, notebook à trous, corrigé
│   ├── s01-introduction-au-machine-learning/
│   ├── s02-regression-lineaire-descente-de-gradient/
│   └── ...
├── data/                 California Housing + Titanic, et d'autres à venir
├── projet-final/         consignes, grille d'évaluation, notebook de départ imposé
├── requirements.txt
└── LICENSE
```

Arborescence complète et commentée : [`ARBORESCENCE.txt`](ARBORESCENCE.txt).

---

## Jeux de données

Deux jeux de données servent d'**ossature** à la formation et reviennent sur plusieurs séances — volontairement, pour ancrer les concepts sur des données déjà familières plutôt que de changer de contexte à chaque fois.

| Dataset | Tâche | Utilisé en |
|---|---|---|
| **California Housing** | Régression (prix médian de logement) | S1, S2, S4, S5, projet final |
| **Titanic** | Classification (survie) | S3, S4, S5 |

Ce ne sont pas les seuls : d'autres jeux de données, choisis selon ce que chaque sujet demande, s'ajoutent déjà ou s'ajouteront au fil des séances restantes — typiquement un dataset plus riche pour la réduction de dimension (S5) et un jeu dédié pour le projet guidé (S6-S8). Ce tableau s'étoffe en conséquence à chaque mise à jour du dépôt.

Détails, sources et licences : README de [`data/`](data/).

---

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

MIT — voir [`LICENSE`](LICENSE).
