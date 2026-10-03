# Généralisation, Validation & Prétraitement

**Statut : contenu prêt (slides ; notebooks en cours).**

## Objectifs

- Comprendre pourquoi l'erreur d'entraînement est un indicateur trompeur, et distinguer sous-apprentissage / sur-apprentissage
- Dériver (pas seulement énoncer) la décomposition biais-variance de l'erreur de généralisation
- Comprendre pourquoi un split train/test unique donne une estimation instable, et comment la validation croisée (K-fold) la stabilise
- Construire la régularisation Ridge (L2) et Lasso (L1) : dérivation de la solution, intuition géométrique, sélection de λ par validation croisée
- Nettoyer et préparer des données réelles : valeurs manquantes, encodage catégoriel, mise à l'échelle
- Comprendre le mécanisme de la fuite de données (data leakage) et pourquoi elle gonfle artificiellement la performance mesurée
- Assembler prétraitement + modèle dans un `Pipeline` scikit-learn pour rendre la discipline anti-fuite automatique

## Plan de séance (2h30 / 150 min)

| Phase | Durée | Contenu |
|---|---|---|
| Rappel & activation | 10 min | Rappel ①②③, remise en question de l'erreur d'entraînement comme objectif |
| **Partie A.1** — Sous/sur-apprentissage | 15 min | Démonstration polynômes sur Latitude, courbe erreur train/test vs complexité |
| **Partie A.2** — Décomposition biais-variance | 20 min | Dérivation complète (ajout/retrait de $\mathbb{E}_D[\hat f(x)]$), visualisation par rééchantillonnage |
| **Partie A.3** — Validation croisée | 15 min | Instabilité du split unique, mécanique du K-fold, gain de stabilité mesuré |
| Pause | 5 min | — |
| **Partie A.4** — Régularisation | 25 min | Ridge/Lasso : fonction de coût, équations normales régularisées, intuition géométrique L1/L2, chemins de régularisation |
| **Partie A.5** — Choisir λ | 10 min | Validation croisée à deux niveaux, `GridSearchCV`, hiérarchie train/validation/test |
| **Partie B.6** — Prétraitement | 15 min | Valeurs manquantes (Titanic), encodage One-Hot vs ordinal |
| **Partie B.7** — Mise à l'échelle | 10 min | Effet sur la descente de gradient (courbes de niveau) et sur l'équité de la régularisation |
| **Partie B.8** — Fuite de données | 10 min | Démonstration : encodage de `Ticket` avec/sans fuite (94.2% vs 70.1% vs baseline 61.6%) |
| **Partie B.9** — Pipelines | 10 min | `ColumnTransformer` + `Pipeline`, pourquoi c'est la bonne pratique |
| TP guidé (démarré en fin de séance, terminé en autonomie) | — | Pipeline complet Titanic + sélection de λ par `GridSearchCV` |
| Débrief & évaluation rapide | 5 min | Synthèse ①②③ augmentée, questions |

## Dataset(s) utilisé(s)

- **California Housing** — sous/sur-apprentissage, biais-variance, régularisation (Ridge/Lasso), choix de λ
- **Titanic** — valeurs manquantes, encodage, mise à l'échelle, fuite de données, Pipelines

## Notebooks

- `notebook_pratique_generalisation.ipynb` — à distribuer avant/pendant la séance
- `corrige_generalisation.ipynb` — à publier après la séance

## Slides

- `slides/s04-slides.md` — 64 slides Marp, 15 figures générées à partir de vraies données (California Housing, Titanic)
