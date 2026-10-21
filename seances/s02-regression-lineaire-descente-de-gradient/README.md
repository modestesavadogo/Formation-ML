# Régression linéaire & descente de gradient

> **Séance 2 · mercredi 30 septembre 2026** — Statut : dispensée

Première séance construite sur la structure en trois briques de tout algorithme d'apprentissage : ① le modèle, ② la fonction de coût, ③ l'optimisation.

## Objectifs

- Reformuler un algorithme d'apprentissage comme modèle, fonction de coût et optimisation
- Écrire le modèle linéaire et lire un coefficient de corrélation avec ses conditions de validité
- Définir l'erreur quadratique moyenne (MSE), la comparer à d'autres coûts (MAE) et mesurer l'effet d'un outlier
- Résoudre le problème par une solution directe : la statistique pour une feature, les équations normales pour plusieurs
- Comprendre la descente de gradient : pente, gradient, règle de mise à jour
- Observer l'effet du learning rate sur la convergence
- Passer à plusieurs features et retrouver le résultat avec scikit-learn

## Déroulé

| Partie | Contenu |
|---|---|
| Rappel S1 | La structure ①②③ commune à tous les algorithmes |
| ① Le modèle | Droite à deux paramètres, corrélation et ses limites |
| ② La fonction de coût | MSE, propriétés d'une bonne fonction de coût, MSE vs MAE |
| ③ Optimisation, solution directe | Pente nulle au minimum, équations normales, intuition géométrique |
| ③ Optimisation, descente de gradient | Dérivée, gradient, règle de mise à jour, learning rate |
| Généralisation | De 1 à p features, utilisation de scikit-learn |
| Débrief | Ce qu'on retient |

## Jeux de données

- **California Housing** — régression du prix médian des logements

## Supports

- [`notebook_pratique_regression.ipynb`](notebook_pratique_regression.ipynb) — notebook à trous
- [`corrige_regression.ipynb`](corrige_regression.ipynb) — corrigé
- [`slides/s02-slides.md`](slides/s02-slides.md) · [`slides/s02-slides.pdf`](slides/s02-slides.pdf)

## Séance précédente et suivante

- Précédente : [S1 — Introduction au Machine Learning](../s01-introduction-au-machine-learning/)
- Suivante : [S3 — Classification & régression logistique](../s03-classification-regression-logistique/)
