# Titanic

Dataset de classification binaire — prédiction de la survie des passagers.

## Source

Dataset public, disponible entre autres via l'entrepôt d'exemples
[OpenML](https://www.openml.org/search?type=data&status=active&search=titanic)
ou la compétition Kaggle "Titanic - Machine Learning from Disaster"
(https://www.kaggle.com/competitions/titanic).

À télécharger et placer ici en `titanic.csv` avant la séance 3
(le fichier n'est pas versionné directement pour éviter tout souci de
redistribution — vérifier la licence de la source choisie).

## Colonnes principales

| Colonne | Description |
|---|---|
| `Pclass` | Classe du billet (1, 2, 3) |
| `Sex` | Sexe |
| `Age` | Âge (valeurs manquantes — utile pour l'imputation, S4) |
| `SibSp` | Nb frères/sœurs/époux à bord |
| `Parch` | Nb parents/enfants à bord |
| `Fare` | Tarif payé |
| `Embarked` | Port d'embarquement |
| `Survived` | **Cible** — 0 = non, 1 = oui |

## Utilisation dans le module

- S3 : classification, régression logistique
- S4 : imputation des valeurs manquantes (`Age`), encodage catégoriel
- S5 : arbres de décision, Random Forest (Gini/entropy plus lisibles ici qu'en régression)

## Attention pédagogique

Dataset très connu — solutions disponibles partout en ligne. Utilisé
uniquement en démo/TP guidé, **pas** comme support du projet final noté
(voir `projet-final/README.md`).
