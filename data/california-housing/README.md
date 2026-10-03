# California Housing

Dataset de régression — prédiction du prix médian des logements par district
en Californie (recensement 1990).

## Source

`sklearn.datasets.fetch_california_housing()` — dérivé du recensement US 1990.
Documentation : https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

Chargement direct (pas besoin de télécharger le CSV séparément) :

```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
df = housing.frame
```

Un export `housing.csv` est aussi versionné ici pour un accès hors-ligne /
cohérence avec le pattern reprise/corrigé des séances.

## Colonnes

| Colonne | Description |
|---|---|
| `MedInc` | Revenu médian du foyer dans le district (dizaines de milliers $) |
| `HouseAge` | Âge médian des logements du district |
| `AveRooms` | Nombre moyen de pièces par logement |
| `AveBedrms` | Nombre moyen de chambres par logement |
| `Population` | Population du district |
| `AveOccup` | Occupation moyenne par logement |
| `Latitude` | Latitude du district |
| `Longitude` | Longitude du district |
| `MedHouseVal` | **Cible** — valeur médiane du logement (centaines de milliers $) |

## Utilisation dans le module

- S1 : exploration initiale
- S2 : régression linéaire, gradient descent
- S4 : Ridge/Lasso, pipelines de preprocessing
- S5 : K-Means/PCA sur `Latitude`/`Longitude` (clustering géographique)
- Projet final : dataset unique du fil rouge

## Licence

Domaine public (données de recensement US).
