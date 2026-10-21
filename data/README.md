# Jeux de données

| Dataset | Tâche | Séances | Versionné |
|---|---|---|---|
| [California Housing](california-housing/README.md) | Régression (prix médian de logement) | S1, S2, S4, S6, projet final | oui (`housing.csv`) |
| [Titanic](titanic/README.md) | Classification (survie) | S3, S4, S5, S6 | oui (`titanic.csv`) |
| [Bank Marketing](bank-marketing/README.md) | Classification (souscription à un dépôt à terme) | S5 | non, téléchargé par le notebook |

Chaque dossier contient un README avec la source, la licence et la description des colonnes.

## Principes

- Les fichiers de petite taille et librement redistribuables sont versionnés pour permettre un travail hors-ligne.
- Les autres sont téléchargés depuis leur source officielle par les notebooks, afin de garder la traçabilité de leur provenance.
- Le projet final porte sur California Housing et impose des contraintes qui évitent de reprendre une solution publique existante (voir [`projet-final/`](../projet-final/README.md)).
