# Jeux de données

| Dataset | Tâche | Séances | Versionné |
|---|---|---|---|
| [California Housing](california-housing/README.md) | Régression (prix médian de logement) | S1, S2, S4 | oui (`housing.csv`) |
| [Titanic](titanic/README.md) | Classification (survie) | S3, S4 | oui (`titanic.csv`) |
| [Bank Marketing](bank-marketing/README.md) | Classification (souscription à un dépôt à terme) | S5, S6 | non, téléchargé par le notebook |
| Wine | Classification à 3 classes (origine d'un vin) | S7 | non, fourni avec scikit-learn (`load_wine`) |

Chaque dossier contient un README avec la source, la licence et la description des colonnes (Wine n'a pas de dossier : il est chargé directement par le notebook du TP).

## Principes

- Les fichiers de petite taille et librement redistribuables sont versionnés pour permettre un travail hors-ligne.
- Les autres sont téléchargés depuis leur source officielle par les notebooks, afin de garder la traçabilité de leur provenance.
- Les jeux du projet final (finance, médecine, santé vétérinaire) ne sont pas redistribués : chaque participant les télécharge depuis la source officielle (voir le [guide d'accès](../Projet%20Final/03_Dataset_Access_Guide.md)).
