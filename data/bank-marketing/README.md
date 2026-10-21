# Bank Marketing

Dataset de classification binaire : prédire si un client souscrit à un dépôt à terme à l'issue d'une campagne de marketing téléphonique d'une banque portugaise (mai 2008 – novembre 2010).

## Source

UCI Machine Learning Repository — [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing).

Référence : S. Moro, P. Cortez, P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, 2014.

Licence : voir la page UCI du dataset (publié sous licence Creative Commons CC BY 4.0).

## Récupération

Le fichier n'est pas versionné dans ce dépôt. Le notebook de la [séance 5](../../seances/s05-pipeline-complet-ml/README.md) télécharge l'archive officielle et lit `bank-additional-full.csv` (41 188 observations, séparateur `;`).

Pour travailler hors-ligne, télécharger l'archive depuis la page UCI, extraire `bank-additional-full.csv` dans ce dossier et renseigner `LOCAL_CSV_PATH` dans le notebook.

## Cible

| Colonne | Description |
|---|---|
| `y` | **Cible** — `yes` si le client a souscrit un dépôt à terme, `no` sinon (classe positive : `yes`, minoritaire) |

## Points d'attention

- **Fuite de données :** certaines variables, comme la durée de l'appel (`duration`), ne sont connues qu'à l'issue du contact. Elles ne peuvent pas servir à prédire avant l'appel. C'est l'un des points de départ du notebook.
- **Classes déséquilibrées :** l'accuracy seule est trompeuse, d'où l'usage de métriques comme le recall, la précision, le F1 et l'aire sous la courbe PR.
- **Portée :** données réelles mais anciennes et propres à un contexte portugais, à ne pas généraliser aux comportements bancaires actuels.

## Utilisation dans le module

- S5 : cadrage, audit, prévention de la fuite, prétraitement, comparaison de modèles, évaluation finale
