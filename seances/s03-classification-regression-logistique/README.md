# Classification & Régression Logistique

**Statut : contenu prêt (notebooks + slides).**

## Objectifs

- Comprendre pourquoi la régression linéaire ne convient pas à un problème de classification
- Construire la régression logistique : sigmoïde, log-loss, descente de gradient
- Comprendre pourquoi le log-loss est préféré à la MSE pour ce type de problème (gradient qui ne s'annule pas)
- Visualiser et interpréter une frontière de décision
- Prendre conscience de l'existence du multiclasse (softmax), hors scope de la séance

## Plan de séance (2h30 / 150 min)

| Phase | Durée | Contenu |
|---|---|---|
| Rappel & activation | 10 min | Rappel structure ①②③, transition vers un problème à target binaire |
| Pourquoi une droite ne suffit plus | 10 min | Démonstration : une droite peut prédire hors de [0,1] |
| ① Le modèle — sigmoïde | 20 min | De la cote (odds) au logit, dérivation de la sigmoïde, propriétés et dérivée |
| Interpréter les coefficients | 10 min | Odds ratio ($e^w$), lecture sur un vrai coefficient entraîné (Fare) |
| ② La fonction de coût — log-loss | 25 min | Maximum de vraisemblance, pourquoi pas la MSE (gradient qui s'annule), calcul à la main |
| Pause | 5 min | — |
| ③ L'optimisation | 15 min | Gradient du log-loss par la règle de la chaîne, mise à jour (identique en forme à S2) |
| Démonstration — frontière de décision | 10 min | Visualisation 2D sur Age/Fare |
| Vers le multiclasse (teaser) | 5 min | Softmax, hors scope détaillé |
| TP guidé | 25 min | Implémentation NumPy, scikit-learn, métriques, frontière |
| Débrief | 10 min | Tableau récapitulatif ①②③ régression vs classification |
| Évaluation rapide | 5 min | Questions Mentimeter |

## Dataset(s) utilisé(s)

- **Titanic** — classification binaire (Survived), première utilisation dans le module

## Notebooks

- `notebook_pratique_classification.ipynb` — à distribuer avant/pendant la séance
- `corrige_classification.ipynb` — à publier après la séance

## Slides

- `slides/s03-slides.md` — 37 slides Marp, 8 figures générées à partir de vraies données Titanic
