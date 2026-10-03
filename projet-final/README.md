# Projet final — California Housing

Le fil rouge du module (California Housing, vu de S1 à S5) devient ici un
projet complet de bout en bout, réalisé en groupe sur les séances 6 et 7,
présenté en séance 8.

## Objectif

Construire un pipeline complet de prédiction du prix médian des logements,
en justifiant chaque décision (préprocessing, choix de modèle,
hyperparamètres, métriques).

## Contraintes obligatoires (anti-copiage)

Un notebook Kaggle générique sur California Housing ne suffit pas — les
éléments suivants sont **imposés** et doivent apparaître explicitement :

1. **Feature engineering spécifique** : au moins 2 features dérivées
   non présentes dans le dataset brut (ex : ratio pièces/occupants,
   distance à la côte calculée depuis lat/long, cluster géographique
   issu du K-Means de la S5 utilisé comme feature catégorielle).
2. **Comparaison d'au moins 3 modèles** vus dans le module (ex :
   régression linéaire régularisée, Random Forest, Gradient Boosting),
   avec justification du choix final.
3. **Justification écrite** (markdown dans le notebook, pas un rapport
   séparé) : pourquoi ce préprocessing, pourquoi cette métrique,
   qu'est-ce qui a été essayé et rejeté.
4. **Analyse d'erreur** : sur quels districts le modèle se trompe le
   plus, et une hypothèse sur pourquoi.

Un notebook qui ne contient que du code sans ces 4 éléments explicites
sera noté en conséquence, indépendamment de la performance brute du modèle.

## Livrables

- Un notebook (`starter.ipynb` comme point de départ imposé)
- Une présentation de 5-7 min en séance 8

## Groupes

À définir — voir `grille-evaluation.md` pour la répartition des points
individuel vs. collectif.

## Deadline

Notebook final à déposer avant la séance 8 (samedi 10 octobre).
