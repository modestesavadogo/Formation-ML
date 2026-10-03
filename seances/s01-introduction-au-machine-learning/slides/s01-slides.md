---
marp: true
theme: formation-ml
paginate: true
math: katex
---

<!-- _class: titre -->

# Introduction au Machine Learning

### Formation ML — Séance 1

Suite logique de la Formation Python 360°

<br>

**26 septembre 2026**

---

## Au programme aujourd'hui

1. Qu'est-ce que le Machine Learning ?
2. Le vocabulaire essentiel
3. Premier contact avec des données réelles
4. Prise en main de l'environnement
5. Travail pratique guidé

<div class="intuition">

**Intuition —** Objectif de la séance : repartir avec une intuition solide du <b>pourquoi</b> du ML, avant d'entrer dans le <b>comment</b> dès la séance 2.

</div>

---

## Le programme complet de la formation

| # | Séance | Date |
|---|---|---|
| 1 | Introduction au Machine Learning | sam. 26 sept. |
| 2 | Régression Linéaire & Descente de Gradient | mer. 30 sept. |
| 3 | Classification & Régression Logistique | sam. 3 oct. |
| 4 | Généralisation, Validation & Prétraitement | mer. 7 oct. |
| 5 | Modèles à base d'Arbres & Réduction de dimension | sam. 10 oct. |
| 6-7 | Projet guidé (en 2 temps) | mer. 14 oct. / sam. 17 oct. |
| 8 | Évaluation & restitution des projets | mer. 21 oct. |

*California Housing et Titanic nous accompagnent tout du long.*

---

<!-- _class: section -->

# 1. Qu'est-ce qu'une donnée ?

---

## D'une table à un problème de ML

| MedInc | HouseAge | AveRooms | ... | MedHouseVal |
|--------|----------|----------|-----|-------------|
| 8.3    | 41       | 6.98     | ... | 4.53        |
| 5.6    | 21       | 6.24     | ... | 3.59        |
| 3.8    | 52       | 5.82     | ... | 3.15        |

<br>

- Chaque **ligne** = une observation (un district californien)
- Chaque **colonne** = une **feature** — sauf une, la **target**
- La target est ce qu'on veut **prédire**

---

## L'espace des features

<div class="intuition">

Chaque observation peut se voir comme un point dans un espace à autant de dimensions que de features.

</div>

- 1 feature → une droite
- 2 features → un plan
- 8 features (California Housing) → un espace qu'on ne peut plus dessiner, mais dont le principe reste identique

<div class="retenir">

**À retenir —** On ne perd rien à raisonner en 2D pour comprendre — les mêmes idées se généralisent, juste sans schéma possible.

</div>

---

<!-- _class: section -->

# 2. Programmation classique vs. apprentissage

---

## Deux façons radicalement différentes de résoudre un problème

<div class="deux-colonnes">
<div>

**Programmation classique**

<div class="flux">
<div class="flux-boite">Règles écrites à la main<br>+ Données</div>
<div class="flux-fleche">↓</div>
<div class="flux-boite">Résultat</div>
</div>

*On code la logique nous-même.*

</div>
<div>

**Machine Learning**

<div class="flux">
<div class="flux-boite">Données<br>+ Résultats (exemples)</div>
<div class="flux-fleche">↓</div>
<div class="flux-boite accent">Règles apprises</div>
</div>

*Le modèle "trouve" la logique.*

</div>
</div>

<div class="attention">

**Attention —** le ML n'est pas magique : il a besoin d'exemples représentatifs pour apprendre une règle qui généralise correctement.

</div>

---

## Pourquoi maintenant ? Les limites du classique

Pour beaucoup de tâches, écrire les règles à la main est **impossible en pratique** :

- Détection de spams, reconnaissance faciale
- Traduction automatique, diagnostic médical
- Prédiction financière, recommandation de contenu

<div class="intuition">

**Intuition —** Point commun : personne ne sait écrire "la formule" qui distingue un spam d'un email normal. On a par contre des <b>milliers d'exemples</b> des deux — c'est cette matière première que le ML exploite.

</div>

---

## Deux définitions qui font référence

<div class="retenir">

**Arthur Samuel (1959)** — le terme lui-même : *"Machine Learning est le domaine d'étude qui donne aux ordinateurs la capacité d'apprendre sans être explicitement programmés."*

</div>

<div class="retenir">

**Tom Mitchell (1997)** — définition opérationnelle : un programme apprend d'une **expérience E**, pour une **tâche T** et une **mesure de performance P**, si sa performance à T, mesurée par P, s'améliore avec E.

</div>

*Exemple : jouer aux échecs (T), % de victoires (P), parties d'entraînement jouées (E).*

---

## Les trois grandes familles

| Type | Principe | Exemple |
|---|---|---|
| **Supervisé** | On connaît la bonne réponse pendant l'entraînement | Prix d'un logement, survie d'un passager |
| **Non-supervisé** | Pas de bonne réponse fournie, on cherche une structure | Regrouper des clients similaires |
| **Renforcement** | Un agent apprend par essai/erreur avec des récompenses | Un robot qui apprend à marcher |

*Non-supervisé, en détail : regroupement (clustering), réduction de dimension, estimation de densité — vu en Séance 5.*

**Ce module se concentre sur le supervisé** (S1-S6), avec un aperçu du non-supervisé en S5.

---

## À l'intérieur du supervisé

<div class="deux-colonnes">
<div>

### Régression

La target est une valeur **continue**

*Exemple : prix d'un logement*

**→ Séance 2**

</div>
<div>

### Classification

La target est une **catégorie**

*Exemple : survécu / n'a pas survécu*

**→ Séance 3**

</div>
</div>

---

<!-- _class: section -->

# 3. Le vocabulaire essentiel

---

## Modèle, paramètres, hyperparamètres

<div class="intuition">

Un <b>modèle</b> est une fonction paramétrée : on choisit sa <i>forme</i> (ex : une droite), et l'entraînement ajuste ses <b>paramètres</b> (ex : la pente) pour qu'elle colle le mieux aux données.

</div>

- **Paramètres** — ajustés automatiquement pendant l'entraînement (ex : $w$, $b$)
- **Hyperparamètres** — choisis par nous, avant l'entraînement (ex : le learning rate)

<div class="retenir">

**À retenir —** On reverra cette distinction à chaque séance — elle est centrale pour comprendre ce qu'on contrôle et ce que le modèle apprend seul.

</div>

---

## Train / Validation / Test

<div class="attention">

**Attention —** On ne peut jamais juger un modèle sur les données qu'il a déjà vues à l'entraînement.

</div>

| Ensemble | Rôle |
|---|---|
| **Train** | Le modèle apprend dessus |
| **Validation** | On compare des modèles / réglages entre eux |
| **Test** | Évaluation finale, une seule fois, à la toute fin |

**Pourquoi ?** → notion de **généralisation** : bien faire sur du jamais-vu, pas seulement sur ce qu'on a mémorisé.

*(On creuse ce point en profondeur en Séance 4.)*

---

<!-- _class: section -->

# 4. Premier contact : California Housing

---

## Le fil rouge du module

<div class="intuition">

California Housing nous accompagne de la Séance 1 au projet final. On l'observe sous des angles différents à chaque séance.

</div>

- **8 features** : revenu médian, âge des logements, nombre de pièces, population, position géographique...
- **1 target** : `MedHouseVal`, la valeur médiane du logement
- **20 640 districts** californiens (recensement 1990)

**Aujourd'hui : on observe, on ne modélise pas encore.**

---

## Ce qu'on va regarder ensemble

1. À quoi ressemblent les données brutes ?
2. Comment se distribue le prix des logements ?
3. Le revenu médian semble-t-il lié au prix ?

<div class="exercice">

**Exercice —** Question à garder en tête : si tu devais deviner le prix d'un logement <b>sans</b> ML, comment t'y prendrais-tu ?

</div>

<!-- Note présentateur (non visible à l'écran) : basculer sur Colab ici pour la démonstration en direct. -->

---

<!-- _class: section -->

# 5. Setup & travail pratique

---

## Environnement de travail

| Outil | Usage |
|---|---|
| **Google Colab** | Exécution des notebooks, aucune installation |
| **GitHub — `formation-ml`** | Notebooks, ressources, structure du module |
| **Google Classroom** | Annonces, ressources complémentaires |
| **WhatsApp** | Questions rapides, entraide |

<div class="retenir">

**À retenir —** Premier réflexe dans Colab : <b>Fichier ▸ Enregistrer une copie dans Drive</b>, avant de toucher au code.

</div>

---

## Travail pratique guidé (35 min)

Dans `notebook_pratique_introduction.ipynb` :

1. Choisir une feature (autre que `MedInc`/`MedHouseVal`)
2. Calculer ses statistiques descriptives
3. Produire une visualisation
4. Rédiger une interprétation en 2-3 phrases

<div class="exercice">

**Exercice —** Bonus si le temps le permet : visualiser Latitude/Longitude — qu'est-ce que ça dessine ?

</div>

---

<!-- _class: section -->

# Débrief

---

## Ce qu'on retient

- Le ML apprend des **règles à partir de données**, plutôt qu'on les écrive à la main
- On a des **features** et une **target**
- On distingue **paramètres** et **hyperparamètres**
- On ne juge jamais un modèle sur ce qu'il a déjà vu

<div class="intuition">

**Intuition —** On a des features, une target — il nous manque une <b>fonction</b> qui relie les deux, et une façon de la trouver <b>automatiquement</b>.

</div>

**→ Séance 2 : régression linéaire & descente de gradient**

---

<!-- _class: titre -->

# Merci !

### Questions ?

Canal WhatsApp — questions ML

