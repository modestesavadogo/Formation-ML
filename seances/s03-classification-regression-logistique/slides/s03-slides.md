---
marp: true
theme: formation-ml
paginate: true
math: katex
---

<!-- _class: titre -->

# Classification & Régression Logistique

### Formation ML — Séance 3

<br>

**3 octobre 2026**

---

## Rappel — Séance 2

- Un algorithme d'apprentissage a toujours 3 briques : ① modèle, ② fonction de coût, ③ optimisation
- On a construit une régression pour prédire une valeur **continue**
- La descente de gradient suit la pente pour minimiser le coût

<div class="intuition">

**Intuition —** Aujourd'hui, la target n'est plus continue mais <b>binaire</b> : survécu ou non, spam ou non, malade ou non. Les 3 mêmes briques vont servir — mais leur contenu change en profondeur.

</div>

---

## Au programme

1. Pourquoi une droite ne suffit plus
2. ① Le modèle : la sigmoïde — et d'où elle vient vraiment
3. Interpréter les coefficients : la cote (odds) et le logit
4. ② La fonction de coût : le log-loss, dérivé et calculé à la main
5. ③ L'optimisation : dériver le gradient, pas juste l'admettre
6. La frontière de décision
7. Vers le multiclasse

---

<!-- _class: section -->

# 1. Pourquoi une droite ne suffit plus

---

## Le nouveau problème : Titanic

- **Target** : `Survived` — 0 ou 1, pas une valeur continue
- On veut en réalité une **probabilité** : $P(\text{survie} \mid x)$
- Une probabilité doit rester dans $[0, 1]$

<div class="attention">

**Attention —** Rien n'empêche une droite $\hat{y} = wx+b$ de sortir de cet intervalle.

</div>

---

## La preuve visuelle

![w:850](figures/01_pourquoi_pas_lineaire.png)

<div class="retenir">

**À retenir —** Il faut une fonction qui <b>écrase</b> n'importe quelle valeur réelle dans $[0, 1]$, quel que soit $w$ et $b$. Reste à savoir : laquelle, précisément, et pourquoi celle-là plutôt qu'une autre courbe en S ?

</div>

---

<!-- _class: section -->

# ① Le modèle : d'où vient la sigmoïde ?

---

## Partir d'une idée plus naturelle : la cote (odds)

<div class="intuition">

Au lieu de modéliser directement une probabilité (coincée entre 0 et 1, difficile à combiner linéairement), les statisticiens ont eu une autre idée : modéliser la <b>cote</b> — ce que les parieurs appellent les "odds".

</div>

$$\text{cote} = \frac{p}{1-p}$$

- $p = 0.5$ → cote $= 1$ (autant de chances des deux côtés)
- $p = 0.8$ → cote $= 4$ (4 fois plus de chances de survivre que non)
- $p \to 1$ → cote $\to \infty$

**La cote peut prendre n'importe quelle valeur positive** — déjà plus flexible qu'une probabilité.

---

## Encore un problème : la cote reste positive

<div class="attention">

**Attention —** On veut modéliser avec une combinaison linéaire $wx+b$, qui peut être négative. La cote (toujours positive) ne peut donc pas être égale à $wx+b$ directement.

</div>

**Solution : passer au logarithme.**

$$\log(\text{cote}) = \log\left(\frac{p}{1-p}\right)$$

Le logarithme d'un nombre positif peut être n'importe quel réel — négatif, nul, positif. **C'est cette quantité qu'on pose égale à $wx+b$.**

---

## La définition qui fonde tout le reste

$$\log\left(\frac{p}{1-p}\right) = wx + b = z$$

En isolant $p$ (quelques lignes d'algèbre : exponentielle des deux côtés, puis réarrangement) :

$$p = \frac{1}{1 + e^{-z}} = \sigma(z)$$

<div class="retenir">

**À retenir —** La sigmoïde n'est donc pas une courbe en S choisie au hasard pour sa forme pratique — c'est la fonction qu'on obtient <b>nécessairement</b> en modélisant le log de la cote de façon linéaire.

</div>

---

## Vérification visuelle : le logit est bien une droite

![w:1050](figures/06_logit_lineaire.png)

<div class="retenir">

**À retenir —** À gauche, $p$ en fonction de $z$ dessine la courbe en S familière. À droite, la même relation vue à travers $\log(p/(1-p))$ redevient une <b>droite parfaite</b> — la preuve que le modèle est bien linéaire "sous le capot".

</div>

---

## Propriétés de la sigmoïde

| Propriété | Conséquence |
|---|---|
| $\sigma(0) = 0.5$ | $z=0$ est le point de bascule |
| $\sigma(-z) = 1-\sigma(z)$ | symétrie autour de $(0, 0.5)$ |
| $\sigma'(z) = \sigma(z)(1-\sigma(z))$ | dérivée simple, utile pour ③ |
| Jamais 0 ni 1 exactement | le modèle n'est jamais "certain à 100%" |

---

## La dérivée, visualisée

![w:750](figures/07_derivee_sigmoide.png)

<div class="intuition">

La dérivée est maximale exactement à $z=0$ (valeur 0.25) — c'est là que la sigmoïde est la plus "sensible" à un changement de $z$. Loin de 0, la dérivée s'écrase vers 0 : la sigmoïde <b>sature</b>.

</div>

---

## L'effet de $w$ et $b$ sur la forme

![w:1050](figures/08_effet_w_b.png)

<div class="retenir">

**À retenir —** $w$ contrôle la <b>brutalité</b> de la transition (grand $w$ = frontière nette, petit $w$ = transition floue). $b$ contrôle <b>où</b> se situe le point de bascule. Ce sont exactement les deux paramètres que l'optimisation va ajuster.

</div>

---

<!-- _class: section -->

# Interpréter les coefficients

---

## Ce que $w$ signifie concrètement

<div class="intuition">

Puisque $\log(\text{cote}) = wx+b$, augmenter $x$ de 1 unité ajoute $w$ au log de la cote — ce qui revient à <b>multiplier la cote par $e^w$</b>.

</div>

$$\text{nouvelle cote} = \text{ancienne cote} \times e^{w}$$

$e^w$ s'appelle l'**odds ratio** — c'est la grandeur que les praticiens (épidémiologistes, data scientists) utilisent réellement pour interpréter un modèle logistique.

---

## Sur de vraies données Titanic

En entraînant une régression logistique sur `Fare` (standardisé) :

$$w \approx 0.846 \qquad b \approx -0.342$$

$$e^{w} = e^{0.846} \approx 2.330$$

<div class="retenir">

**À retenir —** Chaque augmentation d'un écart-type du tarif payé <b>multiplie par ~2.33 la cote de survie</b> — plus qu'un doublement. C'est une phrase qu'on peut dire à quelqu'un qui ne fait pas de ML, contrairement à "le coefficient vaut 0.846".

</div>

---

<!-- _class: section -->

# ② La fonction de coût : le log-loss

---

## D'où vient vraiment cette formule ?

<div class="intuition">

On ne choisit pas le log-loss au hasard parce qu'il "marche bien" — il découle directement d'un principe statistique : le <b>maximum de vraisemblance</b>.

</div>

Pour une observation, le modèle prédit $\hat{y} = P(y=1\mid x)$. La probabilité que le modèle attribue à ce qui s'est **réellement produit** est :

$$P(y \mid x) = \hat{y}^{\,y} \, (1-\hat{y})^{\,1-y}$$

*(vérifie : si $y=1$, ça vaut $\hat{y}$ ; si $y=0$, ça vaut $1-\hat{y}$)*

---

## De la vraisemblance au log-loss

On veut les paramètres qui **maximisent** la probabilité d'observer les vraies données — sur tout le dataset (probabilités indépendantes, donc on multiplie) :

$$\mathcal{L}(w,b) = \prod_{i=1}^n \hat{y}_i^{\,y_i} (1-\hat{y}_i)^{\,1-y_i}$$

<div class="attention">

**Attention —** Multiplier des centaines de petites probabilités donne un nombre proche de zéro — problématique numériquement. On passe au <b>log</b> (transforme les produits en sommes, ne change pas où se situe le maximum).

</div>

---

## Et enfin, minimiser plutôt que maximiser

$$\log \mathcal{L}(w,b) = \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]$$

Par convention, l'optimisation **minimise** — on prend donc l'opposé :

$$\text{Log-loss}(w,b) = -\frac{1}{n}\sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]$$

<div class="retenir">

**À retenir —** Ce n'est pas une formule arbitraire choisie pour ses bonnes propriétés — c'est la conséquence directe de "trouver les paramètres les plus vraisemblables au vu des données observées".

</div>

---

## Pourquoi pas la MSE ? Le vrai problème

Voyons ce qui se passe quand le modèle est **confiant et faux** — $y=1$ mais $\hat{y}$ proche de 0.

![w:1050](figures/05_convexite_logloss_vs_mse.png)

<div class="attention">

**Attention —** Avec la MSE, le gradient devient quasiment nul dans ce cas — <b>rien ne pousse le modèle à se corriger</b>. Le log-loss maintient un gradient fort exactement là où c'est le plus nécessaire.

</div>

---

## Visualiser le log-loss, terme par terme

![w:1050](figures/03_log_loss.png)

<div class="retenir">

**À retenir —** Prédire avec confiance <b>et juste</b> → coût proche de 0. Prédire avec confiance <b>et faux</b> → coût qui explose vers l'infini. C'est exactement le comportement voulu.

</div>

---

## Calcul à la main, sur 4 vraies lignes Titanic

Avec le modèle entraîné plus haut ($w \approx 0.846$, $b \approx -0.342$) :

| Fare | Survived ($y$) | $z$ | $\sigma(z)$ | log-loss |
|---|---|---|---|---|
| 21.08 | 0 | −0.560 | 0.364 | 0.452 |
| 39.69 | 0 | −0.262 | 0.435 | 0.571 |
| 11.13 | 1 | −0.719 | 0.328 | 1.116 |
| 10.50 | 1 | −0.729 | 0.325 | 1.123 |

**Log-loss moyenne** = (0.452 + 0.571 + 1.116 + 1.123) / 4 = **0.815**

<div class="attention">

**Attention —** Les deux passagers ayant survécu (lignes 3-4) ont payé un tarif faible — le modèle leur attribue une <b>faible</b> probabilité de survie (~0.33), d'où un coût élevé : l'erreur du modèle, chiffrée précisément.

</div>

---

<!-- _class: section -->

# ③ L'optimisation : dériver le gradient

---

## Ne pas admettre la formule — la construire

<div class="intuition">

On va calculer $\dfrac{\partial \, \text{Log-loss}}{\partial w}$ par la règle de la chaîne, en 3 étapes : d'abord par rapport à $\hat{y}$, puis par rapport à $z$, puis par rapport à $w$.

</div>

**Étape 1** — dérivée du log-loss par rapport à $\hat{y}$ (pour une observation) :

$$\frac{\partial \, \text{coût}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$

---

## Étape 2 — utiliser la dérivée de la sigmoïde

$$\frac{\partial \hat{y}}{\partial z} = \sigma(z)(1-\sigma(z)) = \hat{y}(1-\hat{y})$$

*(c'est exactement la propriété vue plus haut, page "Propriétés de la sigmoïde")*

En combinant les étapes 1 et 2 par la règle de la chaîne :

$$\frac{\partial \, \text{coût}}{\partial z} = \frac{\partial \, \text{coût}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} = \hat{y} - y$$

<div class="retenir">

**À retenir —** Les termes $\hat{y}(1-\hat{y})$ se simplifient élégamment avec le dénominateur — c'est loin d'être un hasard : c'est précisément <b>pourquoi</b> log-loss et sigmoïde sont faits pour aller ensemble.

</div>

---

## Étape 3 — et enfin par rapport à $w$

Puisque $z = wx+b$, on a $\dfrac{\partial z}{\partial w} = x$. Donc, sur tout le dataset :

$$\frac{\partial \, \text{Log-loss}}{\partial w} = \frac{1}{n}\sum_{i=1}^n x_i(\hat{y}_i - y_i)$$

$$\frac{\partial \, \text{Log-loss}}{\partial b} = \frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)$$

<div class="retenir">

**À retenir —** C'est <b>exactement</b> la même forme qu'en régression linéaire (erreur $\times$ feature) — mais maintenant on sait <b>pourquoi</b>, pas juste "ça se trouve que c'est pareil".

</div>

---

## La règle de mise à jour — inchangée

$$w \leftarrow w - \alpha \frac{\partial \, \text{Log-loss}}{\partial w}$$

$$b \leftarrow b - \alpha \frac{\partial \, \text{Log-loss}}{\partial b}$$

<div class="intuition">

Même réflexe qu'en Séance 2 : avancer dans le sens opposé au gradient, à un pas $\alpha$. Tout ce qu'on a appris sur le learning rate (trop petit / trop grand) s'applique <b>identiquement</b> ici — même mécanique, seule la définition de $\hat{y}$ a changé en amont.

</div>

---

<!-- _class: section -->

# La frontière de décision

---

## Ce que "décider" veut dire géométriquement

<div class="intuition">

Le seuil $\hat{y} = 0.5$ correspond à $z = wx+b = 0$ — une <b>ligne</b> (ou un hyperplan, en dimension supérieure) qui sépare l'espace des features en deux régions.

</div>

Avec deux features (`Age`, `Fare`), on peut la visualiser directement.

---

## La frontière, en pratique

![w:750](figures/04_frontiere_decision.png)

<div class="retenir">

**À retenir —** La ligne pointillée est la frontière de décision — exactement là où $z=0$. Les couleurs de fond montrent la probabilité continue ; la frontière n'est qu'une coupe à 0.5, pas une limite "dure" dans les données elles-mêmes.

</div>

---

<!-- _class: section -->

# Vers le multiclasse

---

## Et s'il y a plus de 2 classes ?

<div class="intuition">

La régression logistique telle qu'on l'a construite est <b>binaire</b>. Pour $K$ classes (ex : type de fleur, catégorie de produit), on généralise avec la fonction <b>softmax</b>, qui étend la sigmoïde à plusieurs sorties.

</div>

$$P(y=k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

<div class="attention">

**Attention —** Chaque classe a son propre $z_k = w_k x + b_k$. Le softmax garantit que toutes les probabilités somment à 1. Hors scope détaillé de cette séance — mais retiens que le principe ①②③ reste identique, seule la brique ① s'enrichit.

</div>

---

<!-- _class: section -->

# Débrief

---

## Ce qu'on retient

| Brique | Régression (S2) | Classification (S3) |
|---|---|---|
| ① Modèle | $\hat{y} = wx+b$ | $\hat{y} = \sigma(wx+b)$, dérivé du logit |
| ② Coût | MSE | Log-loss, dérivé du maximum de vraisemblance |
| ③ Optimisation | Descente de gradient | Descente de gradient (mécanique identique) |

<div class="retenir">

**À retenir —** Rien n'est arbitraire dans ce qu'on a construit aujourd'hui : la sigmoïde découle du choix de modéliser le log de la cote, le log-loss découle du maximum de vraisemblance, et le gradient se dérive proprement — les trois s'emboîtent exactement.

</div>

**→ Séance 4 : généralisation, validation, prétraitement**

---

<!-- _class: titre -->

# Merci !

### Questions ?

Canal WhatsApp — questions ML
