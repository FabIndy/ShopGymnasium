# DAgger + Maskable PPO dans l’atelier de production  
### Analyse, fonctionnement et bilan des expériences

Ce rapport présente le fonctionnement du notebook `dagger_hybrid`, les techniques utilisées (masque d’actions, expert v2, apprentissage par imitation DAgger, renforcement PPO) et les résultats observés.  
L’objectif est d’entraîner un agent capable de gérer un atelier industriel complexe de manière optimale.

---

# 1. Masquage d’actions (Action Masking)

L’environnement `WorkshopEnv` possède 201 actions possibles, mais **toutes ne sont pas valides à un instant donné** :

- Impossible de produire si la machine est déjà occupée  
- Impossible de lancer des productions sans matières premières  
- Impossible de faire STEP2 si aucun P2_inter n’est disponible  
- etc.

Le **masquage d’actions** consiste à interdire automatiquement les actions impossibles pour éviter que le modèle RL apprenne sur du bruit.

Dans ce projet, nous utilisons :

- `env.get_action_mask()` → renvoie un tableau booléen de taille 201  
- `ActionMasker(env, mask_fn)` → wrapper SB3-Contrib  
- `MaskablePPO` → algorithme PPO compatible masques

Grâce à ce mécanisme :

✔ L’agent ne choisit jamais une action impossible  
✔ L’expert utilise aussi le masque pour rester cohérent  

---

# 2. Politique experte v2

L’expert v2 utilise 23 variables d’observation normalisées :

- état des machines (busy, time left)  
- stocks (raw, P1, P2_inter, P2)  
- backlogs  
- livraison de matières premières à venir  
- demande en cours  
- variables additionnelles (minute du jour, reward précédent, etc.)

Même si l’expert n’utilise qu’un sous-ensemble de ces variables, il applique une logique claire :

1. **Si le stock de MP est faible → commander**  
2. **Si M2 est libre et P2_inter > 0 → STEP2 pour finir des P2**  
3. **Si M1 est libre, la demande en P2 > 0 et MP disponible → STEP1 pour P2**  
4. **Sinon : produire P1 si demande P1 > 0 et MP disponible**  
5. **Si aucune de ces actions n’est possible → WAIT**

Sur un épisode de 7 jours (10 080 steps), on obtient :

```text
obs shape : (10080, 23)
actions shape : (10080,)
Reward expert sur cet épisode : 12729.30
Steps joués : 10080

Buffer DAgger initialisé :
dagger_obs : (10080, 23)
dagger_actions : (10080,)
```

 L’expert est **très performant** et produit un dataset de démonstrations riche et cohérent.

---

# 3. DAgger — Imitation Learning avec agrégation

DAgger (Dataset Aggregation) fonctionne en trois phases :

### (1) Imitation supervisée  
L’élève apprend à imiter l’expert sur le buffer courant :

```text
Entrée  : observation (23 variables normalisées)
Cible   : action experte (entier entre 0 et 200)
Perte   : CrossEntropyLoss(logits, action_experte)
```

Le réseau de politique de MaskablePPO est réutilisé tel quel, et on applique une descente de gradient classique pour réduire la CrossEntropy entre les actions prédites et les actions expertes.

### (2) Phase RL (désactivée pour le diagnostic)

Dans la version complète, une phase PPO est insérée ici pour affiner la politique par renforcement.  
**Dans ce rapport, cette phase a été volontairement désactivée** (cellule 9) afin de diagnostiquer séparément :

- la qualité de l’**imitation supervisée seule**,  
- l’impact éventuel de PPO.

Les cellules 11 et 12 du notebook, qui devaient lancer un entraînement PPO « final » indépendant puis le tester, ont été **interrompues** après ce constat : tant que DAgger ne fonctionne pas correctement, il est inutile de poursuivre un long entraînement PPO final.

### (3) Collecte DAgger  
L’élève joue un épisode complet.  
À chaque step :

- l’élève choisit une action (avec masque),
- l’expert est appelé pour corriger l’action,
- la paire `(obs, action_experte)` est ajoutée au buffer DAgger.

On répète ensuite :

1. Imitation sur le buffer agrégé  
2. Collecte de nouvelles données corrigées  
3. Agrégation dans le buffer

L’objectif théorique : **l’élève converge progressivement vers la stratégie experte**.

---

# 4. PPO — Proximal Policy Optimization (contexte)

PPO est un algorithme de renforcement qui :

- maximise le reward attendu  
- limite les mises à jour trop grandes via un terme de clipping (ratio de probabilité)  
- utilise un estimateur de valeur d’état (`V(s)`) pour réduire la variance  
- s’adapte bien aux environnements complexes

Dans ce notebook, PPO est utilisé via `MaskablePPO`, ce qui permet de combiner :

- PPO classique  
- Masquage d’actions  
- Politique et critic partagés avec la phase d’imitation

**Mais dans les expériences décrites ici, la phase PPO a été désactivée pour se concentrer sur un diagnostic de DAgger.**

---

# 5. Résultats expérimentaux (imitation seule)

## 5.1 Dataset expert initial (cellule 4)

Comme rappelé plus haut :

```text
obs shape : (10080, 23)
actions shape : (10080,)
Reward expert sur cet épisode : 12729.30
Steps joués : 10080

Buffer DAgger initialisé :
dagger_obs : (10080, 23)
dagger_actions : (10080,)
```

---

## 5.2 Itération 1 DAgger (cellule 10)

### Phase 1 : Imitation supervisée

```text
  → Entraînement supervisé sur 10080 exemples (20 batches)
    Epoch 1/3 — loss = 5.2627
    Epoch 2/3 — loss = 5.1704
    Epoch 3/3 — loss = 5.0473
  ✓ Entraînement supervisé terminé.
```

La perte baisse légèrement, mais reste **très élevée** pour une classification à 201 classes :  
cela indique que l’élève n’arrive pas à apprendre une politique proche de l’expert.

### Phase 2 : PPO désactivé

```text
===== Phase 2 : (désactivée) =====
⚠ PPO désactivé volontairement pour test de diagnostic.
```

### Phase 3 : Collecte DAgger

```text
Reward élève pendant collecte DAgger : -5835.66 sur 10080 steps
Taille du buffer DAgger après agrégation : (20160, 23)
```

L’élève joue extrêmement mal (reward très négatif), très loin du niveau de l’expert.

### Évaluation élève après itération 1

```text
Reward sur 7 jours : -6309.72
Reward moyen par jour : -901.39
Steps joués : 10080
Modèle sauvegardé dans : maskedppo_dagger_iter_1.zip
```

 Malgré l’imitation supervisée, l’élève reste **largement perdant**.

---

## 5.3 Itération 2 DAgger

### Phase 1 : Imitation supervisée

```text
  → Entraînement supervisé sur 20160 exemples (40 batches)
    Epoch 1/3 — loss = 4.9572
    Epoch 2/3 — loss = 4.6194
    Epoch 3/3 — loss = 4.1639
  ✓ Entraînement supervisé terminé.
```

La CrossEntropyLoss baisse (≈ 4.16 en fin d’époch), mais demeure très élevée :  
le modèle ne parvient toujours pas à reproduire correctement les actions expertes.

### Phase 2 : PPO désactivé

```text
===== Phase 2 : (désactivée) =====
⚠ PPO désactivé volontairement pour test de diagnostic.
```

### Phase 3 : Collecte DAgger

```text
Reward élève pendant collecte DAgger : -5613.00 sur 10080 steps
Taille du buffer DAgger après agrégation : (30240, 23)
```

### Évaluation élève après itération 2

```text
Reward sur 7 jours : -5934.70
Reward moyen par jour : -847.81
Steps joués : 10080
Modèle sauvegardé dans : maskedppo_dagger_iter_2.zip
```

Légère amélioration numérique, mais **aucun changement qualitatif** :  
l’élève reste très loin de l’expert (+12 729).

---

# 6. Conclusion : pourquoi DAgger échoue dans cette configuration ?

Les observations sont claires :

- L’expert v2 est **très performant**  
- Le buffer de démonstrations initial est cohérent  
- Le masque d’actions fonctionne  
- La phase PPO a été désactivée pour isoler l’imitation  
- Les cellules 11 et 12, qui devaient lancer un PPO final indépendant, ont été **interrompues** volontairement après ce diagnostic, puisqu’il est inutile de pousser plus loin le RL tant que l’imitation DAgger elle-même échoue.  
- Malgré tout cela, l’élève **n’arrive pas à imiter l’expert**

Deux symptômes majeurs :

1. **CrossEntropyLoss élevée** (≈ 4–5) même après plusieurs périodes d’entraînement supervisé  
2. **Reward élève très négatif** (environ –6000) après deux itérations DAgger

Nous en concluons que **le problème principal n’est pas PPO, ni le masque**, mais bien :

> **La difficulté de l’élève à approximer la politique experte à partir des 23 variables d’observation.**

Plus précisément :

- Le vecteur d’observation contient de nombreuses variables **non utilisées par l’expert** (minute_of_day, theft_risk, rewards normalisés, identifiant d’action courante, etc.).  
- L’élève reçoit donc une entrée de dimension 23, dont une partie est du bruit ou de l’information non pertinente pour la décision experte.  
- La fonction à approximer (obs → action_experte) devient très complexe, voire non identifiable pour le réseau actuel.

---

# 7. Étapes suivantes : simplifier l’espace d’observation

Pour que DAgger fonctionne réellement, les prochaines étapes sont :

## 7.1 Réduire le nombre de variables

Objectif : passer de 23 variables à un noyau d’environ 10–13 variables **effectivement utilisées par l’expert**, par exemple :

- `time / max_time` (optionnel)  
- `m1_busy`, `m1_time_left`  
- `m2_busy`, `m2_time_left`  
- `stock_raw`, `stock_p1`, `stock_p2_inter`, `stock_p2`  
- `demande_p1`, `demande_p2`  
- `q_raw_incoming`, `next_delivery_countdown`

À l’inverse, il est recommandé de **retirer** (au moins dans un premier temps, côté DAgger) :

- `reward_current_action`, `week_reward`  
- `minute_of_day`  
- `expected_dem_*`  
- `current_action_type`, `current_action_k`, `current_action_id`  
- `theft_risk_level` (si non utilisé par l’expert)

## 7.2 Relancer DAgger en imitation seule

Avec un espace d’observation plus simple :

- la CrossEntropyLoss devrait fortement baisser  
- l’élève devrait commencer à imiter l’expert de manière visible  
- le reward élève devrait remonter significativement (proche de 0 ou positif)

## 7.3 Réactiver PPO dans la boucle DAgger

Une fois l’imitation stabilisée :

- on peut réactiver la phase PPO  
- PPO viendra exploiter les récompenses pour améliorer la stratégie au-delà de l’expert (potentiellement)

## 7.4 (Optionnel) Réintroduire progressivement certaines variables supplémentaires

Une fois l’agent compétent, on peut :

- rajouter certaines variables supplémentaires  
- tester leur impact sur la performance  
- garder seulement celles qui apportent un gain stable

---

# 8. Bilan

Ce travail sur `dagger_hybrid` a permis :

- de valider la **cohérence de l’environnement** (masques, expert, normalisation)  
- de montrer que l’expert v2 est **très performant**  
- d’identifier que **DAgger, dans sa configuration actuelle, n’arrive pas à transférer cette expertise à l’élève**  
- de localiser précisément la cause principale :  
  > un vecteur d’observation trop riche, mal aligné avec les signaux que l’expert utilise réellement.

Les prochaines modifications porteront donc sur :

- la **réduction du nombre de variables d’observation**,  
- puis une **reprise des expériences DAgger + PPO** avec un espace d’état simplifié.

Ce diagnostic constitue une base solide pour les itérations futures sur ce projet d’atelier industriel optimisé par apprentissage par renforcement.
