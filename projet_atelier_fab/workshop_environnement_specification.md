
# Spécification de l’Environnement Workshop – Version RL avancée (23 variables)

Ce document constitue la **spécification officielle** de l’environnement industriel *Workshop* utilisé pour l’apprentissage par renforcement (DQN, PPO, MaskablePPO, etc.).  
Il correspond exactement à la dernière version de `WorkshopEnv` implémentée dans `workshop_env.py` (état de dimension 23).

---

## 1. Structure temporelle

- **1 step = 1 minute**
- 1 journée = 1440 minutes
- 1 épisode = **7 jours = 10080 minutes**
- Les ventes ont lieu **toutes les 15 minutes**.
- Un **vol** a lieu **une fois par jour**, à la minute 1435 de chaque journée.

À chaque step, l’agent choisit une action, les machines avancent, les livraisons sont traitées, la demande/vente peut être mise à jour, un vol peut survenir, puis une nouvelle observation est renvoyée.

---

## 2. Ressources et produits

### 2.1 Types de stocks

- `stock_raw` : matières premières (MP) disponibles.
- `stock_p1` : produit fini P1.
- `stock_p2_inter` : produit intermédiaire pour P2 (étape 1).
- `stock_p2` : produit fini P2.

Capacité nominale utilisée dans le code : environ **50 unités** par type de stock (bornes logiques des observations).

---

## 3. Machines

### 3.1 Machine M1

Transforme la MP en :
- **P1**  
- **P2_STEP1** (intermédiaire pour P2).

Pour un batch de taille `k` :

| Type de batch | Entrée | Sortie | Durée totale |
|---------------|--------|--------|--------------|
| P1_MULTI      | 1 MP par unité | 1 P1 par unité | `3 × k` minutes |
| P2STEP1_MULTI | 1 MP par unité | 1 P2_inter par unité | `10 × k` minutes |

État interne (côté code) : `busy`, `time_left`, `batch_type`, `batch_k`.

### 3.2 Machine M2

Transforme `P2_inter` en produit fini **P2** :

| Type de batch     | Entrée par unité | Sortie par unité | Durée totale |
|-------------------|------------------|------------------|--------------|
| P2STEP2_MULTI     | 1 P2_inter       | 1 P2             | `15 × k` minutes |

État interne similaire : `busy`, `time_left`, `batch_type`, `batch_k`.

### 3.3 Production « au fil de l’eau »

Le temps avance **minute par minute**. À chaque minute, chaque machine décrémente `time_left` et peut **produire une unité** intermédiaire avant la fin totale du batch :

- À chaque minute où une unité est produite :  
  - si `batch_type == "P1_MULTI"` → `stock_p1 += 1`
  - si `batch_type == "P2STEP1_MULTI"` → `stock_p2_inter += 1`
  - si `batch_type == "P2STEP2_MULTI"` → `stock_p2 += 1`
- À la dernière unité (`"last_unit"`), la machine se remet dans un état neutre (batch terminé).

Ainsi, les stocks se remplissent **progressivement**, sans attendre la fin complète du batch.

---

## 4. Commandes de matières premières (MP)

Les actions **150 à 199** correspondent à des commandes de MP :

- `q = action - 149` : quantité commandée.
- coût immédiat dans le reward : `reward -= q`.
- la livraison est planifiée dans environ 120 minutes avec un léger jitter : `120 ± 2 minutes`.
- chaque commande est stockée dans une file `DeliveryQueue` sous forme `(quantité, temps_d_arrivée)`.

À chaque step, la queue est « tickée » : toutes les commandes dont le temps d’arrivée est atteinte sont livrées et leur quantité est ajoutée à `stock_raw`.

---

## 5. Demande client, backlog et ventes

### 5.1 Backlog

On modélise deux backlogs :

- `demande_p1` : demandes P1 non satisfaites,
- `demande_p2` : demandes P2 non satisfaites.

Ce sont des **demandes résiduelles** : chaque nouvelle demande s’ajoute au backlog, et chaque vente le réduit.

### 5.2 Génération de la demande

Toutes les **15 minutes** (`time % 15 == 0`), on génère une nouvelle demande pour P1 et P2, typiquement via un modèle de Poisson jour/nuit (paramètres internes dans `Market`) :

```python
new_d1, new_d2 = market.sample_demand(time, 15)
demande_p1 += new_d1
demande_p2 += new_d2
```

### 5.3 Ventes

Au même moment (toutes les 15 minutes), les ventes sont calculées en fonction du stock disponible et du backlog :

```python
sold_p1 = min(stock_p1, demande_p1)
sold_p2 = min(stock_p2, demande_p2)

stock_p1 -= sold_p1
stock_p2 -= sold_p2

demande_p1 -= sold_p1
demande_p2 -= sold_p2
```

Les ventes génèrent des rewards :

- `reward += 2 * sold_p1 + 20 * sold_p2`.

### 5.4 Pénalité sur le backlog

Après la mise à jour des ventes, on pénalise le backlog total :

```python
backlog_total = demande_p1 + demande_p2
reward -= 0.02 * backlog_total
```

---

## 6. Vol nocturne

Une fois par jour, à la minute `theft_time = 1435` de chaque journée (juste avant minuit), un vol partiel des stocks finis P1 et P2 est appliqué :

```python
if self.time % 1440 == self.theft_time:
    stock_p1 = floor(stock_p1 * 0.9)
    stock_p2 = floor(stock_p2 * 0.9)
```

Une des nouvelles variables d’état, `theft_risk_level`, informe l’agent de la proximité de cet événement (voir section 10).

---

## 7. Reward

Le reward instantané d’un step agrège :

- **Ventes :**
  - `+2` pour chaque unité de P1 vendue.
  - `+20` pour chaque unité de P2 vendue.
- **Commandes de MP :**
  - coût : `-q` pour une commande de `q` MP.
- **Actions de production :**
  - P1 : `+0.5 × k` lors du lancement du batch (si faisable).
  - P2_STEP1 : `+5 × k` lors du lancement.
  - P2_STEP2 : `+15 × k` lors du lancement.
- **WAIT :**
  - `-0.2` par action WAIT.
- **Action impossible (ex. manque de MP) :**
  - `-1`.
- **Backlog :**
  - `-0.02 × (demande_p1 + demande_p2)` à chaque cycle de vente.

Deux nouvelles variables d’état exposent ce reward aux pas de temps :

- `reward_current_week` : somme cumulée de tous les rewards depuis le début de l’épisode.
- `reward_current_action` : reward du **dernier step**.

---

## 8. Espace d’actions (201 actions)

L’espace d’actions est **discret** : `action ∈ {0, …, 200}`.

| Plage | Signification | Paramètre k |
|-------|---------------|------------|
| 0–49 | Lancer un batch P1 sur M1 | `k = action + 1` |
| 50–99 | Lancer un batch P2_STEP1 sur M1 | `k = action − 49` |
| 100–149 | Lancer un batch P2_STEP2 sur M2 | `k = action − 99` |
| 150–199 | Commander des MP | `k = action − 149` |
| 200 | WAIT (aucune production / commande) | 0 |

Des masques d’action peuvent être dérivés via `env.get_action_mask()` pour interdire les actions techniquement impossibles (machine occupée, stock insuffisant, etc.).

---

## 9. Observation – Ancienne version (rappel)

L’ancienne version de l’environnement utilisait un vecteur d’observation de **13 variables** :

```text
[ time,
  m1_busy, m1_time_left,
  m2_busy, m2_time_left,
  stock_raw, stock_p1, stock_p2_inter, stock_p2,
  next_delivery_countdown,
  backlog_p1, backlog_p2,
  q_total_en_route ]
```

Ces 13 premières composantes sont **toujours présentes et dans le même ordre** dans la nouvelle version (indices 0 à 12).

---

## 10. Observation – Nouvelle version (23 variables)

La nouvelle version enrichit l’état avec **10 variables supplémentaires**, pour aider l’agent à mieux comprendre :

- le contexte temporel (dans la journée, avant la prochaine vente),
- le lien entre backlog et reward potentiel,
- le risque de vol,
- l’action courante et le reward associé.

Voici la liste complète du vecteur d’état (dimension 23) :

### 10.1 Partie « historique » (indices 0–12)

| Index | Nom                   | Description |
|-------|------------------------|-------------|
| 0     | `time`                | Minute courante depuis le début de l’épisode (0 à 10080). |
| 1     | `m1_busy`             | 0 ou 1 selon que M1 est occupée. |
| 2     | `m1_time_left`        | Durée restante (en minutes) du batch en cours sur M1. |
| 3     | `m2_busy`             | 0 ou 1 selon que M2 est occupée. |
| 4     | `m2_time_left`        | Durée restante du batch en cours sur M2. |
| 5     | `stock_raw`           | Quantité de MP en stock. |
| 6     | `stock_p1`            | Quantité de P1 en stock. |
| 7     | `stock_p2_inter`      | Quantité de P2_inter en stock. |
| 8     | `stock_p2`            | Quantité de P2 en stock. |
| 9     | `next_delivery_countdown` | Temps (minutes) jusqu’à la prochaine livraison de MP (0 si aucune prévue). |
| 10    | `backlog_p1`          | Backlog P1 (demandes P1 non servies). |
| 11    | `backlog_p2`          | Backlog P2 (demandes P2 non servies). |
| 12    | `q_total_en_route`    | Total de MP actuellement en transit dans la file de livraisons. |

### 10.2 Nouvelles variables (indices 13–22)

| Index | Nom                     | Description |
|-------|--------------------------|-------------|
| 13    | `current_action_type`   | Type d’action courante (celle du step en cours) :<br>0 = M1 produit P1<br>1 = M1 produit P2_STEP1<br>2 = M2 produit P2_STEP2<br>3 = commande de MP<br>4 = WAIT. |
| 14    | `current_action_k`      | Paramètre `k` associé à l’action courante (0 à 50) : nombre d’unités lancées ou commandées. |
| 15    | `current_action_id`     | Index brut de l’action (0 à 200), identique à celui passé par l’agent. |
| 16    | `minute_of_day`         | `time % 1440` : minute dans la journée (0 à 1439), permet de distinguer jour/nuit et la proximité du vol. |
| 17    | `expected_dem_p1_next`  | Reward « potentiel » lié au backlog P1 : `2 × backlog_p1`. Donne une idée du gain maximum possible en satisfaisant tout le backlog P1. |
| 18    | `expected_dem_p2_next`  | Reward potentiel lié au backlog P2 : `20 × backlog_p2`. Idem pour P2. |
| 19    | `theft_risk_level`      | 0 ou 1 : indicateur binaire de risque de vol. Par exemple, 1 si l’on se trouve dans l’heure précédant le vol quotidien, 0 sinon. |
| 20    | `reward_current_week`   | Reward cumulé depuis le début de l’épisode (7 jours). Permet de suivre la performance globale. |
| 21    | `reward_current_action` | Reward généré par le **dernier step**. Permet de relier le signal de reward immédiat à l’état observé. |
| 22    | `time_to_next_sell`     | Nombre de minutes restantes avant la prochaine vente (entre 0 et 15). |

---

## 11. Step : ordre des opérations

À chaque appel à `env.step(action)` :

1. Décodage de `action` → `current_action_type`, `current_action_k`, `current_action_id`.
2. Tentative de lancement d’un batch (P1, P2_STEP1, P2_STEP2) ou d’une commande MP, ou WAIT.  
   - Si l’action n’est pas faisable (machine occupée, stock insuffisant), `reward -= 1`.
3. Avancement d’une minute des machines (`tick()`) et production « au fil de l’eau ».
4. Traitement des livraisons de MP arrivées à `time`.
5. `time += 1` (passage à la minute suivante).
6. Si `time % 15 == 0` : génération de demande, mise à jour du backlog, ventes, reward de vente, pénalité backlog.
7. Si `time % 1440 == theft_time` : application du vol sur P1 et P2.
8. Mise à jour de `reward_current_week` et `reward_current_action`.
9. Construction et renvoi du vecteur d’observation (23 valeurs) + `reward`, `terminated`, `truncated=False`, `info={}`.

---

## 12. Fin d’épisode

L’épisode se termine quand :

```python
terminated = (time >= 10080)
```

Un reset remet toutes les variables internes à zéro (ou à leur état neutre) et reconstruit un nouvel épisode de 7 jours.

---

## 13. Résumé

La version « 23 variables » de l’environnement Workshop :

- conserve la structure de base (temps, stocks, backlogs, actions),
- ajoute un **contexte temporel fin** (minute dans la journée, temps avant la prochaine vente),
- met en évidence le **lien entre backlog et gains potentiels**,
- expose la **proximité du vol nocturne**,
- donne à l’agent une vue explicite sur l’**action courante** et le **reward associé**,
- reste totalement compatible avec des algos d’APRN type DQN, PPO, MaskablePPO.

C’est la **référence** à utiliser pour tout entraînement RL et pour la documentation associée (READ_ME, notebooks de test, etc.).
