
# README — Atelier de Production (Version RL avancée, 23 variables)

Ce document décrit l'état **corrigé, actualisé et entièrement aligné** avec la dernière version de la spécification officielle de l’environnement Workshop (23 variables).  
Cette version remplace complètement l’ancien README basé sur 13 variables.  
Référence : *workshop_environnement_specification.md*.

---

# 1. Objectif général

L’environnement simule un **atelier industriel multitâche** dans lequel un agent RL doit :

- planifier la production de deux produits (P1, P2) répartis sur **deux machines indépendantes** ;
- gérer un **carnet de commandes résiduel (backlog)** mis à jour toutes les 15 minutes ;
- commander de la matière première (MP) avec délai (120 ± 2 minutes) ;
- maintenir des niveaux de stocks pertinents malgré un **vol nocturne quotidien** ;
- maximiser le **reward cumulé sur 7 jours (10080 minutes)**.

L’agent peut utiliser DQN, PPO, MaskablePPO ou tout autre algorithme compatible Gymnasium.

---

# 2. Structure du code (modules)

```
env/
│
├── machines.py         → logique Machine M1/M2, batches asynchrones
├── stock.py            → gestion des stocks P1 / P2 / MP / P2_inter
├── delivery.py         → file FIFO de livraisons de matières premières
├── market.py           → génération demande, backlog, ventes
└── workshop_env.py     → environnement RL complet Gymnasium (23 variables)
```

Tous ces modules collaborent pour produire une simulation cohérente et réaliste.

---

# 3. Dynamique temporelle

- **1 step = 1 minute**
- **1 journée = 1440 minutes**
- **1 épisode = 7 jours = 10080 steps**
- **Ventes : toutes les 15 minutes**
- **Vol nocturne : minute 1435 chaque jour**

À chaque step :

1. L’agent choisit une action parmi 201.
2. Les machines avancent d’une minute.
3. Les productions terminées ajoutent des unités instantanément.
4. Les livraisons planifiées arrivent si leur minute correspond.
5. Toutes les 15 minutes : demande → backlog → ventes → reward.
6. Si minute 1435 : vol sur P1 et P2.
7. L’environnement renvoie l’observation (23 valeurs), le reward et terminated/truncated.

---

# 4. Machines — Production multi-étapes

## M1
- Produit **P1** : durée `3 × k`
- Produit **P2_step1** : durée `10 × k`

## M2
- Produit **P2_step2** : durée `15 × k`

### Production « au fil de l’eau »
Les machines produisent **une unité par minute**, pas uniquement à la fin du batch.

---

# 5. Stocks

- MP (`stock_raw`)
- P1 (`stock_p1`)
- P2 intermédiaire (`stock_p2_inter`)
- P2 final (`stock_p2`)

Chaque stock est borné logiquement autour de 50 unités.

---

# 6. Commandes de matières premières (MP)

Actions **150 à 199** :

```
q = action - 149
coût immédiat = -q
livraison dans 120 ± 2 minutes
```

Livraisons gérées par une file FIFO.

---

# 7. Demande, backlog et ventes

### Demande
Toutes les 15 minutes, génération jour/nuit de `(new_d1, new_d2)` qui sont ajoutées au backlog.

### Backlog résiduel
Le backlog représente la **demande non servie**, jamais remis à zéro.

### Ventes
```
sold_p1 = min(stock_p1, backlog_p1)
sold_p2 = min(stock_p2, backlog_p2)
reward += 2 * sold_p1 + 20 * sold_p2
backlog -= sold
```

### Pénalité backlog
```
reward -= 0.02 * (backlog_p1 + backlog_p2)
```

---

# 8. Vol nocturne

Chaque jour à minute 1435 :
```
stock_p1 = floor(0.9 * stock_p1)
stock_p2 = floor(0.9 * stock_p2)
```

Une variable d’état (`theft_risk_level`) indique la proximité de l’événement.

---

# 9. Espace d’actions (201 actions)

| Actions | Signification | k |
|--------|---------------|---|
| 0–49 | P1 sur M1 | k = a + 1 |
| 50–99 | P2 step1 sur M1 | k = a − 49 |
| 100–149 | P2 step2 sur M2 | k = a − 99 |
| 150–199 | Commander MP | k = a − 149 |
| 200 | WAIT | 0 |

Actions impossibles → WAIT et pénalité.

---

# 10. Observation — **23 variables**

Les 13 premières variables correspondent à l’ancien état.  
Les 10 suivantes enrichissent fortement le contexte temporel, commercial et actionnel.

### 10.1 Anciennes variables (0–12)

1. time  
2. m1_busy  
3. m1_time_left  
4. m2_busy  
5. m2_time_left  
6. stock_raw  
7. stock_p1  
8. stock_p2_inter  
9. stock_p2  
10. next_delivery_countdown  
11. backlog_p1  
12. backlog_p2  
13. q_total_en_route  

### 10.2 Nouvelles variables (13–22)

| idx | Nom | Description |
|-----|------|-------------|
| 13 | current_action_type | (0=P1, 1=P2_step1, 2=P2_step2, 3=command, 4=WAIT) |
| 14 | current_action_k | quantité produite/commandée |
| 15 | current_action_id | action brute ∈ [0,200] |
| 16 | minute_of_day | time mod 1440 |
| 17 | expected_dem_p1_next | 2 × backlog_p1 |
| 18 | expected_dem_p2_next | 20 × backlog_p2 |
| 19 | theft_risk_level | 0/1 selon proximité du vol |
| 20 | reward_current_week | reward cumulatif |
| 21 | reward_current_action | reward du dernier step |
| 22 | time_to_next_sell | minutes restantes avant prochaine vente |

---

# 11. Reward

- +2 × ventes P1  
- +20 × ventes P2  
- − backlog × 0.02  
- +0.5 × k pour batch P1  
- +5 × k pour batch P2_step1  
- +15 × k pour batch P2_step2  
- −q pour commandes MP  
- −0.2 pour WAIT  
- −1 si action impossible  

---

# 12. Compatibilité RL

Compatible **Gymnasium**, utilisable avec :

- SB3 (DQN, PPO, MaskablePPO)
- Ray RLlib
- CleanRL

Recommandations :

- normalisation des observations,
- monitoring backlog + stocks,
- analyser les rewards cumulés/semaine,
- utiliser des action masks.

---

# 13. Résumé

Cette nouvelle version **23 variables** :

- fournit un état véritablement Markovien riche,
- améliore la compréhension temporelle et la gestion du risque,
- relie backlog ↔ reward potentiel,
- expose explicitement la dernière action et son impact,
- permet un RL plus performant et plus stable.

Elle constitue désormais **LA version officielle du Workshop Environment**.



---

## Normalisation des variables d’observation

Dans la version actuelle de `WorkshopEnv` (23 variables), **toutes les composantes du vecteur d’observation sont normalisées** avant d’être renvoyées à l’agent RL.  
L’objectif est d’obtenir des valeurs numériques de même ordre de grandeur (typiquement dans `[0, 1]`) afin de :

- stabiliser l’optimisation (PPO, DQN, etc.) ;
- éviter que certaines features dominent les autres uniquement par leur échelle ;
- faciliter l’agrégation de données pour DAgger et l’apprentissage supervisé.

De façon générale :

- le temps courant est divisé par `max_time` (7 jours en minutes) ;
- les temps restants sur machines sont divisés par une borne maximale fixe (par exemple 100 minutes) ;
- les stocks (MP, P1, P2_inter, P2) sont divisés par leur capacité maximale théorique ;
- les backlogs sont divisés par `1000` (borne haute choisie pour les demandes) ;
- la quantité de MP en livraison (`q_raw_incoming`) est divisée par `1000` ;
- certaines grandeurs de type « score cumulé » (ex. `week_reward`) sont, quand elles sont exposées, divisées par une borne large (par ex. `1e6`).

Cette normalisation est réalisée directement dans la méthode `_get_obs()` de `workshop_env.py`.  
Elle ne change pas la dynamique interne de l’atelier (les variables internes restent en unités « réelles ») mais **uniquement la représentation vue par l’agent RL**.

En pratique, cette étape :

- rend l’entraînement MaskablePPO plus stable,
- facilite l’utilisation d’un même environnement pour DAgger (imitation supervisée) et PPO,
- prépare le terrain pour des variantes de modèles (réseaux plus profonds, autres algos RL).

---

## Synthèse des expériences DAgger + MaskablePPO (notebook `dagger_hybrid`)

Un notebook séparé, `dagger_hybrid`, explore un pipeline complet :

1. **Masque d’actions** : usage systématique de `env.get_action_mask()` et de `MaskablePPO` pour interdire les actions impossibles.
2. **Expert v2** : politique heuristique très performante, obtenant un reward d’environ **+12 729** sur un épisode de 7 jours avec les 23 variables normalisées.
3. **DAgger (Dataset Aggregation)** : apprentissage supervisé de l’élève sur les trajectoires expertes, puis collecte de nouvelles données où l’élève joue et l’expert corrige.
4. **PPO** : dans la version complète, PPO doit affiner la politique après l’imitation. Dans la phase de diagnostic décrite ici, **PPO a été désactivé** pour isoler le comportement de DAgger.

Les résultats clés observés à partir des cellules 4 et 10 du notebook sont :

- Dataset expert initial :  
  - `obs shape : (10080, 23)`  
  - `actions shape : (10080,)`  
  - `Reward expert : 12729.30` (7 jours, 10080 steps)

- DAgger imitation seule (PPO désactivé) sur deux itérations :
  - la CrossEntropyLoss diminue légèrement (≈ 5.26 → 4.16) mais reste très élevée ;
  - les rewards de l’élève restent fortement négatifs (entre **-6300** et **-5900** sur 7 jours) ;
  - même après agrégation de 30 240 exemples, l’élève n’approche pas le niveau de l’expert.

**Conclusion du diagnostic :**

- le masque d’actions et l’expert v2 fonctionnent correctement ;
- le problème observé ne vient pas de PPO (mis de côté dans cette phase) ;
- l’échec de DAgger dans cette configuration vient principalement de la **difficulté de l’élève à approximer la politique experte à partir des 23 variables d’observation**, dont une partie est redondante ou peu informative pour la décision.

**Piste de travail identifiée :**

La prochaine étape consiste à **réduire le vecteur d’observation** exposé à l’élève, en se concentrant sur un sous-ensemble de variables réellement utilisées par l’expert (état des machines, stocks, backlogs, livraisons MP, etc.).  
L’objectif est de rendre la tâche d’imitation plus simple et mieux conditionnée, puis de :

1. relancer DAgger en mode imitation seule avec ce vecteur réduit ;
2. une fois l’élève capable d’imiter correctement l’expert, réactiver PPO dans la boucle ;
3. éventuellement réintroduire certaines variables supplémentaires si elles apportent un gain mesurable.

Ce README, combiné au rapport détaillé `dagger_hybrid_report.md`, sert de référence globale pour l’état actuel de l’environnement Workshop et des premières expérimentations RL avancées.
