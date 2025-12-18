# Workshop Environment Specification — Advanced RL Version (23 variables)

---
# PART 1 — ENGLISH VERSION
---

## Overview

This document is the **official specification** of the industrial *Workshop* environment used for Reinforcement Learning experiments (DQN, PPO, MaskablePPO, etc.).  
It exactly matches the latest implementation of `WorkshopEnv` in `workshop_env.py`, using a **23-dimensional state representation**.

---

## 1. Time Structure

- **1 step = 1 minute**
- 1 day = 1,440 minutes
- 1 episode = **7 days = 10,080 minutes**
- Sales occur **every 15 minutes**
- A **theft event** occurs **once per day**, at minute 1435 of each day

At each step, the agent selects an action, machines advance, deliveries are processed, demand/sales may occur, a theft may happen, and a new observation is returned.

---

## 2. Resources and Products

### 2.1 Stock Types

- `stock_raw`: available raw material
- `stock_p1`: finished product P1
- `stock_p2_inter`: intermediate product for P2 (step 1)
- `stock_p2`: finished product P2

Nominal stock capacity in the code is approximately **50 units** per stock (logical bounds for observations).

---

## 3. Machines

### 3.1 Machine M1

Transforms raw material into:
- **P1**
- **P2_STEP1** (intermediate for P2)

For a batch of size `k`:

| Batch type | Input | Output | Total duration |
|-----------|-------|--------|----------------|
| P1_MULTI | 1 RM per unit | 1 P1 per unit | `3 × k` minutes |
| P2STEP1_MULTI | 1 RM per unit | 1 P2_inter per unit | `10 × k` minutes |

Internal state: `busy`, `time_left`, `batch_type`, `batch_k`.

### 3.2 Machine M2

Transforms `P2_inter` into finished **P2**:

| Batch type | Input per unit | Output per unit | Total duration |
|------------|----------------|-----------------|----------------|
| P2STEP2_MULTI | 1 P2_inter | 1 P2 | `15 × k` minutes |

### 3.3 Continuous Production

Time advances **minute by minute**. Each minute, machines decrement `time_left` and may **produce one unit** before batch completion.

Stocks are therefore filled **progressively**, not only at batch completion.

---

## 4. Raw Material Orders

Actions **150 to 199** correspond to raw material orders:

- `q = action - 149`
- immediate reward cost: `reward -= q`
- delivery scheduled at `120 ± 2` minutes
- stored in a `DeliveryQueue` as `(quantity, arrival_time)`

Delivered quantities are added to `stock_raw` when arrival time is reached.

---

## 5. Demand, Backlog, and Sales

### 5.1 Backlog

Two backlogs are tracked:
- `demande_p1`
- `demande_p2`

They represent unsatisfied customer demand.

### 5.2 Demand Generation

Every **15 minutes**, new demand is generated (Poisson day/night model):

```python
new_d1, new_d2 = market.sample_demand(time, 15)
demande_p1 += new_d1
demande_p2 += new_d2
```

### 5.3 Sales

At the same time:

```python
sold_p1 = min(stock_p1, demande_p1)
sold_p2 = min(stock_p2, demande_p2)
```

Rewards:
- `+2` per P1 sold
- `+20` per P2 sold

### 5.4 Backlog Penalty

```python
reward -= 0.02 * (demande_p1 + demande_p2)
```

---

## 6. Night Theft

Once per day, at minute 1435:

```python
stock_p1 = floor(stock_p1 * 0.9)
stock_p2 = floor(stock_p2 * 0.9)
```

A state variable `theft_risk_level` indicates proximity to this event.

---

## 7. Reward Structure

Reward components:
- Sales revenue
- Raw material order cost
- Production launch bonuses
- WAIT penalty
- Invalid action penalty
- Backlog penalty

Two state variables expose reward dynamics:
- `reward_current_week`
- `reward_current_action`

---

## 8. Action Space (201)

| Range | Meaning |
|------|--------|
| 0–49 | Launch P1 batch on M1 |
| 50–99 | Launch P2_STEP1 batch on M1 |
| 100–149 | Launch P2_STEP2 batch on M2 |
| 150–199 | Order raw material |
| 200 | WAIT |

---

## 9. Observation Space (23 variables)

The first 13 variables correspond exactly to the legacy environment.  
10 additional variables provide temporal context, reward linkage, theft risk, and action awareness.

This enriched state improves long-horizon decision making.

---

## 10. Episode Termination

```python
terminated = (time >= 10080)
```

---

## Conclusion (EN)

This environment provides a **realistic, temporally rich, and constrained** setting for advanced RL methods.  
It is the **reference implementation** for all experiments, evaluations, and documentation.

---
# PART 2 — VERSION FRANÇAISE
---

## Présentation générale

Ce document constitue la **spécification officielle** de l’environnement industriel *Workshop* utilisé pour l’apprentissage par renforcement (DQN, PPO, MaskablePPO, etc.).  
Il correspond exactement à la dernière version de `WorkshopEnv` implémentée dans `workshop_env.py`, avec un **état de dimension 23**.

---

## 1. Structure temporelle

- **1 step = 1 minute**
- 1 journée = 1 440 minutes
- 1 épisode = **7 jours = 10 080 minutes**
- Les ventes ont lieu **toutes les 15 minutes**
- Un **vol nocturne** a lieu **une fois par jour**, à la minute 1435

---

## 2. Ressources et produits

Stocks :
- matières premières
- produits finis P1
- produits intermédiaires P2
- produits finis P2

---

## 3. Machines

M1 et M2 transforment les ressources selon des durées proportionnelles à la taille des batches.  
La production est **progressive, minute par minute**, et non instantanée.

---

## 4. Commandes de matières premières

Les commandes génèrent un coût immédiat et une livraison différée via une file dédiée.

---

## 5. Demande, backlog et ventes

La demande est générée périodiquement, les ventes réduisent le backlog et génèrent du reward, et un coût pénalise les demandes non satisfaites.

---

## 6. Vol nocturne

Un vol partiel quotidien affecte les stocks P1 et P2, avec un indicateur d’état associé.

---

## 7. Reward

Le reward agrège ventes, coûts, bonus de production, pénalités WAIT et backlog.

---

## 8. Espace d’actions

201 actions discrètes avec masquage strict des actions impossibles.

---

## 9. Observation (23 variables)

L’état enrichi apporte :
- contexte temporel fin,
- lien backlog ↔ gain potentiel,
- risque de vol,
- information explicite sur l’action et le reward.

---

## 10. Fin d’épisode

L’épisode se termine après 7 jours simulés.

---

## Conclusion (FR)

L’environnement Workshop constitue un **cadre réaliste et exigeant** pour l’apprentissage par renforcement industriel.  
Il sert de **référence unique** pour tous les entraînements, analyses et comparaisons Expert vs RL.
