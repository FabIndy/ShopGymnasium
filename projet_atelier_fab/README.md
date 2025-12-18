# Atelier de Production — Reinforcement Learning (DAgger + PPO)

---
# PART 1 — ENGLISH VERSION
---

## Overview

This repository presents a **simulated industrial production workshop** and a **complete Reinforcement Learning pipeline** designed to evaluate whether a learned RL agent can **outperform a strong heuristic expert policy** on a realistic production-planning problem.

The work follows a progressive and controlled methodology:

1. definition of an explicit and interpretable **Expert Policy v4**,
2. imitation learning using **DAgger**,
3. reinforcement learning fine-tuning using **safe PPO (MaskablePPO)**,
4. rigorous evaluation over **104 independent production weeks**.

The objective is not only raw performance, but a **methodologically sound comparison between Expert heuristics and Reinforcement Learning**.

---

## 1. Problem Description

The agent controls a **multi-machine production workshop** in order to:
- manufacture two products (P1, P2) through multiple processing steps,
- manage inventories, backlogs, and raw material orders with delivery delays,
- anticipate demand and penalties,
- **maximize cumulative reward over a full production week**.

Time discretization:
- **1 step = 1 minute**
- **1 episode = 7 days = 10,080 steps**

---

## 2. Workshop Environment

The workshop environment is implemented in the `env/` folder.

### Action Space (201 actions)

| Range | Action |
|------|--------|
| 0–49 | Produce P1 (k units) |
| 50–99 | Produce P2 – step 1 |
| 100–149 | Produce P2 – step 2 |
| 150–199 | Order raw material |
| 200 | WAIT |

Invalid actions are **strictly masked** using `env.get_action_mask()`.

### Observation Space (23 features)

The normalized observation vector includes:
- machine states and availability,
- inventory levels (raw material, P1, intermediate P2, final P2),
- P1 / P2 backlogs,
- raw material delivery pipeline,
- temporal variables,
- last-action information.

---

## 3. Expert Policy v4

**Expert v4** is a hand-crafted, deterministic and fully interpretable heuristic policy.

Key principles:
- economic priority given to product P2,
- explicit backlog management,
- raw material orders proportional to anticipated demand,
- strict enforcement of machine and inventory constraints,
- **conservative pre-production strategy** during low-load periods.

This policy does not learn; it encodes domain knowledge directly.

Typical weekly performance:
- **Average reward ≈ 13,300**.

It provides a strong baseline but remains inherently myopic.

---

## 4. Learning Pipeline

### 4.1 DAgger

Starting from Expert v4:
- expert trajectories are generated,
- a supervised policy is trained,
- iterative data aggregation with expert corrections is applied,
- action masking is enforced throughout.

The resulting model faithfully reproduces the expert behavior.

---

### 4.2 PPO v7 (MaskablePPO)

**PPO v7** is initialized from the DAgger policy and then improved using Reinforcement Learning.

Key design choices:
- **MaskablePPO** to guarantee valid actions,
- low learning rate,
- reduced entropy coefficient (controlled exploration),
- conservative clipping range,
- long training horizon (1,000,000 steps).

The goal is to allow **safe local improvements** beyond the expert strategy.

---

## 5. Evaluation: Expert v4 vs PPO v7

### Protocol

- **104 independent weeks**, never seen during training,
- identical random seeds for Expert and PPO,
- strict comparison on full 10,080-step episodes.

### Results

| Model | Mean reward | Std |
|------|------------|-----|
| Expert v4 | ≈ 13,311 | ≈ 553 |
| PPO v7 | ≈ 15,769 | ≈ 387 |

**Average PPO gain: +2,458 reward per week (+18%)**.

- PPO v7 outperforms the expert in **100% of test weeks**,
- improvements are stable and systematic.

---

## 6. Interpretation

PPO v7:
- reproduces most expert decisions,
- selectively deviates when beneficial,
- occasionally sacrifices short-term reward for **higher long-term cumulative gains**.

The observed improvement is therefore structural, not statistical noise.

---

## 7. Conclusion

This work demonstrates that:
- a strong heuristic expert can be **consistently improved using RL**,
- PPO v7 clearly outperforms Expert v4 on this task,
- the Expert → DAgger → PPO pipeline is **stable, interpretable, and effective**.

Importantly, **significant improvement potential remains**:
Expert v4 does **not explicitly account for processing times and long-term temporal effects**, whereas PPO begins to exploit these dynamics.

This opens the door to:
- temporally enriched expert policies,
- more exploratory RL agents,
- or hybrid approaches with even higher performance.

---

# PART 2 — VERSION FRANÇAISE
---

## Présentation générale

Ce dépôt présente un **atelier de production industriel simulé** et un **pipeline complet de Reinforcement Learning** visant à évaluer si un agent RL peut **dépasser une politique experte heuristique** sur un problème réaliste de pilotage de production.

La démarche est progressive et rigoureuse :

1. définition d’une **politique experte v4** explicite et interprétable,
2. apprentissage par imitation (**DAgger**),
3. amélioration par **PPO sécurisé (MaskablePPO)**,
4. évaluation rigoureuse sur **104 semaines indépendantes**.

L’objectif est une **comparaison honnête et argumentée Expert vs RL**.

---

## 1. Description du problème

L’agent contrôle un **atelier de production multi-machines** afin de :
- produire deux produits (P1, P2) en plusieurs étapes,
- gérer les stocks, backlogs et commandes de matières premières avec délais,
- anticiper la demande et les pénalités,
- **maximiser le reward cumulé sur une semaine complète**.

Cadre temporel :
- **1 step = 1 minute**
- **1 épisode = 7 jours = 10 080 steps**

---

## 2. Environnement Workshop

L’environnement est implémenté dans le dossier `env/`.

### Actions (201)

| Plage | Action |
|------|--------|
| 0–49 | Produire P1 |
| 50–99 | Produire P2 – étape 1 |
| 100–149 | Produire P2 – étape 2 |
| 150–199 | Commander la matière première |
| 200 | WAIT |

Les actions impossibles sont **strictement masquées**.

### Observations (23 variables)

Le vecteur d’état inclut notamment :
- états et disponibilités des machines,
- niveaux de stock,
- backlogs P1 / P2,
- pipeline de livraison,
- variables temporelles,
- dernière action.

---

## 3. Politique experte v4

La **politique experte v4** est une heuristique déterministe et interprétable.

Principes :
- priorité économique à P2,
- gestion explicite des backlogs,
- commandes proportionnelles à la demande,
- respect strict des contraintes,
- stratégie prudente en période creuse.

Performance moyenne :
- **Reward ≈ 13 300 par semaine**.

Elle constitue une référence solide mais **myope temporellement**.

---

## 4. Pipeline d’apprentissage

### 4.1 DAgger

- génération de trajectoires expertes,
- apprentissage supervisé,
- agrégation itérative des données,
- masquage strict des actions.

---

### 4.2 PPO v7

Le **PPO v7** est affiné à partir du modèle DAgger.

Choix clés :
- MaskablePPO,
- exploration contrôlée,
- entraînement long.

---

## 5. Résultats

| Modèle | Reward moyen | Écart-type |
|------|-------------|-----------|
| Expert v4 | ≈ 13 311 | ≈ 553 |
| PPO v7 | ≈ 15 769 | ≈ 387 |

**Gain moyen : +2 458 par semaine (+18%)**.

PPO v7 est supérieur à l’expert **dans 100 % des cas testés**.

---

## 6. Interprétation

PPO v7 apprend à :
- suivre l’expert lorsque c’est optimal,
- s’en écarter lorsque le gain cumulé futur est supérieur.

---

## 7. Conclusion

Ce travail montre que le RL permet :
- d’améliorer une politique experte robuste,
- d’exploiter des effets temporels ignorés par l’expert v4,
- d’obtenir une politique plus performante et plus générale.

La **marge de progression reste importante**, notamment en intégrant explicitement les durées et horizons temporels.

---

Projet RL orienté **rigueur méthodologique, interprétabilité et décision industrielle**.
