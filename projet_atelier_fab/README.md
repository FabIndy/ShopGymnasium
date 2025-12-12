# Atelier de Production — Reinforcement Learning (DAgger + PPO)

Ce dépôt présente un **environnement industriel simulé** et un **pipeline RL avancé** combinant :
- une **politique experte heuristique**,
- un apprentissage par imitation **DAgger**,
- un **finetuning PPO sécurisé (MaskablePPO)**,
- et une **analyse fine des divergences Expert vs PPO**.

L’objectif est de montrer **comment le RL peut améliorer une politique experte existante**, de manière contrôlée et interprétable.

---

## 1. Objectif du projet

L’agent pilote un **atelier de production multi-machines** afin de :
- produire deux produits (P1, P2) en plusieurs étapes,
- gérer stocks, backlogs et matières premières avec délais,
- anticiper la demande et les pénalités,
- **maximiser le reward cumulé sur 7 jours (10 080 minutes)**.

Le projet met l’accent sur :
- la **stabilité de l’apprentissage**,
- la **validité des actions** (masques),
- l’**interprétabilité des décisions RL**.

---

## 2. Environnement Workshop (23 variables)

Fichiers .py dedies a l'implementation de l'atelier : dans le dossier `env`

### Dynamique temporelle
- **1 step = 1 minute**
- **1 épisode = 7 jours = 10 080 steps**
- ventes toutes les 15 minutes
- vol nocturne quotidien

### Actions (201)
| Plage   | Action                |
|---------|-----------------------|
| 0–49    | Produire P1 (k = a+1) |
| 50–99   | Produire P2 step1     |
| 100–149 | Produire P2 step2     |
| 150–199 | Commander MP          |
| 200     | WAIT                  |

Les actions impossibles sont **masquées** via `env.get_action_mask()`.

### Observation (23 variables normalisées)
Inclut :
- état machines,
- stocks (MP, P1, P2_inter, P2),
- backlogs P1 / P2,
- pipeline MP,
- variables temporelles,
- reward courant et hebdomadaire,
- dernière action.

---

## 3. Politique experte

Notebook dédié : `dagger_hybrid` (definition de la fonction experte)

Une **politique heuristique v3** sert de référence :
- priorisation économique de P2,
- gestion dynamique des backlogs,
- commandes MP proportionnelles à la demande,
- respect strict des contraintes.

Reward expert typique :
- **≈ 12 900** sur 7 jours.

---

## 4. Pipeline d’apprentissage

### 4.1 DAgger

Notebooks dédiés : 
- `dagger_hybrid` (creation du modele dagger)
- `ppo_finetune` (evaluation du modele dagger)
- `analyse_dagger` (courbes d'entrainement) 

Synthese du contenu de  `dagger_hybrid` :
- génération de trajectoires expertes,
- apprentissage supervisé,
- collecte itérative avec corrections expertes,
- action masking systématique.

### 4.2 PPO safe (MaskablePPO)

Notebook dédié : `ppo_safe_finetune`

- initialisation à partir du modèle DAgger,
- learning rate faible,
- exploration limitée,
- objectif : **ajustements locaux non destructifs**.

---

## 5. Analyse Expert vs PPO

Notebook dédié : `analyse_expert_vs_ppo` + voir le dossier csv

### Résultats clés
- Reward expert : **12 916**
- Reward PPO : **13 113**
- **Gain PPO : +1,5 %**
- Taux de divergence : **19,5 %** (dans 19.5% des actions d'une semaine, PPO decide uen action differente de l'expert)

Le PPO reproduit ~80 % des décisions expertes et s’en affranchit de manière ciblée.

### Nature des divergences
- timing des commandes MP,
- arbitrage production / WAIT,
- priorisation de P2.

Le PPO accepte parfois un **coût immédiat** pour un **gain cumulé supérieur**.

---

## 6. Conclusion

Le pipeline **Expert → DAgger → PPO** :
- ne dégrade pas la politique experte,
- introduit des ajustements stables et utiles,
- améliore le reward global de façon mesurable.

---

## 7. Perspectives

- tester d'autres PPO en variant le nombre de steps,
- analyse comparative approfondie entre expert et modele PPO,
- comparaison avec d’autres algorithmes RL.

---

Projet RL expérimental orienté **stabilité, interprétabilité et décision industrielle**.
