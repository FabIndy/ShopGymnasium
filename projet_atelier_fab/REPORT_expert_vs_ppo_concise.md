# Expert vs PPO — Synthèse d’analyse (post-DAgger)

## Contexte
Cette analyse de `cmp_divergences_only.csv` compare la **politique experte heuristique v3** à un **agent PPO finetuné via DAgger** dans un environnement industriel simulé (stocks, backlogs, machines, délais).
L’objectif est d’évaluer **le degré de divergence** du PPO et **l’impact réel sur le reward**.

---

## Résultats clés

- **Durée analysée** : 1 semaine (10 080 décisions)
- **Reward expert** : 12 916  
- **Reward PPO** : 13 113  
- **Gain PPO** : **+1,5 %**

- **Décisions divergentes** : 1 961  
- **Taux de divergence** : **19,5 %**

Le PPO reproduit ~80 % des décisions expertes et s’en affranchit de manière ciblée.

---

## Analyse des divergences

![Proportion des divergences]

Les divergences concernent principalement :
- le **timing des commandes de matières premières**,
- l’arbitrage **production vs attente (WAIT)**,
- la **priorisation du produit P2**, plus rentable que P1.

![Top divergences]

Le PPO accepte parfois un **coût immédiat** (reward négatif local) afin d’obtenir un **gain cumulé supérieur** à l’échelle hebdomadaire, ce qui correspond au comportement attendu d’un agent RL optimisant le long terme.

---

## Conclusion

Le PPO :
- ne copie pas aveuglément l’expert,
- ne dégrade pas la politique existante,
- apporte des **ajustements locaux pertinents**,
- améliore le reward global de façon mesurable.



---

