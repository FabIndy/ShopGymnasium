#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cellule 1 — Imports et configuration de base

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from sb3_contrib import MaskablePPO
from env.workshop_env import WorkshopEnv

# === Historique des métriques DAgger ===
history = {
    "supervised_loss": [],   # perte moyenne par itération DAgger
    "reward_collect": [],    # reward élève pendant la phase de collecte
    "reward_eval": []        # reward élève en évaluation (7 jours)
}


# Pour rendre les choses un peu reproductibles
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# In[3]:


# Cellule 2 — Politique experte v2 compatible obs=23 variables + normalisation

from env.workshop_env import WorkshopEnv
import numpy as np

def expert_policy(obs: np.ndarray, env: WorkshopEnv) -> int:
    """
    Politique experte v2 avec obs normalisées (23 features).

    On reconstruit d'abord les variables "réelles" à partir de l'observation
    normalisée, en inversant exactement les formules de WorkshopEnv._get_obs(),
    puis on applique la même logique experte qu'avant.

    Rappel de la normalisation dans WorkshopEnv._get_obs() :
      - time              -> obs[0] = time / max_time
      - m1_busy           -> obs[1] = m1.busy
      - m1_time_left      -> obs[2] = m1.time_left / 100
      - m2_busy           -> obs[3] = m2.busy
      - m2_time_left      -> obs[4] = m2.time_left / 100
      - stock_raw         -> obs[5] = stock.raw / raw_capacity
      - stock_p1          -> obs[6] = stock.p1 / raw_capacity
      - stock_p2_inter    -> obs[7] = stock.p2_inter / raw_capacity
      - stock_p2          -> obs[8] = stock.p2 / raw_capacity
      - next_delivery_cd  -> obs[9] = next_delivery_countdown / 10080
      - demande_p1        -> obs[10] = demande_p1 / 1000
      - demande_p2        -> obs[11] = demande_p2 / 1000
      - q_raw_incoming    -> obs[12] = q_total / 1000
      - obs[13:23] = features supplémentaires (non utilisées ici)
    """

    # ============================
    # 1) Dé-normalisation
    # ============================

    # Temps
    time = float(obs[0]) * float(env.max_time)

    # États des machines
    m1_busy = int(round(obs[1]))
    m1_time_left = float(obs[2]) * 100.0

    m2_busy = int(round(obs[3]))
    m2_time_left = float(obs[4]) * 100.0

    # Stocks (raw_capacity est connu dans l'env)
    stock_raw = float(obs[5]) * float(env.raw_capacity)
    stock_p1 = float(obs[6]) * float(env.raw_capacity)
    stock_p2_inter = float(obs[7]) * float(env.raw_capacity)
    stock_p2 = float(obs[8]) * float(env.raw_capacity)

    # Prochaine livraison (normalisée sur 10080)
    next_delivery_cd = float(obs[9]) * 10080.0

    # Backlogs et MP en route (échelle 1000 comme dans _get_obs)
    demande_p1 = float(obs[10]) * 1000.0
    demande_p2 = float(obs[11]) * 1000.0
    q_raw_incoming = float(obs[12]) * 1000.0

    # Les 10 features supplémentaires obs[13:23] sont ignorées pour l’expert v2.

    # Flags machine libres
    m1_free = (m1_busy == 0)
    m2_free = (m2_busy == 0)

    # ============================
    # 2) Logique experte d'origine
    # ============================

    # Règle 1 : si peu de MP (stock + en route), on commande
    if stock_raw + q_raw_incoming < 20:
        q_cmd = 10
        return 149 + q_cmd  # action de commande MP (k = q_cmd)

    # Règle 2 : si M2 libre et on a du P2_inter, on fait STEP2 pour finir P2
    if m2_free and stock_p2_inter > 0:
        k = min(int(stock_p2_inter), 5)
        if k > 0:
            return 99 + k  # action P2_STEP2 (k)

    # Règle 3 : si M1 libre, demande P2 > 0 et MP dispo → STEP1 P2
    if m1_free and demande_p2 > 0 and stock_raw > 0:
        k = min(int(demande_p2), int(stock_raw), 5)
        if k > 0:
            return 49 + k  # action P2_STEP1 (k)

    # Règle 4 : sinon, si M1 libre et demande P1 > 0 → produire P1
    if m1_free and demande_p1 > 0 and stock_raw > 0:
        k = min(max(int(demande_p1), 1), int(stock_raw), 5)
        return k - 1       # action P1 (k = action+1)

    # Règle 5 : sinon WAIT
    return 200


# In[5]:


# Cellule 3 — Politique experte masquée + fonction d'épisode expert

def expert_policy_masked(env: WorkshopEnv, obs: np.ndarray) -> int:
    """
    Version masquée de la politique experte :
    - récupère le mask via env.get_action_mask()
    - propose une action via expert_policy(obs, env)
    - si l'action est invalide, on la remplace par une action valide
    """

    mask = env.get_action_mask()  # bool array de taille 201
    a = expert_policy(obs, env)   # <<< on passe maintenant env à la politique experte

    # Si l'action proposée est invalide, on corrige
    if not mask[a]:
        # Priorité : WAIT si autorisé (200)
        if mask[200]:
            return 200
        # Sinon, on prend la première action valide
        valid_actions = np.nonzero(mask)[0]
        if len(valid_actions) == 0:
            # Cas pathologique : aucune action valide
            return 200
        return int(valid_actions[0])

    return a


def run_expert_episode(env: WorkshopEnv, max_steps: int = 10080):
    """
    Joue un épisode complet avec l'expert masqué.
    Renvoie :
      - obs_array : (T, 23)  # 23 features normalisées
      - act_array : (T,)
      - total_reward
      - nb_steps
    """
    obs, info = env.reset()
    obs_list = []
    act_list = []
    total_reward = 0.0

    for t in range(max_steps):
        action = expert_policy_masked(env, obs)
        obs_list.append(obs.copy())
        act_list.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    obs_array = np.stack(obs_list, axis=0)
    act_array = np.array(act_list, dtype=np.int64)

    return obs_array, act_array, total_reward, t + 1


# In[7]:


# Cellule 4 — Génération du dataset expert initial

env_expert = WorkshopEnv()
expert_obs, expert_actions, R_expert, T_expert = run_expert_episode(env_expert)

print("===== Dataset expert initial =====")
print("obs shape :", expert_obs.shape)
print("actions shape :", expert_actions.shape)
print("Reward expert sur cet épisode :", R_expert)
print("Steps joués :", T_expert)

# Initialisation du buffer DAgger
dagger_obs = expert_obs.copy()
dagger_actions = expert_actions.copy()

print("\nBuffer DAgger initialisé :")
print("dagger_obs :", dagger_obs.shape)
print("dagger_actions :", dagger_actions.shape)


# In[9]:


# Cellule 5 — Environnement avec ActionMasker + Modèle MaskablePPO

import gymnasium as gym
from sb3_contrib.common.wrappers import ActionMasker

# 1) Fonction de masquage (appelée automatiquement par ActionMasker)
def mask_fn(env):
    return env.get_action_mask()

# 2) Environnement enveloppé
env_student = ActionMasker(WorkshopEnv(), mask_fn)

# 3) Modèle MaskablePPO
from sb3_contrib import MaskablePPO

model_student = MaskablePPO(
    policy="MlpPolicy",          # IMPORTANT : pas MultiInputPolicy
    env=env_student,
    verbose=1,
    device="cuda",
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    n_steps=4096,
    batch_size=512,
    clip_range=0.2,
    ent_coef=0.01,
    max_grad_norm=0.5,
    tensorboard_log="./tb_dagger_hybrid"
)

print("Modèle élève initialisé (MaskablePPO + GPU + ActionMasker OK).")


# In[11]:


# ============================================================
# CELLULE 6 — Entraînement supervisé de l'élève (corrigée)
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim

def train_student_supervised(model, obs_array, act_array, epochs=5, batch_size=512):
    """
    Entraînement supervisé : l'élève imite l'expert sur tout le buffer DAgger.
    - obs_array shape = (N, 23)
    - act_array shape = (N,)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = model.policy.to(device)

    X = torch.tensor(obs_array, dtype=torch.float32, device=device)
    y = torch.tensor(act_array, dtype=torch.long, device=device)

    # Sécurité : empêcher tout label hors [0, 200]
    y = torch.clamp(y, 0, policy.action_space.n - 1)

    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    N = len(X)
    nb_batches = (N + batch_size - 1) // batch_size

    print(f"  → Entraînement supervisé sur {N} exemples ({nb_batches} batches)")

    avg_epoch_loss = 0.0

    for epoch in range(epochs):
        perm = torch.randperm(N)
        Xb = X[perm]
        yb = y[perm]

        total_loss = 0.0

        for i in range(nb_batches):
            start = i * batch_size
            end = min(start + batch_size, N)

            xb_i = Xb[start:end]
            yb_i = yb[start:end]

            # Extraction SB3 correcte
            features = policy.extract_features(xb_i)
            pi_latent, _ = policy.mlp_extractor(features)
            logits = policy.action_net(pi_latent)

            loss = loss_fn(logits, yb_i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / nb_batches
        avg_epoch_loss = epoch_loss
        print(f"    Epoch {epoch+1}/{epochs} — loss = {epoch_loss:.4f}")

    print("  ✓ Entraînement supervisé terminé.")
    return avg_epoch_loss




# In[13]:


# Cellule 7 — Test de l'élève sur une semaine (version MlpPolicy)

def run_student_episode(env, model, max_steps: int = 10080):
    obs, info = env.reset()
    total_reward = 0.0

    for t in range(max_steps):
        # On donne simplement l'observation brute au modèle
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    return total_reward, t + 1


# In[15]:


# ============================================================
# CELLULE 8 — Collecte des données DAgger 
# ============================================================

def collect_dagger_data_from_student(model, max_steps=10080):
    """
    L'élève joue un épisode complet.
    L'expert corrige chaque action.
    On renvoie :
    - obs_list  : toutes les observations rencontrées
    - act_list  : actions expertes correspondantes
    - total_reward_student : reward cumulé du student
    - nb_steps_student     : nombre de steps joués
    """

    env = WorkshopEnv()
    obs, info = env.reset()

    obs_list = []
    act_list = []

    total_reward = 0.0

    for t in range(max_steps):

        #  Très important : récupérer le MASQUE
        mask = env.get_action_mask()

        #  predict() DOIT recevoir action_masks pour éviter actions hors borne
        action_student, _ = model.predict(
            obs,
            deterministic=True,
            action_masks=mask
        )

        # L'expert corrige l'action
        action_expert = expert_policy_masked(env, obs)

        obs_list.append(obs.copy())
        act_list.append(action_expert)

        obs, reward, terminated, truncated, info = env.step(action_student)
        total_reward += reward

        if terminated or truncated:
            break

    return (
        np.array(obs_list, dtype=np.float32),
        np.array(act_list, dtype=np.int64),
        float(total_reward),
        t + 1
    )


# In[17]:


# Cellule 9 — Une itération DAgger imitation-seule (SANS PPO)

def dagger_hybrid_iteration(
    model,
    dagger_obs,
    dagger_actions,
    supervised_epochs: int = 3,
    rl_timesteps: int = 10_000
):
    """
    VERSION DIAGNOSTIC : imitation supervisée SEULE
    (on désactive PPO pour voir si le modèle apprend réellement les actions expertes).
    Cette version enregistre les métriques dans le dict global `history`.
    """

    print("\n===== Phase 1 : Imitation supervisée sur buffer DAgger =====")
    loss_value = train_student_supervised(model, dagger_obs, dagger_actions, epochs=supervised_epochs)
    history["supervised_loss"].append(loss_value)

    print("\n===== Phase 2 : (désactivée) =====")
    print("⚠ PPO désactivé volontairement pour test de diagnostic.")

    print("\n===== Phase 3 : Collecte DAgger (élève + expert) =====")
    new_obs, new_actions, R_student, steps_student = collect_dagger_data_from_student(model)
    history["reward_collect"].append(R_student)

    print(f"Reward élève pendant collecte DAgger : {R_student:.2f} sur {steps_student} steps")

    # Agrégation au buffer
    dagger_obs = np.vstack([dagger_obs, new_obs])
    dagger_actions = np.concatenate([dagger_actions, new_actions])

    print("Taille du buffer DAgger après agrégation :", dagger_obs.shape)

    return dagger_obs, dagger_actions




# In[19]:


# Cellule 10 — Boucle principale DAgger hybride + suivi des récompenses

N_ITERS = 2           # nombre d'itérations DAgger
SUPERVISED_EPOCHS = 3
RL_TIMESTEPS = 10_000

for it in range(1, N_ITERS + 1):
    print(f"\n\n================ DAgger Hybride - Itération {it}/{N_ITERS} ================")

    dagger_obs, dagger_actions = dagger_hybrid_iteration(
        model_student,
        dagger_obs,
        dagger_actions,
        supervised_epochs=SUPERVISED_EPOCHS,
        rl_timesteps=RL_TIMESTEPS
    )

    # Test de l'élève sur une semaine
    env_eval = WorkshopEnv()
    reward_eval, steps_eval = run_student_episode(env_eval, model_student)

    # On enregistre le reward de test dans l'historique
    history["reward_eval"].append(reward_eval)

    print(f"\n>>> Évaluation élève après itération {it} :")
    print(f"    Reward sur 7 jours : {reward_eval:.2f}")
    print(f"    Reward moyen par jour : {reward_eval / 7:.2f}")
    print(f"    Steps joués : {steps_eval}")

    # Sauvegarde du modèle
    model_path = f"maskedppo_dagger_iter_{it}.zip"
    model_student.save(model_path)
    print(f"    Modèle sauvegardé dans : {model_path}")


# In[ ]:


# Cellule 11 — PPO final après DAgger (entraînement pur RL)

print("\n===================== PPO FINAL — Phase RL pure =====================\n")

from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env):
    return env.get_action_mask()

# Environnement pour PPO final
env_rl = ActionMasker(WorkshopEnv(), mask_fn)

model_rl = MaskablePPO(
    policy="MlpPolicy",  # MlpPolicy fonctionne parfaitement avec 23 features
    env=env_rl,
    verbose=0,
    device="cuda",
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    n_steps=4096,
    batch_size=512,
    clip_range=0.2,
    ent_coef=0.01,
    max_grad_norm=0.5,
    tensorboard_log="./ppo_final_log/"
)

TOTAL_TIMESTEPS = 300_000  # selon GPU

print(">>> Début entraînement PPO final...")
model_rl.learn(total_timesteps=TOTAL_TIMESTEPS)
print(">>> Fin entraînement PPO final.")

model_rl.save("ppo_final_agent.zip")
print("Modèle PPO final sauvegardé.")


# In[ ]:


# Cellule 12

# ===================== TEST FINAL PPO (Cellule autonome) =====================

import numpy as np
from sb3_contrib import MaskablePPO
from env.workshop_env import WorkshopEnv

print("\n===================== TEST FINAL PPO =====================\n")

# 1) Chargement du modèle entraîné
model_rl = MaskablePPO.load("ppo_final_agent.zip")

# 2) Création d'un environnement nu pour le test
env_test = WorkshopEnv()
obs, info = env_test.reset()

total_reward = 0.0

# 3) Boucle d'évaluation sur 7 jours (10080 minutes)
for t in range(10080):
    mask = env_test.get_action_mask()

    action, _ = model_rl.predict(
        obs,
        deterministic=True,
        action_masks=mask     # <<< IMPORTANT !!!
    )

    obs, reward, terminated, truncated, info = env_test.step(action)
    total_reward += reward

    if terminated or truncated:
        break

# 4) Résultats finaux
print("Reward total PPO :", total_reward)
print("Reward moyen par jour :", total_reward / 7)
print("Steps joués :", t + 1)


# In[ ]:






# In[99]:


# Visualisation des métriques DAgger

import matplotlib.pyplot as plt

if len(history["supervised_loss"]) > 0:
    plt.figure(figsize=(15, 4))

    # 1) Loss d'imitation
    plt.subplot(1, 3, 1)
    plt.plot(history["supervised_loss"], marker='o')
    plt.title("Loss d'imitation (CrossEntropy)")
    plt.xlabel("Itération DAgger")
    plt.ylabel("Loss")

    # 2) Reward pendant la collecte DAgger
    plt.subplot(1, 3, 2)
    plt.plot(history["reward_collect"], marker='o')
    plt.title("Reward élève (collecte DAgger)")
    plt.xlabel("Itération DAgger")
    plt.ylabel("Reward")

    # 3) Reward en évaluation (7 jours)
    plt.subplot(1, 3, 3)
    plt.plot(history["reward_eval"], marker='o')
    plt.title("Reward élève (évaluation 7 jours)")
    plt.xlabel("Itération DAgger")
    plt.ylabel("Reward")

    plt.tight_layout()
    plt.show()
else:
    print("Aucune métrique DAgger enregistrée dans `history` (vérifier que la boucle DAgger a été exécutée).")

