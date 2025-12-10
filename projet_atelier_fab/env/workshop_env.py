# ================================================================
# WORKSHOP ENVIRONMENT — FINAL VERSION (OPTION C, FULLY COMMENTED)
# Production au fil de l'eau + commentaires pédagogiques complets
# + état enrichi (23 variables) pour l'agent RL
# ================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .machines import Machine
from .stock import Stock
from .delivery import DeliveryQueue
from .market import Market


class WorkshopEnv(gym.Env):
    """
    ============================================================
    ENVIRONNEMENT ATELIER — VERSION EXPLICATIVE ET STRUCTURÉE
    ============================================================

    MODELISATION GÉNÉRALE
    ----------------------
    - L'environnement simule un atelier industriel minute par minute.
    - Un épisode complet dure 7 jours : 7 × 24 × 60 = 10 080 minutes.
    - Deux machines :
        M1 : P1 et P2_STEP1
        M2 : P2_STEP2
    - Stock de matières premières (raw), P1, P2_inter, P2.
    - Commandes de matières premières avec délai.
    - Demande client toutes les 15 minutes.
    - Système de backlog pénalisant.
    - Production « au fil de l'eau » : 1 unité visible dès qu'elle est produite.

    OBJECTIFS DE L'AGENT
    ---------------------
    - Produire la bonne quantité au bon moment.
    - Minimiser le backlog (pénalité régulière).
    - Honorer la demande pour gagner des récompenses de vente.
    - Optimiser l'utilisation des machines et des stocks.

    STRUCTURE DU STEP()
    --------------------
    1) Traitement de l'action (production / commande / attente)
    2) Avancement des machines minute par minute + production unitaire
    3) Livraison potentielle de matières premières
    4) Passage du temps (+1 min)
    5) Demande + ventes (toutes les 15 minutes)
    6) Vol nocturne (1 fois par jour)
    7) Pénalité backlog
    8) Construction de l'observation (23 dimensions)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Durée maximale d'un épisode
        self.max_time = 7 * 24 * 60  # 10 080 minutes

        # Capacité des stocks
        self.raw_capacity = 50

        # Vol planifié chaque jour (minute 1435 = 23h55)
        self.theft_time = 1435

        # Initialisation du temps et des backlogs
        self.time = 0
        self.demande_p1 = 0
        self.demande_p2 = 0

        # Machines
        self.m1 = Machine()  # production P1 + STEP1
        self.m2 = Machine()  # production STEP2

        # Différents modules
        self.stock = Stock(capacity=self.raw_capacity)
        self.delivery = DeliveryQueue()
        self.market = Market()

        # -----------------------------------------------------------
        # VARIABLES SUPPLÉMENTAIRES POUR L'ÉTAT RL
        # -----------------------------------------------------------
        # Action courante
        self.current_action_type = 4      # 0=P1(M1), 1=P2_STEP1(M1), 2=P2_STEP2(M2),
                                          # 3=commande MP, 4=WAIT (par défaut)
        self.current_action_k = 0         # nombre d'unités lancées / commandées (0..50)
        self.current_action_id = 200      # index d'action (0..200, 200=WAIT)

        # Rewards dérivés pour l'observation
        self.week_reward = 0.0           # reward cumulé depuis le début de l'épisode
        self.reward_current_action = 0.0 # reward du dernier step

        # -----------------------------------------------------------
        # ESPACE D'OBSERVATION (23 DIMENSIONS)
        # -----------------------------------------------------------
        #  0 : time
        #  1 : m1.busy
        #  2 : m1.time_left
        #  3 : m2.busy
        #  4 : m2.time_left
        #  5 : stock.raw
        #  6 : stock.p1
        #  7 : stock.p2_inter
        #  8 : stock.p2
        #  9 : next_delivery_countdown
        # 10 : backlog_p1 (demande_p1)
        # 11 : backlog_p2 (demande_p2)
        # 12 : q_total_en_route
        # 13 : current_action_type
        # 14 : current_action_k
        # 15 : current_action_id
        # 16 : minute_of_day (time % 1440)
        # 17 : expected_dem_p1_next ≈ 2 * backlog_p1
        # 18 : expected_dem_p2_next ≈ 20 * backlog_p2
        # 19 : theft_risk_level (0 ou 1)
        # 20 : reward_current_week (cumul depuis début épisode)
        # 21 : reward_current_action (reward du dernier step)
        # 22 : time_to_next_sell (minutes avant prochaine vente)
        low = np.zeros(23, dtype=np.float32)
        high = np.array([
            float(self.max_time),          # 0: time
            1.0,                           # 1: m1.busy (0 ou 1)
            100.0,                         # 2: m1.time_left (bornes larges)
            1.0,                           # 3: m2.busy
            100.0,                         # 4: m2.time_left
            float(self.raw_capacity),      # 5: stock.raw
            float(self.raw_capacity),      # 6: stock.p1
            float(self.raw_capacity),      # 7: stock.p2_inter
            float(self.raw_capacity),      # 8: stock.p2
            10080.0,                       # 9: next_delivery_countdown
            1000.0,                        # 10: backlog_p1
            1000.0,                        # 11: backlog_p2
            1000.0,                        # 12: q_total_en_route

            4.0,                           # 13: current_action_type (0..4)
            50.0,                          # 14: current_action_k (0..50)
            200.0,                         # 15: current_action_id (0..200)
            1439.0,                        # 16: minute_of_day (0..1439)
            2000.0,                        # 17: expected_dem_p1_next (≈ 2*backlog_p1)
            20000.0,                       # 18: expected_dem_p2_next (≈ 20*backlog_p2)
            1.0,                           # 19: theft_risk_level
            1_000_000.0,                   # 20: reward_current_week (bornes très larges)
            100_000.0,                     # 21: reward_current_action
            15.0                           # 22: time_to_next_sell (0..15)
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # -----------------------------------------------------------
        # ESPACE D'ACTIONS (201 ACTIONS)
        # -----------------------------------------------------------
        # 0–49    → Produire P1 (k = a+1)
        # 50–99   → Produire P2_STEP1 (k = a-49)
        # 100–149 → Produire P2_STEP2 (k = a-99)
        # 150–199 → Commander MP (k = a-149)
        # 200     → WAIT
        self.action_space = spaces.Discrete(201)

    # ================================================================
    # RESET DE L'ÉPISODE
    # ================================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.time = 0
        self.demande_p1 = 0
        self.demande_p2 = 0

        self.m1.reset()
        self.m2.reset()
        self.stock.reset()
        self.delivery.reset()
        self.market = Market()

        # Reset des variables supplémentaires d'état
        self.current_action_type = 4
        self.current_action_k = 0
        self.current_action_id = 200
        self.week_reward = 0.0
        self.reward_current_action = 0.0

        return self._get_obs(), {}

    # ================================================================
    # ACTION MASK (OPTIONS PERMISES À L'AGENT)
    # ================================================================
    def get_action_mask(self):
        mask = np.ones(201, dtype=bool)

        # Si M1 est occupée, on interdit toutes les actions qui la sollicitent
        if self.m1.busy:
            mask[0:100] = False  # P1 et P2_STEP1

        # Si M2 est occupée, on interdit toutes les actions qui la sollicitent
        if self.m2.busy:
            mask[100:150] = False  # P2_STEP2

        # Contraintes en matières premières pour M1
        for a in range(0, 100):
            k = (a + 1) if a < 50 else (a - 49)
            if self.stock.raw < k:
                mask[a] = False

        # Contraintes en P2_inter pour M2
        for a in range(100, 150):
            k = a - 99
            if self.stock.p2_inter < k:
                mask[a] = False

        return mask

    # ================================================================
    # STEP — UNE MINUTE DE SIMULATION
    # ================================================================
    def step(self, action: int):

        reward = 0.0

        # -----------------------------------------------------------
        # 1) DECODE DE L'ACTION POUR LES VARIABLES D'ÉTAT
        # -----------------------------------------------------------
        self.current_action_id = int(action)

        if action == 200:
            self.current_action_type = 4
            self.current_action_k = 0
        elif 0 <= action <= 49:
            self.current_action_type = 0
            self.current_action_k = action + 1
        elif 50 <= action <= 99:
            self.current_action_type = 1
            self.current_action_k = action - 49
        elif 100 <= action <= 149:
            self.current_action_type = 2
            self.current_action_k = action - 99
        elif 150 <= action <= 199:
            self.current_action_type = 3
            self.current_action_k = action - 149
        else:
            # Sécurité si action hors bornes
            self.current_action_type = 4
            self.current_action_k = 0
            self.current_action_id = 200

        # -----------------------------------------------------------
        # 2) TRAITEMENT DE L'ACTION (REWARD COMME AVANT)
        # -----------------------------------------------------------

        # Action WAIT
        if action == 200:
            reward -= 0.2  # légère pénalité

        # Production P1
        elif 0 <= action <= 49:
            k = action + 1
            if (not self.m1.busy) and self.stock.consume_raw(k):
                duration = 3 * k  # durée totale
                self.m1.start_batch(duration=duration, k=k, batch_type="P1_MULTI")
                reward += 0.5 * k
            else:
                reward -= 1

        # Production P2 — Étape 1
        elif 50 <= action <= 99:
            k = action - 49
            if (not self.m1.busy) and self.stock.consume_raw(k):
                duration = 10 * k
                self.m1.start_batch(duration=duration, k=k, batch_type="P2STEP1_MULTI")
                reward += 5 * k
            else:
                reward -= 1

        # Production P2 — Étape 2
        elif 100 <= action <= 149:
            k = action - 99
            if (not self.m2.busy) and self.stock.p2_inter >= k:
                self.stock.p2_inter -= k
                duration = 15 * k
                self.m2.start_batch(duration=duration, k=k, batch_type="P2STEP2_MULTI")
                reward += 15 * k
            else:
                reward -= 1

        # Commande MP
        elif 150 <= action <= 199:
            k = action - 149
            reward -= float(k)
            jitter = np.random.randint(-2, 3)
            arrival_time = max(self.time + 1, self.time + 120 + jitter)
            self.delivery.schedule(k, arrival_time)

        # -----------------------------------------------------------
        # 3) PRODUCTION AU FIL DE L'EAU — MACHINE M1
        # -----------------------------------------------------------
        res_m1 = self.m1.tick()

        if res_m1 in ("unit", "last_unit"):
            if self.m1.batch_type == "P1_MULTI":
                self.stock.add_p1(1)
            elif self.m1.batch_type == "P2STEP1_MULTI":
                self.stock.add_p2_inter(1)

            if res_m1 == "last_unit":
                self.m1.reset_after_batch()

        # -----------------------------------------------------------
        # 4) PRODUCTION AU FIL DE L'EAU — MACHINE M2
        # -----------------------------------------------------------
        res_m2 = self.m2.tick()

        if res_m2 in ("unit", "last_unit"):
            if self.m2.batch_type == "P2STEP2_MULTI":
                self.stock.add_p2(1)

            if res_m2 == "last_unit":
                self.m2.reset_after_batch()

        # -----------------------------------------------------------
        # 5) LIVRAISONS DE MP
        # -----------------------------------------------------------
        delivered = self.delivery.tick(self.time)
        if delivered > 0:
            self.stock.add_raw(delivered)

        # -----------------------------------------------------------
        # 6) INCRÉMENT DU TEMPS
        # -----------------------------------------------------------
        self.time += 1

        # -----------------------------------------------------------
        # 7) DEMANDE + VENTES — toutes les 15 minutes
        # -----------------------------------------------------------
        if self.time % 15 == 0:

            new_d1, new_d2 = self.market.sample_demand(self.time, 15)
            self.demande_p1 += int(new_d1)
            self.demande_p2 += int(new_d2)

            sold_p1, sold_p2 = self.market.compute_sales(
                self.stock, self.demande_p1, self.demande_p2
            )

            reward += 2.0 * sold_p1 + 20.0 * sold_p2

            self.demande_p1 -= sold_p1
            self.demande_p2 -= sold_p2

            # Pénalité backlog (logique inchangée)
            backlog = self.demande_p1 + self.demande_p2
            reward -= 0.02 * float(backlog)

        # -----------------------------------------------------------
        # 8) VOL QUOTIDIEN (minute 1435)
        # -----------------------------------------------------------
        if self.time % 1440 == self.theft_time:
            self.market.apply_theft(self.stock)

        # -----------------------------------------------------------
        # 9) MISE À JOUR DES REWARDS CUMULÉS POUR L'ÉTAT
        # -----------------------------------------------------------
        self.week_reward += reward
        self.reward_current_action = reward

        # -----------------------------------------------------------
        # 10) TERMINAISON
        # -----------------------------------------------------------
        terminated = self.time >= self.max_time
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

    # ================================================================
    # CONSTRUCTION DE L'OBSERVATION POUR SB3 (23 DIMENSIONS)
    # AVEC NORMALISATION DES VARIABLES
    # ================================================================
    def _get_obs(self):

        # Infos sur la prochaine livraison
        if self.delivery.queue:
            next_t = min(t for (q, t) in self.delivery.queue)
            next_delivery_countdown = max(next_t - self.time, 0)
            q_total = sum(q for (q, t) in self.delivery.queue)
        else:
            next_delivery_countdown = 0
            q_total = 0

        # minute dans la journée
        minute_of_day = self.time % 1440

        # reward potentiel lié au backlog (simple rescaling)
        expected_dem_p1_next = 2.0 * float(self.demande_p1)
        expected_dem_p2_next = 20.0 * float(self.demande_p2)

        # niveau de risque de vol (par exemple : dans l'heure avant le vol)
        minutes_in_day = self.time % 1440
        if 0 <= (self.theft_time - minutes_in_day) <= 60:
            theft_risk_level = 1.0
        else:
            theft_risk_level = 0.0

        # temps jusqu'à la prochaine vente (toutes les 15 minutes)
        time_mod_15 = self.time % 15
        time_to_next_sell = 0 if time_mod_15 == 0 else 15 - time_mod_15

        # ============================
        # NORMALISATION (DIVISION)
        # ============================

        obs = np.array([
            float(self.time) / float(self.max_time),        # 0: time normalisé [0,1]
            float(self.m1.busy),                            # 1: déjà dans {0,1}
            float(self.m1.time_left) / 100.0,               # 2
            float(self.m2.busy),                            # 3
            float(self.m2.time_left) / 100.0,               # 4

            float(self.stock.raw) / float(self.raw_capacity),     # 5
            float(self.stock.p1) / float(self.raw_capacity),      # 6
            float(self.stock.p2_inter) / float(self.raw_capacity),# 7
            float(self.stock.p2) / float(self.raw_capacity),      # 8

            float(next_delivery_countdown) / 10080.0,       # 9

            float(self.demande_p1) / 1000.0,                # 10
            float(self.demande_p2) / 1000.0,                # 11
            float(q_total) / 1000.0,                        # 12

            float(self.current_action_type) / 4.0,          # 13
            float(self.current_action_k) / 50.0,            # 14
            float(self.current_action_id) / 200.0,          # 15

            float(minute_of_day) / 1440.0,                  # 16

            float(expected_dem_p1_next) / 2000.0,           # 17
            float(expected_dem_p2_next) / 20000.0,          # 18
            float(theft_risk_level),                        # 19, déjà 0 ou 1

            float(self.week_reward) / 1_000_000.0,          # 20
            float(self.reward_current_action) / 100_000.0,  # 21
            float(time_to_next_sell) / 15.0                 # 22
        ], dtype=np.float32)

        return obs
