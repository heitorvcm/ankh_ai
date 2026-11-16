"""
ankh_env.py - Wrapper Gymnasium de um jogador contra oponentes embutidos (2 a 5 jogadores).

O agente controla um ASSENTO (`me`, sorteado a cada episodio); os demais assentos jogam
dentro do proprio env, com `opponent(obs, mask) -> acao` (None = aleatorio legal). O cenario
(e com ele o numero de jogadores) e os deuses tambem sao sorteados a cada episodio.
Com merged_reward="zero", os assentos de um deus fundido recebem recompensa 0 (densa e final)
Depois da fusao. Para self-play, troque o oponente por um snapshot do modelo com env.set_opponent(...).

Recompensa (do ponto de vista de `me`), somada a cada passo do agente:

  fim de jogo : +1 vitoria, -1/(P-1) derrota (P=2: -1), draw_reward no Egito ateu
                (padrao -1; com zero_sum=True vale 0 e os pesos dos oponentes = os do agente)
  Devocao     : + dev_gain * (casas de Devocao que EU subi)
                - dev_opp  * (media das casas que os OPONENTES subiram)
  Guerreiros  : + war_gain * (variacao liquida dos MEUS guerreiros no tabuleiro)
                - war_opp  * (media da variacao dos guerreiros dos OPONENTES)

Os termos densos sao recompensas simples, NAO shaping potencial: somados valem aprox. a
Devocao final, entao o agente tambem e premiado por terminar mais alto. Mantenha o total
denso bem abaixo do +-1 da vitoria.

Compativel com sb3-contrib MaskablePPO: env.action_masks().
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ankh_game import AnkhGame
from ankh_map import SCENARIOS
from ankh_obs import Encoder
from ankh_worker import end_reward, merged_seat, reward_parts, snapshot


class AnkhEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario=None, opponent=None, seed=None,
                 dev_gain=0.02, dev_opp=0.01, war_gain=0.005, war_opp=0.0025,
                 draw_reward=-1.0, zero_sum=False, god_pool=None, merged_reward="normal"):
        super().__init__()
        if zero_sum:
            dev_opp, war_opp, draw_reward = dev_gain, war_gain, 0.0
        scenarios = [scenario] if isinstance(scenario, str) else (list(scenario) if scenario else list(SCENARIOS))
        self.g = AnkhGame(scenarios, seed=seed, god_pool=god_pool)
        self.enc = Encoder(self.g)
        self.opponent = opponent
        self.cfg = dict(dev_gain=dev_gain, dev_opp=dev_opp, war_gain=war_gain, war_opp=war_opp, draw=draw_reward,
                        merged_reward=merged_reward)
        self.rng = np.random.default_rng(seed)
        self.me = 0
        dec = self.g.reset()
        self.obs_dim = len(self.enc.encode(dec))
        self.observation_space = spaces.Box(0.0, 1.0, (self.obs_dim,), np.float32)
        self.action_space = spaces.Discrete(self.g.A_SIZE)

    def set_opponent(self, opponent):
        self.opponent = opponent

    def action_masks(self):
        return self.g.decision.mask

    def _opponent_moves(self):
        g = self.g
        while not g.done and g.decision.player != self.me:
            dec = g.decision
            if self.opponent is None:
                a = self.rng.choice(np.flatnonzero(dec.mask))
            else:
                a = int(self.opponent(self.enc.encode(dec), dec.mask))
            g.step(a)

    def reset(self, seed=None, options=None):
        """options: scenario (nome), me (assento), god_me, god_opp (so com 2 jogadores)."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        opts = options or {}
        scen = opts.get("scenario") or self.g.scenarios[int(self.rng.integers(len(self.g.scenarios)))]
        P = SCENARIOS[scen]["n_players"]
        self.me = int(opts.get("me", self.rng.integers(P))) % P
        gods = None
        if "god_me" in opts or "god_opp" in opts:
            pool = list(self.g.god_pool)
            gods = [None] * P
            if "god_me" in opts:
                gods[self.me] = int(opts["god_me"])
            if "god_opp" in opts and P == 2:
                gods[1 - self.me] = int(opts["god_opp"])
            for s in range(P):
                if gods[s] is None:
                    free = [x for x in pool if x not in gods] or pool
                    gods[s] = int(self.rng.choice(free))
        self.g.reset(seed=int(self.rng.integers(2**31)), gods=gods, scenario=scen)
        self._opponent_moves()
        self._prev = snapshot(self.g, self.me)
        return self.enc.encode(self.g.decision), {"me": self.me, "scenario": scen}

    def step(self, action):
        g = self.g
        g.step(int(action))
        self._opponent_moves()
        before, now = self._prev, snapshot(g, self.me)
        self._prev = now
        zero = self.cfg["merged_reward"] == "zero" and merged_seat(g, self.me)
        r_dev, r_war = (0.0, 0.0) if zero else reward_parts(self.cfg, before, now)
        info = {"r_dev": r_dev, "r_war": r_war}
        if g.done:
            r = g.rep[self.me]
            r_end = 0.0 if zero else end_reward(self.cfg, r, g.winner, g.P)
            opp0 = next((e for e in range(g.P) if e != r and g.rep[e] == e), None)
            info.update(r_end=r_end, winner=g.winner, me=self.me, entity=r,
                        won=(g.winner == r), god_me=g.god[r],
                        god_opp=g.god[opp0] if opp0 is not None else None, summary=g.summary(self.me))
            obs = np.zeros(self.obs_dim, np.float32)
            return obs, r_dev + r_war + r_end, True, False, info
        return self.enc.encode(g.decision), r_dev + r_war, False, False, info
