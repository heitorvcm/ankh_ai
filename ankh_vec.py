"""
ankh_vec.py - VecEnv (stable-baselines3) com simulacao em processos de CPU e
inferencia do oponente EM LOTE na GPU.

O que roda onde
---------------
 * Regras do jogo + codificacao da observacao: CPU, em `n_workers` processos
   (cada um com varias partidas). Python puro: nao da para rodar na GPU sem
   reescrever o jogo como operacoes de tensor.
 * Politica do agente (SB3) e do oponente (snapshot): GPU, uma chamada por lote
   com TODAS as partidas que esperam decisao. O PPO tambem atualiza na GPU.

Uso:
    venv = BatchedSelfPlayVecEnv(n_envs=64, n_workers=2)   # crie ANTES do modelo
    venv.set_opponent(None)                  # aleatorio (roda dentro dos workers)
    venv.set_opponent(TorchOpponent(path))   # rede em lote na GPU
"""
import multiprocessing as mp

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from ankh_game import AnkhGame
from ankh_obs import Encoder
from ankh_worker import worker_main


class BatchedSelfPlayVecEnv(VecEnv):
    def __init__(self, n_envs=64, n_workers=2, scenarios=None, seed=0,
                 dev_gain=0.02, dev_opp=0.01, war_gain=0.005, war_opp=0.0025,
                 draw_reward=-1.0, zero_sum=False, god_pool=None, merged_reward="normal",
                 start_method="spawn"):
        # zero_sum=True: o que um jogador ganha o outro perde (pesos simetricos e
        # Egito ateu = 0), para o self-play competitivo depois do pre-treino.
        if zero_sum:
            dev_opp, war_opp, draw_reward = dev_gain, war_gain, 0.0
        from ankh_map import SCENARIOS
        scenarios = list(scenarios) if scenarios else list(SCENARIOS)       # padrao: todos os cenarios validos
        probe = AnkhGame(scenarios)
        self.scenarios = list(probe.scenarios)
        self.skipped = dict(probe.skipped)                                   # cenarios com erro de dados
        probe.reset(seed=0)
        obs_dim, a_size = len(Encoder(probe).encode(probe.decision)), probe.A_SIZE
        super().__init__(n_envs, spaces.Box(0.0, 1.0, (obs_dim,), np.float32), spaces.Discrete(a_size))
        self.obs_dim, self.a_size = obs_dim, a_size
        cfg = dict(dev_gain=dev_gain, dev_opp=dev_opp, war_gain=war_gain, war_opp=war_opp, draw=draw_reward, merged_reward=merged_reward,
                   god_pool=list(god_pool) if god_pool else None)
        ctx = mp.get_context(start_method)    # 'spawn': nao herda estado da GPU
        self.chunks = [list(c) for c in np.array_split(np.arange(n_envs), min(n_workers, n_envs)) if len(c)]
        self.local = {}
        self.pipes, self.procs = [], []
        for w, ids in enumerate(self.chunks):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=worker_main, args=(child, len(ids), seed + 1000 * w, self.scenarios, cfg), daemon=True)
            p.start()
            child.close()
            self.pipes.append(parent)
            self.procs.append(p)
            for i, s in enumerate(ids):
                self.local[s] = (w, i)
        self._obs = np.zeros((n_envs, obs_dim), np.float32)
        self._mask = np.zeros((n_envs, a_size), bool)
        self._ret = np.zeros(n_envs)
        self._len = np.zeros(n_envs, int)
        self._opponent = None
        self._actions = None
        self.set_opponent(None)

    # ---------------- comunicacao ----------------
    def _apply(self, items):
        sent = []
        for w, ids in enumerate(self.chunks):
            mine = [s for s in ids if s in items]
            if mine:
                self.pipes[w].send(("apply", ([self.local[s][1] for s in mine], [items[s] for s in mine])))
                sent.append((w, mine))
        out = {}
        for w, mine in sent:
            for s, r in zip(mine, self.pipes[w].recv()):
                out[s] = r
        return out

    def set_opponent(self, opponent):
        """None = aleatorio nos workers; senao callable(obs[B,D], mask[B,A]) -> acoes[B]."""
        self._opponent = opponent
        for p in self.pipes:
            p.send(("opp_random", opponent is None))
        for p in self.pipes:
            p.recv()

    def _opp_actions(self, obs, mask):
        acts = np.asarray(self._opponent(obs, mask)).astype(int)
        ok = mask[np.arange(len(acts)), acts]
        for k in np.flatnonzero(~ok):                  # rede devolveu acao ilegal: sorteia legal
            acts[k] = np.random.choice(np.flatnonzero(mask[k]))
        return acts

    def _run_until_agent(self, results, rew=None, done=None, infos=None):
        """Aplica em lote as jogadas do oponente ate todas as partidas esperarem o agente."""
        while True:
            opp = []
            for s, (kind, obs, mask, ep, acc) in results.items():
                if ep is not None and rew is not None:
                    rew[s], done[s], infos[s] = ep[0], True, dict(ep[1])
                if kind == "agent":
                    self._obs[s], self._mask[s] = obs, mask
                    if rew is not None and not done[s]:
                        rew[s] = acc
                else:
                    opp.append(s)
            if not opp:
                return
            obs_b = np.stack([results[s][1] for s in opp])
            mask_b = np.stack([results[s][2] for s in opp])
            acts = self._opp_actions(obs_b, mask_b)
            results = self._apply({s: int(a) for s, a in zip(opp, acts)})

    # ---------------- API do VecEnv ----------------
    def reset(self):
        for p in self.pipes:
            p.send(("reset", None))
        results = {}
        for w, ids in enumerate(self.chunks):
            for s, r in zip(ids, self.pipes[w].recv()):
                results[s] = r
        self._run_until_agent(results)
        self._ret[:], self._len[:] = 0, 0
        return self._obs.copy()

    def step_async(self, actions):
        self._actions = np.asarray(actions)

    def step_wait(self):
        n = self.num_envs
        rew, done, infos = np.zeros(n, np.float32), np.zeros(n, bool), [{} for _ in range(n)]
        results = self._apply({s: int(a) for s, a in enumerate(self._actions)})
        self._run_until_agent(results, rew, done, infos)
        self._ret += rew
        self._len += 1
        for s in np.flatnonzero(done):
            infos[s]["terminal_observation"] = np.zeros(self.obs_dim, np.float32)
            infos[s]["episode"] = {"r": float(self._ret[s]), "l": int(self._len[s])}
            self._ret[s], self._len[s] = 0, 0
        return self._obs.copy(), rew, done, infos

    def close(self):
        for p in self.pipes:
            try:
                p.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for pr in self.procs:
            pr.join(timeout=5)

    def _idx(self, indices):
        return range(self.num_envs) if indices is None else ([indices] if isinstance(indices, int) else list(indices))

    def env_method(self, method_name, *args, indices=None, **kwargs):
        idx = self._idx(indices)
        if method_name == "action_masks":
            return [self._mask[i].copy() for i in idx]
        if method_name == "set_opponent":
            self.set_opponent(args[0])
            return [None] * len(idx)
        raise ValueError(f"metodo nao suportado: {method_name}")

    def get_attr(self, attr_name, indices=None):
        return [None] * len(self._idx(indices))

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._idx(indices))

    def get_images(self):
        return [None] * self.num_envs

    def seed(self, seed=None):
        return [None] * self.num_envs


class TorchOpponent:
    """Snapshot do MaskablePPO carregado na GPU; decide em lote."""
    def __init__(self, path, device="cuda"):
        import torch
        from sb3_contrib import MaskablePPO
        self.torch = torch
        self.model = MaskablePPO.load(path, device=device)
        self.policy = self.model.policy
        self.policy.set_training_mode(False)

    def __call__(self, obs, mask):
        torch = self.torch
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.policy.device)
            a, _, _ = self.policy.forward(o, deterministic=False, action_masks=mask)
        return a.cpu().numpy()
