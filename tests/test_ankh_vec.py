import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # raiz do projeto
"""Teste do VecEnv em lote (stubs de gymnasium/sb3 se nao instalados)."""
import sys, time, types
import numpy as np
try:
    import stable_baselines3  # noqa
except ImportError:
    class VecEnv:
        def __init__(self, num_envs, observation_space, action_space):
            self.num_envs, self.observation_space, self.action_space = num_envs, observation_space, action_space
    class Box:
        def __init__(s, lo, hi, shape, dtype): s.shape = shape
    class Discrete:
        def __init__(s, n): s.n = n
    sp = types.ModuleType("gymnasium.spaces"); sp.Box, sp.Discrete = Box, Discrete
    gm = types.ModuleType("gymnasium"); gm.spaces = sp
    sb = types.ModuleType("stable_baselines3"); c = types.ModuleType("stable_baselines3.common")
    ve = types.ModuleType("stable_baselines3.common.vec_env"); ve.VecEnv = VecEnv
    sys.modules.update({"gymnasium": gm, "gymnasium.spaces": sp, "stable_baselines3": sb,
                        "stable_baselines3.common": c, "stable_baselines3.common.vec_env": ve})
from ankh_vec import BatchedSelfPlayVecEnv

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    venv = BatchedSelfPlayVecEnv(n_envs=32, n_workers=2, seed=1)
    obs = venv.reset()
    assert obs.shape == (32, venv.obs_dim) and venv.env_method("action_masks")[0].any()

    def run(ticks, label):
        global obs
        eps, rets, wins, nsteps = 0, [], 0, 0
        t = time.time()
        for _ in range(ticks):
            masks = np.stack(venv.env_method("action_masks"))
            assert masks.any(axis=1).all()
            acts = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
            venv.step_async(acts)
            obs, r, d, infos = venv.step_wait()
            assert np.isfinite(obs).all() and np.isfinite(r).all()
            nsteps += len(acts)
            for k in np.flatnonzero(d):
                eps += 1; rets.append(infos[k]["episode"]["r"]); wins += infos[k]["summary"]["result"] == "win"
        dt = time.time() - t
        print(f"{label}: {nsteps/dt:.0f} passos do agente/s | {eps} episodios, vitorias {wins}, "
              f"retorno medio {np.mean(rets):.2f}")

    run(150, "oponente aleatorio nos workers")
    batches = []
    def fake_net(o, m):                      # imita a rede na GPU: uma chamada por lote
        batches.append(len(o))
        return np.array([np.random.choice(np.flatnonzero(x)) for x in m])
    venv.set_opponent(fake_net)
    run(150, "oponente em lote (rede simulada)")
    print("lotes do oponente: %d chamadas, tamanho medio %.1f" % (len(batches), np.mean(batches)))
    venv.close()
