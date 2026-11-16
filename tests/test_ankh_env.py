import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # raiz do projeto
"""Teste do env sem gymnasium instalado: usa um stub minimo."""
import inspect, sys, time, types
import numpy as np

try:
    import gymnasium  # noqa
except ImportError:
    gm = types.ModuleType("gymnasium"); sp = types.ModuleType("gymnasium.spaces")
    class Env:
        def reset(self, seed=None, options=None): pass
    class Box:
        def __init__(s, lo, hi, shape, dtype): s.shape = shape
    class Discrete:
        def __init__(s, n): s.n = n
    gm.Env, sp.Box, sp.Discrete, gm.spaces = Env, Box, Discrete, sp
    sys.modules["gymnasium"], sys.modules["gymnasium.spaces"] = gm, sp

import ankh_obs
from ankh_env import AnkhEnv

assert "pending" not in inspect.getsource(ankh_obs.Encoder), "observacao nao pode ler escolhas pendentes"
env = AnkhEnv(seed=0)
print("obs_dim", env.obs_dim, "acoes", env.action_space.n)
wins = [0, 0]; ret = []; lens = []; t = time.time(); total = 0; dev = []; war = []
for ep in range(100):
    obs, info = env.reset(seed=ep)
    done, R, n, rd, rw = False, 0.0, 0, 0.0, 0.0
    while not done:
        assert obs.shape == (env.obs_dim,) and np.isfinite(obs).all()
        m = env.action_masks(); assert m.any()
        obs, r, done, trunc, inf = env.step(np.random.choice(np.flatnonzero(m)))
        R += r; n += 1; rd += inf["r_dev"]; rw += inf["r_war"]
    ret.append(R); lens.append(n); total += n; dev.append(rd); war.append(rw)
    wins[0 if inf["won"] else 1] += 1
print("100 episodios: vitorias do agente %d, outras %d (empate conta como derrota)" % tuple(wins))
print("retorno medio %.2f, passos medios %.0f, %.0f passos/s" % (np.mean(ret), np.mean(lens), total / (time.time() - t)))
print("por episodio: recompensa de Devocao %.3f (min %.2f max %.2f), de guerreiros %.3f (min %.2f max %.2f)" % (np.mean(dev), min(dev), max(dev), np.mean(war), min(war), max(war)))
