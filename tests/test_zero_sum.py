"""Propriedade de soma zero das recompensas (funcoes puras do worker)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ankh_worker import step_reward, end_reward

cfg = dict(dev_gain=0.02, dev_opp=0.02, war_gain=0.005, war_opp=0.005, draw=0.0)   # zero-sum
rng = np.random.default_rng(0)
for _ in range(10_000):
    d0, d1 = rng.integers(0, 31, 2); w0, w1 = rng.integers(0, 7, 2)
    nd0, nd1 = min(d0 + rng.integers(0, 4), 30), min(d1 + rng.integers(0, 4), 30)
    nw0, nw1 = rng.integers(0, 7, 2)
    r0 = step_reward(cfg, (d0, d1, w0, w1), (nd0, nd1, nw0, nw1))      # ponto de vista do jogador 0
    r1 = step_reward(cfg, (d1, d0, w1, w0), (nd1, nd0, nw1, nw0))      # ponto de vista do jogador 1
    assert abs(r0 + r1) < 1e-9 and r0 + r1 == r0 + r1
for w in (0, 1, None):
    assert end_reward(cfg, 0, w) + end_reward(cfg, 1, w) == 0, w
nz = dict(cfg, dev_opp=0.01, draw=-1.0)                                 # padrao: NAO e soma zero
assert end_reward(nz, 0, None) + end_reward(nz, 1, None) == -2
print("soma zero ok (step_reward e end_reward)")

for P in (2, 3, 4, 5):                                                  # soma zero tambem com varios jogadores
    for winner in range(P):
        assert abs(sum(end_reward(cfg, me, winner, P) for me in range(P))) < 1e-9, (P, winner)
    assert end_reward(cfg, 0, None, P) == 0
print("soma zero ok para 2 a 5 jogadores")
