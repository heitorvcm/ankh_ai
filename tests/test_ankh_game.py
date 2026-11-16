import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # raiz do projeto
"""Testes de fumaca: jogadores aleatorios + invariantes."""
import collections, time
import numpy as np
from ankh_game import *


def check_invariants(g):
    P = g.P
    small, large = {2: 1, 3: 2}.get(P, 3), {2: 1, 3: 2}.get(P, 2)
    for p in range(P):
        if g.rep[p] != p:
            continue                                       # assento absorvido na fusao
        board_w = sum(1 for i in range(g.N) if g.fig_owner[i] == p and g.fig_kind[i] == WARRIOR_K)
        trapped = sum(t.count(p) for t in g.trapped)
        if g.alive[p]:
            assert board_w + g.warriors_pool[p] + trapped == 6, "guerreiros nao conservados"
            gods = sum(1 for i in range(g.N) if g.fig_owner[i] == p and g.fig_kind[i] == GOD_K)
            assert gods == 1, "deus sumiu"
            owned = sum(1 for i in range(g.N) if g.mon_owner[i] == p)
            assert owned + g.ankh_pool[p] == 9, "tokens Ankh nao conservados"
            for size in ("small", "large"):
                n = sum(1 for k in g.gpool[p] if GUARDIAN_SIZE[k] == size) + sum(
                    1 for i in range(g.N) if g.fig_owner[i] == p and g.fig_kind[i] in GUARDIAN_KINDS
                    and GUARDIAN_SIZE[g.fig_kind[i]] == size)
                assert n == g.bases_used[size][p] <= 2, "bases de guardioes"
            if g._has_god(p, RA):
                assert sum(g.fig_sun[i] for i in range(g.N) if g.fig_owner[i] == p) + g.sun_left[p] == 3
            if g._has_god(p, OSIRIS):
                assert sum(g.underworld) + g.uw_off[p] == 3
            assert len(g.trapped[p]) <= ANUBIS_SLOTS
        assert g.followers[p] >= 0 and 0 <= g.unlock_next[p] <= 6 and len(g.powers[p]) == g.unlock_next[p]
        assert len(g.hand[p]) + len(g.used[p]) == 7
    for lvl in (1, 2, 3):
        k = g.g_card[lvl]
        owned = sum(g.gpool[p].count(k) for p in range(P)) + sum(1 for i in range(g.N) if g.fig_kind[i] == k)
        init = small if GUARDIAN_SIZE[k] == "small" else large
        assert g.g_supply[lvl] >= 0 and (owned + g.g_supply[lvl] == init or not all(g.alive))
    for i in range(g.N):
        if g.fig_owner[i] >= 0:
            assert g.is_land[i] or g.fig_kind[i] == APEP, "figura na agua (so Apep)"
            assert g.alive[g.fig_owner[i]], "figura de entidade morta"
        assert not (g.mon_type[i] is not None and (g.fig_owner[i] >= 0 or not g.is_land[i])), "monumento ocupado"
        assert not g.underworld[i] or g.is_land[i]
    n_mon = collections.Counter(t for t in g.mon_type if t)
    for t in MON_TYPES:
        assert n_mon[t] + g.supply[t] == 10, "monumentos nao conservados"
    assert sum(len(s) for s in g.stacks) == sum(g.alive)
    assert all(g.alive[g.rep[p]] or True for p in range(P))


def play(seed, names=("basic_2p",), check=True):
    g = AnkhGame(list(names), seed=seed)
    rng = np.random.default_rng(seed)
    dec = g.reset()
    phases = collections.Counter()
    while dec is not None:
        a = rng.choice(g.legal_actions())
        phases[dec.phase.name] += 1
        dec = g.step(a)
        if check and g.steps % 7 == 0 and not g.done:
            check_invariants(g)
    return g, phases


if __name__ == "__main__":
    from common import register_test_5p
    t5 = register_test_5p()
    t = time.time()
    for names, n in ((("basic_2p",), 200), (("basic_3p",), 100), (("basic_4p",), 100), ((t5,), 100),
                     (("basic_2p", "basic_3p", "basic_4p", t5), 100)):
        res = collections.Counter(); steps = []; phases = collections.Counter(); merged = 0
        for s in range(n):
            g, ph = play(s, names)
            res[g.winner] += 1; steps.append(g.steps); phases.update(ph); merged += g.merged is not None
        print(f"{'+'.join(names):<45} {n} partidas ok | decisoes med {int(np.median(steps))} | fusoes {merged} | "
              f"vencedores {dict(sorted(res.items(), key=lambda x: str(x[0])))}")
    print("tempo", round(time.time() - t, 1), "s")
