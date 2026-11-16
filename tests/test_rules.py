"""Testes de regras dos deuses, poderes Ankh e Guardioes (estados montados a mao)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ankh_game import *


def fresh(gods=(AMUN, RA), seed=0, scenario="basic_2p"):
    g = AnkhGame([scenario], seed=seed)
    g.reset(seed=seed, gods=gods)
    for i in range(g.N):
        g.fig_owner[i], g.fig_kind[i], g.fig_sun[i], g.claw[i] = -1, 0, False, None
        if g.mon_type[i] is not None:
            g._destroy_monument(i)
    g.warriors_pool = [6] * g.P
    return g


def path(g, n):
    """n casas de terra consecutivas e adjacentes numa mesma regiao."""
    def dfs(p):
        if len(p) == n:
            return p
        for j in g.adj[p[-1]]:
            if j not in p:
                r = dfs(p + [j])
                if r:
                    return r
        return None
    for i in range(g.N):
        if g.is_land[i]:
            r = dfs([i])
            if r:
                return r


def put(g, p, kind, i):
    g.fig_owner[i], g.fig_kind[i] = p, kind


def drive(g, gen, choose=None):
    choose = choose or (lambda d: int(np.flatnonzero(d.mask)[0]))
    first = None
    try:
        dec = next(gen)
        first = dec
        while True:
            a = choose(dec)
            assert dec.mask[a], (dec.phase.name, a)
            dec = gen.send(a)
    except StopIteration:
        pass
    return first


def by_phase(g, queues=None, default=None):
    """Chooser: carta por jogador (queues) e 'nao' nas demais decisoes de sim/nao."""
    queues = queues or {}
    def choose(d):
        if d.phase == Phase.BATTLE_CARD:
            return g.A_CARD + queues[d.player].pop(0)
        if d.phase in (Phase.AMUN_DECLARE, Phase.TIEBREAK, Phase.WORSHIP):
            return g.A_YESNO + (1 if (default == "yes") else 0)
        return int(np.flatnonzero(d.mask)[0])
    return choose


def test_androsphinx():
    g = fresh(); a, b = path(g, 2)
    put(g, 0, ANDRO, a); put(g, 1, WARRIOR_K, b)
    r = g.region_of[a]
    assert g._neutral(b) and g._strength(1, r, []) == 0 and g._strength(0, r, []) == 1
    put(g, 1, ANDRO, b)
    assert g._neutral(a) and g._neutral(b)          # Androsphinx inimigos adjacentes se anulam


def test_cat_mummy():
    g = fresh(); a, b = path(g, 2)
    g.gain_dev(0, 5); g.gain_dev(1, 5)
    put(g, 0, CATMUMMY, a); killed = [0, 0]
    g._kill_pass([a], killed); g._flush_cat()
    assert g.dev_pos == [5, 4] and killed == [1, 0] and CATMUMMY in g.gpool[0]


def test_mummy_revive():
    g = fresh(); a, b = path(g, 2)
    put(g, 0, GOD_K, a); put(g, 0, MUMMY, b)
    g._kill_pass([b], [0, 0])
    mums = [i for i in range(g.N) if g.fig_kind[i] == MUMMY and g.fig_owner[i] == 0]
    assert len(mums) == 1 and mums[0] in g.adj[a] and MUMMY not in g.gpool[0]


def test_anubis_trap():
    g = fresh((ANUBIS, AMUN)); a, b = path(g, 2)
    put(g, 1, WARRIOR_K, a); put(g, 1, WARRIOR_K, b); g.warriors_pool = [6, 4]
    g._kill_pass([a], [0, 0])
    assert g.trapped[0] == [1] and g.warriors_pool[1] == 4          # preso, nao volta ao pool
    g._kill_pass([b], [0, 0])
    assert g.trapped[0] == [1, 1] and g._god_strength(0) == 3
    for _ in range(3):
        g.trapped[0].append(1)
    assert len(g.trapped[0]) <= 5 and g._god_strength(0) == 4        # bonus limitado a +3


def test_isis_protects():
    for gods, survives in (((ISIS, AMUN), True), ((AMUN, RA), False)):
        g = fresh(gods); p = path(g, 5)
        put(g, 0, GOD_K, p[0]); put(g, 0, WARRIOR_K, p[1])
        for i in p[2:]:
            put(g, 1, WARRIOR_K, i)
        g.tiebreak_holder, g.tiebreak_available = 1, False
        drive(g, g._battle(g.region_of[p[0]], [0, 1]), by_phase(g, {0: [CYCLE], 1: [BUILD]}))
        assert (g.fig_owner[p[1]] == 0) == survives, gods


def test_radiant_domination():
    g = fresh((RA, AMUN)); a, = path(g, 1)
    put(g, 0, WARRIOR_K, a); g.fig_sun[a] = True
    drive(g, g._ev_conflict(0))
    assert g.dev_pos[0] == 2                                          # 1 (dominio) + 1 (Radiant)


def test_glorious_and_commanding():
    g = fresh(); p = path(g, 4)
    put(g, 0, GOD_K, p[0]); put(g, 0, WARRIOR_K, p[1]); put(g, 1, WARRIOR_K, p[3])
    g.powers[0] = {GLORIOUS, COMMANDING}
    g.tiebreak_holder, g.tiebreak_available = 1, False
    drive(g, g._battle(g.region_of[p[0]], [0, 1]), by_phase(g, {0: [CHARIOTS], 1: [CYCLE]}))
    assert g.dev_pos[0] == 3 and g.followers[0] == 1 + 3             # 5 vs 1: diferenca >= 3
    assert g.fig_owner[p[3]] == -1


def test_bountiful():
    g = fresh(); g.powers[0] = {BOUNTIFUL}
    g.gain_dev(0, 1); assert g.dev_pos[0] == 2                        # no vermelho: +1
    g.dev_pos_before = g.dev_pos[0]
    g.gain_dev(0, 16); assert g.dev_pos[0] == 2 + 17                  # ainda no vermelho antes: +1
    g.gain_dev(0, 1); assert g.dev_pos[0] == 2 + 17 + 1               # fora do vermelho: sem bonus


def test_underworld():
    g = fresh((OSIRIS, RA)); a, = path(g, 1)
    g.underworld[a] = True
    assert g._can_end(a, 0) and not g._can_end(a, 1)
    assert g._ter(a) == "U"


def test_scorpion_claws():
    g = fresh(); p = path(g, 3)
    a, b = p[1], p[2]                                   # Scorpion no meio; monumento ao lado
    put(g, 0, SCORPION, a); put(g, 0, WARRIOR_K, p[0])
    g.mon_type[b], g.mon_owner[b] = "temple", 1; g.supply["temple"] -= 1; g.ankh_pool[1] -= 1
    k = next(k for k in range(6) if g.cyc_nb[a][k] == b or g.cyc_nb[a][(k + 1) % 6] == b)
    g.claw[a] = k
    drive(g, g._ev_conflict(0))
    assert g.mon_type[b] is None and g.supply["temple"] == 10 and g.ankh_pool[1] == 9


def test_amun_two_cards():
    g = fresh((AMUN, RA)); p = path(g, 4)
    put(g, 0, GOD_K, p[0]); put(g, 0, WARRIOR_K, p[1]); put(g, 1, WARRIOR_K, p[3])
    g.tiebreak_holder, g.tiebreak_available = 1, False
    drive(g, g._battle(g.region_of[p[0]], [0, 1]), by_phase(g, {0: [CHARIOTS, DROUGHT], 1: [CYCLE]}, "yes"))
    assert g.used[0] == [CHARIOTS, DROUGHT] and g.amun_used is False or g.amun_used
    assert g.dev_pos[0] >= 1 and g.fig_owner[p[3]] == -1


def test_unlock_and_guardian():
    g = fresh((AMUN, RA)); g.followers[0] = 6
    for _ in range(3):
        drive(g, g._act_unlock(0))
    assert len(g.powers[0]) == 3 and g.unlock_next[0] == 3
    assert g.followers[0] == 6 - 1 - 1 - 2                           # niveis 1, 1 e 2
    assert g.gpool[0] == [g.g_card[1]] and g.g_supply[1] == 0       # simbolo no 2o desbloqueio do nivel 1
    g.followers[0] = 0
    assert drive(g, g._act_unlock(0)) is None                        # sem Seguidores: sem efeito


def test_satet_push():
    g = fresh(); p = path(g, 3)
    put(g, 0, SATET, p[0]); put(g, 1, WARRIOR_K, p[1])
    def choose(d):
        if d.phase == Phase.MOVE_SELECT:
            return g.A_DONE if g.fig_kind[p[1]] == SATET else g.A_CELL + p[0]
        if d.phase == Phase.MOVE_DEST:
            assert g.A_CELL + p[1] in np.flatnonzero(d.mask)
            return g.A_CELL + p[1]
        return int(np.flatnonzero(d.mask)[0])
    drive(g, g._act_move(0), choose)
    assert g.fig_kind[p[1]] == SATET and g.fig_owner[p[0]] == -1
    enemy = [i for i in range(g.N) if g.fig_owner[i] == 1]
    assert len(enemy) == 1 and g.dist[p[1], enemy[0]] == 1


def test_apep_water():
    g = fresh(); a, = path(g, 1)
    put(g, 0, GOD_K, a); g.gpool[0] = [APEP]
    opts = g._summon_options(0, "regular")
    water = [i for i in range(g.N) if not g.is_land[i]]
    assert APEP - 1 in opts and set(water) <= set(opts[APEP - 1])
    far = max(water, key=lambda w: g.dist[a, w])
    def choose(d):
        if d.phase == Phase.SUMMON_TYPE:
            return g.A_FIG + APEP - 1
        return g.A_CELL + far
    drive(g, g._summon_one(0, "regular"), choose)
    assert g.fig_kind[far] == APEP and g.fig_owner[far] == 0


def test_temple_and_resplendent():
    g = fresh(); p = path(g, 3)
    put(g, 0, GOD_K, p[0])
    g.mon_type[p[1]], g.mon_owner[p[1]] = "temple", 0
    r = g.region_of[p[0]]
    base = g._strength(0, r, [])
    g.powers[0] = {TEMPLE_ATT}
    assert g._strength(0, r, []) == base + 2
    g.powers[0] = {RESPLENDENT}
    for i in [i for i in range(g.N) if g._empty_land(i)][:3]:
        g.mon_type[i], g.mon_owner[i] = "obelisk", 0
    assert g._god_strength(0) == 3


def test_pyramid_attuned_extra_summon():
    g = fresh(); p = path(g, 2)
    put(g, 0, GOD_K, p[0]); g.mon_type[p[1]], g.mon_owner[p[1]] = "pyramid", 0
    g.powers[0] = {PYRAMID_ATT}
    d = drive(g, g._act_summon(0), lambda d: g.A_DONE if d.phase == Phase.SUMMON_TYPE else int(np.flatnonzero(d.mask)[0]))
    assert d.phase == Phase.SUMMON_TYPE and d.info["mode"] == "pyramid"


def test_obs_hides_secrets():
    import inspect
    src = inspect.getsource(__import__("ankh_obs").Encoder)
    assert "pending" not in src


def test_merge_gods():
    g = fresh((AMUN, RA, ISIS), scenario="basic_3p"); p = path(g, 6)
    g.gain_dev(0, 10); g.gain_dev(1, 3); g.gain_dev(2, 6)           # mais baixo: 1 (Ra); 2o mais baixo: 2 (Isis)
    put(g, 0, GOD_K, p[0]); put(g, 1, GOD_K, p[2]); put(g, 1, WARRIOR_K, p[3]); put(g, 2, GOD_K, p[4])
    m = p[5]; g.mon_type[m], g.mon_owner[m] = "temple", 1; g.supply["temple"] -= 1; g.ankh_pool[1] -= 1
    g.followers[1], g.followers[2] = 4, 2
    g._merge_gods()
    assert g.merged == (2, 1) and g.rep[1] == 2 and not g.alive[1] and g.alive[2]
    assert g._has_god(2, RA) and g._has_god(2, ISIS) and g.sun_left[2] == 3      # herda a habilidade do Ra
    assert g.mon_type[m] is None and g.supply["temple"] == 10                  # monumentos do absorvido caem
    assert not any(g.fig_owner[i] == 1 for i in range(g.N))                    # figuras do absorvido saem
    assert g.followers[2] == 6 and g.dev_pos[2] == 3                           # Seguidores somam; Devocao = a do mais baixo
    assert sorted(x for st in g.stacks for x in st) == [0, 2]


def test_merged_god_single_action_and_turns():
    g = AnkhGame(["basic_3p"], seed=1); dec = g.reset(seed=1, gods=(AMUN, RA, ISIS))
    g._merge_gods()                                                   # Devocoes iguais: funde os assentos 2 e 1
    assert g.merged == (1, 2)
    rng = np.random.default_rng(1); per_turn, last, double0 = {}, None, 0
    while dec is not None:
        if g.cur != last:
            per_turn, last = {}, g.cur
        if dec.phase == Phase.CHOOSE_ACTION:
            per_turn[g.cur] = per_turn.get(g.cur, 0) + 1
            if g.cur in (1, 2):
                assert per_turn[g.cur] <= 1, "deus fundido fez 2 acoes no mesmo turno"
            elif per_turn[g.cur] == 2:
                double0 += 1
        dec = g.step(rng.choice(g.legal_actions()))
    assert double0 > 0                                                # o assento sem fusao ainda faz 2 acoes


def test_three_way_tie_all_lose():
    g = fresh((AMUN, RA, ISIS), scenario="basic_3p"); p = path(g, 6)
    for q, i in zip((0, 1, 2), (p[0], p[2], p[4])):
        put(g, q, WARRIOR_K, i)
    g.tiebreak_holder, g.tiebreak_available = 0, False
    drive(g, g._battle(g.region_of[p[0]], [0, 1, 2]), by_phase(g, {0: [CYCLE], 1: [CYCLE], 2: [CYCLE]}))
    assert all(g.fig_owner[i] == -1 for i in (p[0], p[2], p[4]))
    assert g.dev_pos == [0, 0, 0]


def test_three_player_winner_kills_everyone_else():
    g = fresh((AMUN, RA, ISIS), scenario="basic_3p"); p = path(g, 7)
    put(g, 0, WARRIOR_K, p[0]); put(g, 0, WARRIOR_K, p[1])
    put(g, 1, WARRIOR_K, p[3]); put(g, 2, WARRIOR_K, p[5])
    g.tiebreak_holder, g.tiebreak_available = 1, False
    drive(g, g._battle(g.region_of[p[0]], [0, 1, 2]), by_phase(g, {0: [CYCLE], 1: [CYCLE], 2: [CYCLE]}))
    assert g.fig_owner[p[0]] == 0 and g.fig_owner[p[1]] == 0
    assert g.fig_owner[p[3]] == -1 and g.fig_owner[p[5]] == -1 and g.dev_pos[0] == 1


def test_cat_mummy_hits_all_other_players():
    g = fresh((AMUN, RA, ISIS), scenario="basic_3p"); a, = path(g, 1)
    for q in range(3):
        g.gain_dev(q, 5)
    put(g, 1, CATMUMMY, a)
    g._kill_pass([a], [0, 0, 0]); g._flush_cat()
    assert g.dev_pos == [4, 5, 4]


def test_forgotten_merged_god_removes_both_seats():
    g = AnkhGame(["basic_3p"], seed=2); g.reset(seed=2, gods=(AMUN, RA, ISIS))
    g._merge_gods(); g.gain_dev(0, 25)                                 # entidade 0 fora do vermelho; fundida fica no vermelho
    try:
        g._forget_gods()
    except GameEnd:
        pass
    assert g.winner == 0 and not g.alive[1] and not g.alive[2]


def test_guardian_supply_by_players():
    for sc, small, large in (("basic_2p", 1, 1), ("basic_3p", 2, 2), ("basic_4p", 3, 2)):
        g = fresh(None, scenario=sc)
        for lvl in (1, 2, 3):
            exp = small if GUARDIAN_SIZE[g.g_card[lvl]] == "small" else large
            assert g.g_supply[lvl] == exp, (sc, lvl)



if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print("ok", name)
    print(n, "testes de regras ok")
