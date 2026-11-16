"""A fusao em si nao pode gerar nem tirar recompensa de ninguem; modo estrito zera os assentos fundidos."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ankh_game import *
from ankh_map import build_board
from ankh_worker import Slot, merged_seat, reward_parts, snapshot

CFG = dict(dev_gain=0.02, dev_opp=0.01, war_gain=0.005, war_opp=0.0025, draw=-1.0, merged_reward="normal")


def test_merge_gives_no_reward_to_anyone():
    g = AnkhGame(["basic_3p"], seed=0)
    g.reset(seed=0, gods=(AMUN, RA, ISIS))
    g.gain_dev(0, 10); g.gain_dev(1, 3); g.gain_dev(2, 6)             # mais baixo: 1; 2o mais baixo: 2
    extra = [i for i in range(g.N) if g._empty_land(i)][:4]
    for e, i in zip((0, 1, 2, 2), extra):                             # guerreiros diferentes por deus
        g.fig_owner[i], g.fig_kind[i] = e, WARRIOR_K
    before = [snapshot(g, s) for s in range(3)]
    dev_before = g.dev_pos[2]
    g._merge_gods()
    assert g.merged == (2, 1) and g.dev_pos[2] != dev_before          # a posicao MUDOU (o artefato existia)...
    for s in range(3):                                                # ...mas ninguem recebe recompensa por isso
        assert reward_parts(CFG, before[s], snapshot(g, s)) == (0.0, 0.0), s


def test_real_gains_still_rewarded_after_merge():
    g = AnkhGame(["basic_3p"], seed=0)
    g.reset(seed=0, gods=(AMUN, RA, ISIS))
    g.gain_dev(0, 10); g.gain_dev(1, 3); g.gain_dev(2, 6)             # o assento 0 fica de fora da fusao
    g._merge_gods()
    before = snapshot(g, 0)
    g.gain_dev(0, 4)                                                  # ganho de Devocao de verdade (assento 0)
    r_dev, _ = reward_parts(CFG, before, snapshot(g, 0))
    assert abs(r_dev - 0.02 * 4) < 1e-9


def _run_slot(mode, seed):
    cfg = dict(CFG, merged_reward=mode)
    slot = Slot(build_board(), ["basic_3p"], seed, cfg)
    slot.g.reset(seed=seed, gods=(AMUN, RA, ISIS), scenario="basic_3p")
    slot.me = 1
    slot.g._merge_gods()                                              # assentos 1 (superior) e 2 (absorvido)
    assert merged_seat(slot.g, 1) and merged_seat(slot.g, 2) and not merged_seat(slot.g, 0)
    slot.fresh, slot.prev, slot.acc = False, slot.snap(), 0.0
    rng = np.random.default_rng(seed)
    while True:
        ep = slot.apply(rng.choice(np.flatnonzero(slot.g.decision.mask)))
        if ep is not None:
            return ep[0]


def test_strict_mode_zeroes_merged_seats():
    for seed in range(3):
        assert _run_slot("zero", seed) == 0.0
        assert _run_slot("normal", seed) != 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("recompensa e fusao ok")
