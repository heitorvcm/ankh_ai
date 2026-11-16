"""
ankh_worker.py - Processo que roda varias partidas de Ankh (CPU) para o VecEnv.

Nao depende de gymnasium nem de stable-baselines3 (para o processo 'spawn'
iniciar rapido e sem tocar na GPU). Cada "slot" e uma partida com um ASSENTO do
agente (`me`, sorteado por episodio entre os P assentos do cenario sorteado) e
os demais assentos controlados por um oponente (a rede na GPU, em lote, ou
jogadas aleatorias dentro do proprio worker). O worker devolve a proxima
decisao pendente:
  kind='agent' -> o agente deve agir;  kind='opp' -> um oponente deve agir.

Recompensa do agente: fim de jogo (+1 vitoria, -1/(P-1) derrota, `draw` no Egito
ateu; com P=2 a derrota vale -1); + dev_gain por casa de Devocao que o agente subiu,
- dev_opp pela media das casas que os oponentes subiram; idem guerreiros no
tabuleiro (war_gain / war_opp). Se ele esta num deus fundido, a recompensa e da
entidade fundida.
"""
import numpy as np

from ankh_game import AnkhGame
from ankh_map import build_board
from ankh_obs import Encoder


def _unpack(t):
    """Aceita o formato antigo (dev_eu, dev_op, guerr_eu, guerr_op) e o novo (com dicts por oponente)."""
    if len(t) == 4:
        return t[0], {0: t[1]}, t[2], {0: t[3]}, None
    return t


def reward_parts(c, prev, now):
    """(r_dev, r_war) de uma transicao, do ponto de vista do agente."""
    pd, pod, pw, pow_, pe = _unpack(prev)
    nd, nod, nw, now_, ne = _unpack(now)
    if pe != ne:                                   # a entidade do agente mudou (fusao): sem delta nesse passo
        return 0.0, 0.0
    common = [e for e in nod if e in pod]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    d_opp = mean([nod[e] - pod[e] for e in common])
    w_opp = mean([now_[e] - pow_[e] for e in common])
    return (c["dev_gain"] * (nd - pd) - c["dev_opp"] * d_opp,
            c["war_gain"] * (nw - pw) - c["war_opp"] * w_opp)


def step_reward(c, prev, now):
    """Recompensa densa total de uma transicao."""
    a, b = reward_parts(c, prev, now)
    return a + b


def end_reward(c, me, winner, n_players=2):
    """+1 vitoria, -1/(P-1) derrota (P=2: -1), c['draw'] se todos os deuses foram esquecidos."""
    if winner is None:
        return c["draw"]
    return 1.0 if winner == me else -1.0 / (n_players - 1)


def snapshot(g, me):
    """Estado relevante para a recompensa do assento `me`. Usa a Devocao acumulada `rdev` (que a
    fusao NAO altera) e os guerreiros no tabuleiro: assim a fusao em si nao gera nem tira recompensa."""
    r = g.rep[me]
    opps = [e for e in range(g.P) if e != r and g.rep[e] == e and g.alive[e]]
    return (g.rdev[r], {e: g.rdev[e] for e in opps}, g.board_warriors(r),
            {e: g.board_warriors(e) for e in opps}, r)


def merged_seat(g, me):
    """O assento `me` pertence a um deus fundido (superior ou absorvido)?"""
    return g.merged is not None and me in g.merged


class Slot:
    def __init__(self, board, scenarios, seed, cfg):
        self.g = AnkhGame(scenarios, seed=seed, board=board, god_pool=cfg.get("god_pool"))
        self.enc = Encoder(self.g)
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.start()

    def start(self):
        self.g.reset(seed=int(self.rng.integers(2**31)))          # sorteia cenario e deuses
        self.me = int(self.rng.integers(self.g.P))                # assento do agente
        self.fresh, self.acc, self.prev = True, 0.0, None         # aberturas dos oponentes nao dao recompensa

    def snap(self):
        return snapshot(self.g, self.me)

    def apply(self, a):
        g, c = self.g, self.cfg
        g.step(int(a))
        zero = c.get("merged_reward") == "zero" and merged_seat(g, self.me)     # modo estrito
        if not self.fresh:
            now = self.snap()
            if not zero:
                self.acc += step_reward(c, self.prev, now)
            self.prev = now
        if g.done:
            r = g.rep[self.me]
            end = 0.0 if zero else end_reward(c, r, g.winner, g.P)
            opp0 = next((e for e in range(g.P) if e != r and g.rep[e] == e), None)
            ep = (self.acc + end, {"winner": g.winner, "me": self.me, "r_end": end,
                                   "god_me": g.god[r], "god_opp": g.god[opp0] if opp0 is not None else None,
                                   "summary": g.summary(self.me)})
            self.start()
            return ep
        return None

    def result(self, ep, random_opp):
        g = self.g
        while True:
            if g.decision.player == self.me:
                if self.fresh:
                    self.prev, self.fresh, self.acc = self.snap(), False, 0.0
                kind = "agent"
                break
            if not random_opp:
                kind = "opp"
                break
            e = self.apply(self.rng.choice(np.flatnonzero(g.decision.mask)))
            ep = e if e is not None else ep
        dec = g.decision
        acc = 0.0
        if kind == "agent":
            acc, self.acc = self.acc, 0.0
        return (kind, self.enc.encode(dec), dec.mask, ep, acc)


def worker_main(conn, n_slots, seed, scenarios, cfg):
    board = build_board()
    slots = [Slot(board, scenarios, seed + i, cfg) for i in range(n_slots)]
    random_opp = True
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                for s in slots:
                    s.start()
                conn.send([s.result(None, random_opp) for s in slots])
            elif cmd == "apply":
                ids, actions = data
                conn.send([slots[i].result(slots[i].apply(a), random_opp) for i, a in zip(ids, actions)])
            elif cmd == "opp_random":
                random_opp = bool(data)
                conn.send(True)
            elif cmd == "close":
                break
    finally:
        conn.close()
