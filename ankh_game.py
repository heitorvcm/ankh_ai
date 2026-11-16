"""
ankh_game.py - Nucleo de regras do Ankh: Gods of Egypt (v2).

Escopo
------
 * 2 jogadores (sem fusao de deuses), cenarios de ankh_map.SCENARIOS.
 * Os 5 deuses com suas habilidades (Amun, Anubis, Isis, Osiris, Ra).
 * Os 12 Poderes Ankh e a acao Desbloquear de verdade (custo em Seguidores).
 * Os 6 Guardioes (Mummy, Satet, Apep, Giant Scorpion, Androsphinx, Cat-Mummy).
 * As 7 cartas de batalha.

Interface
---------
    g = AnkhGame("basic_2p", seed=0)
    dec = g.reset()              # Decision(player, phase, mask, info) ou None
    while dec is not None:
        a = escolha entre np.flatnonzero(dec.mask)
        dec = g.step(a)
    g.winner                     # 0, 1 ou None (empate / Egito ateu)

Cada step e UMA decisao de UM jogador. Escolhas secretas (carta, lance do Plague)
sao pedidas em sequencia; o observador deve esconder g.pending_card / g.pending_bid.
O jogo e um gerador Python: o estado NAO pode ser copiado (sem deepcopy/MCTS).

DADOS E INTERPRETACOES A CONFIRMAR (marcados com [ASSUMIDO])
------------------------------------------------------------
 * GUARDIAN_CARDS: quais Guardioes formam cada carta de nivel (o manual nao diz).
 * GUARDIAN_SLOT: em qual das 2 casas da fileira inferior do painel de cada deus
   fica o simbolo de Guardiao de cada nivel (so o nivel 1 do Amun aparece no manual).
 * ANUBIS_SLOTS = 3 casas para guerreiros presos.
 * Escolhas automaticas (sem decisao do jogador): Anubis sempre prende 1 guerreiro
   quando ha vaga; Isis sempre salva figuras protegidas; Ra coloca o Sol no figura
   invocada quando ha token; Mummy volta na menor casa livre ao lado do deus; Satet
   empurrando um Giant Scorpion aponta as garras para o melhor alvo.
 * Obelisk Attuned: cada jogador faz todos os seus movimentos de uma vez (o manual
   manda alternar 1 figura por vez).
 * Sem cartas na mao, o jogador nao joga carta; lance do Plague ate BID_MAX; o
   evento de Camelo pode ser recusado; uma rodada de lances por carta Plague.
 * Underworld: so o MOVIMENTO e bloqueado para figuras de outros deuses (como no texto).
"""

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from ankh_map import DIRECTIONS, SCENARIOS, build_board, components, edge, load_scenario, nbrs

# ---------------------------------------------------------------------------
# Constantes de regras
# ---------------------------------------------------------------------------
TRACK_LEN = 31            # casas da Devocao; a ultima (indice 30) vence o jogo
TOP = TRACK_LEN - 1
RED_SPACES = 19           # casas vermelhas (indices 0..RED_SPACES-1) - confirmado
ACTION_END = 6
ACTION_START = {2: 3, 3: 2, 4: 1, 5: 0}
EVENTS = (["ctrl"] * 3 + ["conf", "camel"] + ["ctrl"] * 2 + ["conf", "camel"]
          + ["ctrl"] * 2 + ["conf", "camel"] + ["ctrl"] * 2 + ["conf", "ctrl", "conf"])
assert len(EVENTS) == 18
N_TOKENS = 8
CAMELS_TOTAL = 30
MAX_LINE = 6
MIN_REGION = 6
BID_MAX = 15
K_LINES = 128
N_CHOICE = 16

# cartas de batalha
PLAGUE, BUILD, CHARIOTS, CYCLE, DROUGHT, FLOOD, MIRACLE = range(7)
CARD_NAMES = ["plague", "build", "chariots", "cycle", "drought", "flood", "miracle"]
CARD_BONUS = [1, 0, 3, 0, 1, 0, 0]
MON_TYPES = ["obelisk", "temple", "pyramid"]

# deuses
GODS = ["amun", "anubis", "isis", "osiris", "ra"]
AMUN, ANUBIS, ISIS, OSIRIS, RA = range(5)

# poderes Ankh (os mesmos 12 para todos os deuses)
(COMMANDING, INSPIRING, OMNIPRESENT, REVERED, RESPLENDENT, OBELISK_ATT, TEMPLE_ATT, PYRAMID_ATT,
 GLORIOUS, MAGNANIMOUS, BOUNTIFUL, WORSHIPFUL) = range(12)
POWER_NAMES = ["commanding", "inspiring", "omnipresent", "revered", "resplendent", "obelisk_attuned",
               "temple_attuned", "pyramid_attuned", "glorious", "magnanimous", "bountiful", "worshipful"]
POWERS_BY_LEVEL = {1: [0, 1, 2, 3], 2: [4, 5, 6, 7], 3: [8, 9, 10, 11]}

# tipos de figura
GOD_K, WARRIOR_K, MUMMY, SATET, APEP, SCORPION, ANDRO, CATMUMMY = range(1, 9)
KIND_NAMES = {1: "deus", 2: "guerreiro", 3: "mummy", 4: "satet", 5: "apep",
              6: "giant_scorpion", 7: "androsphinx", 8: "cat_mummy"}
GUARDIAN_KINDS = (MUMMY, SATET, APEP, SCORPION, ANDRO, CATMUMMY)
GUARDIAN_SIZE = {MUMMY: "small", SATET: "small", CATMUMMY: "small",
                 APEP: "large", SCORPION: "large", ANDRO: "large"}
GUARDIAN_CARDS = {1: [SATET, MUMMY], 2: [APEP, SCORPION], 3: [ANDRO, CATMUMMY]}   # [ASSUMIDO]
GUARDIAN_SLOT = {g: [1, 1, 1] for g in range(5)}   # [ASSUMIDO] casa (0 ou 1) que revela o simbolo
ANUBIS_SLOTS = 3                                   # [ASSUMIDO]
CYCLIC = ["N", "NE", "SE", "S", "SW", "NW"]        # ordem das 6 direcoes ao redor de um hexagono

# Espacos RESERVADOS (formato fixo para crescer sem invalidar pesos)
N_POWERS = 12
N_GODS = 5
N_FIG_TYPES = 8           # 0 guerreiro (reserva), 1 guerreiro preso, 2..7 guardioes (tipo - 1)
N_EXTRA = 32


class Phase(IntEnum):
    CHOOSE_ACTION = 0
    MOVE_SELECT = 1
    MOVE_DEST = 2
    SUMMON_DEST = 3
    CONTROL_PICK = 4
    CAMEL_LINE = 5
    CAMEL_KEEP = 6
    CAMEL_SWAP = 7
    BATTLE_CARD = 8
    BUILD_TYPE = 9
    BUILD_CELL = 10
    BID = 11
    TIEBREAK = 12
    UNLOCK_PICK = 13
    SUMMON_TYPE = 14
    AMUN_DECLARE = 15
    CLAW_POINT = 16
    PUSH_DEST = 17
    UNDERWORLD_SRC = 18
    UNDERWORLD_DEST = 19
    ATTUNE_SRC = 20
    ATTUNE_DEST = 21
    WORSHIP = 22
    RESERVED_A = 23
    RESERVED_B = 24
    RESERVED_C = 25
    RESERVED_D = 26
    RESERVED_E = 27
    RESERVED_F = 28
    RESERVED_G = 29
    RESERVED_H = 30
    RESERVED_I = 31


@dataclass
class Decision:
    player: int
    phase: Phase
    mask: np.ndarray
    info: dict = field(default_factory=dict)


class GameEnd(Exception):
    pass


class AnkhGame:
    def __init__(self, scenario="basic_2p", seed=None, board=None, god_pool=None):
        self.board = board if board is not None else build_board()   # Board e somente leitura
        names = [scenario] if isinstance(scenario, str) else list(scenario)
        self.scenarios, self.skipped = [], {}
        for n in names:                                  # cenarios com erro de dados sao ignorados
            errs = load_scenario(self.board, SCENARIOS[n]).errors
            if errs:
                self.skipped[n] = errs
            else:
                self.scenarios.append(n)
        if not self.scenarios:
            raise ValueError(f"nenhum cenario valido: {self.skipped}")
        self.scenario = self.scenarios[0]
        self.P = SCENARIOS[self.scenario]["n_players"]
        self._scenario_override = None
        self.god_pool = list(god_pool) if god_pool else list(range(N_GODS))
        self.rng = np.random.default_rng(seed)
        b = self.board
        self.N = len(b.cells)
        self.ax = [b.axial(rc) for rc in b.cells]
        self.ax2i = {a: i for i, a in enumerate(self.ax)}
        self.terrain = [b.terrain[rc] for rc in b.cells]
        self.is_land = [t != "W" for t in self.terrain]
        self.dist = b.move_dist
        self.nb = [[b.idx[b.step(rc, d)] for d in DIRECTIONS if b.step(rc, d) is not None]
                   for rc in b.cells]
        self.cyc_nb = [[(b.idx[b.step(rc, d)] if b.step(rc, d) is not None else -1) for d in CYCLIC]
                       for rc in b.cells]
        N = self.N
        self.A_TRACK = 0
        self.A_CELL = 4
        self.A_DONE = self.A_CELL + N
        self.A_CARD = self.A_DONE + 1
        self.A_BID = self.A_CARD + 7
        self.A_YESNO = self.A_BID + BID_MAX + 1
        self.A_BTYPE = self.A_YESNO + 2
        self.A_CAMEL = self.A_BTYPE + 4
        self.A_CHOICE = self.A_CAMEL + K_LINES + 1
        self.A_POWER = self.A_CHOICE + N_CHOICE
        self.A_FIG = self.A_POWER + N_POWERS
        self.A_EXTRA = self.A_FIG + N_FIG_TYPES
        self.A_SIZE = self.A_EXTRA + N_EXTRA
        self.decision = None
        self.done = True
        self._gods_override = None

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def reset(self, seed=None, gods=None, scenario=None):
        """gods: um deus por jogador ou None (sorteia do god_pool); scenario: nome ou None (sorteia)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._gods_override = gods
        self._scenario_override = scenario
        self._setup()
        self.done = False
        self.winner = None
        self.steps = 0
        self._gen = self._main()
        self.decision = self._advance(None)
        return self.decision

    def step(self, action):
        if self.done:
            raise RuntimeError("jogo terminou; chame reset()")
        d = self.decision
        if not (0 <= action < self.A_SIZE and d.mask[action]):
            raise ValueError(f"acao ilegal {action} na fase {d.phase.name}")
        self.steps += 1
        self.decision = self._advance(int(action))
        return self.decision

    def legal_actions(self):
        return np.flatnonzero(self.decision.mask)

    def _advance(self, a):
        try:
            return self._gen.send(a)
        except (GameEnd, StopIteration):
            self.done = True
            return None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _setup(self):
        N, b = self.N, self.board
        name = self._scenario_override or self.scenarios[int(self.rng.integers(len(self.scenarios)))]
        self.scenario = name
        P = self.P = SCENARIOS[name]["n_players"]
        setup = load_scenario(b, SCENARIOS[name])
        assert not setup.errors, setup.errors
        self.fig_owner = [-1] * N
        self.fig_kind = [0] * N
        self.fig_sun = [False] * N               # Ra: figura Radiant
        self.claw = [None] * N                   # Giant Scorpion: orientacao das garras (0..5)
        self.mon_type = [None] * N
        self.mon_owner = [-1] * N
        self.warriors_pool = [6] * P
        self.followers = [1] * P
        self.ankh_pool = [9] * P
        self.supply = {t: 10 for t in MON_TYPES}
        for kind, rc, owner in setup.monuments:
            i = b.idx[rc]
            self.mon_type[i] = kind
            self.supply[kind] -= 1
            if owner is not None:
                self.mon_owner[i] = owner
                self.ankh_pool[owner] -= 1
        for p, figs in setup.figures.items():
            for kind, rc in figs:
                i = b.idx[rc]
                self.fig_owner[i] = p
                self.fig_kind[i] = GOD_K if kind == "god" else WARRIOR_K
                if kind == "warrior":
                    self.warriors_pool[p] -= 1
        self.camel_edges = set(setup.camel_edges)
        self.camel_supply = CAMELS_TOTAL - len(self.camel_edges)
        self.tokens = {}
        for ri, tok in setup.conflict_order.items():
            self.tokens[frozenset(self.ax2i[a] for a in setup.regions[ri])] = tok
        self._recompute_regions()
        # devocao
        self.dev_pos = [0] * P
        self.stacks = [[] for _ in range(TRACK_LEN)]
        for p in reversed(range(P)):
            self.stacks[0].append(p)
        self.rdev = [0] * P                      # Devocao acumulada para RECOMPENSA (a fusao nao a altera)
        self.alive = [True] * P                  # por ENTIDADE (deus); a fusao junta dois assentos
        self.rep = list(range(P))                # assento -> entidade que ele controla
        self.god2 = [None] * P                   # segundo deus de uma entidade fundida
        self.merged = None                       # (entidade, assento absorvido) apos a fusao
        self.markers = [ACTION_START[P]] * 4
        self.event_idx = 0
        self.conflicts_done = 0
        self.hand = [set(range(7)) for _ in range(P)]
        self.used = [[] for _ in range(P)]
        self.pending_card = [None] * P
        self.pending_bid = [None] * P
        self.tiebreak_holder = None
        self.tiebreak_available = False
        self.turn_first = None
        self.cur_region = None
        self.cur_cards = None
        self.cur = 0
        # deuses, poderes, guardioes
        gods = self._gods_override
        if gods is None:
            gods = self.rng.choice(self.god_pool, size=P, replace=len(self.god_pool) < P)
        assert len(gods) == P, f"{P} deuses necessarios, recebi {len(gods)}"
        self.god = [int(x) for x in gods]
        self.powers = [set() for _ in range(P)]
        self.unlock_next = [0] * P
        self.g_card = {l: GUARDIAN_CARDS[l][int(self.rng.integers(2))] for l in (1, 2, 3)}
        small, large = {2: 1, 3: 2}.get(P, 3), {2: 1, 3: 2}.get(P, 2)
        self.g_supply = {l: (small if GUARDIAN_SIZE[self.g_card[l]] == "small" else large) for l in (1, 2, 3)}
        self.gpool = [[] for _ in range(P)]
        self.bases_used = {"small": [0] * P, "large": [0] * P}
        self.sun_left = [3 if g == RA else 0 for g in self.god]
        self.uw_off = [3 if g == OSIRIS else 0 for g in self.god]
        self.underworld = [False] * N
        self.trapped = [[] for _ in range(P)]    # trapped[a] = donos dos guerreiros presos por Anubis a
        self.amun_used = False
        self.amun_declared = {}
        self.cat_pending = []
        self.stats = self._new_stats()

    def _new_stats(self):
        P = self.P
        return {"end_reason": None, "actions": [[0] * 4 for _ in range(P)],
                "dev_src": [{} for _ in range(P)], "dev_lost": [0] * P,
                "battles": 0, "battle_wins": [0] * P, "battle_ties": 0, "dominations": [0] * P,
                "cards": [[0] * 7 for _ in range(P)], "kills": [0] * P,
                "unlocks": [[] for _ in range(P)], "guardians": [[] for _ in range(P)],
                "followers_action": [0] * P, "monuments_destroyed": 0, "tiebreak_used": 0,
                "amun_double": 0, "pushes": 0, "underworld_moves": 0, "builds": [0] * P,
                "plague_rounds": 0}

    def summary(self, me=0):
        """Resumo da partida do ponto de vista do ASSENTO `me`: pares [eu, media dos oponentes]."""
        r = self.rep[me]
        opps = [e for e in range(self.P) if e != r and self.rep[e] == e]
        st = self.stats
        mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
        num = lambda x: [x[r], mean([x[e] for e in opps])]
        vec = lambda x: [list(x[r]), [mean([x[e][k] for e in opps]) for k in range(len(x[r]))]]

        def dsrc(x):
            keys = set().union(*[set(x[e]) for e in opps]) if opps else set()
            return [dict(x[r]), {k: mean([x[e].get(k, 0) for e in opps]) for k in keys}]

        result = "draw" if self.winner is None else ("win" if self.winner == r else "loss")
        return {
            "result": result, "me_first": me == 0, "me_seat": me, "n_players": self.P,
            "scenario": self.scenario, "merged": self.merged is not None,
            "merge_role": None if self.merged is None else (
                "superior" if me == self.merged[0] else "absorvido" if me == self.merged[1] else None),
            "god": [self.god[r], self.god[opps[0]] if opps else None], "steps": self.steps,
            "end_reason": st["end_reason"], "events": self.event_idx, "conflicts": self.conflicts_done,
            "dev_final": num(self.dev_pos), "followers": num(self.followers),
            "monuments": [sum(1 for m in range(self.N) if self.mon_owner[m] == e) for e in [r] + opps[:1]]
            if len(opps) == 1 else [sum(1 for m in range(self.N) if self.mon_owner[m] == r),
                                    mean([sum(1 for m in range(self.N) if self.mon_owner[m] == e) for e in opps])],
            "powers": [sorted(self.powers[r]), sorted(set().union(*[self.powers[e] for e in opps])) if opps else []],
            "actions": vec(st["actions"]), "dev_src": dsrc(st["dev_src"]), "dev_lost": num(st["dev_lost"]),
            "battles": st["battles"], "battle_wins": num(st["battle_wins"]), "battle_ties": st["battle_ties"],
            "dominations": num(st["dominations"]), "cards": vec(st["cards"]), "kills": num(st["kills"]),
            "unlocks": [list(st["unlocks"][r]), [w for e in opps for w in st["unlocks"][e]]],
            "guardians": [list(st["guardians"][r]), [k for e in opps for k in st["guardians"][e]]],
            "builds": num(st["builds"]), "followers_action": num(st["followers_action"]),
            "monuments_destroyed": st["monuments_destroyed"], "tiebreak_used": st["tiebreak_used"],
            "amun_double": st["amun_double"], "pushes": st["pushes"], "camels": len(self.camel_edges),
            "regions": len(self.regions),
        }

    def _recompute_regions(self):
        blocked = self.board.river_edges | self.camel_edges
        comps = components(self.board.land, blocked)
        self.regions = [frozenset(self.ax2i[a] for a in c) for c in comps]
        self.region_of = {i: r for r in self.regions for i in r}
        self.adj = [[] for _ in range(self.N)]           # terra-terra sem rio/camelo
        for i in range(self.N):
            if not self.is_land[i]:
                continue
            for j in self.nb[i]:
                if self.is_land[j] and edge(self.ax[i], self.ax[j]) not in blocked:
                    self.adj[i].append(j)
        # adjacencia completa: agua e adjacente a todas as casas ao redor
        self.adjl = [(self.adj[i] + [j for j in self.nb[i] if not self.is_land[j]]) if self.is_land[i]
                     else list(self.nb[i]) for i in range(self.N)]
        self.adjs = [set(x) for x in self.adjl]
        # regiao + agua vizinha (figuras na agua participam de todas as regioes adjacentes)
        self.region_full = {r: frozenset(set(r) | {w for i in r for w in self.nb[i] if not self.is_land[w]})
                            for r in self.regions}

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    def _dec(self, player, phase, idxs, **info):
        mask = np.zeros(self.A_SIZE, dtype=bool)
        mask[list(idxs)] = True
        return Decision(player, phase, mask, info)

    def dev_key(self, p):
        pos = self.dev_pos[p]
        return (pos, self.stacks[pos].index(p))

    def _asc(self, players):
        return sorted(players, key=self.dev_key)

    def gain_dev(self, p, n, bountiful=True, src="outros"):
        """Sobe n casas. Bountiful: +1 por 'instancia' de ganho enquanto no vermelho."""
        if n <= 0 or not self.alive[p]:
            return
        ds = self.stats["dev_src"][p]
        if bountiful and BOUNTIFUL in self.powers[p] and self.dev_pos[p] < RED_SPACES:
            n += 1
            ds["bountiful"] = ds.get("bountiful", 0) + 1
            ds[src] = ds.get(src, 0) + n - 1
        else:
            ds[src] = ds.get(src, 0) + n
        old = self.dev_pos[p]
        new = min(old + n, TOP)
        self.stacks[old].remove(p)
        self.stacks[new].append(p)
        self.dev_pos[p] = new
        self.rdev[p] += new - old
        if new == TOP:
            self.winner = p
            self.stats["end_reason"] = "topo"
            raise GameEnd

    def lose_dev(self, p, n):
        if n <= 0 or not self.alive[p]:
            return
        self.stats["dev_lost"][p] += n
        old = self.dev_pos[p]
        new = max(old - n, 0)
        self.stacks[old].remove(p)
        self.stacks[new].append(p)
        self.dev_pos[p] = new
        self.rdev[p] += new - old

    def board_warriors(self, e):
        return sum(1 for i in range(self.N) if self.fig_owner[i] == e and self.fig_kind[i] == WARRIOR_K)

    def _has_god(self, q, g):
        return self.god[q] == g or self.god2[q] == g

    def _empty_land(self, i):
        return self.is_land[i] and self.fig_owner[i] == -1 and self.mon_type[i] is None

    def _can_end(self, i, p):
        """Casa onde a figura de p pode terminar um movimento (Underworld bloqueia outros deuses)."""
        return self._empty_land(i) and (not self.underworld[i] or self._has_god(p, OSIRIS))

    def _ter(self, i):
        return "U" if self.underworld[i] else self.terrain[i]

    def _figs_in(self, region, p):
        return [i for i in self.region_full[region] if self.fig_owner[i] == p]

    def _move_fig(self, s, d):
        self.fig_owner[d], self.fig_kind[d] = self.fig_owner[s], self.fig_kind[s]
        self.fig_sun[d], self.claw[d] = self.fig_sun[s], self.claw[s]
        self.fig_owner[s], self.fig_kind[s], self.fig_sun[s], self.claw[s] = -1, 0, False, None

    def _place_figure(self, p, kind, i):
        """Coloca uma figura de p em i (invocacao): Ra atribui o Sol se houver token."""
        self.fig_owner[i], self.fig_kind[i], self.claw[i] = p, kind, None
        self.fig_sun[i] = False
        if self._has_god(p, RA) and self.sun_left[p] > 0:
            self.fig_sun[i] = True
            self.sun_left[p] -= 1

    def _neutral(self, i):
        """Figura cuja forca nao conta: adjacente a um Androsphinx inimigo."""
        o = self.fig_owner[i]
        return any(self.fig_kind[j] == ANDRO and self.fig_owner[j] not in (-1, o) for j in self.adjl[i])

    def _god_strength(self, q):
        s = 1
        if RESPLENDENT in self.powers[q]:
            for t in MON_TYPES:
                if sum(1 for m in range(self.N) if self.mon_type[m] == t and self.mon_owner[m] == q) >= 3:
                    s = 3
        if self._has_god(q, ANUBIS):
            s += min(3, len(self.trapped[q]))
        return s

    def _strength(self, q, region, cards):
        figs = self._figs_in(region, q)
        if not figs:
            return 0
        total = 0
        for i in figs:
            if self._neutral(i):
                continue
            total += self._god_strength(q) if self.fig_kind[i] == GOD_K else 1
        if TEMPLE_ATT in self.powers[q]:
            for m in region:
                if (self.mon_type[m] == "temple" and self.mon_owner[m] == q
                        and any(self.fig_owner[j] == q for j in self.adjl[m])):
                    total += 2
        return total + sum(CARD_BONUS[c] for c in cards)

    def _majorities(self, region):
        counts = [{t: 0 for t in MON_TYPES} for _ in range(self.P)]
        for i in region:
            if self.mon_type[i] is not None and self.mon_owner[i] >= 0:
                counts[self.mon_owner[i]][self.mon_type[i]] += 1
        maj = [0] * self.P
        for t in MON_TYPES:
            for p in range(self.P):
                c = counts[p][t]
                if c > 0 and all(c > counts[q][t] for q in range(self.P) if q != p):
                    maj[p] += 1
        return maj

    def _destroy_monument(self, m):
        if self.mon_owner[m] >= 0:
            self.ankh_pool[self.mon_owner[m]] += 1
        self.supply[self.mon_type[m]] += 1
        self.mon_type[m], self.mon_owner[m] = None, -1

    def _kill_pass(self, cells, killed):
        """Mata as figuras em `cells` (deuses nunca morrem) e aplica os efeitos de morte:
        Anubis prende 1 guerreiro por momento, Mummy volta, Cat-Mummy agenda a penalidade,
        o Sol de Ra volta ao painel."""
        warriors, revive = [], []
        for i in cells:
            o, k = self.fig_owner[i], self.fig_kind[i]
            if o < 0 or k == GOD_K:
                continue
            killed[o] += 1
            if k == WARRIOR_K:
                warriors.append(o)
            else:
                self.gpool[o].append(k)
                if k == CATMUMMY:
                    self.cat_pending.append(o)
                if k == MUMMY:
                    revive.append(o)
            if self.fig_sun[i]:
                self.sun_left[o] += 1
            self.fig_owner[i], self.fig_kind[i], self.fig_sun[i], self.claw[i] = -1, 0, False, None
        for a in range(self.P):
            if self.alive[a] and self._has_god(a, ANUBIS) and len(self.trapped[a]) < ANUBIS_SLOTS:
                for k, o in enumerate(warriors):
                    if o != a:
                        self.trapped[a].append(warriors.pop(k))
                        break
        for o in warriors:
            self.warriors_pool[o] += 1
        for o in revive:
            self._revive_mummy(o)

    def _revive_mummy(self, o):
        if MUMMY not in self.gpool[o]:
            return
        gc = next((i for i in range(self.N) if self.fig_owner[i] == o and self.fig_kind[i] == GOD_K), None)
        if gc is None:
            return
        cand = [j for j in self.adj[gc] if self._can_end(j, o)]
        if cand:
            self.gpool[o].remove(MUMMY)
            self._place_figure(o, MUMMY, min(cand))

    def _flush_cat(self):
        for o in self.cat_pending:
            for q in range(self.P):
                if q != o:
                    self.lose_dev(q, 1)
        self.cat_pending = []

    def _auto_claw(self, i):
        """Garras do Scorpion apontadas para o par de casas com mais monumentos inimigos."""
        o, best, bk = self.fig_owner[i], -1, 0
        for k in range(6):
            score = 0
            for d in (k, (k + 1) % 6):
                j = self.cyc_nb[i][d]
                if j >= 0 and self.mon_type[j] is not None and j in self.adjs[i]:
                    score += 2 if self.mon_owner[j] != o else -1
            if score > best:
                best, bk = score, k
        self.claw[i] = bk

    # ------------------------------------------------------------------
    # Laco principal
    # ------------------------------------------------------------------
    def _main(self):
        while True:
            p = self.cur                                   # assento da vez
            r = self.rep[p]                                # entidade que ele controla
            single = self.merged is not None and r == self.merged[0]
            self.turn_first = None
            first = yield self._dec(p, Phase.CHOOSE_ACTION, range(4))
            self.turn_first = first
            if (yield from self._perform(p, first)):
                pass
            elif first < 3 and not single:                 # deus fundido: so 1 acao por turno
                second = yield self._dec(p, Phase.CHOOSE_ACTION, range(first + 1, 4))
                yield from self._perform(p, second)
            self.cur = (self.cur + 1) % self.P
            while not self.alive[self.rep[self.cur]]:
                self.cur = (self.cur + 1) % self.P

    def _perform(self, p, line):
        r = self.rep[p]
        self.markers[line] += 1
        self.stats["actions"][r][line] += 1
        if line == 0:
            yield from self._act_move(r)
        elif line == 1:
            yield from self._act_summon(r)
        elif line == 2:
            self._act_followers(r)
        else:
            yield from self._act_unlock(r)
        if self.markers[line] < ACTION_END:
            return False
        kind = EVENTS[self.event_idx]
        self.event_idx += 1
        if kind == "ctrl":
            yield from self._ev_control(r)
        elif kind == "camel":
            yield from self._ev_camel(r)
        else:
            yield from self._ev_conflict(r)
        self.markers[line] = ACTION_START[self.P]
        if self.event_idx == len(EVENTS):
            alive = [q for q in range(self.P) if self.alive[q]]
            self.winner = max(alive, key=self.dev_key)
            self.stats["end_reason"] = "ultimo_evento"
            raise GameEnd
        return True

    # ------------------------------------------------------------------
    # Acoes
    # ------------------------------------------------------------------
    def _push_opts(self, j):
        """Casas para onde Satet pode empurrar a figura inimiga em j (1 casa)."""
        q = self.fig_owner[j]
        return [k for k in range(self.N) if k != j and self.dist[j, k] == 1 and self._can_end(k, q)]

    def _point_claws(self, p, i, seat=None):
        seat = self.cur if seat is None else seat
        a = yield self._dec(seat, Phase.CLAW_POINT, [self.A_CHOICE + k for k in range(6)], src=i)
        self.claw[i] = a - self.A_CHOICE

    def _act_move(self, p):
        moved = set()
        while True:
            srcs = [i for i in range(self.N) if self.fig_owner[i] == p and i not in moved]
            if not srcs:
                return
            a = yield self._dec(self.cur, Phase.MOVE_SELECT, [self.A_CELL + i for i in srcs] + [self.A_DONE])
            if a == self.A_DONE:
                return
            s = a - self.A_CELL
            kind = self.fig_kind[s]
            dests = [s] + [j for j in range(self.N) if j != s and self.dist[s, j] <= 3 and self._can_end(j, p)]
            pushes = []
            if kind == SATET:
                pushes = [j for j in range(self.N) if j != s and self.dist[s, j] <= 3 and self.is_land[j]
                          and self.fig_owner[j] not in (-1, p) and self._push_opts(j)]
            a = yield self._dec(self.cur, Phase.MOVE_DEST, [self.A_CELL + j for j in dests + pushes], src=s)
            d = a - self.A_CELL
            if d in pushes:
                opts = self._push_opts(d)
                a = yield self._dec(self.cur, Phase.PUSH_DEST, [self.A_CELL + k for k in opts], src=d)
                k = a - self.A_CELL
                self.stats["pushes"] += 1
                self._move_fig(d, k)
                if self.fig_kind[k] == SCORPION:
                    self._auto_claw(k)
            if d != s:
                self._move_fig(s, d)
                if kind == SCORPION:
                    yield from self._point_claws(p, d)
            moved.add(d)

    def _summon_options(self, p, mode, pyr=None):
        """{tipo: [casas legais]}; tipo 0 = guerreiro do pool, 1 = guerreiro preso por Anubis, k-1 = guardiao k."""
        N = self.N
        if mode == "regular":
            cells = [i for i in range(N) if self._empty_land(i) and any(
                self.fig_owner[j] == p or (self.mon_type[j] is not None and self.mon_owner[j] == p)
                for j in self.adjl[i])]
        elif mode == "pyramid":
            cells = [j for j in self.adjl[pyr] if self._empty_land(j)]
        else:  # underworld
            cells = [i for i in range(N) if self.underworld[i] and self._empty_land(i)]
        types = []
        if self.warriors_pool[p] > 0:
            types.append((0, WARRIOR_K))
        if self.followers[p] >= 1 and any(p in tl for a, tl in enumerate(self.trapped) if a != p):
            types.append((1, WARRIOR_K))
        for k in sorted(set(self.gpool[p])):
            types.append((k - 1, k))
        out = {}
        for t, k in types:
            c = list(cells)
            if k == APEP:
                if mode == "regular":
                    c += [i for i in range(N) if not self.is_land[i] and self.fig_owner[i] == -1]
                elif mode == "pyramid":
                    c += [j for j in self.adjl[pyr] if not self.is_land[j] and self.fig_owner[j] == -1]
            if c:
                out[t] = c
        return out

    def _summon_one(self, p, mode, pyr=None):
        opts = self._summon_options(p, mode, pyr)
        if not opts:
            return
        optional = mode != "regular"
        if len(opts) == 1 and not optional:
            t = next(iter(opts))
        else:
            idxs = [self.A_FIG + t for t in opts] + ([self.A_DONE] if optional else [])
            a = yield self._dec(self.cur, Phase.SUMMON_TYPE, idxs, mode=mode)
            if a == self.A_DONE:
                return
            t = a - self.A_FIG
        a = yield self._dec(self.cur, Phase.SUMMON_DEST, [self.A_CELL + c for c in opts[t]], fig=t, mode=mode)
        i = a - self.A_CELL
        if t <= 1:
            kind = WARRIOR_K
            if t == 0:
                self.warriors_pool[p] -= 1
            else:                                   # resgata guerreiro preso: paga 1 Seguidor a Anubis
                an = next(a2 for a2, tl in enumerate(self.trapped) if a2 != p and p in tl)
                self.trapped[an].remove(p)
                self.followers[p] -= 1
                self.followers[an] += 1
        else:
            kind = t + 1
            self.gpool[p].remove(kind)
        self._place_figure(p, kind, i)
        if kind == SCORPION:
            yield from self._point_claws(p, i)

    def _act_summon(self, p):
        if PYRAMID_ATT in self.powers[p]:
            for pyr in [m for m in range(self.N) if self.mon_type[m] == "pyramid" and self.mon_owner[m] == p]:
                yield from self._summon_one(p, "pyramid", pyr)
        if self._has_god(p, OSIRIS) and any(self.underworld):
            yield from self._summon_one(p, "underworld")
        yield from self._summon_one(p, "regular")

    def _act_followers(self, p):
        n = sum(1 for m in range(self.N) if self.mon_type[m] is not None and self.mon_owner[m] in (-1, p)
                and any(self.fig_owner[j] == p for j in self.adjl[m]))
        if REVERED in self.powers[p]:
            n += 1
        self.followers[p] += n
        self.stats["followers_action"][p] += n

    def _gain_guardian(self, p, level):
        k = self.g_card[level]
        size = GUARDIAN_SIZE[k]
        if self.g_supply[level] <= 0 or self.bases_used[size][p] >= 2:
            return
        self.g_supply[level] -= 1
        self.bases_used[size][p] += 1
        self.gpool[p].append(k)
        self.stats["guardians"][p].append(k)

    def _act_unlock(self, p):
        n = self.unlock_next[p]
        if n >= 6:
            return
        level = n // 2 + 1
        if self.followers[p] < level:
            return
        avail = [w for w in POWERS_BY_LEVEL[level] if w not in self.powers[p]]
        a = yield self._dec(self.cur, Phase.UNLOCK_PICK, [self.A_POWER + w for w in avail], level=level)
        self.followers[p] -= level
        self.powers[p].add(a - self.A_POWER)
        self.stats["unlocks"][p].append(a - self.A_POWER)
        self.unlock_next[p] += 1
        if n % 2 == GUARDIAN_SLOT[self.god[p]][level - 1]:
            self._gain_guardian(p, level)

    # ------------------------------------------------------------------
    # Eventos
    # ------------------------------------------------------------------
    def _ev_control(self, p):
        if self.ankh_pool[p] == 0:
            return
        mons = [m for m in range(self.N) if self.mon_type[m] is not None]
        neutral = any(self.mon_owner[m] == -1 for m in mons)
        cand = []
        for m in mons:
            owner = self.mon_owner[m]
            ok = owner == -1 if neutral else owner not in (-1, p)
            if ok and any(self.fig_owner[j] == p for j in self.adjl[m]):
                cand.append(m)
        if not cand:
            return
        a = yield self._dec(self.cur, Phase.CONTROL_PICK, [self.A_CELL + m for m in cand])
        m = a - self.A_CELL
        if self.mon_owner[m] >= 0:
            self.ankh_pool[self.mon_owner[m]] += 1
        self.mon_owner[m] = p
        self.ankh_pool[p] -= 1

    # --- camelos ---
    def legal_camel_lines(self):
        """Todas as linhas legais de camelos (lista de frozensets de lados)."""
        land = self.board.land
        blocked = self.board.river_edges | self.camel_edges
        max_len = min(MAX_LINE, self.camel_supply)

        def verts(e):
            a, c = tuple(e)
            return [frozenset((a, c, x)) for x in nbrs(a) & nbrs(c)]

        anchors = set()
        for e in blocked:
            anchors.update(verts(e))
        ev, v_edges = {}, {}
        for a in land:
            for c in nbrs(a) & land:
                if a < c:
                    e = edge(a, c)
                    if e in blocked:
                        continue
                    ev[e] = verts(e)
                    for v in ev[e]:
                        v_edges.setdefault(v, []).append(e)
                        if any(x not in land for x in v):    # toca agua ou borda do mapa
                            anchors.add(v)

        def dfs(v, path, visited):
            if len(path) == max_len:
                return
            for e in v_edges.get(v, []):
                if e in path:
                    continue
                u = ev[e][0] if ev[e][1] == v else ev[e][1]
                if u in visited:
                    continue
                new = path + [e]
                if u in anchors:
                    yield frozenset(new)
                else:
                    yield from dfs(u, new, visited | {u})

        raw = {l for a in anchors for l in dfs(a, [], {a})}
        reg_of = {x: r for r in components(land, blocked) for x in r}
        legal = []
        for line in raw:
            e0 = next(iter(line))
            region = reg_of[next(iter(e0))]
            parts = components(region, blocked | line)
            if len(parts) == 2 and all(len(c) >= MIN_REGION for c in parts):
                legal.append(line)
        legal.sort(key=lambda l: sorted(sorted(e) for e in l))
        return legal

    def _ev_camel(self, p):
        lines = self.legal_camel_lines()[:K_LINES]
        idxs = [self.A_CAMEL] + [self.A_CAMEL + 1 + k for k in range(len(lines))]
        a = yield self._dec(self.cur, Phase.CAMEL_LINE, idxs, lines=lines)
        k = a - self.A_CAMEL - 1
        if k < 0:
            return
        line = lines[k]
        old_regions = set(self.regions)
        self.camel_edges |= line
        self.camel_supply -= len(line)
        self._recompute_regions()
        new = sorted((r for r in self.regions if r not in old_regions), key=min)
        old = next(r for r in old_regions if r not in set(self.regions))
        tok = self.tokens.pop(old)
        a = yield self._dec(self.cur, Phase.CAMEL_KEEP, [self.A_CHOICE, self.A_CHOICE + 1])
        keep = new[a - self.A_CHOICE]
        other = new[1 - (a - self.A_CHOICE)]
        self.tokens[keep] = tok
        self.tokens[other] = min(set(range(1, N_TOKENS + 1)) - set(self.tokens.values()))
        ops = [None]
        for x in new:
            for y in self.regions:
                if y != x and not (y in new and new.index(y) < new.index(x)):
                    ops.append((x, y))
        ops = ops[:N_CHOICE]
        a = yield self._dec(self.cur, Phase.CAMEL_SWAP, [self.A_CHOICE + k for k in range(len(ops))])
        op = ops[a - self.A_CHOICE]
        if op is not None:
            x, y = op
            self.tokens[x], self.tokens[y] = self.tokens[y], self.tokens[x]

    # --- conflito ---
    def _ev_conflict(self, trigger):
        self.tiebreak_holder, self.tiebreak_available = trigger, True
        self.amun_used = False
        for q in self._asc([q for q in range(self.P) if self.alive[q]]):
            if OMNIPRESENT in self.powers[q]:
                self.followers[q] += sum(1 for r in self.regions if self._figs_in(r, q))
        for i in range(self.N):                                    # garras do Giant Scorpion
            if self.fig_kind[i] == SCORPION and self.claw[i] is not None:
                for d in (self.claw[i], (self.claw[i] + 1) % 6):
                    j = self.cyc_nb[i][d]
                    if j >= 0 and self.mon_type[j] is not None and j in self.adjs[i]:
                        self._destroy_monument(j)
                        self.stats["monuments_destroyed"] += 1
        for region in sorted(self.regions, key=lambda r: self.tokens[r]):
            players = [q for q in range(self.P) if self.alive[q] and self._figs_in(region, q)]
            if not players:
                continue
            if len(players) == 1:
                q = players[0]
                maj = self._majorities(region)[q]
                if maj > 0:
                    self.gain_dev(q, maj, src="maioria")
                self.stats["dominations"][q] += 1
                radiant = 1 if any(self.fig_sun[i] for i in self._figs_in(region, q)) else 0
                self.gain_dev(q, 1 + radiant, src="dominio")
            else:
                yield from self._battle(region, players)
        self.tiebreak_holder, self.tiebreak_available = None, False
        self.conflicts_done += 1
        if self.conflicts_done == 3 and self.P >= 3:
            self._merge_gods()
        if self.conflicts_done == 4:
            self._forget_gods()

    def _merge_gods(self):
        """Fim do 3o Conflito (3+ jogadores): os 2 deuses de menor Devocao se fundem.
        Entidade resultante = deus 'superior' (2o mais baixo); o mais baixo e absorvido."""
        alive = [q for q in range(self.P) if self.alive[q]]
        if len(alive) < 2:
            return
        order = sorted(alive, key=self.dev_key)
        L, H = order[0], order[1]
        for m in range(self.N):                                     # monumentos de L sao destruidos
            if self.mon_owner[m] == L:
                self._destroy_monument(m)
        for i in range(self.N):                                     # figuras: Guardioes passam para H
            if self.fig_owner[i] == L:
                k = self.fig_kind[i]
                size = GUARDIAN_SIZE.get(k)
                if size and self.bases_used[size][H] < 2:
                    self.bases_used[size][H] += 1
                    self.fig_owner[i] = H
                    if self.fig_sun[i]:
                        self.fig_sun[i] = False
                        self.sun_left[L] += 1
                else:
                    if self.fig_sun[i]:
                        self.sun_left[L] += 1
                    self.fig_owner[i], self.fig_kind[i], self.fig_sun[i], self.claw[i] = -1, 0, False, None
        for k in self.gpool[L]:
            size = GUARDIAN_SIZE[k]
            if self.bases_used[size][H] < 2:
                self.bases_used[size][H] += 1
                self.gpool[H].append(k)
        self.gpool[L] = []
        self.followers[H] += self.followers[L]
        self.followers[L] = 0
        self.hand[L], self.used[L] = set(), []
        self.warriors_pool[L] = 0
        for a in range(self.P):                                     # guerreiros de L presos por Anubis somem
            self.trapped[a] = [v for v in self.trapped[a] if v != L]
        for v in self.trapped[L]:                                   # os que L prendeu voltam aos donos
            self.warriors_pool[v] += 1
        self.trapped[L] = []
        self.sun_left[H] += self.sun_left[L]
        self.uw_off[H] += self.uw_off[L]
        self.sun_left[L] = self.uw_off[L] = 0
        pos = self.dev_pos[L]                                       # os dois tokens passam a andar juntos
        self.stacks[pos].remove(L)
        self.stacks[self.dev_pos[H]].remove(H)
        self.dev_pos[H] = pos
        self.stacks[pos].insert(0, H)
        self.alive[L] = False
        self.rep[L] = H
        self.god2[H] = self.god[L]
        self.merged = (H, L)
        self.stats["merge"] = (H, L)

    def _forget_gods(self):
        for q in range(self.P):
            if self.alive[q] and self.dev_pos[q] < RED_SPACES:
                self.alive[q] = False
                self.stacks[self.dev_pos[q]].remove(q)
                for i in range(self.N):
                    if self.fig_owner[i] == q:
                        self.fig_owner[i], self.fig_kind[i], self.fig_sun[i], self.claw[i] = -1, 0, False, None
                    if self.mon_owner[i] == q:
                        self._destroy_monument(i)
                self.followers[q] = 0
                if self._has_god(q, OSIRIS):
                    self.underworld = [False] * self.N
                    self.uw_off[q] = 0
                for v in self.trapped[q]:
                    self.warriors_pool[v] += 1
                self.trapped[q] = []
        alive = [q for q in range(self.P) if self.alive[q]]
        if len(alive) == 1:
            self.winner = alive[0]
            self.stats["end_reason"] = "ultimo_deus"
            raise GameEnd
        if not alive:
            self.winner = None
            self.stats["end_reason"] = "egito_ateu"
            raise GameEnd

    def _attune(self, q, region):
        """Obelisk Attuned: move figuras de qualquer lugar para casas vazias ao lado dos seus Obeliscos."""
        obel = [m for m in region if self.mon_type[m] == "obelisk" and self.mon_owner[m] == q]
        if not obel:
            return
        moved = set()
        while True:
            dests = sorted({j for m in obel for j in self.adj[m] if self._can_end(j, q)})
            srcs = [i for i in range(self.N) if self.fig_owner[i] == q and i not in moved]
            if not dests or not srcs:
                return
            a = yield self._dec(q, Phase.ATTUNE_SRC, [self.A_CELL + i for i in srcs] + [self.A_DONE])
            if a == self.A_DONE:
                return
            s = a - self.A_CELL
            a = yield self._dec(q, Phase.ATTUNE_DEST, [self.A_CELL + j for j in dests], src=s)
            d = a - self.A_CELL
            kind = self.fig_kind[s]
            self._move_fig(s, d)
            moved.add(d)
            if kind == SCORPION:
                yield from self._point_claws(q, d, seat=q)

    def _underworld(self, q, region):
        """Osiris perdeu a batalha: coloca ou realoca 1 token Underworld na regiao."""
        cells = [i for i in region if self._empty_land(i) and not self.underworld[i]]
        on = [i for i in range(self.N) if self.underworld[i]]
        if not cells or (self.uw_off[q] == 0 and not on):
            return
        src = None
        if self.uw_off[q] == 0:
            if len(on) == 1:
                src = on[0]
            else:
                a = yield self._dec(q, Phase.UNDERWORLD_SRC, [self.A_CELL + i for i in on] + [self.A_DONE])
                if a == self.A_DONE:
                    return
                src = a - self.A_CELL
        a = yield self._dec(q, Phase.UNDERWORLD_DEST, [self.A_CELL + i for i in cells] + [self.A_DONE])
        if a == self.A_DONE:
            return
        if src is None:
            self.uw_off[q] -= 1
        else:
            self.underworld[src] = False
        self.underworld[a - self.A_CELL] = True

    def _battle(self, region, players):
        rf = self.region_full[region]
        killed = [0] * self.P
        self.stats["battles"] += 1
        self.cur_region, self.cur_cards = region, None
        # 0. Obelisk Attuned
        for q in self._asc([q for q in players if OBELISK_ATT in self.powers[q]]):
            yield from self._attune(q, region)
        # 1. Amun anuncia (publico); depois escolha secreta e revelacao simultanea
        self.amun_declared = {}
        for q in players:
            if self._has_god(q, AMUN) and not self.amun_used and len(self.hand[q]) >= 2:
                a = yield self._dec(q, Phase.AMUN_DECLARE, [self.A_YESNO, self.A_YESNO + 1])
                if a == self.A_YESNO + 1:
                    self.amun_declared[q] = True
                    self.stats["amun_double"] += 1
                    self.amun_used = True
        picks = {q: [] for q in players}
        for q in players:
            for _ in range(2 if self.amun_declared.get(q) else 1):
                avail = [c for c in sorted(self.hand[q]) if c not in picks[q]]
                if not avail:
                    break
                a = yield self._dec(q, Phase.BATTLE_CARD, [self.A_CARD + c for c in avail])
                picks[q].append(a - self.A_CARD)
                self.pending_card[q] = list(picks[q])
        cards = {}
        for q in players:
            self.pending_card[q] = None
            cards[q] = picks[q]
            for c in picks[q]:
                self.hand[q].discard(c)
                self.used[q].append(c)
                self.stats["cards"][q][c] += 1
        self.cur_cards = {q: list(c) for q, c in cards.items()}
        protected = set()
        for q in players:
            if FLOOD in cards[q]:
                fert = [i for i in self._figs_in(region, q) if self._ter(i) == "F"]
                self.followers[q] += len(fert)
                protected.update(fert)
        # 2. Build Monument (ordem crescente de Devocao)
        for q in self._asc([q for q in players if BUILD in cards[q]]):
            yield from self._build(q, region)
        # 3. Plague of Locusts
        for _ in range(sum(1 for q in players if PLAGUE in cards[q])):
            yield from self._plague(region, players, killed)
        # 4. maiorias de monumentos
        maj = self._majorities(region)
        for q in self._asc(players):
            if self._figs_in(region, q) and maj[q] > 0:
                self.gain_dev(q, maj[q], src="maioria")
        # 5. resolucao
        nfig = {q: len(self._figs_in(region, q)) for q in players}
        strength = {q: self._strength(q, region, cards[q]) for q in players}
        cands = [q for q in players if nfig[q] > 0]
        winner = None
        if cands:
            top = max(strength[q] for q in cands)
            tops = [q for q in cands if strength[q] == top]
            if len(tops) == 1:
                winner = tops[0]
            elif self.tiebreak_available and self.tiebreak_holder in tops:
                a = yield self._dec(self.tiebreak_holder, Phase.TIEBREAK, [self.A_YESNO, self.A_YESNO + 1])
                if a == self.A_YESNO + 1:
                    winner = self.tiebreak_holder
                    self.tiebreak_available = False
                    self.stats["tiebreak_used"] += 1
        prot = set(protected)
        for q in players:                                          # Isis: figuras protegidas
            if self._has_god(q, ISIS):
                for i in self._figs_in(region, q):
                    if any(self.fig_owner[j] not in (-1, q) for j in self.adjl[i]):
                        prot.add(i)
        self._kill_pass([i for i in rf if self.fig_owner[i] >= 0 and self.fig_owner[i] != winner
                         and i not in prot], killed)
        if winner is None:
            self.stats["battle_ties"] += 1
        else:
            self.stats["battle_wins"][winner] += 1
        for q in players:
            self.stats["kills"][q] += killed[q]
        if winner is not None:
            others = [strength[q] for q in players if q != winner]
            gap = strength[winner] - (max(others) if others else 0)
            gain = 3 if (GLORIOUS in self.powers[winner] and gap >= 3) else 1
            if DROUGHT in cards[winner]:
                gain += sum(1 for i in self._figs_in(region, winner) if self._ter(i) == "D")
            if any(self.fig_sun[i] for i in self._figs_in(region, winner)):
                gain += 1
            self.gain_dev(winner, gain, src="vitoria")
            if COMMANDING in self.powers[winner]:
                self.followers[winner] += 3
        for q in self._asc([q for q in players if q != winner and nfig[q] >= 2]):
            if MAGNANIMOUS in self.powers[q]:
                self.gain_dev(q, 2, src="magnanimous")
        self._flush_cat()
        for q in players:                                          # Osiris perdeu: Underworld
            if q != winner and self._has_god(q, OSIRIS):
                yield from self._underworld(q, region)
        # pos-resolucao: Miracle, Worshipful, Cycle of Ma'at
        for q in self._asc([q for q in players if MIRACLE in cards[q]]):
            self.gain_dev(q, killed[q], src="miracle")
        for q in self._asc([q for q in players if WORSHIPFUL in self.powers[q] and cards[q]]):
            if self.followers[q] >= 2:
                a = yield self._dec(q, Phase.WORSHIP, [self.A_YESNO, self.A_YESNO + 1])
                if a == self.A_YESNO + 1:
                    self.followers[q] -= 2
                    self.gain_dev(q, 1, src="worshipful")
        for q in players:
            if CYCLE in cards[q]:
                self.hand[q] |= set(self.used[q])
                self.used[q] = []
        self.cur_region, self.cur_cards, self.amun_declared = None, None, {}

    def _build(self, q, region):
        cost = 0 if INSPIRING in self.powers[q] else 3
        cells = [i for i in region if self._empty_land(i) and not self.underworld[i]]
        types = [t for t in range(3) if self.supply[MON_TYPES[t]] > 0]
        if not (self.followers[q] >= cost and self.ankh_pool[q] > 0 and cells and types):
            return
        a = yield self._dec(q, Phase.BUILD_TYPE, [self.A_BTYPE] + [self.A_BTYPE + 1 + t for t in types])
        t = a - self.A_BTYPE - 1
        if t < 0:
            return
        a = yield self._dec(q, Phase.BUILD_CELL, [self.A_CELL + i for i in cells])
        i = a - self.A_CELL
        self.stats["builds"][q] += 1
        self.followers[q] -= cost
        self.supply[MON_TYPES[t]] -= 1
        self.ankh_pool[q] -= 1
        self.mon_type[i] = MON_TYPES[t]
        self.mon_owner[i] = q

    def _plague(self, region, players, killed):
        for q in players:
            top = min(self.followers[q], BID_MAX)
            a = yield self._dec(q, Phase.BID, [self.A_BID + k for k in range(top + 1)])
            self.pending_bid[q] = a - self.A_BID
        bids = {q: self.pending_bid[q] for q in players}
        self.pending_bid = [None] * self.P
        for q, k in bids.items():
            self.followers[q] -= k
        m = max(bids.values())
        top = [q for q in players if bids[q] == m]
        spared = top[0] if len(top) == 1 else None
        self._kill_pass([i for i in self.region_full[region] if self.fig_owner[i] >= 0
                         and self.fig_owner[i] != spared and self.fig_kind[i] != GOD_K], killed)
        self._flush_cat()
