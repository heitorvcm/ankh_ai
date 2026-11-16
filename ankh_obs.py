"""
ankh_obs.py - Codificacao da observacao (do ponto de vista de um jogador).

NAO le g.pending_card nem g.pending_bid: escolhas secretas ainda nao reveladas
nunca vazam para o adversario. Cartas reveladas (g.cur_cards), cartas usadas,
poderes desbloqueados, deuses e Guardioes sao publicos, como no jogo.
"""
import numpy as np

from ankh_game import (K_LINES, MON_TYPES, EVENTS, ACTION_START, TOP, RED_SPACES, Phase, N_TOKENS,
                       N_POWERS, N_GODS, GUARDIAN_KINDS, GUARDIAN_CARDS, ANUBIS_SLOTS, CYCLIC)
from ankh_map import DIRECTIONS, edge, components

N_CELL_CH = 48            # 44 em uso + 4 reservados
MAX_OPP = 4               # oponentes visiveis (2 a 5 jogadores)
ENT_FEATS = 17 + N_POWERS + 2 * N_GODS + 7 + 7 + 6 + 6   # features por entidade (sem a mao)
LINE_FEATS = 11
N_PHASES = len(Phase)
EVENT_KINDS = ["ctrl", "camel", "conf", "end"]
N_RESERVED_GLOBALS = 24


class Encoder:
    def __init__(self, g):
        self.g = g
        b = g.board
        N = g.N
        self.dir_nb = [[(b.idx[b.step(rc, d)] if b.step(rc, d) is not None else -1)
                        for d in DIRECTIONS] for rc in b.cells]
        self.terr = np.zeros((N, 3), np.float32)
        for i, t in enumerate(g.terrain):
            self.terr[i, "FDW".index(t)] = 1
        self._blk_key = None
        self._flags = np.zeros((N, 6), np.float32)

    def _blocked_flags(self):
        g = self.g
        key = frozenset(g.camel_edges)
        if key != self._blk_key:
            blocked = g.board.river_edges | g.camel_edges
            f = np.zeros((g.N, 6), np.float32)
            for i in range(g.N):
                if not g.is_land[i]:
                    continue
                for d, j in enumerate(self.dir_nb[i]):
                    if j >= 0 and g.is_land[j] and edge(g.ax[i], g.ax[j]) in blocked:
                        f[i, d] = 1
            self._flags, self._blk_key = f, key
        return self._flags

    def line_feats(self, dec, me):
        g = self.g
        out = np.zeros((K_LINES, LINE_FEATS), np.float32)
        lines = dec.info.get("lines") if dec.phase == Phase.CAMEL_LINE else None
        if not lines:
            return out
        blocked = g.board.river_edges | g.camel_edges
        reg_ax = {r: frozenset(g.ax[i] for i in r) for r in g.regions}
        for k, line in enumerate(lines[:K_LINES]):
            a = next(iter(next(iter(line))))
            parts = components(reg_ax[g.region_of[g.ax2i[a]]], blocked | line)
            out[k, 0] = len(line) / 6
            col = 1
            for part in sorted(parts, key=len):
                idx = [g.ax2i[x] for x in part]
                own = sum(g.fig_owner[i] == me for i in idx)
                en = sum(g.fig_owner[i] not in (-1, me) for i in idx)
                om = sum(g.mon_type[i] is not None and g.mon_owner[i] == me for i in idx)
                em = sum(g.mon_type[i] is not None and g.mon_owner[i] not in (-1, me) for i in idx)
                out[k, col:col + 5] = [len(idx) / 40, own / 7, en / 7, om / 3, em / 3]
                col += 5
        return out

    def encode(self, dec):
        """Observacao do assento dec.player: 'eu' = a entidade que ele controla; ate 4 oponentes
        (entidades), ordenados pela ordem de turno a partir de mim, com zeros nos espacos vazios."""
        g = self.g
        seat = dec.player
        me = g.rep[seat]
        N = g.N
        opps = []
        for k in range(1, g.P):
            e = g.rep[(seat + k) % g.P]
            if e != me and e not in opps:
                opps.append(e)
        opps = opps[:MAX_OPP]
        slot = {e: k for k, e in enumerate(opps)}

        c = np.zeros((N, N_CELL_CH), np.float32)
        c[:, 0:3] = self.terr
        for i in range(N):
            ow = g.fig_owner[i]
            if ow >= 0:
                if ow == me:
                    c[i, 3 + g.fig_kind[i] - 1] = 1
                else:
                    c[i, 11 + g.fig_kind[i] - 1] = 1
                    c[i, 19 + slot.get(ow, 0)] = 1
                if g.claw[i] is not None:                                 # alvos das garras do Scorpion
                    for d in (g.claw[i], (g.claw[i] + 1) % 6):
                        j = g.cyc_nb[i][d]
                        if j >= 0:
                            c[j, 42 if ow == me else 43] = 1
            if g.fig_sun[i]:
                c[i, 41] = 1
            t = g.mon_type[i]
            if t is not None:
                c[i, 23 + MON_TYPES.index(t)] = 1
                mo = g.mon_owner[i]
                c[i, 26 + (0 if mo == -1 else 1 if mo == me else 2 + slot.get(mo, 0))] = 1
            if g.underworld[i]:
                c[i, 40] = 1
        for r, tok in g.tokens.items():
            for i in r:
                c[i, 32] = tok / N_TOKENS
        c[:, 33:39] = self._blocked_flags()
        if g.cur_region is not None:
            for i in g.cur_region:
                c[i, 39] = 1

        alive_ents = sorted((e for e in range(g.P) if g.rep[e] == e and g.alive[e]), key=g.dev_key)
        rc = g.cur_cards or {}

        def ent(e, mine):
            if e is None:
                return [0.0] * (ENT_FEATS + (7 if mine else 0))
            rank = alive_ents.index(e) / max(len(alive_ents) - 1, 1) if e in alive_ents else 0.0
            f = [min(g.followers[e], 10) / 10, g.warriors_pool[e] / 6, g.ankh_pool[e] / 9, g.dev_pos[e] / TOP,
                 rank, float(g.dev_pos[e] >= RED_SPACES), float(g.alive[e]), g.unlock_next[e] / 6,
                 g.sun_left[e] / 3, g.uw_off[e] / 3, len(g.trapped[e]) / ANUBIS_SLOTS,
                 sum(t.count(e) for t in g.trapped) / ANUBIS_SLOTS, g.bases_used["small"][e] / 2,
                 g.bases_used["large"][e] / 2, len(g.hand[e]) / 7, float(g.amun_declared.get(e, False)),
                 float(g.merged is not None and g.merged[0] == e)]
            f += [float(k in g.powers[e]) for k in range(N_POWERS)]
            f += [float(g.god[e] == k) for k in range(N_GODS)]
            f += [float(g.god2[e] == k) for k in range(N_GODS)]
            f += [float(k in g.used[e]) for k in range(7)]
            f += [float(k in rc.get(e, [])) for k in range(7)]
            f += [float(g.gpool[e].count(k)) for k in GUARDIAN_KINDS]
            f += [sum(1 for i in range(N) if g.fig_kind[i] == k and g.fig_owner[i] == e) / 2 for k in GUARDIAN_KINDS]
            if mine:
                f += [float(k in g.hand[e]) for k in range(7)]
            return f

        gl = ent(me, True)
        for k in range(MAX_OPP):
            gl += ent(opps[k] if k < len(opps) else None, False)
        gl += [float(g.P == p) for p in (2, 3, 4, 5)]
        gl += [float(g.merged is not None), float(g.merged is not None and g.merged[0] == me)]
        gl += [seat / max(g.P - 1, 1), ((g.cur - seat) % g.P) / g.P, float(g.cur == seat)]
        start = ACTION_START[g.P]
        gl += [(m - start) / (6 - start) for m in g.markers]
        nxt = EVENTS[g.event_idx] if g.event_idx < len(EVENTS) else "end"
        gl += [float(nxt == k) for k in EVENT_KINDS]
        gl += [g.event_idx / len(EVENTS), g.conflicts_done / 5]
        gl += [float(g.tiebreak_available), float(g.tiebreak_holder == me), float(g.amun_used)]
        gl += [g.supply[t] / 10 for t in MON_TYPES]
        gl += [g.camel_supply / 30, len(g.regions) / 8]
        tf = g.turn_first
        gl += [float(tf is None)] + [float(tf == k) for k in range(4)]
        for lvl in (1, 2, 3):
            gl += [float(g.g_card[lvl] == GUARDIAN_CARDS[lvl][0]), float(g.g_card[lvl] == GUARDIAN_CARDS[lvl][1]),
                   g.g_supply[lvl] / 3]
        gl += [0.0] * N_RESERVED_GLOBALS
        ph = np.zeros(N_PHASES, np.float32)
        ph[int(dec.phase)] = 1
        ctx = np.zeros(N, np.float32)
        if "src" in dec.info and dec.phase in (Phase.MOVE_DEST, Phase.PUSH_DEST, Phase.ATTUNE_DEST, Phase.CLAW_POINT):
            ctx[dec.info["src"]] = 1
        v = np.concatenate([c.ravel(), np.asarray(gl, np.float32), ph, ctx,
                            self.line_feats(dec, me).ravel()])
        return np.clip(v, 0.0, 1.0)

    @staticmethod
    def potential(g, me):
        """Phi(s) do shaping potencial: vantagem de Devocao + progresso ate sair do vermelho."""
        others = [g.dev_pos[e] for e in range(g.P) if e != me and g.rep[e] == e]
        return ((g.dev_pos[me] - (max(others) if others else 0)) / TOP
                + min(g.dev_pos[me], RED_SPACES) / RED_SPACES)
