"""
jogar.py - Jogue Ankh contra as IAs no navegador.

    python jogar.py --models ~/ankh_runs/run1/checkpoints/round_0040.zip
    python jogar.py                       # sem --models: as IAs jogam aleatoriamente

Abre http://127.0.0.1:8765 . As regras sao as MESMAS do treino (ankh_game.py): este arquivo so
expoe o estado em JSON, aceita a sua jogada e deixa as IAs jogarem os outros assentos.
Precisa de jogar.html na mesma pasta. So os modelos exigem torch + sb3-contrib; o resto usa a
biblioteca padrao + numpy.

Cada jogada das IAs vira um QUADRO (frame) com o estado depois dela; a pagina os reproduz um a um.

Informacao secreta: as cartas escolhidas e os lances do Plague de outros jogadores NAO sao enviados
ao navegador (so as cartas ja reveladas e as ja usadas, que sao publicas no jogo).
"""
import argparse
import json
import os
import random
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from ankh_game import (ACTION_END, ACTION_START, AnkhGame, CARD_BONUS, CARD_NAMES, EVENTS, GODS,
                       GUARDIAN_KINDS, KIND_NAMES, MON_TYPES, N_CHOICE, POWER_NAMES, RED_SPACES, TOP,
                       ANUBIS_SLOTS, Phase)
from ankh_map import SCENARIOS, build_board
from ankh_obs import Encoder
import replay
from replay import describe

replay.ACTIONS[3] = "Desbloquear poder Ankh"       # o rotulo antigo do replay.py diz 'sem efeito'

HERE = os.path.dirname(os.path.abspath(__file__))
SECRET_PHASES = (Phase.BATTLE_CARD, Phase.BID)        # escolhas ocultas ate serem reveladas
# escolhas so de selecao (sem efeito visivel): nao geram quadro; o destino mostra a jogada inteira
SELECT_ONLY = (Phase.MOVE_SELECT, Phase.SUMMON_TYPE, Phase.ATTUNE_SRC, Phase.UNDERWORLD_SRC)
EVENT_TEXT = {"ctrl": "Controlar Monumento", "camel": "Caravana de Camelos", "conf": "CONFLITO"}


def seat_label(g, p):
    return f"J{p} ({GODS[g.god[p]]})"


class Session:
    def __init__(self, models, device, deterministic):
        self.lock = threading.RLock()
        self.board = build_board()
        self.models = [os.path.expanduser(m) for m in models]
        self.device = device
        self.deterministic = deterministic
        self._policies = {}
        self.g = None
        self.human = 0
        self.ai_kind = "random"
        self.log = []
        self.camel_new = None
        self.warnings = []
        self.frames = []

    # ---------------- IA ----------------
    def policy(self, idx):
        if idx in self._policies:
            return self._policies[idx]
        try:
            from sb3_contrib import MaskablePPO
            m = MaskablePPO.load(self.models[idx], device=self.device)
            det = self.deterministic

            def act(obs, mask):
                a, _ = m.predict(obs, action_masks=mask, deterministic=det)
                return int(a)
        except Exception as e:                                     # noqa: BLE001
            self.warnings.append(f"modelo {self.models[idx]} nao carregou ({e}); IA aleatoria")
            print("AVISO:", self.warnings[-1])
            act = None
        self._policies[idx] = act
        return act

    def ai_action(self, dec):
        mask = dec.mask
        fn = None
        if self.ai_kind != "random":
            fn = self.policy(int(self.ai_kind))
        if fn is None:
            return int(random.choice(np.flatnonzero(mask)))
        a = fn(self.enc.encode(dec), mask)
        return a if mask[a] else int(random.choice(np.flatnonzero(mask)))

    # ---------------- jogo ----------------
    def new_game(self, scenario, human, god, ai, seed):
        with self.lock:
            seed = int(seed) if seed not in (None, "") else int(time.time()) % 10**6
            self.g = AnkhGame(scenario, seed=seed, board=self.board)
            P = SCENARIOS[scenario]["n_players"]
            self.human = int(human) % P if human not in (None, "", -1) else int(np.random.default_rng(seed).integers(P))
            rng = np.random.default_rng(seed + 1)
            gods = [None] * P
            if god not in (None, "", -1):
                gods[self.human] = int(god)
            for s in range(P):
                if gods[s] is None:
                    free = [x for x in range(len(GODS)) if x not in gods]
                    gods[s] = int(rng.choice(free))
            self.g.reset(seed=seed, gods=gods, scenario=scenario)
            self.enc = Encoder(self.g)
            self.ai_kind = "random" if ai in (None, "", "random") else str(int(ai))
            self.log, self.camel_new, self.seed, self.frames = [], None, seed, []
            self.say(f"Nova partida: {scenario} ({P} jogadores), seed {seed}. Voce e {seat_label(self.g, self.human)}.")
            self._run_ai()

    def say(self, text):
        self.log.append(text)
        del self.log[:-400]

    def _before(self, dec):
        g = self.g
        if dec.phase == Phase.CAMEL_KEEP:
            self.camel_new = sorted((r for r in g.regions if r not in g.tokens), key=min)

    def _do(self, dec, a):
        """Executa uma jogada (de qualquer assento) e, se ela tem efeito visivel, grava um QUADRO."""
        g = self.g
        p = dec.player
        self._before(dec)
        ph = dec.phase
        a = int(a)
        skip = ph in SELECT_ONLY or a == g.A_DONE and ph in (Phase.MOVE_SELECT, Phase.ATTUNE_SRC, Phase.UNDERWORLD_SRC)
        if ph == Phase.MOVE_DEST and a - g.A_CELL == dec.info.get("src"):
            skip = True                                           # a figura ficou onde estava
        hl = {"from": None, "to": None, "line": None, "edges": None}
        if g.A_CELL <= a < g.A_DONE:
            c = a - g.A_CELL
            if ph in (Phase.MOVE_DEST, Phase.PUSH_DEST, Phase.ATTUNE_DEST):
                hl["from"], hl["to"] = dec.info.get("src"), c
            else:
                hl["to"] = c
        if ph == Phase.CHOOSE_ACTION:
            hl["line"] = a - g.A_TRACK
        if ph == Phase.CAMEL_LINE and a > g.A_CAMEL:
            hl["edges"] = [[g.ax2i[x] for x in tuple(e)] for e in dec.info["lines"][a - g.A_CAMEL - 1]]
        if ph in SECRET_PHASES and p != self.human:
            text = "escolhe uma carta (em segredo)" if ph == Phase.BATTLE_CARD else "faz um lance (em segredo)"
        else:
            try:
                text = describe(g, dec, a)
            except Exception:                                      # noqa: BLE001
                text = f"acao {a}"
        dev0, alive0, ev0, cards0 = list(g.dev_pos), list(g.alive), g.event_idx, g.cur_cards
        merged0 = g.merged
        lines = [f"{seat_label(g, p)}: {text}"]
        g.step(a)
        if g.event_idx > ev0 and ev0 < len(EVENTS):
            lines.append(f">>> EVENTO {ev0 + 1}/18: {EVENT_TEXT[EVENTS[ev0]]}")
        if g.cur_cards is not None and g.cur_cards is not cards0 and g.cur_cards:
            shown = ", ".join(f"{seat_label(g, e)}: {'+'.join(CARD_NAMES[c] for c in cs)}"
                              for e, cs in g.cur_cards.items() if cs)
            if shown:
                lines.append(f"    cartas reveladas -> {shown}")
        diff = [(q, g.dev_pos[q] - dev0[q]) for q in range(g.P) if g.dev_pos[q] != dev0[q]]
        if diff and g.merged is merged0:
            lines.append("    Devocao: " + ", ".join(f"J{q} {d:+d} (-> {g.dev_pos[q]})" for q, d in diff))
        if g.merged is not merged0 and g.merged is not None:
            lines.append(f"    FUSAO: J{g.merged[1]} foi absorvido por J{g.merged[0]}")
        for q in range(g.P):
            if alive0[q] and not g.alive[q] and (g.merged is None or q not in g.merged or g.rep[q] == q):
                if not (g.merged is not None and g.merged[1] == q):
                    lines.append(f"    {seat_label(g, q)} foi ESQUECIDO")
        if skip and len(lines) == 1:
            return
        if skip:
            lines = lines[1:]
        for ln in lines:
            self.say(ln)
        self.frames.append({"actor": p, "phase": ph.name, "lines": lines, "hl": hl,
                            "slow": 0.55 if ph in (Phase.MOVE_DEST, Phase.ATTUNE_DEST) else 1.0,
                            "state": self.state_json(light=True)})

    def _run_ai(self):
        g = self.g
        guard = 0
        while not g.done and g.decision.player != self.human:
            dec = g.decision
            self._do(dec, self.ai_action(dec))
            guard += 1
            if guard > 100000:
                raise RuntimeError("loop das IAs")
        if g.done:
            self._announce_end()

    def _announce_end(self):
        g = self.g
        if getattr(self, "_end_said", None) is g:
            return
        self._end_said = g
        if g.winner is None:
            self.say("FIM: Egito ateu (todos os deuses foram esquecidos).")
        else:
            self.say(f"FIM: vence {seat_label(g, g.winner)}.")

    def act(self, a):
        with self.lock:
            g = self.g
            if g is None or g.done or g.decision.player != self.human:
                raise ValueError("nao e a sua vez")
            dec = g.decision
            if not (0 <= a < g.A_SIZE and dec.mask[a]):
                raise ValueError("acao ilegal")
            self.frames = []
            self._do(dec, a)
            self._run_ai()

    def payload(self):
        with self.lock:
            out = {"frames": self.frames, "state": self.state_json()}
            self.frames = []
            return out

    # ---------------- exportacao ----------------
    def board_json(self):
        b, g = self.board, AnkhGame("basic_2p", board=self.board)
        cells = [{"i": i, "col": rc[0], "line": rc[1], "r2": b._pos[rc][0], "ter": g.terrain[i]}
                 for i, rc in enumerate(b.cells)]
        river = []
        for e in b.river_edges:
            a, c = tuple(e)
            if a in g.ax2i and c in g.ax2i:
                river.append([g.ax2i[a], g.ax2i[c]])
        return {"cells": cells, "river": river, "ncols": b.ncols,
                "gods": GODS, "kinds": KIND_NAMES, "cards": CARD_NAMES, "card_bonus": CARD_BONUS,
                "powers": POWER_NAMES, "events": EVENTS, "top": TOP, "red": RED_SPACES,
                "action_start": ACTION_START, "action_end": ACTION_END}

    def options(self):
        return {"scenarios": {n: s["n_players"] for n, s in SCENARIOS.items()},
                "gods": GODS, "models": [os.path.basename(m) for m in self.models],
                "warnings": self.warnings}

    def _anchor(self, region):
        pos = [self.board._pos[self.board.cells[i]] for i in region]
        cr = sum(p[0] for p in pos) / len(pos)
        cc = sum(p[1] for p in pos) / len(pos)
        land = [i for i in region if self.g.is_land[i]]
        return min(land, key=lambda i: (self.board._pos[self.board.cells[i]][0] - cr) ** 2 * 0.5
                   + (self.board._pos[self.board.cells[i]][1] - cc) ** 2 * 2)

    def decision_json(self):
        g = self.g
        dec = g.decision
        if g.done or dec is None or dec.player != self.human:
            return None
        acts = []
        N = g.N
        ops = None
        if dec.phase == Phase.CAMEL_SWAP and self.camel_new:
            new = self.camel_new
            ops = [None]
            for x in new:
                for y in g.regions:
                    if y != x and not (y in new and new.index(y) < new.index(x)):
                        ops.append((x, y))
            ops = ops[:N_CHOICE]
        for a in np.flatnonzero(dec.mask):
            a = int(a)
            item = {"a": a}
            try:
                item["text"] = describe(g, dec, a)
            except Exception:                                      # noqa: BLE001
                item["text"] = f"acao {a}"
            if g.A_CELL <= a < g.A_DONE:
                item["cell"] = a - g.A_CELL
            elif dec.phase == Phase.BID:
                item["bid"] = a - g.A_BID
            elif dec.phase == Phase.BATTLE_CARD:
                item["card"] = a - g.A_CARD
            elif dec.phase == Phase.CAMEL_LINE and a > g.A_CAMEL:
                line = dec.info["lines"][a - g.A_CAMEL - 1]
                item["edges"] = [[g.ax2i[x] for x in tuple(e)] for e in line]
            elif dec.phase == Phase.CAMEL_KEEP and self.camel_new:
                item["region"] = sorted(self.camel_new[a - g.A_CHOICE])
                item["text"] = f"a regiao em destaque mantem o token de ordem antigo ({'A' if a == g.A_CHOICE else 'B'})"
            elif dec.phase == Phase.CAMEL_SWAP and ops:
                op = ops[a - g.A_CHOICE]
                if op is None:
                    item["text"] = "nao trocar tokens de ordem"
                else:
                    x, y = op
                    item["regions"] = [sorted(x), sorted(y)]
                    item["text"] = f"trocar token {g.tokens[x]} (regiao A) com token {g.tokens[y]} (regiao B)"
            elif dec.phase == Phase.UNLOCK_PICK:
                item["power"] = a - g.A_POWER
            acts.append(item)
        out = {"phase": dec.phase.name, "actions": acts, "src": dec.info.get("src"),
               "level": dec.info.get("level")}
        return out

    def state_json(self, light=False):
        with self.lock:
            g = self.g
            if g is None:
                return {"started": False}
            N = g.N
            cells = {}
            for i in range(N):
                c = {}
                if g.fig_owner[i] >= 0:
                    c["f"] = [g.fig_owner[i], g.fig_kind[i], int(g.fig_sun[i])]
                if g.claw[i] is not None:
                    d = g.claw[i]
                    c["cl"] = [j for j in (g.cyc_nb[i][d], g.cyc_nb[i][(d + 1) % 6]) if j >= 0]
                if g.mon_type[i] is not None:
                    c["m"] = [MON_TYPES.index(g.mon_type[i]), g.mon_owner[i]]
                if g.underworld[i]:
                    c["u"] = 1
                if c:
                    cells[str(i)] = c
            camels = [[g.ax2i[x] for x in tuple(e)] for e in g.camel_edges]
            tokens = [[self._anchor(r), t] for r, t in g.tokens.items()]
            players = []
            for p in range(g.P):
                e = g.rep[p]
                players.append({
                    "seat": p, "god": g.god[p], "god2": g.god2[e], "ent": e, "alive": bool(g.alive[e]),
                    "absorbed": e != p, "superior": bool(g.merged is not None and g.merged[0] == p),
                    "dev": g.dev_pos[e], "followers": g.followers[e], "pool": g.warriors_pool[e],
                    "board_w": g.board_warriors(e), "ankh": g.ankh_pool[e],
                    "powers": sorted(g.powers[e]), "unlock_next": g.unlock_next[e],
                    "hand_n": len(g.hand[e]), "used": list(g.used[e]), "gpool": list(g.gpool[e]),
                    "sun_left": g.sun_left[e], "uw_off": g.uw_off[e],
                    "trapped": len(g.trapped[e]), "amun_declared": bool(g.amun_declared.get(e, False)),
                    "dev_order": g.dev_key(e)[1] if g.alive[e] else -1,
                })
            cur_cards = None
            if g.cur_cards:
                cur_cards = {str(e): list(cs) for e, cs in g.cur_cards.items()}
            hand = sorted(g.hand[g.rep[self.human]])
            s = {
                "started": True, "scenario": g.scenario, "P": g.P, "human": self.human, "seed": self.seed,
                "cells": cells, "camels": camels, "tokens": tokens, "players": players, "hand": hand,
                "markers": list(g.markers), "event_idx": g.event_idx, "conflicts_done": g.conflicts_done,
                "cur": g.cur, "merged": list(g.merged) if g.merged is not None else None,
                "tiebreak_holder": g.tiebreak_holder, "tiebreak_available": bool(g.tiebreak_available),
                "supply": g.supply, "camel_supply": g.camel_supply,
                "g_card": {str(k): v for k, v in g.g_card.items()},
                "g_supply": {str(k): v for k, v in g.g_supply.items()},
                "cur_region": sorted(g.cur_region) if g.cur_region is not None else None,
                "cur_cards": cur_cards,
                "done": bool(g.done), "winner": g.winner,
                "decision": None if light else self.decision_json(), "warnings": self.warnings,
                "ai": "aleatoria" if self.ai_kind == "random" else os.path.basename(self.models[int(self.ai_kind)]),
                "anubis_slots": ANUBIS_SLOTS,
            }
            if not light:
                s["log"] = self.log[-150:]
            if g.done:
                s["end_reason"] = g.stats.get("end_reason")
            return s


class Handler(BaseHTTPRequestHandler):
    session = None

    def log_message(self, *a):                       # silencia o log por requisicao
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        s = Handler.session
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "jogar.html"), "rb") as f:
                self._send(200, f.read(), "text/html")
        elif self.path == "/api/board":
            self._send(200, s.board_json())
        elif self.path == "/api/options":
            self._send(200, s.options())
        elif self.path == "/api/state":
            self._send(200, {"frames": [], "state": s.state_json()})
        else:
            self._send(404, {"erro": "nao encontrado"})

    def do_POST(self):
        s = Handler.session
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/new":
                s.new_game(body.get("scenario", "basic_2p"), body.get("human"), body.get("god"),
                           body.get("ai"), body.get("seed"))
            elif self.path == "/api/act":
                s.act(int(body["a"]))
            else:
                return self._send(404, {"erro": "nao encontrado"})
            self._send(200, s.payload())
        except Exception as e:                       # noqa: BLE001
            self._send(400, {"erro": str(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="checkpoints .zip separados por virgula (vazio = IA aleatoria)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--deterministic", action="store_true", help="a IA escolhe sempre a acao mais provavel")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    models = [m for m in a.models.split(",") if m]
    for m in models:
        if not os.path.exists(os.path.expanduser(m)):
            print("AVISO: arquivo nao encontrado:", m)
    Handler.session = Session(models, a.device, a.deterministic)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}"
    print("Ankh rodando em", url, "(Ctrl+C para sair)")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
