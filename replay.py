"""
replay.py - Joga partidas e imprime em TEXTO cada decisao e cada mudanca de estado,
para conferir as regras a olho (e achar bugs no jogo antes de o agente explora-los).

    python replay.py                         # 1 partida, os dois jogadores aleatorios
    python replay.py --seed 7 --maps         # com o mapa depois de cada Conflito
    python replay.py --scenario basic_4p --seed 7
    python replay.py --models run1/checkpoints/round_0003.zip,run1/snapshots/snap_0002.zip

Mostra TUDO, inclusive as cartas e os lances secretos (e ferramenta de depuracao).
Sem --model0/--model1 o jogador correspondente joga aleatoriamente.
"""
import argparse
from types import SimpleNamespace

import numpy as np

from ankh_game import (AnkhGame, CARD_NAMES, EVENTS, GODS, KIND_NAMES, MON_TYPES, POWER_NAMES, Phase,
                       RED_SPACES)
from ankh_map import render
from ankh_obs import Encoder

ACTIONS = ["Mover figuras", "Invocar guerreiro", "Ganhar seguidores", "Desbloquear poder (sem efeito nesta versao)"]
EVENT_NAMES = {"ctrl": "Controlar Monumento", "camel": "Caravana de Camelos", "conf": "CONFLITO"}


def cell(g, i):
    return "(%d,%d)" % g.board.cells[i]


def describe(g, dec, a):
    ph = dec.phase
    if ph == Phase.CHOOSE_ACTION:
        return "escolhe a acao: " + ACTIONS[a - g.A_TRACK]
    if ph == Phase.MOVE_SELECT:
        return "termina o movimento" if a == g.A_DONE else f"seleciona a figura em {cell(g, a - g.A_CELL)}"
    if ph == Phase.MOVE_DEST:
        d, s = a - g.A_CELL, dec.info["src"]
        return f"fica em {cell(g, s)}" if d == s else f"move {cell(g, s)} -> {cell(g, d)}"
    if ph == Phase.SUMMON_TYPE:
        if a == g.A_DONE:
            return "nao faz a invocacao extra"
        t = a - g.A_FIG
        return "invoca: " + ("guerreiro" if t == 0 else "guerreiro preso por Anubis" if t == 1 else KIND_NAMES[t + 1])
    if ph == Phase.SUMMON_DEST:
        t = dec.info.get("fig", 0)
        nome = "guerreiro" if t <= 1 else KIND_NAMES[t + 1]
        return f"invoca {nome} em {cell(g, a - g.A_CELL)}" + (f" (extra: {dec.info['mode']})" if dec.info.get("mode") not in (None, "regular") else "")
    if ph == Phase.UNLOCK_PICK:
        return f"desbloqueia o poder {POWER_NAMES[a - g.A_POWER]} (nivel {dec.info['level']})"
    if ph == Phase.AMUN_DECLARE:
        return "ANUNCIA 2 cartas (Mysterious Forms)" if a == g.A_YESNO + 1 else "nao anuncia 2 cartas"
    if ph == Phase.CLAW_POINT:
        return f"aponta as garras do Scorpion em {cell(g, dec.info['src'])} (orientacao {a - g.A_CHOICE})"
    if ph == Phase.PUSH_DEST:
        return f"Satet empurra o inimigo de {cell(g, dec.info['src'])} para {cell(g, a - g.A_CELL)}"
    if ph == Phase.UNDERWORLD_SRC:
        return "nao realoca o Underworld" if a == g.A_DONE else f"realoca o token Underworld de {cell(g, a - g.A_CELL)}"
    if ph == Phase.UNDERWORLD_DEST:
        return "nao coloca Underworld" if a == g.A_DONE else f"Underworld em {cell(g, a - g.A_CELL)}"
    if ph == Phase.ATTUNE_SRC:
        return "termina Obelisk Attuned" if a == g.A_DONE else f"Obelisk Attuned: pega a figura em {cell(g, a - g.A_CELL)}"
    if ph == Phase.ATTUNE_DEST:
        return f"Obelisk Attuned: leva para {cell(g, a - g.A_CELL)}"
    if ph == Phase.WORSHIP:
        return "SACRIFICA 2 seguidores (Worshipful)" if a == g.A_YESNO + 1 else "nao usa Worshipful"
    if ph == Phase.CONTROL_PICK:
        m = a - g.A_CELL
        return f"passa a controlar o {g.mon_type[m]} em {cell(g, m)}"
    if ph == Phase.CAMEL_LINE:
        k = a - g.A_CAMEL - 1
        if k < 0:
            return "nao coloca camelos"
        edges = ["%s|%s" % tuple(cell(g, g.ax2i[x]) for x in sorted(e)) for e in dec.info["lines"][k]]
        return f"coloca {len(edges)} camelos entre: " + " ".join(edges)
    if ph == Phase.CAMEL_KEEP:
        return f"a regiao {a - g.A_CHOICE} mantem o token antigo"
    if ph == Phase.CAMEL_SWAP:
        return "nao troca tokens" if a == g.A_CHOICE else f"troca de tokens (opcao {a - g.A_CHOICE})"
    if ph == Phase.BATTLE_CARD:
        return f"escolhe a carta {CARD_NAMES[a - g.A_CARD]}"
    if ph == Phase.BUILD_TYPE:
        t = a - g.A_BTYPE - 1
        return "nao constroi" if t < 0 else f"constroi um {MON_TYPES[t]}"
    if ph == Phase.BUILD_CELL:
        return f"constroi em {cell(g, a - g.A_CELL)}"
    if ph == Phase.BID:
        return f"da um lance de {a - g.A_BID} seguidores"
    if ph == Phase.TIEBREAK:
        return "USA o token de desempate" if a == g.A_YESNO + 1 else "nao usa o token de desempate"
    return f"acao {a}"


def snapshot(g):
    mons = sorted((cell(g, i), g.mon_type[i], g.mon_owner[i]) for i in range(g.N) if g.mon_type[i])
    return {
        "devocao": list(g.dev_pos), "seguidores": list(g.followers),
        "guerreiros no tabuleiro": [6 - w for w in g.warriors_pool],
        "marcadores de acao": list(g.markers), "cartas usadas": [list(map(lambda c: CARD_NAMES[c], u)) for u in g.used],
        "camelos no tabuleiro": len(g.camel_edges), "monumentos": mons, "vivos": list(g.alive),
        "tokens de ordem": sorted(g.tokens.values()),
        "poderes": [sorted(POWER_NAMES[w] for w in g.powers[p]) for p in range(g.P)],
        "guardioes no pool": [[KIND_NAMES[k] for k in gp] for gp in g.gpool],
        "guardioes em campo": sorted((p, KIND_NAMES[g.fig_kind[i]], cell(g, i)) for i in range(g.N)
                                     for p in [g.fig_owner[i]] if p >= 0 and g.fig_kind[i] > 2),
        "presos por Anubis": [list(t) for t in g.trapped],
        "Radiant": sorted(cell(g, i) for i in range(g.N) if g.fig_sun[i]),
        "Underworld": sorted(cell(g, i) for i in range(g.N) if g.underworld[i]),
    }


def show_map(g):
    mons = [(g.mon_type[i], g.board.cells[i], None if g.mon_owner[i] < 0 else g.mon_owner[i])
            for i in range(g.N) if g.mon_type[i]]
    figs = {p: [("god" if g.fig_kind[i] == 1 else "warrior", g.board.cells[i])   # guardioes aparecem como 'w'
                for i in range(g.N) if g.fig_owner[i] == p] for p in range(g.P)}
    setup = SimpleNamespace(monuments=mons, figures=figs, region_of={})
    print(render(g.board, setup))


def load_policy(path, device):
    from sb3_contrib import MaskablePPO
    m = MaskablePPO.load(path, device=device)
    def act(obs, mask):
        a, _ = m.predict(obs, action_masks=mask, deterministic=False)
        return int(a)
    return act


def play(seed, policies, maps, scenario="basic_2p"):
    g = AnkhGame(scenario, seed=seed)
    enc, rng = Encoder(g), np.random.default_rng(seed)
    dec = g.reset(seed=seed)
    policies = (list(policies) + [None] * g.P)[:g.P]          # assentos sem modelo jogam aleatoriamente
    seats = ", ".join(f"J{p} ({GODS[g.god[p]]}): {'modelo' if policies[p] else 'aleatorio'}" for p in range(g.P))
    print(f"===== partida seed={seed} | cenario {g.scenario} ({g.P} jogadores) | {seats} =====")
    print(f"      guardioes: nivel 1 = {KIND_NAMES[g.g_card[1]]}, nivel 2 = {KIND_NAMES[g.g_card[2]]}, nivel 3 = {KIND_NAMES[g.g_card[3]]}")
    if maps:
        show_map(g)
    prev, n = snapshot(g), 0
    while dec is not None:
        p = dec.player
        fn = policies[p]
        a = fn(enc.encode(dec), dec.mask) if fn else int(rng.choice(np.flatnonzero(dec.mask)))
        n += 1
        text = describe(g, dec, a)
        ev_before, conf_before = g.event_idx, g.conflicts_done
        print(f"[{n:3d}] J{p} {dec.phase.name:<13} {text}")
        dec = g.step(a)
        if g.event_idx > ev_before:
            print(f"      >>> EVENTO {g.event_idx}/18: {EVENT_NAMES[EVENTS[g.event_idx - 1]]}")
        cur = snapshot(g)
        for k in cur:
            if cur[k] != prev[k]:
                print(f"      · {k}: {prev[k]} -> {cur[k]}")
        prev = cur
        if maps and g.conflicts_done > conf_before:
            show_map(g)
    res = "empate (Egito ateu)" if g.winner is None else f"vence o deus do jogador {g.winner}" + (
        f" (fundido com o jogador {g.merged[1]})" if g.merged and g.merged[0] == g.winner else "")
    print(f"----- fim: {res} | devocao {g.dev_pos} (vermelho: < {RED_SPACES}) | {n} decisoes -----")
    if maps:
        show_map(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenario", default="basic_2p", help="basic_2p, basic_3p, basic_4p ou basic_5p")
    ap.add_argument("--models", default="", help="checkpoints por assento, separados por virgula (vazio = aleatorio)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--maps", action="store_true", help="imprime o mapa no inicio, apos cada Conflito e no fim")
    a = ap.parse_args()
    pol = [load_policy(m, a.device) if m else None for m in (a.models.split(",") if a.models else [])]
    for i in range(a.games):
        play(a.seed + i, pol, a.maps, a.scenario)


if __name__ == "__main__":
    main()
