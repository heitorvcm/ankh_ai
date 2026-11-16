"""
ankh_report.py - Estatisticas detalhadas por partida, por round e relatorio em Markdown.

  Agg                  acumula resumos de partidas (AnkhGame.summary, do ponto de vista do agente)
  make_stats_callback  callback do SB3: agrega as partidas de TREINO e grava escalares no TensorBoard
  log_round            acrescenta o registro de um round em report.jsonl
  render_markdown      gera relatorio.md a partir de report.jsonl (e curvas.png se houver matplotlib)

Tudo sem dependencias pesadas no import (o SB3 so e importado em make_stats_callback).
"""
import collections
import json
import os

from ankh_game import GODS, GUARDIAN_KINDS, KIND_NAMES, POWER_NAMES

ACTION_NAMES = ["mover", "invocar", "seguidores", "desbloquear"]
CARD_NAMES_PT = ["plague", "build", "chariots", "cycle", "drought", "flood", "miracle"]
END_NAMES = {"topo": "chegou ao topo da Devocao", "ultimo_deus": "unico deus apos o 4o Conflito",
             "ultimo_evento": "maior Devocao no ultimo evento", "egito_ateu": "Egito ateu (todos esquecidos)"}


class Agg:
    """Acumulador de resumos de partida (cada resumo e do ponto de vista do agente)."""

    def __init__(self):
        self.n = 0
        self.c = collections.Counter()
        self.end = collections.Counter()
        self.dev_src = [collections.Counter(), collections.Counter()]
        self.cards = [[0] * 7, [0] * 7]
        self.actions = [[0] * 4, [0] * 4]
        self.power_games = [collections.Counter(), collections.Counter()]
        self.guardians = [collections.Counter(), collections.Counter()]
        self.by_god = {}                       # deus do agente -> [partidas, vitorias]
        self.by_players = {}                   # numero de jogadores -> [partidas, vitorias, egito ateu]
        self.seat_idx = {}                     # "P-assento" -> [partidas, vitorias]
        self.roles = {}                        # papel na fusao -> [partidas, vitorias]
        self.matrix = {}                       # "deus_agente-deus_oponente" -> [partidas, vitorias]
        self.seat = {"first": [0, 0], "second": [0, 0]}

    def add(self, s):
        self.n += 1
        c = self.c
        c[s["result"]] += 1
        for k in ("steps", "battles", "battle_ties", "tiebreak_used", "amun_double", "monuments_destroyed",
                  "pushes", "camels", "conflicts", "events"):
            c[k] += s[k]
        for i, who in enumerate(("me", "opp")):
            c["dev_" + who] += s["dev_final"][i]
            c["mon_" + who] += s["monuments"][i]
            c["fol_" + who] += s["followers"][i]
            c["bw_" + who] += s["battle_wins"][i]
            c["dom_" + who] += s["dominations"][i]
            c["kills_" + who] += s["kills"][i]
            c["builds_" + who] += s["builds"][i]
            c["unl_" + who] += len(s["unlocks"][i])
            c["lost_" + who] += s["dev_lost"][i]
            self.dev_src[i].update(s["dev_src"][i])
            for k in range(7):
                self.cards[i][k] += s["cards"][i][k]
            for k in range(4):
                self.actions[i][k] += s["actions"][i][k]
            self.power_games[i].update(set(s["powers"][i]))
            self.guardians[i].update(s["guardians"][i])
        self.end[s["end_reason"]] += 1
        win = s["result"] == "win"
        g = self.by_god.setdefault(s["god"][0], [0, 0])
        g[0] += 1
        g[1] += win
        if s["n_players"] == 2 and s["god"][1] is not None:
            m = self.matrix.setdefault(f"{s['god'][0]}-{s['god'][1]}", [0, 0])
            m[0] += 1
            m[1] += win
        bp = self.by_players.setdefault(str(s["n_players"]), [0, 0, 0])
        bp[0] += 1
        bp[1] += win
        bp[2] += s["result"] == "draw"
        if s["n_players"] > 2:
            ro = self.roles.setdefault(s.get("merge_role") or "fora da fusao", [0, 0])
            ro[0] += 1
            ro[1] += win
        sp = self.seat_idx.setdefault(f"{s['n_players']}-{s['me_seat']}", [0, 0])
        sp[0] += 1
        sp[1] += win
        seat = self.seat["first" if s["me_first"] else "second"]
        seat[0] += 1
        seat[1] += win

    def to_dict(self):
        n = max(self.n, 1)
        c = self.c
        per = lambda k: c[k] / n
        return {
            "n": self.n, "win": c["win"] / n, "loss": c["loss"] / n, "draw": c["draw"] / n,
            "steps": per("steps"), "events": per("events"), "conflicts": per("conflicts"), "camels": per("camels"),
            "dev_me": per("dev_me"), "dev_opp": per("dev_opp"), "monuments_me": per("mon_me"),
            "monuments_opp": per("mon_opp"), "followers_me": per("fol_me"), "followers_opp": per("fol_opp"),
            "battles": per("battles"), "battle_wins_me": per("bw_me"), "battle_wins_opp": per("bw_opp"),
            "battle_ties": per("battle_ties"), "dominations_me": per("dom_me"), "dominations_opp": per("dom_opp"),
            "kills_me": per("kills_me"), "kills_opp": per("kills_opp"), "builds_me": per("builds_me"),
            "builds_opp": per("builds_opp"), "unlocks_me": per("unl_me"), "unlocks_opp": per("unl_opp"),
            "dev_lost_me": per("lost_me"), "tiebreak_used": per("tiebreak_used"), "amun_double": per("amun_double"),
            "monuments_destroyed": per("monuments_destroyed"), "pushes": per("pushes"),
            "end": {k: v / n for k, v in self.end.items()},
            "dev_src_me": {k: v / n for k, v in self.dev_src[0].items()},
            "dev_src_opp": {k: v / n for k, v in self.dev_src[1].items()},
            "cards_me": [x / n for x in self.cards[0]], "cards_opp": [x / n for x in self.cards[1]],
            "actions_me": [x / n for x in self.actions[0]], "actions_opp": [x / n for x in self.actions[1]],
            "power_games_me": {POWER_NAMES[k]: v / n for k, v in self.power_games[0].items()},
            "power_games_opp": {POWER_NAMES[k]: v / n for k, v in self.power_games[1].items()},
            "guardians_me": {KIND_NAMES[k]: v / n for k, v in self.guardians[0].items()},
            "guardians_opp": {KIND_NAMES[k]: v / n for k, v in self.guardians[1].items()},
            "by_god": {str(k): v for k, v in self.by_god.items()},
            "matrix": self.matrix, "seat": self.seat, "by_players": self.by_players, "seat_idx": self.seat_idx,
            "merge_roles": self.roles,
        }


def make_stats_callback(agg_round):
    """Callback do SB3: agrega as partidas de treino em `agg_round` e loga escalares no TensorBoard."""
    from stable_baselines3.common.callbacks import BaseCallback

    class StatsCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.roll = Agg()

        def _on_step(self):
            for info in self.locals.get("infos", []):
                s = info.get("summary")
                if s is not None:
                    agg_round.add(s)
                    self.roll.add(s)
            return True

        def _on_rollout_end(self):
            if self.roll.n:
                d = self.roll.to_dict()
                for k in ("win", "draw", "steps", "dev_me", "dev_opp", "battles", "battle_wins_me",
                          "unlocks_me", "monuments_me"):
                    self.logger.record(f"ankh/{k}", d[k])
                for k, v in d["end"].items():
                    self.logger.record(f"ankh/fim_{k}", v)
                self.roll = Agg()

    return StatsCallback()


def log_round(report_dir, record):
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "report.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


def read_rounds(report_dir):
    path = os.path.join(report_dir, "report.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------
pct = lambda x: f"{100 * x:.0f}%"
f1 = lambda x: f"{x:.1f}"
f2 = lambda x: f"{x:.2f}"


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _block(title, d):
    """Tabela-resumo de um agregado (treino ou avaliacao)."""
    if not d or not d["n"]:
        return ""
    rows = [["partidas", d["n"]], ["vitorias / derrotas / Egito ateu", f"{pct(d['win'])} / {pct(d['loss'])} / {pct(d['draw'])}"],
            ["decisoes por partida", f1(d["steps"])], ["Devocao final (eu / oponente)", f"{f1(d['dev_me'])} / {f1(d['dev_opp'])}"],
            ["monumentos no fim (eu / oponente)", f"{f1(d['monuments_me'])} / {f1(d['monuments_opp'])}"],
            ["batalhas por partida (vitorias eu / oponente / empates)",
             f"{f1(d['battles'])} ({f1(d['battle_wins_me'])} / {f1(d['battle_wins_opp'])} / {f1(d['battle_ties'])})"],
            ["dominios por partida (eu / oponente)", f"{f1(d['dominations_me'])} / {f1(d['dominations_opp'])}"],
            ["figuras minhas / do oponente mortas", f"{f1(d['kills_me'])} / {f1(d['kills_opp'])}"],
            ["poderes desbloqueados (eu / oponente)", f"{f1(d['unlocks_me'])} / {f1(d['unlocks_opp'])}"]]
    return f"**{title}**\n\n" + _table(["metrica", "valor"], rows) + "\n"


def _alerts(rounds):
    out = []
    last = rounds[-1]
    er = last.get("eval_random")
    if er and er["draw"] > 0.30:
        out.append(f"{pct(er['draw'])} das partidas de avaliacao terminam em Egito ateu: o agente ainda nao sai do vermelho de forma consistente.")
    prevs = [r["eval_prev"]["win"] for r in rounds[-3:] if r.get("eval_prev")]
    if len(prevs) == 3 and min(prevs) >= 0.95:
        out.append("O modelo vence ~100% contra o snapshot anterior por 3 rounds seguidos: o snapshot pode estar velho/fraco (aumente a frequencia de snapshots) ou o treino esta muito a frente.")
    if er and er.get("by_god"):
        rates = {int(k): v[1] / max(v[0], 1) for k, v in er["by_god"].items()}
        if len(rates) > 1 and max(rates.values()) - min(rates.values()) > 0.30:
            lo = min(rates, key=rates.get)
            out.append(f"Diferenca grande entre deuses (contra aleatorio): {GODS[lo]} esta em {pct(rates[lo])} contra {pct(max(rates.values()))} do melhor.")
    tr = last.get("train")
    if tr and tr["n"] >= 20:
        f_, s_ = tr["seat"]["first"], tr["seat"]["second"]
        if f_[0] and s_[0] and abs(f_[1] / f_[0] - s_[1] / s_[0]) > 0.15:
            out.append(f"Vantagem de posicao no treino: {pct(f_[1] / f_[0])} como 1o jogador contra {pct(s_[1] / s_[0])} como 2o.")
        used = tr["cards_me"]
        tot = sum(used) or 1
        for k, u in enumerate(used):
            if u / tot < 0.03:
                out.append(f"A carta {CARD_NAMES_PT[k]} quase nunca e jogada ({pct(u / tot)} das jogadas).")
        if tr["end"].get("egito_ateu", 0) > 0.40:
            out.append(f"{pct(tr['end']['egito_ateu'])} das partidas de TREINO acabam em Egito ateu.")
    return out


def render_markdown(report_dir, plot=True):
    rounds = read_rounds(report_dir)
    if not rounds:
        return None
    last = rounds[-1]
    L = [f"# Relatorio do treino ({os.path.basename(os.path.abspath(report_dir))})", "",
         f"Round {last['round']} | {last['steps']:,} passos | {last.get('minutes', 0):.0f} min | "
         f"recompensas: {last.get('rewards', '?')} | deuses: {last.get('gods', '?')}", ""]
    alerts = _alerts(rounds)
    if alerts:
        L += ["## Alertas", ""] + [f"- {a}" for a in alerts] + [""]
    L += ["## Evolucao (ultimos rounds)", ""]
    rows = []
    for r in rounds[-12:]:
        e, p, t = r.get("eval_random"), r.get("eval_prev"), r.get("train")
        rows.append([r["round"], f"{r['steps']:,}", pct(e["win"]) if e else "-", pct(e["draw"]) if e else "-",
                     pct(p["win"]) if p else "-", pct(t["win"]) if t and t["n"] else "-",
                     pct(t["draw"]) if t and t["n"] else "-", f1(t["steps"]) if t and t["n"] else "-",
                     f"{f1(t['dev_me'])}/{f1(t['dev_opp'])}" if t and t["n"] else "-", r.get("opponent", "")])
    L += [_table(["round", "passos", "vit. vs aleat.", "ateu (aleat.)", "vit. vs anterior", "vit. treino",
                  "ateu treino", "decisoes/partida", "Devocao eu/opp (treino)", "treinou contra"], rows), ""]
    L += ["## Ultimo round em detalhe", ""]
    for title, key in (("Partidas de treino (contra o oponente do round)", "train"),
                       ("Avaliacao contra jogador aleatorio", "eval_random"),
                       ("Avaliacao contra o snapshot anterior", "eval_prev")):
        b = _block(title, last.get(key))
        if b:
            L += [b]
    tr = last.get("train")
    if tr and tr["n"]:
        L += ["## Como as partidas terminam (treino)", "",
              _table(["motivo", "fracao"], [[END_NAMES.get(k, k), pct(v)] for k, v in sorted(tr["end"].items(), key=lambda x: -x[1])]), ""]
        L += ["## Origem da Devocao por partida (treino)", "",
              _table(["fonte", "eu", "oponente"],
                     [[k, f2(tr["dev_src_me"].get(k, 0)), f2(tr["dev_src_opp"].get(k, 0))]
                      for k in sorted(set(tr["dev_src_me"]) | set(tr["dev_src_opp"]))] +
                     [["Devocao perdida (Cat-Mummy)", f2(tr["dev_lost_me"]), "-"]]), ""]
        L += ["## Acoes e cartas por partida (treino)", "",
              _table(["", *ACTION_NAMES], [["eu", *[f1(x) for x in tr["actions_me"]]], ["oponente", *[f1(x) for x in tr["actions_opp"]]]]), "",
              _table(["", *CARD_NAMES_PT], [["eu", *[f2(x) for x in tr["cards_me"]]], ["oponente", *[f2(x) for x in tr["cards_opp"]]]]), ""]
        pw = sorted(tr["power_games_me"].items(), key=lambda x: -x[1])
        L += ["## Poderes e Guardioes (fracao das partidas em que o agente os teve)", "",
              _table(["poder", "eu", "oponente"], [[k, pct(v), pct(tr["power_games_opp"].get(k, 0))] for k, v in pw]), ""]
        gd = [[KIND_NAMES[k], f2(tr["guardians_me"].get(KIND_NAMES[k], 0)), f2(tr["guardians_opp"].get(KIND_NAMES[k], 0))] for k in GUARDIAN_KINDS]
        L += [_table(["guardiao", "eu (por partida)", "oponente"], gd), ""]
        extra = [["Amun com 2 cartas", f2(tr["amun_double"])], ["desempates usados", f2(tr["tiebreak_used"])],
                 ["monumentos destruidos por Scorpion", f2(tr["monuments_destroyed"])], ["empurroes de Satet", f2(tr["pushes"])],
                 ["camelos no tabuleiro ao fim", f1(tr["camels"])]]
        L += ["## Eventos raros (por partida, treino)", "", _table(["evento", "valor"], extra), ""]
        s = tr["seat"]
        L += ["## Primeiro x segundo jogador (treino)", "",
              _table(["", "partidas", "vitorias"], [[k, v[0], pct(v[1] / v[0]) if v[0] else "-"] for k, v in s.items()]), ""]
    for title, key in (("treino", "train"), ("avaliacao contra aleatorio", "eval_random")):
        d = last.get(key)
        if d and d.get("by_players"):
            rows = [[p, v[0], pct(v[1] / max(v[0], 1)), pct(1 / int(p)), pct(v[2] / max(v[0], 1))]
                    for p, v in sorted(d["by_players"].items(), key=lambda x: int(x[0]))]
            L += [f"## Por numero de jogadores ({title})", "",
                  _table(["jogadores", "partidas", "vitorias", "acaso (1/P)", "Egito ateu"], rows), ""]
            seats = sorted(d["seat_idx"].items(), key=lambda x: (int(x[0].split("-")[0]), int(x[0].split("-")[1])))
            L += [_table(["jogadores-assento", "partidas", "vitorias"],
                         [[k, v[0], pct(v[1] / max(v[0], 1))] for k, v in seats]), ""]
    for title, key in (("treino", "train"), ("avaliacao contra aleatorio", "eval_random")):
        d = last.get(key)
        if d and d.get("merge_roles"):
            tot = sum(v[0] for v in d["merge_roles"].values()) or 1
            L += [f"## Fusao de deuses ({title}; partidas com 3+ jogadores)", "",
                  _table(["papel do agente", "partidas", "fracao", "vitorias"],
                         [[k, v[0], pct(v[0] / tot), pct(v[1] / max(v[0], 1))] for k, v in sorted(d["merge_roles"].items())]),
                  "", "Se o agente passar a ser 'absorvido' ou 'superior' com frequencia muito acima do esperado, ele pode estar buscando a fusao.", ""]
    er = last.get("eval_random")
    if er and er.get("by_god"):
        rows = [[GODS[int(k)], v[0], pct(v[1] / max(v[0], 1))] for k, v in sorted(er["by_god"].items(), key=lambda x: int(x[0]))]
        L += ["## Por deus (avaliacao contra aleatorio)", "", _table(["deus", "partidas", "vitorias"], rows), ""]
    mats = [r for r in rounds if r.get("matrix")]
    if mats:
        m = mats[-1]["matrix"]
        pool = sorted({int(k.split("-")[0]) for k in m} | {int(k.split("-")[1]) for k in m})
        rows = []
        for a in pool:
            rows.append([GODS[a]] + [("-" if a == b or f"{a}-{b}" not in m else
                                      f"{pct(m[f'{a}-{b}'][1] / max(m[f'{a}-{b}'][0], 1))} ({m[f'{a}-{b}'][0]})") for b in pool])
        L += [f"## Confrontos entre deuses (round {mats[-1]['round']}; linha = deus do modelo, coluna = deus do oponente; vitorias do modelo contra {mats[-1].get('matrix_vs', '?')})", "",
              _table(["", *[GODS[b] for b in pool]], rows), ""]
    text = "\n".join(L)
    with open(os.path.join(report_dir, "relatorio.md"), "w") as f:
        f.write(text)
    if plot:
        try:
            _plot(rounds, report_dir)
        except Exception:
            pass
    return text


def _plot(rounds, report_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["steps"] for r in rounds]
    get = lambda key, f: [f(r[key]) if r.get(key) and r[key]["n"] else float("nan") for r in rounds]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0, 0].plot(x, get("eval_random", lambda d: d["win"]), "o-", label="vs aleatorio")
    ax[0, 0].plot(x, get("eval_prev", lambda d: d["win"]), "o-", label="vs snapshot anterior")
    ax[0, 0].plot(x, get("train", lambda d: d["win"]), "o-", label="treino")
    ax[0, 0].set_title("vitorias"); ax[0, 0].set_ylim(0, 1); ax[0, 0].legend()
    ax[0, 1].plot(x, get("eval_random", lambda d: d["draw"]), "o-", label="aleatorio")
    ax[0, 1].plot(x, get("train", lambda d: d["draw"]), "o-", label="treino")
    ax[0, 1].set_title("Egito ateu"); ax[0, 1].set_ylim(0, 1); ax[0, 1].legend()
    ax[1, 0].plot(x, get("train", lambda d: d["steps"]), "o-"); ax[1, 0].set_title("decisoes por partida (treino)")
    ax[1, 1].plot(x, get("train", lambda d: d["dev_me"]), "o-", label="eu")
    ax[1, 1].plot(x, get("train", lambda d: d["dev_opp"]), "o-", label="oponente")
    ax[1, 1].set_title("Devocao final (treino)"); ax[1, 1].legend()
    for a in ax.flat:
        a.set_xlabel("passos")
    fig.tight_layout()
    fig.savefig(os.path.join(report_dir, "curvas.png"), dpi=110)
    plt.close(fig)
