"""
ankh_eval.py - Avaliacao: o modelo joga cada deus, em cada cenario (2 a 5 jogadores) e em todos os
assentos, contra oponentes (None = aleatorios) que recebem deuses diferentes.
"""
from ankh_env import AnkhEnv
from ankh_map import SCENARIOS


def evaluate(model, n_per_god=4, opponent=None, god_pool=None, seed=12345, agg=None, scenarios=None):
    """Devolve (vitorias, egito_ateu, vitorias_por_deus {deus: taxa}) do modelo (deterministico).
    `n_per_god` = partidas por deus E por cenario (os assentos se alternam).
    Se `agg` (ankh_report.Agg) for dado, acumula o resumo detalhado de cada partida."""
    env = AnkhEnv(scenario=scenarios, seed=seed, opponent=opponent, god_pool=god_pool)
    pool, scens = list(env.g.god_pool), list(env.g.scenarios)
    wins = draws = total = 0
    by_god = {}
    for god in pool:
        w = n = 0
        for sc in scens:
            P = SCENARIOS[sc]["n_players"]
            for ep in range(n_per_god):
                obs, _ = env.reset(options={"scenario": sc, "me": ep % P, "god_me": god})
                done = False
                while not done:
                    a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
                    obs, r, done, _, info = env.step(int(a))
                w += bool(info["won"])
                n += 1
                draws += info["winner"] is None
                if agg is not None:
                    agg.add(info["summary"])
        by_god[god] = w / max(n, 1)
        wins += w
        total += n
    return wins / total, draws / total, by_god


def evaluate_matrix(model, opponent=None, god_pool=None, n_per_pair=3, seed=777, scenarios=None):
    """Vitorias do modelo para cada par (deus do modelo, deus do oponente) em cenarios de 2 jogadores:
    {"a-b": [partidas, vitorias]}."""
    env = AnkhEnv(scenario=scenarios, seed=seed, opponent=opponent, god_pool=god_pool)
    two = [s for s in env.g.scenarios if SCENARIOS[s]["n_players"] == 2]
    if not two:
        return {}
    pool = list(env.g.god_pool)
    out = {}
    for a in pool:
        for b in pool:
            if a == b and len(pool) > 1:
                continue
            w = 0
            for ep in range(n_per_pair):
                obs, _ = env.reset(options={"scenario": two[0], "me": ep % 2, "god_me": a, "god_opp": b})
                done = False
                while not done:
                    act, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
                    obs, r, done, _, info = env.step(int(act))
                w += bool(info["won"])
            out[f"{a}-{b}"] = [n_per_pair, int(w)]
    return out
