"""Gera um relatorio de ponta a ponta com partidas reais do simulador e confere o conteudo."""
import os, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
try:
    import gymnasium  # noqa
except ImportError:
    class Env:
        def reset(self, seed=None, options=None): pass
    class Box:
        def __init__(s, *a): s.shape = a[2]
    class Discrete:
        def __init__(s, n): s.n = n
    sp = types.ModuleType("gymnasium.spaces"); sp.Box, sp.Discrete = Box, Discrete
    gm = types.ModuleType("gymnasium"); gm.Env, gm.spaces = Env, sp
    sys.modules.update({"gymnasium": gm, "gymnasium.spaces": sp})
from ankh_env import AnkhEnv
from ankh_eval import evaluate, evaluate_matrix
from ankh_report import Agg, log_round, render_markdown


class RandomModel:
    def predict(self, obs, action_masks=None, deterministic=False):
        return np.random.choice(np.flatnonzero(action_masks)), None


if __name__ == "__main__":
    m, tmp = RandomModel(), tempfile.mkdtemp()
    for r in (1, 2):
        ar, ap, at = Agg(), Agg(), Agg()
        evaluate(m, 2, agg=ar)
        evaluate(m, 1, lambda o, mk: int(np.random.choice(np.flatnonzero(mk))), agg=ap)
        env = AnkhEnv(seed=r)
        for _ in range(6):
            env.reset(); done = False
            while not done:
                _, _, done, _, info = env.step(np.random.choice(np.flatnonzero(env.action_masks())))
            at.add(info["summary"])
        log_round(tmp, {"round": r, "steps": r * 1000, "minutes": 1.0, "opponent": "x", "rewards": "padrao",
                        "gods": ["amun"], "train": at.to_dict(), "eval_random": ar.to_dict(),
                        "eval_prev": ap.to_dict(), "matrix": evaluate_matrix(m, None, None, 1) if r == 2 else None,
                        "matrix_vs": "aleatorio"})
    text = render_markdown(tmp, plot=False)
    for secao in ("Evolucao", "Como as partidas terminam", "Origem da Devocao", "Poderes e Guardioes",
                  "Por deus", "Confrontos entre deuses", "Primeiro x segundo", "Fusao de deuses"):
        assert secao in text, secao
    print("relatorio ok:", len(text.splitlines()), "linhas")
