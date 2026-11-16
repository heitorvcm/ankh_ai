"""
train_local.py - Treino PPO do Ankh no PC (CPU em varios processos + GPU em lote).

Instalar:   pip install gymnasium stable-baselines3 sb3-contrib psutil
            (e o PyTorch com CUDA: https://pytorch.org/get-started/locally/)
Rodar:      python train_local.py --dir ~/ankh_runs/run1
Retomar:    rode o mesmo comando (usa o ultimo checkpoint de --dir).
Fase 2 (soma zero):
            python train_local.py --dir ~/ankh_runs/fase2 --zero-sum \
                   --init-from ~/ankh_runs/run1/checkpoints/round_0040.zip
Parar:      Ctrl+C (salva um checkpoint antes de sair).

Deixe os arquivos ankh_*.py na mesma pasta deste script. Os processos de CPU
usam 'spawn', por isso o bloco `if __name__ == "__main__"` no fim e obrigatorio.

Saidas em --dir: checkpoints/, snapshots/ (oponentes do self-play), tb/ e metrics.csv.
"""
import argparse
import csv
import gc
import glob
import os
import random
import shutil
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from ankh_eval import evaluate, evaluate_matrix
from ankh_report import Agg, log_round, make_stats_callback, render_markdown
from ankh_game import GODS, AnkhGame
from ankh_map import SCENARIOS
from ankh_vec import BatchedSelfPlayVecEnv, TorchOpponent


def default_workers():
    """Nucleos fisicos - 1 (deixa um para o processo principal, que cuida da GPU)."""
    n = 0
    try:
        import psutil
        n = psutil.cpu_count(logical=False) or 0
    except ImportError:
        pass
    if not n:
        n = max((os.cpu_count() or 2) // 2, 1)
    return max(n - 1, 1)


class DiskCheckpoint(BaseCallback):
    """Salva o modelo a cada `every` passos e apaga os mais antigos."""
    def __init__(self, every, directory, keep=3):
        super().__init__()
        self.every, self.dir, self.keep, self.last = every, directory, keep, 0

    def _on_training_start(self):
        self.last = self.num_timesteps

    def _on_step(self):
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            self.model.save(f"{self.dir}/step_{self.num_timesteps:010d}.zip")
            for old in sorted(glob.glob(f"{self.dir}/step_*.zip"))[:-self.keep]:
                os.remove(old)
        return True


def newest(directory):
    files = glob.glob(f"{directory}/*.zip")
    return max(files, key=os.path.getmtime) if files else None


def single(opp):
    """Adapta um oponente em lote (TorchOpponent) para decisoes individuais."""
    return lambda obs, mask: int(opp(obs[None], mask[None])[0])


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="ankh_run", help="pasta de saida (checkpoints, snapshots, tb, metrics)")
    p.add_argument("--workers", type=int, default=0, help="processos de CPU (0 = nucleos fisicos - 1)")
    p.add_argument("--envs", type=int, default=0, help="partidas em paralelo (0 = 16 por worker)")
    p.add_argument("--round-steps", type=int, default=400_000)
    p.add_argument("--total-rounds", type=int, default=40)
    p.add_argument("--save-every", type=int, default=100_000)
    p.add_argument("--keep-step-ckpts", type=int, default=3)
    p.add_argument("--keep-rounds", type=int, default=10)
    p.add_argument("--snapshot-every", type=int, default=1)
    p.add_argument("--eval-per-god", type=int, default=8, help="partidas de avaliacao por deus E por cenario")
    p.add_argument("--scenarios", default="all", help="cenarios usados (nomes separados por virgula) ou all")
    p.add_argument("--players", default="2,3,4,5", help="numeros de jogadores treinados, ex.: 2 ou 2,3")
    p.add_argument("--matrix-every", type=int, default=5, help="a cada N rounds, matriz de confrontos entre deuses (0 = nunca)")
    p.add_argument("--matrix-games", type=int, default=3, help="partidas por par de deuses na matriz")
    p.add_argument("--god-pool", default="0,1,2,3,4",
                   help="deuses sorteados (0 Amun, 1 Anubis, 2 Isis, 3 Osiris, 4 Ra), ex.: 4 ou 0,4")
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--merged-reward", choices=["normal", "zero"], default="normal",
                   help="zero: os assentos de um deus fundido recebem recompensa 0 depois da fusao")
    p.add_argument("--zero-sum", action="store_true",
                   help="recompensas de soma zero (pesos simetricos, Egito ateu = 0)")
    p.add_argument("--init-from", default="",
                   help="checkpoint .zip de uma fase anterior: usa os pesos como ponto de partida "
                        "e o semeia como primeiro oponente do self-play")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    root = os.path.abspath(os.path.expanduser(a.dir))
    ckpt, snap, tb, metrics = (f"{root}/checkpoints", f"{root}/snapshots", f"{root}/tb", f"{root}/metrics.csv")
    for d in (ckpt, snap, tb):
        os.makedirs(d, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("AVISO: GPU nao encontrada; treinando na CPU (bem mais lento).")
    workers = a.workers or default_workers()
    n_envs = a.envs or 16 * workers
    batch = max((a.n_steps * n_envs) // 4, 64)
    print("saida em:", root, "| relatorio:", f"{root}/relatorio.md")
    print(f"dispositivo: {device} | workers: {workers} | partidas: {n_envs} | lote de coleta: {a.n_steps * n_envs}")

    god_pool = [int(x) for x in a.god_pool.split(",")]
    print("deuses sorteados:", [GODS[x] for x in god_pool])
    wanted = [int(x) for x in a.players.split(",")]
    names = list(SCENARIOS) if a.scenarios == "all" else a.scenarios.split(",")
    names = [n for n in names if SCENARIOS[n]["n_players"] in wanted]
    probe = AnkhGame(names)
    scenarios = list(probe.scenarios)
    print("cenarios usados:", {n: SCENARIOS[n]["n_players"] for n in scenarios})
    for n, errs in probe.skipped.items():
        print(f"AVISO: cenario {n} ignorado (erro nos dados): {errs[0]}")

    venv = BatchedSelfPlayVecEnv(n_envs=n_envs, n_workers=workers, seed=int(time.time()) % 100000,
                                 zero_sum=a.zero_sum, god_pool=god_pool, scenarios=scenarios,
                                 merged_reward=a.merged_reward)
    print("recompensas:", "SOMA ZERO" if a.zero_sum else "padrao (nao simetricas)")
    last = newest(ckpt)
    if last:
        model = MaskablePPO.load(last, env=venv, device=device, tensorboard_log=tb)
        print("RETOMANDO de", os.path.basename(last), "| passos ja feitos:", model.num_timesteps)
    else:
        model = MaskablePPO("MlpPolicy", venv, n_steps=a.n_steps, batch_size=batch, n_epochs=4, gamma=0.995,
                            gae_lambda=0.95, ent_coef=0.01, learning_rate=a.lr, verbose=0, device=device,
                            tensorboard_log=tb, policy_kwargs=dict(net_arch=[512, 512]))
        print("treino NOVO")
        if a.init_from:
            model.set_parameters(os.path.expanduser(a.init_from), device=device)
            print("pesos iniciais carregados de", a.init_from)
            if not glob.glob(f"{snap}/snap_*.zip"):
                shutil.copy(os.path.expanduser(a.init_from), f"{snap}/snap_0000.zip")

    if not os.path.exists(metrics):
        with open(metrics, "w", newline="") as f:
            csv.writer(f).writerow(["timesteps", "win_vs_prev", "atheist_vs_prev", "prev_snapshot",
                                    "train_opponent", "steps_per_s", "minutes"]
                                   + [f"w_{g}" for g in GODS])

    goal, t0, opp = a.total_rounds * a.round_steps, time.time(), None
    try:
        while model.num_timesteps < goal:
            todo = a.round_steps - model.num_timesteps % a.round_steps
            del opp
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            
            # Seleção do oponente de treino
            pool = sorted(glob.glob(f"{snap}/snap_*.zip"))
            if not pool:
                opp, opp_name = None, "aleatorio"
            elif len(pool) == 1:
                path = pool[0]
                opp, opp_name = TorchOpponent(path, device=device), os.path.basename(path)
            else:
                if random.random() < 0.5:
                    path = pool[-1]  # 50% último modelo
                else:
                    candidates = pool[:-1]
                    if len(candidates) <= 4:
                        path = random.choice(candidates)
                    else:
                        if random.random() < 0.8:
                            path = random.choice(candidates[-4:])
                        else:
                            path = random.choice(candidates[:-4])
                opp, opp_name = TorchOpponent(path, device=device), os.path.basename(path)
            
            venv.set_opponent(opp)
            t_round = time.time()
            agg_train = Agg()
            model.learn(todo, reset_num_timesteps=False, tb_log_name="ppo", progress_bar=False,
                        callback=[DiskCheckpoint(a.save_every, ckpt, a.keep_step_ckpts), make_stats_callback(agg_train)])
            sps = todo / (time.time() - t_round)
            r = model.num_timesteps // a.round_steps
            model.save(f"{ckpt}/round_{r:04d}.zip")
            for old in sorted(glob.glob(f"{ckpt}/round_*.zip"))[:-a.keep_rounds]:
                os.remove(old)
            
            # Avaliação única contra o snapshot anterior
            agg_p = Agg()
            prev_files = sorted(glob.glob(f"{snap}/snap_*.zip"))  # ANTES de salvar o novo snapshot
            if prev_files:
                prev_name = os.path.basename(prev_files[-1])
                prev_fn = single(TorchOpponent(prev_files[-1], device=device))
                pw, pa, by_god = evaluate(model, a.eval_per_god, prev_fn, god_pool=god_pool, agg=agg_p, scenarios=scenarios)
            else:
                prev_name, pw, pa, prev_fn, by_god = "", 0.0, 0.0, None, {}

            if r % a.snapshot_every == 0:
                model.save(f"{snap}/snap_{r:04d}.zip")
            minutes = (time.time() - t0) / 60
            matrix = None
            if a.matrix_every and r % a.matrix_every == 0 and prev_fn:
                matrix = evaluate_matrix(model, prev_fn, god_pool, a.matrix_games, scenarios=scenarios)
            
            log_round(root, {"round": r, "steps": model.num_timesteps, "minutes": minutes, "opponent": opp_name,
                             "prev_snapshot": prev_name, "rewards": "soma zero" if a.zero_sum else "padrao",
                             "gods": [GODS[x] for x in god_pool], "scenarios": {n: SCENARIOS[n]["n_players"] for n in scenarios},
                             "train": agg_train.to_dict(),
                             "eval_prev": agg_p.to_dict() if prev_name else None,
                             "matrix": matrix, "matrix_vs": prev_name or "nenhum"})
            render_markdown(root)
            
            with open(metrics, "a", newline="") as f:
                csv.writer(f).writerow([model.num_timesteps, pw, pa, prev_name, opp_name,
                                        round(sps), round(minutes, 1)]
                                       + [round(by_god.get(g, float("nan")), 2) for g in range(len(GODS))])
                                       
            vs_prev = f"vs {prev_name}: vitorias {pw:.0%}, Egito ateu {pa:.0%}" if prev_name else "vs anterior: (sem snapshot ainda)"
            per_god = " ".join(f"{GODS[g][:3]} {w:.0%}" for g, w in by_god.items()) if by_god else ""
            god_str = f" ({per_god})" if per_god else ""
            print(f"round {r:3d} | passos {model.num_timesteps:>10,} | {sps:,.0f} passos/s | {vs_prev}{god_str} | treinou contra: {opp_name} | {minutes:.0f} min")
        print("treino concluido")
    except KeyboardInterrupt:
        path = f"{ckpt}/step_{model.num_timesteps:010d}.zip"
        model.save(path)
        print("\ninterrompido; checkpoint salvo em", path)
    finally:
        venv.close()


if __name__ == "__main__":
    main()