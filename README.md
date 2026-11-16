# Ankh: Gods of Egypt - IA com PPO (2 a 5 jogadores)

Ambiente de regras do Ankh para **2 a 5 jogadores**, com os 5 deuses, os 12 Poderes Ankh, os 6 Guardiões, as 7 cartas de batalha e a **fusão de deuses** (3+ jogadores). Treinamento com `MaskablePPO` (sb3-contrib), self-play por snapshots, CPU em vários processos + GPU em lote. Uma única rede joga qualquer número de jogadores.

## Estrutura

```text
ankh_map.py      mapa (GRID_COLUMNS), rios (RIVER) e cenários (SCENARIOS)
ankh_game.py     regras: ações, eventos, camelos, conflito, batalha, deuses, poderes, guardiões, fusão
ankh_obs.py      codificação da observação (eu + até 4 oponentes; esconde escolhas secretas)
ankh_env.py      env Gymnasium de um assento + recompensas
ankh_worker.py   processo de CPU que roda várias partidas
ankh_vec.py      VecEnv em lote (SB3) + TorchOpponent (snapshot na GPU)
ankh_eval.py     avaliação por deus, cenário e assento; matriz de confrontos (2 jogadores)
ankh_report.py   estatísticas por partida, report.jsonl e relatório.md
report.py        regenera o relatório de uma run
train_local.py   treinamento no PC
replay.py        imprime partidas em texto (depurar regras)
tests/           testes de jogo, regras, env, VecEnv, soma zero e relatório
```

O código fica todo na raiz de propósito: os processos de CPU importam esses módulos.

## Treinar no PC

```bash
pip install -r requirements.txt        # + PyTorch com CUDA
python train_local.py --dir ~/ankh_runs/run1
```

Rode o mesmo comando para retomar; Ctrl+C salva um checkpoint.

- `--players 2` (ou `2,3`) restringe os números de jogadores; `--scenarios basic_2p,basic_4p` escolhe cenários. Por padrão, sorteia, a cada partida, um cenário válido (2 a 5 jogadores), o assento do agente e os deuses.
- `--god-pool 4` treina só com o Ra (0 Amun, 1 Anubis, 2 Isis, 3 Osiris, 4 Ra). Os deuses são sempre diferentes entre si.
- Todos os cenários disponíveis são validados antes do treinamento. Cenários com erros de dados são ignorados com um aviso. Para conferir a validade dos mapas e cenários, execute `python ankh_map.py`.
- `python train_local.py --help` lista as demais opções.

## Como funciona com 3 a 5 jogadores

- **Assentos e entidades:** cada partida tem P assentos. O agente ocupa um assento sorteado; os outros são jogados pelo oponente (aleatório ou snapshot da rede), inclusive o parceiro depois de uma fusão.
- **Observação:** sempre do ponto de vista de quem decide, com 4 espaços para oponentes ordenados pela ordem de turno (zeros se houver menos). O formato não muda com o número de jogadores.
- **Fusão de deuses** (fim do 3º Conflito, 3+ jogadores): o de menor Devoção é absorvido pelo 2º menor. O absorvido perde monumentos, figuras e cartas, e passa Seguidores e Guardiões ao outro; os dois tokens andam juntos; o deus fundido tem as habilidades dos dois e cada um dos dois jogadores faz **1 ação** por turno. Nas batalhas, decide o jogador do deus "superior". Vencem ou perdem juntos.
- **Recompensa:** vitória +1, derrota -1/(P-1) (com 2 jogadores, -1), Egito ateu = `draw`; mais os termos densos de Devoção e guerreiros (média dos oponentes). Com `--zero-sum`, os pesos ficam simétricos e o Egito ateu vale 0.
- **A fusão não gera recompensa:** a recompensa densa usa a Devoção *acumulada* (`rdev`), que a fusão não altera, e os guerreiros no tabuleiro. Assim, a queda de Devoção do deus "superior" (que passa a ter a posição do absorvido) e a saída do absorvido não pagam nem cobram nada de ninguém, nem do próprio agente nem dos oponentes. Ganhos reais de Devoção depois da fusão continuam premiados. O relatório traz uma tabela "Fusão de deuses" com a frequência com que o agente é absorvido/superior e a taxa de vitória em cada papel, para você conferir que ele não está buscando a fusão.
- **Modo estrito `--merged-reward zero`:** os dois assentos de um deus fundido recebem recompensa **0** (densa e final) depois da fusão. Cuidado: sem sinal de recompensa, o agente não aprende a jogar bem como deus fundido (que é uma parte grande das partidas com 3+ jogadores), e jogadores fracos passam a preferir a fusão a arriscar uma derrota. Por isso, o padrão é `normal`.
- **Self-play:** um assento aprende e os demais jogam com um snapshot congelado (ou aleatório). Com mais de 2 jogadores, não existe "soma zero" estrita entre dois lados; a recompensa acima é a aproximação usada.

## Relatório detalhado de cada run

A cada round, `--dir` recebe `report.jsonl` e o **`relatorio.md`** (mais `curvas.png` se houver matplotlib). Para regenerar:

```bash
python report.py --dir ~/ankh_runs/run1
```

Traz: evolução dos últimos rounds; como as partidas terminam; Devoção final; batalhas e domínios; **origem da Devoção**; ações, cartas, poderes e Guardiões; eventos raros; vantagem de posição; vitórias **por deus**; vitórias **por número de jogadores e por assento** (comparadas com o acaso, 1/P); matriz de confrontos entre deuses (cenários de 2 jogadores); alertas automáticos. Os mesmos números vão para o TensorBoard em `ankh/...`.

## Como ler o metrics.csv

- `win_vs_random` / `atheist_vs_random`: o modelo contra oponentes ALEATÓRIOS (média de todos os deuses, cenários e assentos). Com P jogadores, o acaso é 1/P; compare com a tabela por número de jogadores do relatório. `w_amun ... w_ra`: a mesma taxa por deus.
- `win_vs_prev` / `atheist_vs_prev`: o modelo contra o ÚLTIMO snapshot salvo (`prev_snapshot`).
- `train_opponent`: contra quem o round foi TREINADO.

## Conferir as regras em texto

```bash
python replay.py --scenario basic_4p --seed 7 --maps
python replay.py --models run1/checkpoints/round_0003.zip,run1/snapshots/snap_0002.zip
```

Mostra tudo (inclusive cartas e lances secretos): cada decisão, os eventos e o que mudou. Assentos sem modelo jogam aleatoriamente.

```bash
python tests/test_ankh_game.py     # partidas aleatórias de 2 a 5 jogadores + invariantes
python tests/test_rules.py         # deuses, poderes, guardiões, fusão e batalhas com 3 jogadores
python tests/test_ankh_env.py
python tests/test_ankh_vec.py
python tests/test_zero_sum.py
python tests/test_merge_reward.py  # a fusão não gera recompensa; modo estrito
python tests/test_report.py
