"""
ankh_map.py - Mapa e cenarios do Ankh em forma extensa (para editar a mao).

LAYOUT
------
Hexagonos "flat-top" organizados em COLUNAS, exatamente como no seu codigo:
    coordenada publica = (col, linha_na_coluna), ambas a partir de 0,
    linha_na_coluna 0 = topo da coluna.
As colunas ficam centralizadas na vertical, entao colunas com menos celulas
sao deslocadas (COLUMN_COUNTS [8,9,10,9,...] -> colunas de 9 ficam meia
celula abaixo das de 10; as de 8 sobram 1 celula em cima e 1 embaixo).
Internamente usa-se "coordenadas dobradas": r2 = 2*linha + (max - n_da_coluna).

COMO EDITAR
-----------
1) GRID_COLUMNS: uma string por coluna, uma letra por celula, de cima p/ baixo.
       F fertil     D deserto     W agua
   O tamanho de cada string define quantas celulas a coluna tem.

2) RIVER: lados com rio, como (col, linha, direcao).
       direcoes: N, S, NE, SE, NW, SW
   (c, r, 'SE') = rio no lado sudeste da celula (c, r), isto e, entre ela e a
   vizinha a sudeste. Declare cada lado uma vez.
   Confira o numero de regioes que o validador imprime: ele e o teste de que
   o rio realmente separa (vazamento por um lado faltando e facil de cometer).

3) SCENARIOS: monumentos, figuras, camelos e Conflict Order iniciais.

Rode `python ankh_map.py` para validar e ver o mapa em ASCII.
"""

from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# 1) GRADE POR COLUNAS  (traduzida das listas water_hexes / fertile_hexes)
#    ATENCAO: no codigo original o deserto vinha de _initialize_desert();
#    aqui assumi "deserto = toda celula que nao e agua nem fertil".
# ---------------------------------------------------------------------------
GRID_COLUMNS = [
    "FFDDDDDD",      # col 0   (8)
    "FFFDDDDDD",     # col 1   (9)
    "FFFFDWWDDD",    # col 2  (10)
    "FFFFDDWDD",     # col 3   (9)
    "WFFFFDWFFF",    # col 4  (10)
    "WFFFFFFFF",     # col 5   (9)
    "WFFFFFFDDD",    # col 6  (10)
    "WFFFDDDDD",     # col 7   (9)
    "WFFFDDDWDD",    # col 8  (10)
    "WFFWDDWDD",     # col 9   (9)
    "WFDWWWWW",      # col 10  (8)
]
# Deslocamento vertical de cada coluna em meias celulas. None = centralizar.
COLUMN_OFFSET_HALF = None

# ---------------------------------------------------------------------------
# 2) RIOS: (col, linha, direcao)   -- preencher
# ---------------------------------------------------------------------------
RIVER = [
    (0, 0, 'NE'),
    (0, 0, 'SE'),   # estava 'E' na sua lista (nao existe em flat-top); lido como SE
    (0, 1, 'NE'),
    (1, 1, 'S'),
    (2, 2, 'SW'),
    (2, 2, 'S'),
    (3, 2, 'SW'),
    (3, 2, 'S'),
    (4, 3, 'SW'),
    (4, 3, 'S'),
    (4, 7, 'NE'),
    (4, 7, 'SE'),
    (4, 8, 'NE'),
    (4, 8, 'SE'),
    (4, 9, 'NE'),
    (4, 9, 'SE'),   # na borda do mapa: ignorado pelo carregador
    (5, 3, 'S'),
    (5, 3, 'SW'),
    (5, 3, 'SE'),
    (5, 4, 'NE'),
    (5, 4, 'SE'),
    (5, 5, 'NE'),
    (5, 5, 'SE'),
    (5, 5, 'S'),
    (6, 3, 'S'),
    (6, 3, 'SE'),
    (7, 2, 'S'),
    (7, 2, 'SE'),
    (8, 2, 'S'),
    (8, 2, 'SE'),
    (9, 1, 'S'),
    (9, 1, 'SE'),
]

# ---------------------------------------------------------------------------
# 3) CENARIOS
# ---------------------------------------------------------------------------
SCENARIOS = {
    "basic_2p": {
        "n_players": 2,
        # (tipo, (col, linha), dono)   tipo: 'obelisk' | 'temple' | 'pyramid'
        #                              dono: None (neutro) ou 0..n_players-1
        "monuments": [
            ("obelisk", (0, 4), 0),
            ("pyramid", (1, 8), None),
            ("temple", (3, 0), None),
            ("pyramid", (5, 2), None),
            ("temple", (5, 6), 1),
            ("obelisk", (8, 1), None),
            ("pyramid", (8, 5), None),
        ],
        # jogador -> lista de (tipo, (col, linha))   tipo: 'god' | 'warrior'
        "figures": {
            0: [
                ("god", (1, 5)),
                ("warrior", (1, 4)),
            ],
            1: [
                ("god", (6, 7)),
                ("warrior", (6, 6)),
            ],
        },
        # lados com camelo no inicio: (col, linha, direcao)
        "camels": [],
        # numero do Conflict Order token -> qualquer celula (col, linha) da regiao
        "conflict_order": {
            1: (0, 0),
            2: (5, 6),
            3: (1, 0),
        },
    },
    "basic_3p": {
        "n_players": 3,
        "monuments": [
            ("pyramid", (0, 2), None),
            ("obelisk", (2, 7), None),
            ("pyramid", (3, 1), 2),
            ("temple", (4, 5), 0),
            ("temple", (5, 3), None),
            ("obelisk", (5, 6), 1),
            ("pyramid", (7, 3), None),
            ("obelisk", (8, 1), None),
            ("temple", (9, 5), None),
        ],
        "figures": {
            0: [
                ("god", (3, 4)),
                ("warrior", (4, 4)),
            ],
            1: [
                ("god", (6, 7)),
                ("warrior", (6, 6)),
            ],
            2: [
                ("god", (4, 2)),
                ("warrior", (3, 2)),
            ],
        },
        "camels": [],
        "conflict_order": {
            1: (0, 0),
            2: (5, 6),
            3: (1, 0),
        },
    },
    "basic_4p": {
        "n_players": 4,
        "monuments": [
            ("pyramid", (0, 6), None),
            ("obelisk", (1, 4), None),
            ("temple", (2, 4), 0),
            ("temple", (3, 1), None),
            ("temple", (4, 8), 1),
            ("obelisk", (5, 3), 2),
            ("pyramid", (5, 5), None),
            ("temple", (6, 8), None),
            ("pyramid", (7, 5), 3),
            ("pyramid", (8, 2), None),
            ("obelisk", (8, 3), None),
        ],
        "figures": {
            0: [
                ("god", (1, 3)),
                ("warrior", (2, 3)),
            ],
            1: [
                ("god", (3, 7)),
                ("warrior", (4, 7)),
            ],
            2: [
                ("god", (5, 2)),
                ("warrior", (4, 3)),
            ],
            3: [
                ("god", (7, 4)),
                ("warrior", (6, 5)),
            ],
        },
        "camels": [(0, 2, "S"),
                   (1, 3, "SW"),
                   (1, 3, "S"),
                   (2, 4, "SW"),
                   ],
        "conflict_order": {
            1: (0, 0),
            2: (0, 3),
            3: (1, 0),
            4: (5, 6),
        },
    },
    "basic_5p": {
        "n_players": 5,
        "monuments": [
            ("temple", (1, 2), 0),
            ("obelisk", (1, 4), None),
            ("pyramid", (1, 7), 4),
            ("obelisk", (2, 0), None),
            ("temple", (4, 8), None),
            ("pyramid", (5, 3), 3),
            ("pyramid", (5, 4), None),
            ("temple", (6, 1), None),
            ("pyramid", (6, 6), None),
            ("obelisk", (6, 7), None),
            ("obelisk", (8, 3), 1),
            ("temple", (8, 9), 2),
        ],
        "figures": {
            0: [
                ("god", (1, 3)),
                ("warrior", (2, 3)),
            ],
            1: [
                ("god", (8, 4)),
                ("warrior", (7, 3)),
            ],
            2: [
                ("god", (8, 8)),
                ("warrior", (7, 8)),
            ],
            3: [
                ("god", (4, 3)),
                ("warrior", (5, 2)),
            ],
            4: [
                ("god", (1, 6)),
                ("warrior", (2, 7)),
            ],
        },
        "camels": [(0, 2, "S"),
                   (1, 3, "SW"),
                   (1, 3, "S"),
                   (2, 4, "SW"),
                   (5, 6, "S"),
                   (5, 6, "SE"),
                   (6, 6, "S"),
                   (6, 6, "SE"),
                   (7, 5, "S"),
                   (8, 6, "SW"),
                   ],
        "conflict_order": {
            1: (0, 0),
            2: (5, 6),
            3: (5, 7),
            4: (1, 0),
            5: (0, 3),
        },
    },
}

# ===========================================================================
# Geometria (flat-top, coordenadas dobradas internamente)
# ===========================================================================
AXIAL_DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

# (d_r2, d_col) em coordenadas dobradas
_DELTA = {"N": (-2, 0), "S": (2, 0), "NE": (-1, 1), "SE": (1, 1), "NW": (-1, -1), "SW": (1, -1)}
DIRECTIONS = tuple(_DELTA)
TERRAIN_CHARS = {"F", "D", "W"}
MONUMENT_TYPES = {"obelisk", "temple", "pyramid"}
FIGURE_TYPES = {"god", "warrior"}


def nbrs(ax):
    """Vizinhas em coordenadas axiais (funciona tambem fora do mapa)."""
    return {(ax[0] + dq, ax[1] + dr) for dq, dr in AXIAL_DIRS}


def edge(a, b):
    """Lado de hexagono entre duas celulas (coordenadas axiais)."""
    return frozenset((a, b))


def components(land, blocked_edges):
    """Regioes: componentes conexos de celulas de terra, sem cruzar rio/camelo."""
    seen, comps = set(), []
    for s in land:
        if s in seen:
            continue
        stack, comp = [s], {s}
        seen.add(s)
        while stack:
            u = stack.pop()
            for v in nbrs(u) & land:
                if v not in seen and edge(u, v) not in blocked_edges:
                    seen.add(v)
                    comp.add(v)
                    stack.append(v)
        comps.append(frozenset(comp))
    return comps


# ===========================================================================
# Tabuleiro
# ===========================================================================
class Board:
    """Estrutura estatica do tabuleiro.

    terrain      dict (col,linha) -> 'F'|'D'|'W'
    cells        lista ordenada de (col,linha); indice = id da celula
    idx          dict (col,linha) -> indice
    land, water  conjuntos de coords AXIAIS (para regioes e camelos)
    river_edges  frozenset de lados (frozenset de 2 coords axiais)
    move_dist    np.int8 [N,N]: distancia de movimento (atravessa tudo no mapa)
    axial(rc)    (col,linha) -> coord axial
    step(rc, d)  vizinha de rc na direcao d, ou None se fora do mapa
    """

    def __init__(self, grid_columns, offsets_half, river, warnings):
        self.counts = [len(s) for s in grid_columns]
        self.ncols = len(grid_columns)
        self.warnings = warnings
        self.terrain = {}
        self._pos, self._inv = {}, {}
        for c, col in enumerate(grid_columns):
            for r, ch in enumerate(col):
                if ch not in TERRAIN_CHARS:
                    raise ValueError(f"caractere invalido {ch!r} em {(c, r)}")
                r2 = 2 * r + offsets_half[c]
                if (r2 + c) % 2:
                    raise ValueError(
                        f"coluna {c}: deslocamento incompativel (paridade); "
                        "colunas vizinhas precisam alternar meia celula")
                self.terrain[(c, r)] = ch
                self._pos[(c, r)] = (r2, c)
                self._inv[(r2, c)] = (c, r)
        self.max_r2 = max(p[0] for p in self._pos.values())
        self.cells = sorted(self.terrain)
        self.idx = {rc: i for i, rc in enumerate(self.cells)}
        self.land = {self.axial(rc) for rc, t in self.terrain.items() if t != "W"}
        self.water = {self.axial(rc) for rc, t in self.terrain.items() if t == "W"}

        edges = set()
        for (c, r, d) in river:
            if d not in DIRECTIONS:
                raise ValueError(f"direcao invalida {d!r} em rio {(c, r, d)}")
            a, b = (c, r), self.step((c, r), d)
            if a not in self.terrain:
                raise ValueError(f"rio {(c, r, d)}: celula {a} nao existe")
            if b is None:
                # lado na borda do mapa: nao separa nada, so ignorar
                warnings.append(f"rio {(c, r, d)} fica na borda do mapa (ignorado)")
                continue
            if "W" in (self.terrain[a], self.terrain[b]):
                warnings.append(f"rio {(c, r, d)} toca agua; rios normalmente separam terra de terra")
            edges.add(edge(self.axial(a), self.axial(b)))
        self.river_edges = frozenset(edges)
        self.move_dist = self._distances()

    def axial(self, rc):
        r2, c = self._pos[rc]
        return (c, (r2 - c) // 2)

    def step(self, rc, d):
        r2, c = self._pos[rc]
        dr, dc = _DELTA[d]
        return self._inv.get((r2 + dr, c + dc))

    def _distances(self):
        n = len(self.cells)
        dist = np.full((n, n), 99, dtype=np.int8)
        for s in range(n):
            dist[s, s] = 0
            q = deque([self.cells[s]])
            while q:
                u = q.popleft()
                du = dist[s, self.idx[u]]
                for d in DIRECTIONS:
                    v = self.step(u, d)
                    if v is not None and dist[s, self.idx[v]] == 99:
                        dist[s, self.idx[v]] = du + 1
                        q.append(v)
        return dist

    def selfcheck(self):
        """Vizinhas por direcao == vizinhas axiais (garante a geometria)."""
        for rc in self.cells:
            got = {self.axial(v) for v in (self.step(rc, d) for d in DIRECTIONS) if v}
            want = nbrs(self.axial(rc)) & {self.axial(x) for x in self.cells}
            assert got == want, f"geometria inconsistente em {rc}"


def build_board(grid_columns=None, river=None, offsets_half=None):
    grid_columns = GRID_COLUMNS if grid_columns is None else grid_columns
    river = RIVER if river is None else river
    offsets_half = COLUMN_OFFSET_HALF if offsets_half is None else offsets_half
    if not grid_columns or not any(grid_columns):
        raise ValueError("GRID_COLUMNS vazio")
    if offsets_half is None:
        m = max(len(s) for s in grid_columns)
        offsets_half = [m - len(s) for s in grid_columns]
    return Board(grid_columns, offsets_half, river, [])


# ===========================================================================
# Cenario
# ===========================================================================
class Setup:
    """Estado inicial validado.

    monuments  : lista de (tipo, (col,linha), dono)
    figures    : dict jogador -> lista de (tipo, (col,linha))
    camel_edges: frozenset de lados (axiais)
    regions    : lista de frozensets de coords axiais
    region_of  : dict coord axial -> indice da regiao
    conflict_order: dict indice da regiao -> numero do token
    errors     : lista de mensagens de erro
    """


def load_scenario(board, sc):
    errors = []
    n_players = sc["n_players"]

    camel_edges = set()
    for (c, r, d) in sc.get("camels", []):
        a, b = (c, r), board.step((c, r), d) if (c, r) in board.terrain else None
        if b is None or "W" in (board.terrain[a], board.terrain[b]):
            errors.append(f"camelo {(c, r, d)} deve ficar entre duas celulas de terra")
            continue
        camel_edges.add(edge(board.axial(a), board.axial(b)))

    regions = components(board.land, board.river_edges | camel_edges)
    region_of = {ax: i for i, comp in enumerate(regions) for ax in comp}

    occupied = {}

    def occupy(rc, what):
        if rc not in board.terrain:
            errors.append(f"{what} em {rc}: fora do mapa")
        elif board.terrain[rc] == "W":
            errors.append(f"{what} em {rc}: celula de agua")
        elif rc in occupied:
            errors.append(f"{what} em {rc}: ja ocupada por {occupied[rc]}")
        else:
            occupied[rc] = what

    monuments = []
    for (kind, rc, owner) in sc.get("monuments", []):
        if kind not in MONUMENT_TYPES:
            errors.append(f"tipo de monumento invalido {kind!r}")
        if owner is not None and not (0 <= owner < n_players):
            errors.append(f"dono invalido {owner} no monumento em {rc}")
        occupy(rc, f"monumento {kind}")
        monuments.append((kind, rc, owner))

    figures = {}
    for p, figs in sc.get("figures", {}).items():
        if not (0 <= p < n_players):
            errors.append(f"jogador invalido {p} em figures")
            continue
        figures[p] = list(figs)
        n_god = sum(1 for k, _ in figs if k == "god")
        n_war = sum(1 for k, _ in figs if k == "warrior")
        if n_god > 1:
            errors.append(f"jogador {p} tem {n_god} deuses")
        if n_war > 6:
            errors.append(f"jogador {p} tem {n_war} guerreiros (max 6)")
        for kind, rc in figs:
            if kind not in FIGURE_TYPES:
                errors.append(f"tipo de figura invalido {kind!r}")
            occupy(rc, f"figura {kind} do jogador {p}")

    conflict_order = {}
    tokens = sc.get("conflict_order", {})
    for token, rc in tokens.items():
        if rc not in board.terrain or board.terrain[rc] == "W":
            errors.append(f"token {token} em {rc}: nao e celula de terra")
            continue
        ri = region_of[board.axial(rc)]
        if ri in conflict_order:
            errors.append(f"regiao de {rc} recebeu mais de um token")
        conflict_order[ri] = token
    missing = [i for i in range(len(regions)) if i not in conflict_order]
    if missing:
        errors.append(f"regioes sem Conflict Order token: {missing}")

    s = Setup()
    s.monuments, s.figures = monuments, figures
    s.camel_edges, s.regions, s.region_of = frozenset(camel_edges), regions, region_of
    s.conflict_order, s.errors = conflict_order, errors
    return s


# ===========================================================================
# Visualizacao ASCII
# ===========================================================================
def render(board, setup=None, show_regions=False, labels=False):
    """Mapa em layout escalonado (uma linha de texto por meia celula).

    Cada celula ocupa 4 colunas: terreno, simbolo, dono. Ex.: 'FT0 '.
    Simbolos: O/T/P = obelisco/templo/piramide (dono '-' = neutro);
              g/w = deus/guerreiro (dono = jogador).
    labels=True mostra 'col:linha' de cada celula (util para escrever rios).
    show_regions=True troca o terreno pela letra da regiao."""
    overlay = {}
    if setup is not None:
        for kind, rc, owner in setup.monuments:
            overlay[rc] = kind[0].upper() + ("-" if owner is None else str(owner))
        for p, figs in setup.figures.items():
            for kind, rc in figs:
                overlay[rc] = ("g" if kind == "god" else "w") + str(p)
    w = 6 if labels else 4
    out = ["      " + "".join(f"{c:<{w}d}" for c in range(board.ncols))]
    for r2 in range(board.max_r2 + 1):
        line = f"{r2 / 2:>4.1f}  "
        for c in range(board.ncols):
            rc = board._inv.get((r2, c))
            if rc is None:
                line += " " * w
            elif labels:
                line += f"{rc[0]}:{rc[1]}".ljust(w)
            else:
                ch = board.terrain[rc]
                if show_regions and setup is not None and ch != "W":
                    ch = chr(ord("A") + setup.region_of[board.axial(rc)] % 26)
                line += ch + overlay.get(rc, "  ") + " "
        out.append(line.rstrip())
    return "\n".join(out)


def validate_all(verbose=True):
    try:
        board = build_board()
        board.selfcheck()
    except (ValueError, AssertionError) as e:
        print("ERRO:", e)
        return False
    for w in board.warnings:
        print("AVISO:", w)
    n_land, n_water = len(board.land), len(board.water)
    print(f"{len(board.cells)} celulas ({n_land} terra, {n_water} agua); "
          f"{len(board.river_edges)} lados de rio")
    ok = True
    for name, sc in SCENARIOS.items():
        setup = load_scenario(board, sc)
        sizes = sorted((len(r) for r in setup.regions), reverse=True)
        if verbose:
            print(f"\n=== {name} ===")
            print(f"{len(setup.regions)} regioes, tamanhos (so terra): {sizes}")
            print(render(board, setup))
            print("\nRegioes:")
            print(render(board, setup, show_regions=True))
        for e in setup.errors:
            print("ERRO:", e)
            ok = False
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if validate_all() else 1)
