"""Grid geometry for the tag game (Pineau, Gordon & Thrun, IJCAI 2003, Fig. 2b).

Cells are numbered bottom row first, left to right, exactly as in the paper:

    y=4            26 27 28
    y=3            23 24 25
    y=2            20 21 22
    y=1   10 11 12 13 14 15 16 17 18 19
    y=0    0  1  2  3  4  5  6  7  8  9
"""
from __future__ import annotations

import numpy as np

UP, DOWN, LEFT, RIGHT, TAG = range(5)
ACTION_NAMES = ("Up", "Down", "Left", "Right", "Tag")
MOVE_DELTAS = {UP: (0, 1), DOWN: (0, -1), LEFT: (-1, 0), RIGHT: (1, 0)}

# Map rows listed top (largest y) to bottom; '.' = free cell, '#' = wall.
def open_layout(width: int, height: int | None = None) -> tuple:
    """Layout of an obstacle-free width x height grid (cell = y * width + x)."""
    return tuple("." * width for _ in range(height or width))


PINEAU_LAYOUT = (
    "#####...##",
    "#####...##",
    "#####...##",
    "..........",
    "..........",
)


class TagGrid:
    """Free cells of a rectangular map, with move and distance tables."""

    def __init__(self, layout=PINEAU_LAYOUT):
        rows = [r for r in layout]
        self.height = len(rows)
        self.width = max(len(r) for r in rows)
        coords = []
        # Number cells bottom-up, left-to-right (paper convention).
        for y in range(self.height):
            row = rows[self.height - 1 - y]
            for x, ch in enumerate(row):
                if ch == ".":
                    coords.append((x, y))
        if not coords:
            raise ValueError("layout has no free cells")
        self.coords = np.array(coords, dtype=int)  # [n, 2] (x, y)
        self.n = len(coords)
        self._index = {c: i for i, c in enumerate(coords)}

        # move[cell, action] -> next cell (bumping into a wall = stay put).
        self.move = np.zeros((self.n, 4), dtype=int)
        for i, (x, y) in enumerate(coords):
            for a, (dx, dy) in MOVE_DELTAS.items():
                self.move[i, a] = self._index.get((x + dx, y + dy), i)

        diff = self.coords[:, None, :] - self.coords[None, :, :]
        self._dist = {
            "manhattan": np.abs(diff).sum(-1).astype(float),
            "euclidean": np.sqrt((diff**2).sum(-1)),
            "chebyshev": np.abs(diff).max(-1).astype(float),
        }

        self.distances = self._dist
        self._away_cache: dict = {}

    @property
    def away(self) -> dict:
        """Opponent "move away" tables (Pineau Tag only), built on first use."""
        if not self._away_cache:
            self._away_cache = {rule: self._away_table(rule) for rule in ("axis", "greedy")}
        return self._away_cache

    def _away_table(self, rule: str):
        """away[r][o] -> tuple of (next_cell, weight); weights sum to 1 and describe where
        the opponent goes *given that it tries to move* (it stays with prob. 1 - p_move).

        "axis"   (default; standard Tag implementations, e.g. APPL / pomdp-py): half of the
                 mass on the x-move away from the robot, half on the y-move; if aligned on an
                 axis, that half is split between both directions. Blocked moves = stay.
        "greedy" all mass split evenly over neighbours that increase the Manhattan distance
                 to the robot; stays if there is none.
        Both reproduce the paper's example (robot s0, opponent s15 -> 0.4 s16, 0.4 s20, 0.2 s15).
        """
        man = self._dist["manhattan"]
        table = [[() for _ in range(self.n)] for _ in range(self.n)]
        for r in range(self.n):
            rx, ry = self.coords[r]
            for o in range(self.n):
                ox, oy = self.coords[o]
                w: dict[int, float] = {}
                if rule == "axis":
                    for d_r, d_o, pos, neg in ((rx, ox, RIGHT, LEFT), (ry, oy, UP, DOWN)):
                        dirs = [pos] if d_r < d_o else [neg] if d_r > d_o else [pos, neg]
                        for a in dirs:
                            c = int(self.move[o, a])
                            w[c] = w.get(c, 0.0) + 0.5 / len(dirs)
                elif rule == "greedy":
                    cands = sorted({int(self.move[o, a]) for a in MOVE_DELTAS} - {o})
                    cands = [c for c in cands if man[c, r] > man[o, r]] or [o]
                    w = {c: 1.0 / len(cands) for c in cands}
                else:
                    raise ValueError(f"unknown opponent rule {rule!r}")
                table[r][o] = tuple(sorted(w.items()))
        return table

    def index(self, x: int, y: int) -> int:
        return self._index[(x, y)]

    def distance(self, metric: str = "manhattan") -> np.ndarray:
        """[n, n] matrix of distances between cells."""
        try:
            return self._dist[metric]
        except KeyError:
            raise ValueError(f"unknown distance metric {metric!r}; use one of {list(self._dist)}")
