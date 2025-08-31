from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class TSPInstance:
    coords: List[Tuple[float, float]]
    name: str = "euclidean_tsp"

    @staticmethod
    def random_euclidean(n: int, seed: Optional[int] = None, square_size: float = 100.0, name: str = "random_euclidean"):
        rng = random.Random(seed)
        coords = [(rng.uniform(0, square_size), rng.uniform(0, square_size)) for _ in range(n)]
        return TSPInstance(coords=coords, name=name)

    def n_cities(self) -> int:
        return len(self.coords)

    def distance(self, i: int, j: int) -> float:
        (x1, y1), (x2, y2) = self.coords[i], self.coords[j]
        return math.hypot(x1 - x2, y1 - y2)

    def distance_matrix(self) -> List[List[float]]:
        n = self.n_cities()
        D = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                d = self.distance(i, j)
                D[i][j] = D[j][i] = d
        return D

    def tour_length(self, tour: List[int]) -> float:
        n = self.n_cities()
        dist = 0.0
        for k in range(n):
            i, j = tour[k], tour[(k + 1) % n]
            dist += self.distance(i, j)
        return dist