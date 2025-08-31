from __future__ import annotations
from typing import List, Set
from .aco_base import ACOBase, ACOConfig, argmax

class AntColonySystem(ACOBase):
    """Ant Colony System (ACS) with q0 exploitation and local pheromone updates."""
    def __init__(self, dist_matrix: List[List[float]], cfg: ACOConfig):
        super().__init__(dist_matrix, cfg)
        if cfg.q0 is None or cfg.phi is None:
            raise ValueError("ACS requires q0 and phi parameters.")

    def _choose_next_acs(self, current: int, unvisited: Set[int]) -> int:
        alpha, beta, q0 = self.cfg.alpha, self.cfg.beta, self.cfg.q0
        best = argmax(unvisited, key=lambda j: (self.tau[current][j] ** alpha) * (self.eta[current][j] ** beta))
        if self.rng.random() < q0:
            return best
        return self._prob_next(current, unvisited)

    def _local_update(self, i: int, j: int):
        phi = self.cfg.phi
        tau0 = self.cfg.tau0 if self.cfg.tau0 is not None else self._default_tau0()
        self.tau[i][j] = (1 - phi) * self.tau[i][j] + phi * tau0
        self.tau[j][i] = self.tau[i][j]

    def _tour_construction(self):
        n = self.n
        tour = []
        start = self.rng.randrange(n)
        tour.append(start)
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        while unvisited:
            nxt = self._choose_next_acs(current, unvisited)
            tour.append(nxt)
            self._local_update(current, nxt)
            unvisited.remove(nxt)
            current = nxt
        return tour