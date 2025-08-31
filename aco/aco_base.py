from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional

def argmax(items, key):
    best = None
    best_val = None
    for it in items:
        v = key(it)
        if best is None or v > best_val:
            best, best_val = it, v
    return best

@dataclass
class ACOConfig:
    alpha: float = 1.0          # pheromone influence
    beta: float = 2.0           # heuristic influence
    rho: float = 0.5            # evaporation rate
    Q: float = 1.0              # pheromone deposit factor
    tau0: Optional[float] = None # initial pheromone; if None, set later based on heuristic
    q0: Optional[float] = None  # ACS parameter for exploitation probability
    phi: Optional[float] = None # ACS local pheromone decay (0<phi<=1)
    n_ants: int = 20
    n_iterations: int = 100
    mmas: bool = False
    tau_min: Optional[float] = None
    tau_max: Optional[float] = None
    seed: Optional[int] = None

@dataclass
class ACOResult:
    best_tour: List[int]
    best_length: float
    history_best_lengths: List[float]
    config: ACOConfig
    elapsed_sec: float

class ACOBase:
    def __init__(self, dist_matrix: List[List[float]], cfg: ACOConfig):
        self.D = dist_matrix
        self.n = len(dist_matrix)
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        # heuristic 1/d
        self.eta = [[0.0]*self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.D[i][j] > 0:
                    self.eta[i][j] = 1.0 / self.D[i][j]

        tau0 = cfg.tau0 if cfg.tau0 is not None else self._default_tau0()
        self.tau = [[tau0 if i != j else 0.0 for j in range(self.n)] for i in range(self.n)]

        self.best_tour = None
        self.best_length = math.inf
        # per-iteration history for visualization
        self.history_best_lengths: List[float] = []
        self.history_best_tours: List[List[int]] = []

    def _default_tau0(self) -> float:
        # Simple heuristic: tau0 = 1 / (n * avg_dist)
        total = 0.0; count = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                total += self.D[i][j]; count += 1
        avg = total / max(1, count)
        return 1.0 / (self.n * avg) if avg > 0 else 1.0

    def _prob_next(self, current: int, unvisited: set) -> int:
        alpha, beta = self.cfg.alpha, self.cfg.beta
        weights = []
        total = 0.0
        for j in unvisited:
            w = (self.tau[current][j] ** alpha) * (self.eta[current][j] ** beta)
            weights.append((j, w))
            total += w
        if total == 0.0:
            return self.rng.choice(list(unvisited))
        r = self.rng.random() * total
        acc = 0.0
        for j, w in weights:
            acc += w
            if acc >= r:
                return j
        return weights[-1][0]

    def _tour_construction(self) -> List[int]:
        n = self.n
        tour = []
        start = self.rng.randrange(n)
        tour.append(start)
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        while unvisited:
            nxt = self._prob_next(current, unvisited)
            tour.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        return tour

    def _evaporate(self):
        rho = self.cfg.rho
        for i in range(self.n):
            for j in range(self.n):
                self.tau[i][j] *= (1.0 - rho)

    def _deposit(self, tours: List[List[int]], lengths: List[float]):
        Q = self.cfg.Q
        for tour, L in zip(tours, lengths):
            dta = Q / L if L > 0 else 0.0
            for k in range(self.n):
                i, j = tour[k], tour[(k+1)%self.n]
                self.tau[i][j] += dta
                self.tau[j][i] += dta

    def _apply_bounds_if_needed(self):
        # Overridden by MMAS
        pass

    def run(self) -> ACOResult:
        import time
        start = time.time()
        # reset history
        self.history_best_lengths = []
        self.history_best_tours = []

        for it in range(self.cfg.n_iterations):
            tours = [self._tour_construction() for _ in range(self.cfg.n_ants)]
            lengths = [self._tour_length(t) for t in tours]
            # update global best
            for t, L in zip(tours, lengths):
                if L < self.best_length:
                    self.best_length = L
                    self.best_tour = t

            self._evaporate()
            self._deposit(tours, lengths)
            self._apply_bounds_if_needed()

            # record history for visualization
            self.history_best_lengths.append(self.best_length)
            self.history_best_tours.append(list(self.best_tour))

        elapsed = time.time() - start
        return ACOResult(best_tour=self.best_tour, best_length=self.best_length,
                         history_best_lengths=self.history_best_lengths, config=self.cfg, elapsed_sec=elapsed)

    def _tour_length(self, tour: List[int]) -> float:
        dist = 0.0
        for k in range(self.n):
            i, j = tour[k], tour[(k+1)%self.n]
            dist += self.D[i][j]
        return dist