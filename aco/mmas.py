from __future__ import annotations
from typing import List
from .aco_base import ACOBase, ACOConfig

class MaxMinAntSystem(ACOBase):
    """Max-Min Ant System (MMAS): deposit only by best ant, enforce tau_min <= tau <= tau_max."""
    def __init__(self, dist_matrix: List[List[float]], cfg: ACOConfig):
        super().__init__(dist_matrix, cfg)
        if cfg.tau_min is None or cfg.tau_max is None:
            raise ValueError("MMAS requires tau_min and tau_max.")
        if cfg.tau_min > cfg.tau_max:
            raise ValueError("tau_min must be <= tau_max.")
        # initialize within bounds
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.tau[i][j] = min(max(self.tau[i][j], cfg.tau_min), cfg.tau_max)

    def _deposit(self, tours: List[List[int]], lengths: List[float]):
        # only best ant deposits
        best_idx = min(range(len(tours)), key=lambda k: lengths[k])
        best_tour = tours[best_idx]
        best_L = lengths[best_idx]
        dta = self.cfg.Q / best_L if best_L > 0 else 0.0
        for k in range(self.n):
            i, j = best_tour[k], best_tour[(k+1)%self.n]
            self.tau[i][j] += dta
            self.tau[j][i] += dta

    def _apply_bounds_if_needed(self):
        tau_min, tau_max = self.cfg.tau_min, self.cfg.tau_max
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    if self.tau[i][j] < tau_min:
                        self.tau[i][j] = tau_min
                    elif self.tau[i][j] > tau_max:
                        self.tau[i][j] = tau_max