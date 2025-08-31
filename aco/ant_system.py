from __future__ import annotations
from typing import List
from .aco_base import ACOBase, ACOConfig

class AntSystem(ACOBase):
    """Classic Ant System (AS): global update with all ants."""
    def __init__(self, dist_matrix: List[List[float]], cfg: ACOConfig):
        super().__init__(dist_matrix, cfg)