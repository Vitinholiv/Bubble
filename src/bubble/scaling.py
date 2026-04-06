"""Influencer scaling functions."""

import numpy as np
from typing import Callable

def constant_scaling(qt: int) -> Callable[[int], int]:
    """Regra Constante: k = qt"""
    def const_rule(num_nodes: int) -> int:
        return max(2, qt)
    return const_rule

def linear_scaling(alpha: float) -> Callable[[int], int]:
    """Regra Linear: k = ceil(alpha * N)"""
    def lin_rule(num_nodes: int) -> int:
        return max(2, int(np.ceil(alpha * num_nodes)))
    return lin_rule

def sqrt_scaling(alpha: float = 1.0) -> Callable[[int], int]:
    """Regra de Raiz Quadrada: k = ceil(alpha * sqrt(N))"""
    def sqrt_rule(num_nodes: int) -> int:
        return max(2, int(np.ceil(alpha * np.sqrt(num_nodes))))
    return sqrt_rule

def log_scaling(alpha: float = 2.0, base: float = 2.0) -> Callable[[int], int]:
    """Regra Logarítmica: k = ceil(alpha * log_base(N))"""
    def log_rule(num_nodes: int) -> int:
        val = max(1, num_nodes)
        return max(2, int(np.ceil(alpha * (np.log(val) / np.log(base)))))
    return log_rule

def power_law_scaling(alpha: float = 2.0, exponent: float = 0.5) -> Callable[[int], int]:
    """Regra de Potência: k = ceil(alpha * N^exponent)"""
    def power_rule(num_nodes: int) -> int:
        return max(2, int(np.ceil(alpha * (num_nodes ** exponent))))
    return power_rule