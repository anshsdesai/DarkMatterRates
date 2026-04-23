"""Shared detector-response helpers."""

from __future__ import annotations


def build_probability_table(rate, max_ne=20):
    """Build the semiconductor probability table for a configured ``DMeRate``."""
    import torch

    prob_fn_tiled = []
    for ne in torch.arange(1, max_ne + 1):
        temp = rate.ionization_func(ne)
        temp = torch.where(torch.isnan(temp), 0, temp)
        prob_fn_tiled.append(temp)
    return torch.stack(prob_fn_tiled)


def rebuild_step_probability_table(rate, max_ne=20):
    """Switch a configured semiconductor rate object to step probabilities."""
    rate.ionization_func = rate.step_probabilities
    rate.probabilities = build_probability_table(rate, max_ne=max_ne)
    return rate.probabilities
