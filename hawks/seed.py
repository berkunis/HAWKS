"""Seed management for reproducible simulation randomness."""

from __future__ import annotations

import numpy as np


class SeedManager:
    """Deterministically derive independent RNGs for each simulation component.

    Each module receives its own np.random.Generator at construction.
    Modules never share RNGs. This ensures full reproducibility from a single
    master_seed and component-level independence.
    """

    def __init__(self, master_seed: int) -> None:
        self._seq = np.random.SeedSequence(master_seed)
        self._children: dict[str, np.random.Generator] = {}
        self._spawn_count = 0

    def spawn(self, name: str) -> np.random.Generator:
        """Deterministically derive a child RNG for a named component."""
        if name in self._children:
            return self._children[name]
        child_seq = self._seq.spawn(self._spawn_count + 1)[self._spawn_count]
        self._spawn_count += 1
        rng = np.random.default_rng(child_seq)
        self._children[name] = rng
        return rng
