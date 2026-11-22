"""The "core": which experts of each layer stay in RAM."""

import json

import numpy as np


def build_core(counts, frac=None, k=None, previous=None, stickiness=0.0):
    """Pick the hottest experts per layer.

    counts     [L, E] activation counts (any non-negative weights)
    frac / k   budget: share or number of experts kept per layer
    previous   optional previous core; with stickiness > 0 its members get a
               small bonus so the core does not thrash between near-ties
    """
    counts = np.asarray(counts, dtype=np.float64)
    L, E = counts.shape
    if k is None:
        k = max(1, int(round(E * (frac if frac is not None else 0.25))))
    k = min(k, E)
    score = counts.copy()
    if previous is not None and stickiness > 0:
        for layer, ids in enumerate(previous):
            score[layer, list(ids)] *= 1.0 + stickiness
    score += np.random.default_rng(0).random((L, E)) * 1e-9  # stable tie-break
    order = np.argsort(-score, axis=1)[:, :k]
    return [sorted(int(e) for e in row) for row in order]


def coverage(core, counts):
    """Share of activations in `counts` that the core serves from RAM."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    hit = sum(counts[layer, ids].sum() for layer, ids in enumerate(core))
    return float(hit / total)


def diff(old, new):
    """Number of experts swapped in when moving from old core to new core."""
    if old is None:
        return sum(len(x) for x in new)
    return sum(len(set(n) - set(o)) for o, n in zip(old, new))


def save_core(core, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"core": core}, f)


def load_core(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["core"]
