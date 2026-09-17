"""Distance matrices and perfect matchings (prereg §7-§8).

All algorithms operate ONLY on distances — gold labels are never visible here.
Everything is seeded and deterministic given (pairing_seed, seed, qid).
"""
import random

import numpy as np


def cosine_distance_matrix(embeddings):
    """D1 (primary): cosine distance over embeddings. Returns (N, N) array."""
    X = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    sim = Xn @ Xn.T
    dist = 1.0 - np.clip(sim, -1.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _check_even(n):
    if n % 2 != 0:
        raise ValueError("perfect matching requires even N, got %d" % n)


def _canon(pairs):
    """Canonicalize: each pair (i, j) with i < j; list sorted."""
    return sorted(tuple(sorted(p)) for p in pairs)


def random_matching(n, rng):
    """Seeded random perfect matching (prereg §7 'Random')."""
    _check_even(n)
    agents = list(range(n))
    rng.shuffle(agents)
    return _canon([(agents[i], agents[i + 1]) for i in range(0, n, 2)])


def greedy_matching(dist, rng, farthest):
    """RGFM (farthest=True, prereg §7 primary) / nearest-greedy (False).

    Loop: pick a uniformly random unpaired agent i, pair with
    argmax/argmin_k d(i, k) over unpaired k != i. Ties: lowest index
    (numpy argmax/argmin). Order-dependent; the seed is recorded upstream.
    """
    n = dist.shape[0]
    _check_even(n)
    unpaired = set(range(n))
    pairs = []
    while unpaired:
        i = rng.choice(sorted(unpaired))
        unpaired.discard(i)
        best_j, best_d = None, None
        for j in sorted(unpaired):
            d = dist[i, j]
            if best_j is None or (d > best_d if farthest else d < best_d):
                best_j, best_d = j, d
        unpaired.discard(best_j)
        pairs.append((i, best_j))
    return _canon(pairs)


def maxweight_matching(dist):
    """Exact maximum-weight perfect matching, exhaustive over the (N-1)!!
    matchings (N<=8 => <=105 matchings; prereg §7 'Max-weight').
    Tie-breaking: first best in recursion order (deterministic)."""
    n = dist.shape[0]
    _check_even(n)
    if n > 8:
        raise ValueError("exhaustive max-weight matching requires N<=8")
    best = {"w": -np.inf, "pairs": None}

    def rec(agents, pairs, w):
        if not agents:
            if w > best["w"]:
                best["w"] = w
                best["pairs"] = list(pairs)
            return
        i = agents[0]
        for k in range(1, len(agents)):
            j = agents[k]
            rest = agents[1:k] + agents[k + 1:]
            rec(rest, pairs + [(i, j)], w + dist[i, j])

    rec(list(range(n)), [], 0.0)
    return _canon(best["pairs"])


def self_pairs(n):
    """C9 pseudo-pairing: each agent paired with itself (reviewer == reviewee)."""
    return [(i, i) for i in range(n)]


ALGOS = {"random", "nearest", "rgfm", "maxweight"}


def compute_matching(algo, dist, rng):
    if algo == "random":
        return random_matching(dist.shape[0], rng)
    if algo == "nearest":
        return greedy_matching(dist, rng, farthest=False)
    if algo == "rgfm":
        return greedy_matching(dist, rng, farthest=True)
    if algo == "maxweight":
        return maxweight_matching(dist)
    raise ValueError("unknown matching algorithm: %s" % algo)


def matching_diagnostics(dist, pairs):
    """Distance-only diagnostics for a matching."""
    ds = [float(dist[i, j]) for i, j in pairs]
    return {
        "total_weight": round(sum(ds), 6),
        "mean_pair_distance": round(sum(ds) / len(ds), 6) if ds else None,
        "min_pair_distance": round(min(ds), 6) if ds else None,
        "max_pair_distance": round(max(ds), 6) if ds else None,
    }


def is_perfect_matching(pairs, n):
    """Validation helper used by the smoke test."""
    flat = [a for p in pairs for a in p]
    return sorted(flat) == list(range(n))
