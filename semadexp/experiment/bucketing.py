"""Three bucketing policies:
A) traditional structured stratification (tier x category),
B) semantic-profile stratification (semantic clusters x tier),
C) competition-graph community randomization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ..config import BucketingPolicy
from ..features.competition_graph import CompetitionGraph
from ..features.semantic import profile_matrix


def assign_buckets(
    profiles: pd.DataFrame,
    graph: CompetitionGraph,
    policy: BucketingPolicy,
    treatment_share: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(profiles)
    if policy == BucketingPolicy.STRUCTURED_STRATIFIED:
        strata = list(zip(profiles["tier"], profiles["category"]))
        return _stratified_assign(strata, n, treatment_share, rng)
    if policy == BucketingPolicy.SEMANTIC_STRATIFIED:
        mat = profile_matrix(profiles)
        k = min(max(6, n // 15), n // 2)
        labels = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(mat)
        strata = list(zip(labels, profiles["tier"]))
        return _stratified_assign(strata, n, treatment_share, rng)
    if policy == BucketingPolicy.GRAPH_CLUSTER:
        return _graph_cluster_assign(graph, n, treatment_share, rng)
    raise ValueError(f"unknown policy {policy}")


def _stratified_assign(
    strata: list[tuple], n: int, share: float, rng: np.random.Generator
) -> np.ndarray:
    assignment = np.zeros(n, dtype=int)
    order = rng.permutation(n)
    groups: dict[tuple, list[int]] = {}
    for i in order:
        groups.setdefault(strata[i], []).append(i)
    for members in groups.values():
        k = int(round(len(members) * share))
        for j in members[:k]:
            assignment[j] = 1
    return assignment


def _graph_cluster_assign(
    graph: CompetitionGraph, n: int, share: float, rng: np.random.Generator
) -> np.ndarray:
    assignment = np.zeros(n, dtype=int)
    units = sorted(graph.communities.values(), key=len, reverse=True)
    sizes = np.zeros(2)
    for unit in units:
        arm = int(np.argmin(sizes))
        if rng.random() < 0.1:  # small random tie-break
            arm = int(rng.integers(0, 2))
        for node in unit:
            assignment[node] = arm
        sizes[arm] += len(unit)
    # correct drift from the target share
    diff = int(round(n * share)) - int(assignment.sum())
    if diff != 0:
        idx = np.where(assignment == (0 if diff > 0 else 1))[0]
        if len(idx):
            rng.shuffle(idx)
            flip = idx[: abs(diff)]
            assignment[flip] = 1 - assignment[flip]
    return assignment


def assignment_metrics(
    profiles: pd.DataFrame,
    graph: CompetitionGraph,
    assignment: np.ndarray,
) -> dict:
    """Balance (max/mean SMD of embedding + tier/category one-hots) and cross-arm edge ratio."""
    mat = profile_matrix(profiles)
    t = assignment == 1
    smd = []
    for col in range(mat.shape[1]):
        a, b = mat[t, col], mat[~t, col]
        if a.std() + b.std() < 1e-12:
            continue
        smd.append(abs(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12))
    smd = np.array(smd) if smd else np.array([0.0])
    return {
        "max_smd": float(smd.max()),
        "mean_smd": float(smd.mean()),
        "n_treatment": int(t.sum()),
        "n_control": int((~t).sum()),
        "cross_arm_edge_ratio": graph.cross_arm_edge_ratio(assignment),
        "internal_density": graph.internal_density(),
        "cross_density": graph.cross_density(),
    }

