"""Semantic competition graph: advertisers as nodes, competition strength as edge weight,
community detection for graph-cluster bucketing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CompetitionGraph:
    advertiser_ids: list[int]
    matrix: np.ndarray
    edges: list[tuple[int, int, float]] = field(default_factory=list)
    communities: dict[int, list[int]] = field(default_factory=dict)

    def adjacency(self) -> np.ndarray:
        return self.matrix

    def internal_density(self, threshold: float = 0.3) -> float:
        m = (self.matrix > threshold).astype(float)
        total = 0.0
        count = 0
        for comm in self.communities.values():
            if len(comm) > 1:
                sub = m[np.ix_(comm, comm)]
                total += sub.sum() / (len(comm) * (len(comm) - 1))
                count += 1
        return total / max(count, 1)

    def cross_density(self, threshold: float = 0.3) -> float:
        m = (self.matrix > threshold).astype(float)
        total = 0.0
        count = 0
        comms = list(self.communities.values())
        for a in range(len(comms)):
            for b in range(a + 1, len(comms)):
                i, j = comms[a], comms[b]
                total += m[np.ix_(i, j)].mean()
                count += 1
        return total / max(count, 1)

    def cross_arm_edge_ratio(
        self, assignment: np.ndarray, threshold: float = 0.3
    ) -> float:
        """Fraction of strong competition edges that cross the control/treatment split."""
        m = self.matrix > threshold
        cross = assignment[:, None] != assignment[None, :]
        strong = m.sum()
        if strong == 0:
            return 0.0
        return float((m & cross).sum() / strong)


def build_competition_graph(
    profiles: pd.DataFrame,
    top_k: int = 8,
    min_weight: float = 0.15,
    audience_overlap_weight: float = 0.5,
    budget_proximity_weight: float = 0.25,
) -> CompetitionGraph:
    """Edge weight = semantic cosine sim x (0.5+0.5*audience Jaccard) x (0.75+0.25*budget proximity)."""
    emb = np.vstack(profiles["embedding"].values)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sim = emb @ emb.T
    np.fill_diagonal(sim, 0.0)
    n = len(profiles)
    aud = [set(tags) for tags in profiles["audience_tags"]]
    overlap = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            u = aud[i] & aud[j]
            o = len(u) / max(len(aud[i] | aud[j]), 1)
            overlap[i, j] = overlap[j, i] = o
    tier = np.array([{"S": 0, "M": 1, "L": 2}[t] for t in profiles["tier"]])
    prox = 1.0 - np.abs(tier[:, None] - tier[None, :]) / 2.0
    w = sim * (0.5 + audience_overlap_weight * overlap) * (0.75 + budget_proximity_weight * prox)
    np.fill_diagonal(w, 0.0)
    # keep top-k edges per node, symmetrize
    mask = np.zeros_like(w, dtype=bool)
    for i in range(n):
        idx = np.argsort(w[i])[::-1][:top_k]
        mask[i, idx] = True
    w = np.where(mask | mask.T, w, 0.0)
    w[w < min_weight] = 0.0
    edges = [
        (int(i), int(j), float(w[i, j]))
        for i in range(n)
        for j in range(i + 1, n)
        if w[i, j] > 0
    ]
    graph = CompetitionGraph(advertiser_ids=profiles["advertiser_id"].tolist(), matrix=w, edges=edges)
    graph.communities = detect_communities(graph)
    return graph


def detect_communities(graph: CompetitionGraph, max_community_size: int = 40) -> dict[int, list[int]]:
    """Louvain-style greedy modularity (networkx) with spectral fallback; oversized
    communities are recursively balanced-split so they can serve as randomization units."""
    n = len(graph.advertiser_ids)
    try:
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(range(n))
        for i, j, w in graph.edges:
            g.add_edge(i, j, weight=w)
        comps = nx.community.greedy_modularity_communities(g, weight="weight")
        communities = {ci: sorted(list(c)) for ci, c in enumerate(comps)}
    except Exception:
        communities = _spectral_communities(graph.matrix, k=min(max(6, n // 12), n // 2))
    if not communities:
        communities = {0: list(range(n))}
    # recursive balance split
    final: dict[int, list[int]] = {}
    cid = 0
    for nodes in communities.values():
        parts = _balanced_split(graph.matrix, list(nodes), max_community_size)
        for part in parts:
            if part:
                final[cid] = part
                cid += 1
    return final


def _spectral_communities(adj: np.ndarray, k: int) -> dict[int, list[int]]:
    from sklearn.cluster import SpectralClustering

    try:
        model = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=0)
        labels = model.fit_predict(np.maximum(adj, adj.T))
    except Exception:
        labels = np.zeros(adj.shape[0], dtype=int)
    out: dict[int, list[int]] = {}
    for ci in np.unique(labels):
        out[int(ci)] = np.where(labels == ci)[0].tolist()
    return out


def _balanced_split(adj: np.ndarray, nodes: list[int], max_size: int) -> list[list[int]]:
    if len(nodes) <= max_size:
        return [nodes]
    sub = adj[np.ix_(nodes, nodes)]
    from sklearn.cluster import SpectralClustering

    k = max(2, int(np.ceil(len(nodes) / max_size)))
    try:
        labels = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=0).fit_predict(
            np.maximum(sub, sub.T)
        )
    except Exception:
        labels = np.zeros(len(nodes), dtype=int)
    parts: list[list[int]] = []
    for ci in np.unique(labels):
        child = [nodes[i] for i in np.where(labels == ci)[0]]
        parts.extend(_balanced_split(adj, child, max_size))
    return parts

