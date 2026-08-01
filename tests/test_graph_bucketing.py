import numpy as np

from semadexp.config import BucketingPolicy
from semadexp.data.corpora import generate_advertisers
from semadexp.experiment.bucketing import assign_buckets, assignment_metrics
from semadexp.features.competition_graph import build_competition_graph
from semadexp.features.semantic import build_profiles
from semadexp.llm.client import DeterministicLLM


def test_graph_communities_balanced():
    ads = generate_advertisers(60, 3)
    profiles = build_profiles(ads, DeterministicLLM())
    graph = build_competition_graph(profiles)
    sizes = [len(v) for v in graph.communities.values()]
    assert max(sizes) <= 40
    assert sum(sizes) == 60


def test_graph_cluster_reduces_cross_edges():
    ads = generate_advertisers(60, 3)
    profiles = build_profiles(ads, DeterministicLLM())
    graph = build_competition_graph(profiles)
    a_rand = assign_buckets(profiles, graph, BucketingPolicy.STRUCTURED_STRATIFIED, seed=1)
    a_graph = assign_buckets(profiles, graph, BucketingPolicy.GRAPH_CLUSTER, seed=1)
    m_rand = assignment_metrics(profiles, graph, a_rand)
    m_graph = assignment_metrics(profiles, graph, a_graph)
    assert m_graph["cross_arm_edge_ratio"] <= m_rand["cross_arm_edge_ratio"] + 1e-9
    assert abs(a_graph.sum() - 30) <= 3

