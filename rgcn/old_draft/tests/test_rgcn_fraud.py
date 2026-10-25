"""Checks for the pieces of rgcn_fraud that are easy to get subtly wrong.

Run with `python tests/test_rgcn_fraud.py` or `pytest tests`.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rgcn_fraud as rf  # noqa: E402


def tiny_graph() -> rf.Graph:
    """8 nodes: types 0..5, then one more node of type 0 and one of type 1."""
    node_type = np.array([0, 1, 2, 3, 4, 5, 0, 1])
    edges = np.array([[0, 1], [0, 2], [1, 2], [6, 1], [6, 7], [2, 3], [3, 4], [4, 5], [0, 5]])
    return rf.Graph(edges=edges, node_type=node_type, payments={}, missing=np.arange(6), unknown=np.arange(6), info={})


def test_layer_matches_naive_loop():
    """The sparse implementation must equal h_v' = W0 h_v + sum_r mean_{u in N_r(v)} W_r h_u."""
    torch.manual_seed(0)
    g = tiny_graph()
    adj, names = rf.build_relations(g)
    layer = rf.RGCNLayer(4, 3, n_rel=len(names))
    h = torch.randn(g.n_nodes, 4)
    fast = layer(h, adj).detach()

    src = np.concatenate([g.edges[:, 0], g.edges[:, 1]])
    dst = np.concatenate([g.edges[:, 1], g.edges[:, 0]])
    pair = g.node_type[src] * rf.N_TYPES + g.node_type[dst]
    rel_of = {p: i for i, p in enumerate(np.unique(pair))}
    expected = layer.self_loop(h).detach().clone()
    for v in range(g.n_nodes):
        for r in range(len(names)):
            senders = [u for u, d, p in zip(src, dst, pair) if d == v and rel_of[p] == r]
            if senders:
                msg = sum(h[u] @ layer.weight[r] for u in senders) / len(senders)
                expected[v] += msg.detach()
    assert torch.allclose(fast, expected, atol=1e-5)


def test_basis_sharing_has_same_shape_and_fewer_parameters():
    full, shared = rf.RGCNLayer(8, 8, n_rel=30), rf.RGCNLayer(8, 8, n_rel=30, n_bases=4)
    assert full.relation_weights().shape == shared.relation_weights().shape == (30, 8, 8)
    assert sum(p.numel() for p in shared.parameters()) < sum(p.numel() for p in full.parameters())


def test_relation_rows_are_normalised():
    """Per (receiver, link type) the message weights must sum to 1 (the 1 / c_{v,r} factor)."""
    g = tiny_graph()
    adj, _ = rf.build_relations(g)
    index, values = adj.indices(), adj.values()
    rel = index[1] // g.n_nodes
    sums = pd.Series(values.numpy()).groupby([index[0].numpy(), rel.numpy()]).sum()
    assert np.allclose(sums.values, 1.0)


def test_unknown_and_missing_mapping():
    train = pd.DataFrame({c: ["a", "b", None] for c in rf.ENTITY_COLS})
    index = rf.build_node_index(train)
    new = pd.DataFrame({c: ["a", "zzz", None] for c in rf.ENTITY_COLS})
    nodes = rf.map_entities(new, index)
    for j, col in enumerate(rf.ENTITY_COLS):
        assert nodes[0, j] == index[col]["values"]["a"]
        assert nodes[1, j] == index[col]["unknown"]  # never seen in training
        assert nodes[2, j] == index[col]["missing"]


def test_edges_are_each_link_once_and_independent_of_direction():
    nodes = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [6, 1, 2, 3, 4, 5]])  # first two payments identical
    edges = rf.build_edges(nodes, n_nodes=7)
    # 15 links for the first two (identical) payments; the third only swaps the first entity, adding 5 new links
    assert len(edges) == len(list(combinations(range(6), 2))) + 5
    both = np.concatenate([edges, edges[:, ::-1]])
    node_type = np.array([0, 1, 2, 3, 4, 5, 0])
    assert (rf.canonical_edges(both, node_type) == rf.canonical_edges(edges, node_type)).all()


def test_fast_bootstrap_metrics_match_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.05).astype(np.float32)
    score = np.round(rng.normal(size=5000) + 1.0 * y, 1)  # many tied scores
    levels = np.unique(-score, return_inverse=True)[1]
    counts = [np.bincount(levels, weights=w) for w in (y, 1 - y)]
    auc, ap = rf._curve_metrics(*counts)
    ref_auc, ref_ap = rf.roc_pr(y, score)
    assert abs(auc - ref_auc) < 1e-9 and abs(ap - ref_ap) < 1e-9


def test_review_budget():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    score = np.array([0.9, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # top 10% = the 0.9 payment (a fraud); top 20% adds the 0.3 payment (the other fraud)
    assert rf.review_budget(y, score, shares=(0.1, 0.2)) == {0.1: 0.5, 0.2: 1.0}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
