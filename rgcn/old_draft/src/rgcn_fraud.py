"""R-GCN fraud scoring on the team's shared IEEE-CIS entity graph.

The team plan fixes the data, the split, the graph and the training rules, so
the only thing this module varies is the graph layer: an R-GCN layer with one
weight matrix per link type. The module covers

    1. the split file and the entity graph (build for development, load for the
       team's official files, check the counts),
    2. the R-GCN model,
    3. the training loop (one run = one seed on one graph),
    4. evaluation helpers (review budget, paired bootstrap).
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

ENTITY_COLS = ["card1", "addr1", "P_emaildomain", "R_emaildomain", "DeviceInfo", "DeviceType"]
SPLITS = ("train", "val", "test")
N_TYPES = len(ENTITY_COLS)  # every entity column is its own node type (this reproduces 14,239 nodes)


# --------------------------------------------------------------------------- split file
def load_split(path, columns=None) -> pd.DataFrame:
    """Load the team's ieee_sorted_split.parquet (only the columns we need)."""
    columns = columns or ["TransactionID", "isFraud", "TransactionDT", "split", *ENTITY_COLS]
    return pd.read_parquet(path, columns=columns)


def content_hash(df: pd.DataFrame) -> str:
    """Short fingerprint of a table's values, so teammates can confirm they hold the same split."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- graph
@dataclass
class Graph:
    """Entity graph plus the entity nodes of every payment."""

    edges: np.ndarray  # (E, 2) every link once, as (node of the lower type, node of the higher type)
    node_type: np.ndarray  # (N,) entity type 0..5 of every node
    payments: dict  # split -> DataFrame[TransactionID, isFraud, 6 node columns]
    missing: np.ndarray  # (6,) node number of the "missing" node of each type
    unknown: np.ndarray  # (6,) node number of the "unknown" node of each type
    info: dict

    @property
    def n_nodes(self) -> int:
        return len(self.node_type)

    def nodes(self, split: str) -> np.ndarray:
        return self.payments[split][[f"node_{c}" for c in ENTITY_COLS]].to_numpy(np.int64)

    def labels(self, split: str) -> np.ndarray:
        return self.payments[split]["isFraud"].to_numpy(np.float32)


def build_node_index(train: pd.DataFrame) -> dict:
    """Node numbers per entity column, learned from the training payments only.

    Layout of every column: its training values, then one "missing" node, then one
    "unknown" node (used for values never seen in training).
    """
    index, offset = {}, 0
    for col in ENTITY_COLS:
        values = sorted(train[col].dropna().astype(str).unique())
        index[col] = {
            "values": {v: offset + i for i, v in enumerate(values)},
            "missing": offset + len(values),
            "unknown": offset + len(values) + 1,
        }
        offset += len(values) + 2
    index["n_nodes"] = offset
    return index


def map_entities(df: pd.DataFrame, index: dict) -> np.ndarray:
    """(n, 6) node numbers: NaN -> missing node, unseen value -> unknown node."""
    out = np.empty((len(df), N_TYPES), np.int64)
    for j, col in enumerate(ENTITY_COLS):
        mapped = df[col].astype(str).map(index[col]["values"])  # NaN where the value is unseen (or missing)
        out[:, j] = np.where(df[col].isna(), index[col]["missing"], np.where(mapped.isna(), index[col]["unknown"], mapped))
    return out


def build_edges(nodes: np.ndarray, n_nodes: int) -> np.ndarray:
    """Link the 6 entities of every payment to each other; keep every link once."""
    parts = []
    for i, j in combinations(range(N_TYPES), 2):
        key = np.unique(nodes[:, i] * n_nodes + nodes[:, j])
        parts.append(np.stack([key // n_nodes, key % n_nodes], axis=1))
    return np.concatenate(parts).astype(np.int64)


def shuffle_training_payments(nodes: np.ndarray, seed: int) -> np.ndarray:
    """Control graph: shuffle each entity column over the training payments independently."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(nodes[:, j]) for j in range(N_TYPES)], axis=1)


def build_development_graph(split_df: pd.DataFrame, out_dir, shuffle_seeds=(42, 1, 2)) -> dict:
    """Rebuild the graph from the split file following the team plan, for development.

    The official graph comes from the SOA member. This builder follows the same
    rules and is used to develop and test the pipeline and to cross-check the
    official counts. Final results must use the official files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = split_df[split_df["split"] == "train"]
    index = build_node_index(train)
    n_nodes = index["n_nodes"]

    node_type = np.concatenate(
        [np.full(len(index[c]["values"]) + 2, t, np.int64) for t, c in enumerate(ENTITY_COLS)]
    )
    train_nodes = map_entities(train, index)
    edges = build_edges(train_nodes, n_nodes)
    np.save(out_dir / "node_type.npy", node_type)
    np.save(out_dir / "graph_edges.npy", edges)
    for seed in shuffle_seeds:
        shuffled = build_edges(shuffle_training_payments(train_nodes, seed), n_nodes)
        np.save(out_dir / f"graph_shuf_seed{seed}_edges.npy", shuffled)

    for name in SPLITS:
        part = split_df[split_df["split"] == name]
        table = pd.DataFrame(map_entities(part, index), columns=[f"node_{c}" for c in ENTITY_COLS])
        table.insert(0, "isFraud", part["isFraud"].to_numpy())
        table.insert(0, "TransactionID", part["TransactionID"].to_numpy())
        table.to_parquet(out_dir / f"payment_nodes_{name}.parquet", index=False)

    info = {
        "source": "development build from the split file (not the SOA member's graph)",
        "entity_columns": ENTITY_COLS,
        "n_nodes": int(n_nodes),
        "n_links_before_repeats": int(len(train) * len(list(combinations(range(N_TYPES), 2))) * 2),
        "n_links": int(len(edges)),
        "missing_node": {c: int(index[c]["missing"]) for c in ENTITY_COLS},
        "unknown_node": {c: int(index[c]["unknown"]) for c in ENTITY_COLS},
    }
    (out_dir / "graph_info.json").write_text(json.dumps(info, indent=2))
    return info


def load_graph(graph_dir, shuffled_seed: int | None = None) -> Graph:
    """Load a graph folder (official files or the development build).

    With shuffled_seed the links come from graph_shuf_seed<N>_edges.npy; the
    payment node files always stay real.
    """
    graph_dir = Path(graph_dir)
    info = json.loads((graph_dir / "graph_info.json").read_text())
    edge_file = "graph_edges.npy" if shuffled_seed is None else f"graph_shuf_seed{shuffled_seed}_edges.npy"
    node_type = np.load(graph_dir / "node_type.npy")
    payments = {s: _read_payment_nodes(graph_dir / f"payment_nodes_{s}.parquet") for s in SPLITS}
    return Graph(
        edges=canonical_edges(np.load(graph_dir / edge_file), node_type),
        node_type=node_type,
        payments=payments,
        missing=_special_nodes(info, "missing"),
        unknown=_special_nodes(info, "unknown"),
        info=info,
    )


def canonical_edges(edges: np.ndarray, node_type: np.ndarray) -> np.ndarray:
    """Every link once, as (node of the lower entity type, node of the higher type).

    Works whether the file stores each link once or in both directions.
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    swap = node_type[edges[:, 0]] > node_type[edges[:, 1]]
    edges = np.where(swap[:, None], edges[:, ::-1], edges)
    return np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)


def _read_payment_nodes(path) -> pd.DataFrame:
    """Payment node file -> TransactionID, isFraud and one node_<column> per entity column."""
    table = pd.read_parquet(path)
    entity = [c for c in table.columns if c not in ("TransactionID", "isFraud")]
    if len(entity) != N_TYPES:
        raise ValueError(f"{path}: expected {N_TYPES} entity node columns, found {entity}")
    by_name = {c: next((e for e in entity if col_matches(e, c)), None) for c in ENTITY_COLS}
    ordered = list(by_name.values()) if None not in by_name.values() else entity  # fall back to file order
    out = table[["TransactionID", "isFraud"]].copy()
    for col, source in zip(ENTITY_COLS, ordered):
        out[f"node_{col}"] = table[source].to_numpy(np.int64)
    return out


def col_matches(file_col: str, entity_col: str) -> bool:
    return entity_col.lower() in file_col.lower()


def _special_nodes(info: dict, kind: str) -> np.ndarray:
    """Node numbers of the "missing" / "unknown" node of each entity column, from graph_info.json."""
    value = next((info[k] for k in info if kind in k.lower() and isinstance(info[k], (dict, list))), None)
    if value is None:
        raise KeyError(f"graph_info.json has no '{kind}' node entry; keys: {list(info)}")
    if isinstance(value, dict):
        return np.array([value[next(k for k in value if col_matches(k, c))] for c in ENTITY_COLS])
    return np.asarray(value, dtype=np.int64)


def link_counts_by_type_pair(graph: Graph) -> pd.DataFrame:
    """Number of links (each once) between every pair of entity types."""
    t = graph.node_type[graph.edges]
    pair = pd.Series([f"{ENTITY_COLS[a]} - {ENTITY_COLS[b]}" for a, b in t], name="links")
    return pair.value_counts().rename_axis("link type").reset_index().sort_values("link type", ignore_index=True)


def check_graph(graph: Graph, expected_nodes: int = 14_239, expected_links_before_repeats: int = 12_401_340) -> dict:
    """The checks the plan asks for before anyone trains."""
    n_train = len(graph.payments["train"])
    links_before = n_train * len(list(combinations(range(N_TYPES), 2))) * 2
    nodes_all = np.concatenate([graph.nodes(s) for s in SPLITS])
    report = {
        "nodes": graph.n_nodes,
        "nodes_match_plan": graph.n_nodes == expected_nodes,
        "links_before_repeats": links_before,
        "links_before_repeats_match_plan": links_before == expected_links_before_repeats,
        "links_after_repeats_removed": len(graph.edges),
        "edge_ids_in_range": bool(graph.edges.min() >= 0 and graph.edges.max() < graph.n_nodes),
        "payment_nodes_in_range": bool(nodes_all.min() >= 0 and nodes_all.max() < graph.n_nodes),
        "payment_rows": {s: len(graph.payments[s]) for s in SPLITS},
        "node_type_counts": np.bincount(graph.node_type, minlength=N_TYPES).tolist(),
    }
    # every payment node must have the entity type of its column
    report["payment_node_types_ok"] = bool(
        all((graph.node_type[graph.nodes(s)[:, j]] == j).all() for s in SPLITS for j in range(N_TYPES))
    )
    return report


# --------------------------------------------------------------------------- R-GCN
def build_relations(graph: Graph, edges: np.ndarray | None = None):
    """Sparse, normalised adjacency of all link types, shape (N, R * N).

    A link type is the ordered pair (entity type of the sender, entity type of
    the receiver), so card -> address and address -> card have separate weights.
    Every link is used in both directions; messages are averaged per (receiver,
    link type), which is the usual R-GCN normalisation 1 / c_{v,r}.
    """
    edges = graph.edges if edges is None else edges
    n = graph.n_nodes
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    pair = graph.node_type[src] * N_TYPES + graph.node_type[dst]
    pair_ids, rel = np.unique(pair, return_inverse=True)  # compact numbering of the link types in use
    _, inv, count = np.unique(dst * len(pair_ids) + rel, return_inverse=True, return_counts=True)
    weight = (1.0 / count[inv]).astype(np.float32)
    index = torch.as_tensor(np.stack([dst, rel * n + src]), dtype=torch.long)
    adj = torch.sparse_coo_tensor(index, torch.as_tensor(weight), (n, len(pair_ids) * n)).coalesce()
    names = [f"{ENTITY_COLS[p // N_TYPES]} -> {ENTITY_COLS[p % N_TYPES]}" for p in pair_ids]
    return adj, names


class RGCNLayer(nn.Module):
    """h_v' = W_0 h_v + sum_r (1 / c_{v,r}) sum_{u in N_r(v)} W_r h_u.

    n_bases shares the relation weights through a small set of basis matrices
    (W_r = sum_b a_rb V_b); None gives every link type its own full matrix.
    """

    def __init__(self, d_in: int, d_out: int, n_rel: int, n_bases: int | None = None):
        super().__init__()
        self.n_rel, self.n_bases = n_rel, n_bases
        self.self_loop = nn.Linear(d_in, d_out)
        if n_bases:
            self.basis = nn.Parameter(torch.empty(n_bases, d_in, d_out))
            self.coef = nn.Parameter(torch.empty(n_rel, n_bases))
            nn.init.xavier_uniform_(self.basis.view(n_bases, -1))
            nn.init.xavier_uniform_(self.coef)
        else:
            self.weight = nn.Parameter(torch.empty(n_rel, d_in, d_out))
            for w in self.weight:
                nn.init.xavier_uniform_(w)

    def relation_weights(self) -> torch.Tensor:
        if self.n_bases:
            return (self.coef @ self.basis.flatten(1)).view(self.n_rel, *self.basis.shape[1:])
        return self.weight

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        messages = torch.einsum("nd,rde->rne", h, self.relation_weights()).reshape(self.n_rel * h.size(0), -1)
        return self.self_loop(h) + torch.sparse.mm(adj, messages)


@dataclass
class Config:
    """Training settings. The first block is fixed by the team plan, the second is ours."""

    dim: int = 64
    layers: int = 2
    epochs: int = 12
    patience: int = 10
    unknown_rate: float = 0.10
    # free choices (chosen once on validation, then the same for every seed and graph)
    lr: float = 3e-3
    dropout: float = 0.2
    n_bases: int | None = None
    batch_size: int = 4096


class RGCNFraud(nn.Module):
    """Learned node + entity-type embeddings -> R-GCN layers -> payment score."""

    def __init__(self, graph: Graph, n_rel: int, cfg: Config):
        super().__init__()
        self.node_emb = nn.Embedding(graph.n_nodes, cfg.dim)
        self.type_emb = nn.Embedding(N_TYPES, cfg.dim)
        self.register_buffer("node_type", torch.as_tensor(graph.node_type, dtype=torch.long))
        self.layers = nn.ModuleList(RGCNLayer(cfg.dim, cfg.dim, n_rel, cfg.n_bases) for _ in range(cfg.layers))
        self.head = nn.Sequential(
            nn.Linear(N_TYPES * cfg.dim, 128), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(128, 1)
        )

    def encode(self, adj: torch.Tensor) -> torch.Tensor:
        h = self.node_emb.weight + self.type_emb(self.node_type)
        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            if i < len(self.layers) - 1:
                h = torch.relu(h)
        return h

    def score(self, embeddings: torch.Tensor, payment_nodes: torch.Tensor) -> torch.Tensor:
        return self.head(embeddings[payment_nodes].flatten(1)).squeeze(-1)


def adj_without_link_type(adj: torch.Tensor, link_type: int, n_nodes: int) -> torch.Tensor:
    """Copy of the adjacency with one link type switched off (used to see what the model relies on)."""
    index, values = adj.indices(), adj.values()
    keep = (index[1] // n_nodes) != link_type
    return torch.sparse_coo_tensor(index[:, keep], values[keep], adj.shape).coalesce()


def link_type_weight_norms(model: RGCNFraud) -> np.ndarray:
    """(layers, link types) Frobenius norm of every link type's weight matrix."""
    return np.stack([layer.relation_weights().detach().flatten(1).norm(dim=1).cpu().numpy() for layer in model.layers])


# --------------------------------------------------------------------------- metrics
def roc_pr(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def review_budget(y: np.ndarray, score: np.ndarray, shares=(0.01, 0.05)) -> dict:
    """Share of all frauds caught when the top `share` of payments by score is reviewed."""
    order = np.argsort(-score, kind="stable")
    return {s: float(y[order[: int(round(s * len(y)))]].sum() / y.sum()) for s in shares}


def _curve_metrics(counts_pos: np.ndarray, counts_neg: np.ndarray) -> tuple[float, float]:
    """ROC-AUC and average precision from per-score-level weighted counts (descending score)."""
    tp, fp = np.cumsum(counts_pos), np.cumsum(counts_neg)
    p, n = tp[-1], fp[-1]
    tpr, fpr = np.concatenate([[0.0], tp / p]), np.concatenate([[0.0], fp / n])
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    ap = float(np.sum(np.diff(np.concatenate([[0.0], tp / p])) * precision))
    return auc, ap


def paired_bootstrap(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """95% interval of metric(a) - metric(b) over bootstrap resamples of the payments (same resamples for a and b)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    levels = []
    for s in (score_a, score_b):
        uniq, inv = np.unique(-s, return_inverse=True)  # ascending in -score = descending in score
        levels.append((inv, len(uniq)))
    pos = y.astype(bool)
    diffs = np.empty((n_boot, 2))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        res = []
        for inv, k in levels:
            cp = np.bincount(inv, weights=w * pos, minlength=k)
            cn = np.bincount(inv, weights=w * ~pos, minlength=k)
            res.append(_curve_metrics(cp, cn))
        diffs[b] = np.array(res[0]) - np.array(res[1])
    out = {}
    for i, name in enumerate(["roc_auc", "pr_auc"]):
        lo, hi = np.percentile(diffs[:, i], [2.5, 97.5])
        out[name] = {"diff": float(np.mean(diffs[:, i])), "lo": float(lo), "hi": float(hi)}
    return out


# --------------------------------------------------------------------------- training
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: RGCNFraud, adj: torch.Tensor, payment_nodes: torch.Tensor, batch: int = 65536) -> np.ndarray:
    """Raw model output (logit) for every payment."""
    model.eval()
    embeddings = model.encode(adj)
    out = [model.score(embeddings, payment_nodes[i : i + batch]) for i in range(0, len(payment_nodes), batch)]
    return torch.cat(out).float().cpu().numpy()


def train_run(graph: Graph, name: str, seed: int, cfg: Config, out_dir=None, score_test: bool = True,
              device: str = "cuda", verbose: bool = True) -> dict:
    """One run: train on the training payments, keep the epoch with the best validation ROC-AUC.

    The test payments are scored once, after training, with that epoch's weights.
    Writes <name>_val.parquet, <name>_test.parquet and <name>_history.json to out_dir.
    """
    set_seed(seed)
    start = time.time()
    adj, rel_names = build_relations(graph)
    adj = adj.to(device)
    model = RGCNFraud(graph, n_rel=len(rel_names), cfg=cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    nodes = {s: torch.as_tensor(graph.nodes(s), device=device) for s in SPLITS}
    y = {s: graph.labels(s) for s in SPLITS}
    y_train = torch.as_tensor(y["train"], device=device)
    pos_weight = torch.tensor(float(np.sqrt((y["train"] == 0).sum() / (y["train"] == 1).sum())), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    unknown = torch.as_tensor(graph.unknown, device=device)

    history, best = [], {"roc": -1.0, "epoch": 0, "state": None}
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(len(y_train), device=device)
        total = 0.0
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            batch_nodes = nodes["train"][idx]
            hide = torch.rand(batch_nodes.shape, device=device) < cfg.unknown_rate  # teach the "unknown" nodes
            batch_nodes = torch.where(hide, unknown.expand_as(batch_nodes), batch_nodes)
            loss = loss_fn(model.score(model.encode(adj), batch_nodes), y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        val_roc, val_pr = roc_pr(y["val"], predict(model, adj, nodes["val"]))
        history.append({"epoch": epoch, "loss": total / len(y_train), "val_roc_auc": val_roc, "val_pr_auc": val_pr})
        if verbose:
            print(f"  {name} epoch {epoch:2d}  loss {total / len(y_train):.4f}  val ROC-AUC {val_roc:.4f}  PR-AUC {val_pr:.4f}")
        if val_roc > best["roc"]:
            best = {"roc": val_roc, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        elif epoch - best["epoch"] >= cfg.patience:
            break

    model.load_state_dict(best["state"])
    scores = {"val": predict(model, adj, nodes["val"])}
    if score_test:
        scores["test"] = predict(model, adj, nodes["test"])  # the only time the test set is scored
    minutes = (time.time() - start) / 60

    row = {"run": name, "seed": seed, "best_epoch": best["epoch"], "minutes": round(minutes, 2),
           "val_roc_auc": best["roc"], "val_pr_auc": history[best["epoch"] - 1]["val_pr_auc"]}
    if score_test:
        row["test_roc_auc"], row["test_pr_auc"] = roc_pr(y["test"], scores["test"])

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, s in scores.items():
            table = graph.payments[split][["TransactionID", "isFraud"]].copy()
            table["score"] = s.astype(np.float32)
            table.to_parquet(out_dir / f"{name}_{split}.parquet", index=False)
        meta = {"model": "R-GCN", "column_set": "none (graph only)", "run": name, "seed": seed,
                "settings": asdict(cfg), "link_types": rel_names, "chosen_epoch": best["epoch"],
                "graph_source": graph.info.get("source", "official"), "epochs": history}
        (out_dir / f"{name}_history.json").write_text(json.dumps(meta, indent=2))
    return {"row": row, "scores": scores, "history": history, "model": model, "adj": adj, "link_types": rel_names}
