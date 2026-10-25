"""Assemble notebooks/rgcn_fraud.ipynb from src/rgcn_fraud.py and the cells below.

The first notebook cell writes the module next to the notebook, so the notebook
runs on Colab without cloning anything. Usage: python tools/build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md("""
# R-GCN on the team's shared entity graph

**New member 2 (R-GCN)** · Team plan: *graph models with and without the V columns* · IEEE-CIS fraud data

The team compares four graph models (GraphSAGE, GAT, R-GCN, HAN) under identical data, split, graph and training
rules, so that only the graph layer differs. This notebook is the R-GCN part. It

1. checks the shared split and graph against the plan,
2. builds R-GCN (one weight matrix per link type),
3. chooses the free settings on the validation set only,
4. trains 3 seeds on the real graph and 3 seeds on the shuffled graphs (the plan's 6 runs), plus a no-message-passing
   ablation as an extra,
5. saves the files the plan asks for (logit scores, history, results row),
6. records what we observe.

R-GCN uses **no payment columns**: it scores a payment from its six entity nodes only (card, address, two e-mail
domains, device info, device type). Combining it with the tabular models is done later by the evaluation member.
""")

md("""
**Parameters.** The official graph files from the SOA member go into `graph/`. While they are missing, the notebook
rebuilds the graph from the split file with the same rules (`graph_dev/`) so the pipeline can be developed and
tested; every output is then stored under `results/development/` and must not be submitted. With the official files
present, everything is stored under `results/official/` and the development build is used only to cross-check the
official counts.
""")

SOURCE = (ROOT / "src" / "rgcn_fraud.py").read_text(encoding="utf-8")
code("%%writefile rgcn_fraud.py\n" + SOURCE)

code("""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import rgcn_fraud as rf

QUICK = False  # True: 2 epochs and 2 grid points, only to test the notebook end to end

SPLIT_FILE = next(p for p in (Path("ieee_sorted_split.parquet"), Path("split_entities.parquet")) if p.exists())
GRAPH_DIR = Path("graph")          # official files from the SOA member
DEV_GRAPH_DIR = Path("graph_dev")  # development build with the same rules
OFFICIAL = (GRAPH_DIR / "graph_info.json").exists()
ACTIVE_GRAPH_DIR = GRAPH_DIR if OFFICIAL else DEV_GRAPH_DIR
RESULTS = Path("results") / ("official" if OFFICIAL else "development")
RESULTS.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 1, 2)
EPOCHS = 2 if QUICK else 12

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pd.set_option("display.width", 160)
print("device:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu (slow)")
print("graph :", "OFFICIAL files from the SOA member" if OFFICIAL else "DEVELOPMENT build from the split file (preliminary results)")
""")

md("""
## 1. The shared split file

Everyone loads Reset's `ieee_sorted_split.parquet`; nobody makes their own split. R-GCN needs only the identifiers, the
label, the split and the six entity columns, so only those are read. The content hash lets teammates confirm they hold
the same data.
""")

code("""
split = rf.load_split(SPLIT_FILE)
days = (split.groupby("split")["TransactionDT"].agg(first_day="min", last_day="max") / 86400).round(1)
summary = split.groupby("split").agg(payments=("isFraud", "size"), fraud_rate=("isFraud", "mean")).join(days)
summary = summary.loc[list(rf.SPLITS)]
summary["plan_payments"] = [413_378, 88_581, 88_581]
display(summary.round(4))
assert (summary["payments"] == summary["plan_payments"]).all(), "split sizes differ from the plan"
print("sorted by TransactionDT:", split["TransactionDT"].is_monotonic_increasing, "| total payments:", len(split))
print("content hash of the loaded columns:", rf.content_hash(split))
""")

md("""
## 2. The entity graph

The plan fixes the graph: six entity columns, each its own node type, one *missing* and one *unknown* node per type,
and a link between every two entities that occur in the same training payment. The plan gives two numbers to check
before anyone trains: 14,239 nodes and 12,401,340 links before repeats are removed.
""")

code("""
if OFFICIAL:
    graph = rf.load_graph(GRAPH_DIR)
else:
    rf.build_development_graph(split, DEV_GRAPH_DIR)
    graph = rf.load_graph(DEV_GRAPH_DIR)

report = rf.check_graph(graph)
checks = pd.DataFrame(
    [("nodes", report["nodes"], 14_239),
     ("links before repeats are removed", report["links_before_repeats"], 12_401_340)],
    columns=["check", "observed", "plan"],
)
checks["ok"] = checks["observed"] == checks["plan"]
display(checks)
print("links after repeats are removed:", f"{report['links_after_repeats_removed']:,}")
print("node numbers in range:", report["edge_ids_in_range"], report["payment_nodes_in_range"],
      "| every payment node has the entity type of its column:", report["payment_node_types_ok"])
assert checks["ok"].all() and report["payment_node_types_ok"], "graph does not match the plan: stop and compare with Reset"

types = pd.DataFrame({"entity type": rf.ENTITY_COLS, "nodes": report["node_type_counts"],
                      "missing node": graph.missing, "unknown node": graph.unknown})
display(types)
""")

code("""
# Cross-check of the official graph: rebuild it from the split file and compare link counts per link type.
if OFFICIAL:
    rf.build_development_graph(split, DEV_GRAPH_DIR)
    rebuilt = rf.load_graph(DEV_GRAPH_DIR)
    both = rf.link_counts_by_type_pair(graph).merge(
        rf.link_counts_by_type_pair(rebuilt), on="link type", suffixes=("_official", "_rebuilt"))
    both["equal"] = both["links_official"] == both["links_rebuilt"]
    display(both)
    print("official graph agrees with the rebuild on every link type:", bool(both["equal"].all()))
else:
    print("development graph in use: no official files to compare with yet")
""")

code("""
# The three shuffled link files: how different are they from the real graph?
def link_keys(g):
    return set((g.edges[:, 0] * g.n_nodes + g.edges[:, 1]).tolist())

real_keys = link_keys(graph)
rows = []
for seed in SEEDS:
    shuffled = rf.load_graph(ACTIVE_GRAPH_DIR, shuffled_seed=seed)
    keys = link_keys(shuffled)
    rows.append({"seed": seed, "links": len(keys), "links also in the real graph": len(keys & real_keys),
                 "share (%)": round(100 * len(keys & real_keys) / len(keys), 2)})
display(pd.DataFrame(rows))
""")

md("""
## 3. What the model can see

R-GCN receives a payment only as six node numbers. Two things limit what it can learn: how often an entity is
*missing* (no information) or *unknown* (never seen in training), and how many links an entity has.
""")

code("""
rows = []
for s in rf.SPLITS:
    nodes = graph.nodes(s)
    for j, col in enumerate(rf.ENTITY_COLS):
        rows.append({"split": s, "entity": col,
                     "missing %": 100 * (nodes[:, j] == graph.missing[j]).mean(),
                     "unknown %": 100 * (nodes[:, j] == graph.unknown[j]).mean()})
coverage = pd.DataFrame(rows).pivot(index="entity", columns="split").reindex(rf.ENTITY_COLS)
display(coverage.round(1))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, kind in zip(axes, ("missing %", "unknown %")):
    coverage[kind][list(rf.SPLITS)].plot.bar(ax=ax, rot=30, width=0.8)
    ax.set_title(f"Payments whose entity is {kind.split()[0]}")
    ax.set_ylabel("% of payments")
fig.tight_layout()
plt.show()
""")

code("""
display(rf.link_counts_by_type_pair(graph).rename(columns={"links": "links (each once)"}))

degree = np.bincount(graph.edges.ravel(), minlength=graph.n_nodes)
deg = pd.DataFrame({"type": [rf.ENTITY_COLS[t] for t in graph.node_type], "degree": degree})
display(deg.groupby("type")["degree"].describe(percentiles=[0.5, 0.95])[["count", "50%", "95%", "max"]].reindex(rf.ENTITY_COLS))
""")

md("""
## 4. The model

Every node starts as a learned embedding plus an embedding of its entity type (64 dimensions). Two R-GCN layers
update the nodes, then a payment's six entity embeddings are concatenated (6 × 64 = 384) and scored by
`Linear(384, 128) → ReLU → dropout → Linear(128, 1)`.

One R-GCN layer updates node *v* as

$$h_v' = W_0 h_v + \\sum_{r} \\frac{1}{c_{v,r}} \\sum_{u \\in \\mathcal{N}_r(v)} W_r h_u ,$$

with one weight matrix $W_r$ per link type $r$ and $c_{v,r}$ the number of neighbours of $v$ under $r$. A link type is
the ordered pair *(entity type of the sender, entity type of the receiver)*, so card→address and address→card have
separate weights. With six entity types there are 30 link types. Every link is used in both directions.

| Fixed by the plan | Value |
| --- | --- |
| Layers, embedding size | 2, 64 |
| Epochs, model selection | 12, best validation ROC-AUC (patience 10) |
| Optimizer, loss | Adam, weighted BCE with fraud weight √(N0/N1) from the training payments |
| Unknown-node training | each epoch 10 % of the training payments' entities are replaced by their unknown node |
| Seeds, test set | 42, 1, 2; the test set is scored once, at the end of each run |

| Free choices (made once, on validation) | Options |
| --- | --- |
| Learning rate | 1e-3, 3e-3, 1e-2 |
| Dropout in the payment head | 0.2, 0.4 |
| Weight sharing across link types | none (a full matrix per link type) or 8 basis matrices |
""")

code("""
adj, link_types = rf.build_relations(graph)
print(f"{len(link_types)} link types in use; {adj._nnz():,} directed links after repeats are removed")
model = rf.RGCNFraud(graph, n_rel=len(link_types), cfg=rf.Config())
print("trainable parameters:", f"{sum(p.numel() for p in model.parameters()):,}",
      "| of which node embeddings:", f"{model.node_emb.weight.numel():,}")
del model
""")

md("""
## 5. Free choices, on the validation set only

Learning rate, dropout and weight sharing are not given by the paper, so they are chosen once, with seed 42 on the real
graph, by validation ROC-AUC. The test set is not scored in this search. The chosen values are then used for every seed
and for the shuffled graphs.
""")

code("""
GRID = [{"lr": lr, "dropout": do, "n_bases": nb} for lr in (1e-3, 3e-3, 1e-2) for do in (0.2, 0.4) for nb in (None, 8)]
if QUICK:
    GRID = GRID[:2]
search = []
for g in GRID:
    out = rf.train_run(graph, "search", seed=42, cfg=rf.Config(**g, epochs=EPOCHS), score_test=False,
                       device=DEVICE, verbose=False)
    search.append({**g, **{k: out["row"][k] for k in ("best_epoch", "val_roc_auc", "val_pr_auc", "minutes")}})
search = pd.DataFrame(search).sort_values("val_roc_auc", ascending=False, ignore_index=True)
search.to_csv(RESULTS / "free_choice_search.csv", index=False)
display(search.round(4))

best = search.iloc[0]
CFG = rf.Config(lr=float(best["lr"]), dropout=float(best["dropout"]),
                n_bases=None if pd.isna(best["n_bases"]) else int(best["n_bases"]), epochs=EPOCHS)
(RESULTS / "free_choices.json").write_text(json.dumps(
    {"lr": CFG.lr, "dropout": CFG.dropout, "n_bases": CFG.n_bases, "batch_size": CFG.batch_size,
     "chosen_by": "validation ROC-AUC, seed 42, real graph"}, indent=2))
print("chosen:", {"lr": CFG.lr, "dropout": CFG.dropout, "n_bases": CFG.n_bases})
""")

md("""
## 6. Final runs

For each seed: R-GCN on the real graph (`rgcn_seed<N>`) and on the shuffled link file of the same seed
(`rgcn_shuf_seed<N>`); these are the plan's 6 runs. As an extra, the same network without any message passing
(`noprop_seed<N>`, zero graph layers: the payment head sees only the learned node and type embeddings) shows how much
the links add on top of the entities themselves. Every run writes `<run>_val.parquet`, `<run>_test.parquet` (logit,
`TransactionID`, `isFraud`) and `<run>_history.json`.
""")

code("""
RUNS = {}

def run(name, g, seed, cfg):
    out = rf.train_run(g, name, seed, cfg, out_dir=RESULTS, device=DEVICE, verbose=False)
    r = out["row"]
    print(f"{name:18s} best epoch {r['best_epoch']:2d} | val ROC {r['val_roc_auc']:.4f} | "
          f"test ROC {r['test_roc_auc']:.4f}  PR {r['test_pr_auc']:.4f} | {r['minutes']:.1f} min")
    RUNS[name] = out

for seed in SEEDS:
    run(f"rgcn_seed{seed}", graph, seed, CFG)
    run(f"rgcn_shuf_seed{seed}", rf.load_graph(ACTIVE_GRAPH_DIR, shuffled_seed=seed), seed, CFG)
for seed in SEEDS:
    run(f"noprop_seed{seed}", graph, seed, rf.Config(**{**CFG.__dict__, "layers": 0}))

cols = ["run", "test_roc_auc", "test_pr_auc", "best_epoch", "minutes"]
table = pd.DataFrame([RUNS[k]["row"] for k in RUNS])[cols]
table[~table["run"].str.startswith("noprop")].to_csv(RESULTS / "results_table.csv", index=False)
table[table["run"].str.startswith("noprop")].to_csv(RESULTS / "extra_runs_table.csv", index=False)
""")

md("""
## 7. Results

All numbers are test-set scores of the six plan runs plus the extra ablation. The range in brackets is the lowest and
highest seed. Intervals are 95 % paired bootstrap intervals over the test payments (2,000 resamples; the same
resamples for both models of a pair).
""")

code("""
GROUPS = {"R-GCN, real graph": "rgcn_seed", "R-GCN, shuffled graph": "rgcn_shuf_seed", "no message passing": "noprop_seed"}
rows = []
for label, prefix in GROUPS.items():
    keys = [k for k in RUNS if k.startswith(prefix)]
    row = {"model": label}
    for metric, name in (("val_roc_auc", "val ROC-AUC"), ("test_roc_auc", "test ROC-AUC"), ("test_pr_auc", "test PR-AUC")):
        v = np.array([RUNS[k]["row"][metric] for k in keys])
        row[name] = f"{v.mean():.4f} ({v.min():.4f}-{v.max():.4f})"
    budget = np.array([[*rf.review_budget(graph.labels("test"), RUNS[k]["scores"]["test"]).values()] for k in keys])
    row["frauds caught, top 1%"], row["frauds caught, top 5%"] = (f"{100 * budget[:, 0].mean():.1f}%", f"{100 * budget[:, 1].mean():.1f}%")
    rows.append(row)
SUMMARY = pd.DataFrame(rows).set_index("model")
display(SUMMARY)
""")

code("""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
styles = {"rgcn_seed": ("tab:blue", "real graph"), "rgcn_shuf_seed": ("tab:orange", "shuffled graph"), "noprop_seed": ("tab:green", "no message passing")}
for prefix, (colour, label) in styles.items():
    for i, k in enumerate(k for k in RUNS if k.startswith(prefix)):
        h = pd.DataFrame(RUNS[k]["history"])
        ax[0].plot(h["epoch"], h["loss"], color=colour, alpha=0.8, label=label if i == 0 else None)
        ax[1].plot(h["epoch"], h["val_roc_auc"], color=colour, alpha=0.8, label=label if i == 0 else None)
ax[0].set(title="Training loss", xlabel="epoch", ylabel="weighted BCE")
ax[1].set(title="Validation ROC-AUC", xlabel="epoch", ylabel="ROC-AUC")
ax[1].legend()
fig.tight_layout()
fig.savefig(RESULTS / "training_curves.png", dpi=150)
plt.show()
""")

code("""
# Paired bootstrap per seed: is the real graph better than the shuffled one (RQ3), and does message passing add anything?
y_test = graph.labels("test")
rows = []
for seed in SEEDS:
    real = RUNS[f"rgcn_seed{seed}"]["scores"]["test"]
    for other, label in ((f"rgcn_shuf_seed{seed}", "real - shuffled graph"), (f"noprop_seed{seed}", "R-GCN - no message passing")):
        res = rf.paired_bootstrap(y_test, real, RUNS[other]["scores"]["test"], n_boot=200 if QUICK else 2000, seed=seed)
        rows.append({"comparison": label, "seed": seed,
                     "ROC-AUC diff": res["roc_auc"]["diff"], "ROC 95% CI": f"({res['roc_auc']['lo']:+.4f}, {res['roc_auc']['hi']:+.4f})",
                     "PR-AUC diff": res["pr_auc"]["diff"], "PR 95% CI": f"({res['pr_auc']['lo']:+.4f}, {res['pr_auc']['hi']:+.4f})"})
BOOT = pd.DataFrame(rows)
BOOT.to_csv(RESULTS / "paired_bootstrap.csv", index=False)
display(BOOT.round(4))
""")

md("""
## 8. What the model learned

Two views of which link types matter: the size of each link type's weight matrix in the trained model (seed 42), and
how much the *validation* ROC-AUC falls when one link type is switched off at prediction time (the model is not
retrained). Only validation payments are used here.
""")

code("""
main = RUNS["rgcn_seed42"]
model, adj, names = main["model"], main["adj"], main["link_types"]
val_nodes = torch.as_tensor(graph.nodes("val"), device=DEVICE)
y_val = graph.labels("val")
full_roc = rf.roc_pr(y_val, rf.predict(model, adj, val_nodes))[0]

drop = []
for r, name in enumerate(names):
    roc = rf.roc_pr(y_val, rf.predict(model, rf.adj_without_link_type(adj, r, graph.n_nodes), val_nodes))[0]
    drop.append(full_roc - roc)
norms = rf.link_type_weight_norms(model)
LINKS = pd.DataFrame({"link type": names, "weight norm, layer 1": norms[0], "weight norm, layer 2": norms[1],
                      "val ROC-AUC drop when removed": drop}).sort_values("val ROC-AUC drop when removed", ascending=False, ignore_index=True)
LINKS.to_csv(RESULTS / "link_type_importance.csv", index=False)
print(f"validation ROC-AUC of the full model: {full_roc:.4f}")
display(LINKS.head(10).round(4))

fig, ax = plt.subplots(figsize=(7, 5.5))
LINKS.iloc[::-1].plot.barh(x="link type", y="val ROC-AUC drop when removed", ax=ax, legend=False, color="tab:blue")
ax.set(xlabel="drop in validation ROC-AUC", title="Link types the model relies on (seed 42)")
fig.tight_layout()
plt.show()
""")

md("""
## 9. Where the graph can help

A payment whose card was never seen in training reaches R-GCN as the *unknown* card node, so the model has to rely on
the other five entities. The table splits the test payments by that, and by whether the device is known at all.
""")

code("""
nodes_test = graph.nodes("test")
j_card, j_dev = rf.ENTITY_COLS.index("card1"), rf.ENTITY_COLS.index("DeviceInfo")
groups = {"card seen in training": nodes_test[:, j_card] != graph.unknown[j_card],
          "card unseen in training": nodes_test[:, j_card] == graph.unknown[j_card],
          "device info present": nodes_test[:, j_dev] != graph.missing[j_dev],
          "device info missing": nodes_test[:, j_dev] == graph.missing[j_dev]}
rows = []
for label, mask in groups.items():
    row = {"test payments": label, "share (%)": round(100 * mask.mean(), 1), "fraud rate (%)": round(100 * y_test[mask].mean(), 2)}
    for title, prefix in (("R-GCN", "rgcn_seed"), ("shuffled", "rgcn_shuf_seed"), ("no message passing", "noprop_seed")):
        keys = [k for k in RUNS if k.startswith(prefix)]
        row[f"ROC-AUC {title}"] = np.mean([rf.roc_pr(y_test[mask], RUNS[k]["scores"]["test"][mask])[0] for k in keys])
        row[f"PR-AUC {title}"] = np.mean([rf.roc_pr(y_test[mask], RUNS[k]["scores"]["test"][mask])[1] for k in keys])
    rows.append(row)
SEGMENTS = pd.DataFrame(rows)
SEGMENTS.to_csv(RESULTS / "segments.csv", index=False)
display(SEGMENTS.round(3))
""")

md("""
## 10. Reference numbers from the team plan

Test-set scores quoted on page 1 of the plan. R-GCN scores a payment from entities only, so the closest reference is
*GraphSAGE alone*. The reference paper's models use all payment columns.
""")

code("""
mean = lambda prefix, m: np.mean([RUNS[k]["row"][m] for k in RUNS if k.startswith(prefix)])
reference = pd.DataFrame([
    ("GraphSAGE alone, entities only (1 seed, plan)", 0.741, 0.138),
    ("THGNN, with V (3 seeds, plan)", 0.843, 0.392),
    ("TabNet, all columns (1 seed, plan)", 0.852, 0.395),
    ("Paper: TabNet-GraphSAGE hybrid", 0.791, 0.206),
    ("Paper: LightGBM", 0.834, 0.394),
    ("R-GCN, real graph (this notebook, mean of 3 seeds)", mean("rgcn_seed", "test_roc_auc"), mean("rgcn_seed", "test_pr_auc")),
], columns=["model", "test ROC-AUC", "test PR-AUC"])
display(reference.round(3))
""")

md("""
## 11. Observations

The numbers below are computed from the runs above; the written discussion is in `docs/observations.md`.
""")

code("""
real = [k for k in RUNS if k.startswith("rgcn_seed")]
shuf = [k for k in RUNS if k.startswith("rgcn_shuf")]
nop = [k for k in RUNS if k.startswith("noprop")]
v = lambda keys, m: np.array([RUNS[k]["row"][m] for k in keys])
print("graph used:", "official" if OFFICIAL else "DEVELOPMENT build (preliminary)")
print(f"real graph    : test ROC-AUC {v(real,'test_roc_auc').mean():.4f}, PR-AUC {v(real,'test_pr_auc').mean():.4f}  (seed range ROC {np.ptp(v(real,'test_roc_auc')):.4f}, PR {np.ptp(v(real,'test_pr_auc')):.4f})")
print(f"shuffled graph: test ROC-AUC {v(shuf,'test_roc_auc').mean():.4f}, PR-AUC {v(shuf,'test_pr_auc').mean():.4f}")
print(f"no propagation: test ROC-AUC {v(nop,'test_roc_auc').mean():.4f}, PR-AUC {v(nop,'test_pr_auc').mean():.4f}")
for label in ("real - shuffled graph", "R-GCN - no message passing"):
    part = BOOT[BOOT["comparison"] == label]
    print(f"{label}: ROC-AUC diff {part['ROC-AUC diff'].mean():+.4f} (per seed {', '.join(f'{x:+.4f}' for x in part['ROC-AUC diff'])}); "
          f"PR-AUC diff {part['PR-AUC diff'].mean():+.4f} (per seed {', '.join(f'{x:+.4f}' for x in part['PR-AUC diff'])})")
print("link types with the largest effect on validation ROC-AUC:", ", ".join(LINKS['link type'].head(3)))
json.dump({"graph": "official" if OFFICIAL else "development", "summary": SUMMARY.to_dict(), "chosen": CFG.__dict__},
          open(RESULTS / "summary.json", "w"), indent=2, default=str)
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}}
out = ROOT / "notebooks" / "rgcn_fraud.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, len(cells), "cells")
