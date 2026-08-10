"""
Experiment 2 — Temporal stickiness & domain specialization of expert routing.

Reads traces/<domain>.npz produced by 01_log_routing.py and computes:

  H1 (stickiness):  P(expert active at t+lag | active at t) and Jaccard
                    overlap of active-expert sets across lags, per layer.
  H2 (working set): how many experts cover 90/95/99% of activations,
                    per layer and per domain.
  H3 (domain specialization): cosine similarity between per-domain expert
                    frequency vectors — low off-diagonal = specialized experts.

Outputs: results/stickiness.csv, results/working_set.csv,
         results/domain_similarity.csv, plots in results/*.png

Usage: python 02_stickiness.py --traces traces --out results
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LAGS = [1, 2, 4, 8, 16, 32]


def load_trace(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    return z["experts"], meta          # (L, T, k), dict


def membership(experts, n_experts):
    """(L, T, k) int -> (L, T, E) bool one-hot of active experts."""
    L, T, k = experts.shape
    m = np.zeros((L, T, n_experts), dtype=bool)
    l_idx = np.arange(L)[:, None, None]
    t_idx = np.arange(T)[None, :, None]
    m[l_idx, t_idx, experts] = True
    return m


def stickiness_stats(m):
    """Per-layer reuse probability and Jaccard at each lag."""
    L, T, E = m.shape
    rows = []
    k_per_tok = m[:, 0].sum(-1).astype(float)          # (L,)
    for lag in LAGS:
        if lag >= T:
            continue
        a, b = m[:, :-lag], m[:, lag:]
        inter = (a & b).sum(-1).astype(float)          # (L, T-lag)
        union = (a | b).sum(-1).astype(float)
        for l in range(L):
            rows.append(dict(layer=l, lag=lag,
                             reuse_prob=float(inter[l].mean() / k_per_tok[l]),
                             jaccard=float((inter[l] / union[l]).mean())))
    return pd.DataFrame(rows)


def chance_reuse_explicit(m):
    """P(e in A_{t+lag} | e in A_t) under independence = sum_e p_e^2 / sum_e p_e."""
    p = m.mean(axis=1)                                  # (L, E)
    return (p ** 2).sum(-1) / p.sum(-1)


def working_set(m, targets=(0.90, 0.95, 0.99)):
    """Experts needed to cover X% of activations, per layer."""
    L, T, E = m.shape
    rows = []
    freq = m.sum(axis=1)                                # (L, E)
    for l in range(L):
        f = np.sort(freq[l])[::-1].astype(float)
        cum = np.cumsum(f) / f.sum()
        for tgt in targets:
            n = int(np.searchsorted(cum, tgt) + 1)
            rows.append(dict(layer=l, coverage=tgt, experts_needed=n,
                             frac_of_experts=n / E))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="traces")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.traces, "*.npz")))
    if not paths:
        raise SystemExit(f"No .npz traces found in {args.traces}/")

    stick_all, ws_all, freq_vectors, domains = [], [], [], []

    for path in paths:
        domain = os.path.splitext(os.path.basename(path))[0]
        experts, meta = load_trace(path)
        E = meta["num_experts"]
        m = membership(experts, E)
        print(f"{domain}: layers={m.shape[0]} tokens={m.shape[1]} experts={E}")

        df = stickiness_stats(m)
        df["domain"] = domain
        df["chance_reuse"] = df["layer"].map(
            dict(enumerate(chance_reuse_explicit(m))))
        stick_all.append(df)

        for row in working_set(m):
            row["domain"] = domain
            ws_all.append(row)

        freq_vectors.append(m.mean(axis=1).ravel())     # (L*E,)
        domains.append(domain)

    stick = pd.concat(stick_all, ignore_index=True)
    stick.to_csv(os.path.join(args.out, "stickiness.csv"), index=False)
    ws = pd.DataFrame(ws_all)
    ws.to_csv(os.path.join(args.out, "working_set.csv"), index=False)

    # cross-domain cosine similarity of expert usage
    F = np.stack(freq_vectors)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    sim = F @ F.T
    pd.DataFrame(sim, index=domains, columns=domains).to_csv(
        os.path.join(args.out, "domain_similarity.csv"))

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(6, 4))
    for domain, g in stick.groupby("domain"):
        gg = g.groupby("lag")["reuse_prob"].mean()
        ax.plot(gg.index, gg.values, marker="o", label=domain)
    ax.axhline(stick["chance_reuse"].mean(), ls="--", c="gray",
               label="chance (indep.)")
    ax.set_xlabel("lag (tokens)"); ax.set_ylabel("P(reuse)")
    ax.set_xscale("log", base=2); ax.legend(fontsize=8)
    ax.set_title("Expert reuse probability vs lag (mean over layers)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "reuse_vs_lag.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    for domain, g in ws[ws.coverage == 0.95].groupby("domain"):
        ax.plot(g["layer"], g["frac_of_experts"], marker=".", label=domain)
    ax.set_xlabel("layer"); ax.set_ylabel("frac of experts for 95% coverage")
    ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title("Working-set size per layer")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "working_set.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(sim, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(domains)), domains, rotation=45, ha="right")
    ax.set_yticks(range(len(domains)), domains)
    fig.colorbar(im); ax.set_title("Domain expert-usage similarity")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "domain_similarity.png"), dpi=150)

    print(f"\nWrote CSVs and plots to {args.out}/")
    print("\nKey numbers:")
    lag1 = stick[stick.lag == 1]
    print(f"  mean P(reuse, lag=1) = {lag1.reuse_prob.mean():.3f}  "
          f"vs chance {lag1.chance_reuse.mean():.3f}")
    w95 = ws[ws.coverage == 0.95]
    print(f"  mean working set @95% = {w95.frac_of_experts.mean():.1%} of experts")


if __name__ == "__main__":
    main()
