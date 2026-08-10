"""
Experiment 3 — Look-ahead expert predictability.

Question: given the hidden state at token t, can a cheap probe predict which
experts the router will select at token t+lag? If yes, prefetching experts
from slow storage before they're needed is feasible.

Trains a multi-label linear probe (logistic regression, NumPy Adam — no torch
needed for analysis) per lag, and compares against two baselines:
  * persistence: predict the same experts that fired at t
  * frequency:   always predict the globally most frequent experts

Metric: recall@k — fraction of the true top-k experts covered when the probe
proposes its top-m (default m=k) experts.

Requires traces saved with --save_hidden in 01_log_routing.py.
Saves per-token predictions for the best lag to results/predictions_<domain>.npz
(consumed by 04_cache_sim.py's probe-prefetch policy).

Usage: python 03_probe.py --trace traces/code.npz --layer auto --lags 1 2 4 8
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def train_linear_probe(X, Y, epochs=4, bs=512, lr=1e-3, wd=1e-4, seed=0):
    """Multi-label logistic regression with Adam. X:(N,d) f32, Y:(N,E) {0,1}."""
    rng = np.random.default_rng(seed)
    N, d = X.shape
    E = Y.shape[1]
    W = np.zeros((d, E), dtype=np.float32)
    b = np.zeros(E, dtype=np.float32)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    for ep in range(epochs):
        idx = rng.permutation(N)
        for s in range(0, N, bs):
            batch = idx[s: s + bs]
            xb, yb = X[batch], Y[batch]
            p = sigmoid(xb @ W + b)
            g = (p - yb) / len(batch)                       # BCE grad
            gW = xb.T @ g + wd * W
            gb = g.sum(0)
            step += 1
            for (P, G, M, V) in ((W, gW, mW, vW), (b, gb, mb, vb)):
                M *= beta1; M += (1 - beta1) * G
                V *= beta2; V += (1 - beta2) * G * G
                mh = M / (1 - beta1 ** step)
                vh = V / (1 - beta2 ** step)
                P -= lr * mh / (np.sqrt(vh) + eps)
    return W, b


def recall_at_m(scores, true_sets, m):
    """scores:(N,E) -> take top-m; recall vs true top-k sets:(N,k)."""
    pred = np.argpartition(-scores, m - 1, axis=1)[:, :m]   # (N, m)
    hits = np.zeros(len(scores))
    for i in range(len(scores)):
        hits[i] = len(np.intersect1d(pred[i], true_sets[i],
                                     assume_unique=False))
    return float(hits.mean() / true_sets.shape[1]), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="one traces/<domain>.npz")
    ap.add_argument("--layer", default="auto",
                    help="MoE layer index to predict for; 'auto' = middle")
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--pred_m", type=int, default=None,
                    help="probe proposes top-m experts; default = 2*k")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = np.load(args.trace, allow_pickle=True)
    if "hidden" not in z:
        raise SystemExit("Trace has no hidden states — rerun 01 with --save_hidden")
    meta = json.loads(str(z["meta"]))
    experts = z["experts"]                 # (L, T, k)
    H = z["hidden"].astype(np.float32)     # (T, d)
    L, T, k = experts.shape
    E = meta["num_experts"]
    layer = L // 2 if args.layer == "auto" else int(args.layer)
    m = args.pred_m or 2 * k
    domain = os.path.splitext(os.path.basename(args.trace))[0]
    print(f"{domain}: T={T} d={H.shape[1]} layer={layer} k={k} E={E} m={m}")

    # standardize features
    mu, sd = H.mean(0, keepdims=True), H.std(0, keepdims=True) + 1e-6
    H = (H - mu) / sd

    split = int(T * args.train_frac)       # contiguous split = no leakage
    rows = []
    best = None

    for lag in args.lags:
        Xtr, Xte = H[: split - lag], H[split: T - lag]
        tgt_tr = experts[layer, lag: split]            # (Ntr, k)
        tgt_te = experts[layer, split + lag: T]        # (Nte, k)
        cur_te = experts[layer, split: T - lag]        # persistence source

        Ytr = np.zeros((len(Xtr), E), dtype=np.float32)
        Ytr[np.arange(len(Xtr))[:, None], tgt_tr] = 1.0

        W, b = train_linear_probe(Xtr, Ytr, epochs=args.epochs)
        scores = Xte @ W + b
        probe_rec, pred_sets = recall_at_m(scores, tgt_te, m)

        # persistence baseline: experts at t as prediction for t+lag
        pers_hits = [len(np.intersect1d(cur_te[i], tgt_te[i]))
                     for i in range(len(tgt_te))]
        pers_rec = float(np.mean(pers_hits) / k)

        # frequency baseline: global top-m experts on train split
        freq = np.bincount(experts[layer, :split].ravel(), minlength=E)
        top_freq = np.argsort(-freq)[:m]
        freq_hits = [len(np.intersect1d(top_freq, tgt_te[i]))
                     for i in range(len(tgt_te))]
        freq_rec = float(np.mean(freq_hits) / k)

        print(f"  lag={lag:>2}  probe recall@{m}={probe_rec:.3f}  "
              f"persistence={pers_rec:.3f}  frequency={freq_rec:.3f}")
        rows.append(dict(domain=domain, layer=layer, lag=lag, pred_m=m,
                         probe=probe_rec, persistence=pers_rec,
                         frequency=freq_rec))
        if best is None or lag == args.lags[0]:
            best = (lag, pred_sets, split)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out, f"probe_{domain}.csv")
    df.to_csv(out_csv, index=False)

    lag, pred_sets, split = best
    np.savez_compressed(
        os.path.join(args.out, f"predictions_{domain}.npz"),
        pred=pred_sets.astype(np.int16),   # aligned to tokens split+lag .. T-1
        meta=json.dumps(dict(domain=domain, layer=layer, lag=int(lag),
                             start_token=int(split + lag), pred_m=int(m))))
    print(f"Wrote {out_csv} and predictions_{domain}.npz "
          f"(lag={lag}, aligned from token {split + lag})")


if __name__ == "__main__":
    main()
