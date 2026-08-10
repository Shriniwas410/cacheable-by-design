"""
Experiment 4 — Expert cache simulation (the payoff, no GPU needed).

Replays a routing trace through a simulated expert cache of capacity C and
compares eviction/prefetch policies:

  lru     : evict least-recently-used expert
  lfu     : evict least-frequently-used (cumulative counts)
  static  : pin the top-C most frequent experts from a warmup prefix
  belady  : clairvoyant oracle (evict the expert needed farthest in future)
  probe   : LRU + prefetch the experts predicted by 03_probe.py one lag ahead

Reports demand-miss rate vs capacity and converts it to estimated tokens/sec
via a simple latency model:
    token_time = compute_ms + demand_misses * (expert_mb / ssd_gbps)
(1 GB/s == 1 MB/ms, prefetch loads are assumed overlapped with compute).

Usage:
  python 04_cache_sim.py --trace traces/code.npz --layer auto \
      --predictions results/predictions_code.npz \
      --capacities 0.1 0.2 0.3 0.5 0.7 0.9
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- policies
class BaseCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = set()
        self.clock = 0

    def full(self):
        return len(self.cache) >= self.cap

    def access(self, needed, prefetch=()):
        """Return (#demand_misses, #prefetch_loads) for this token."""
        self.clock += 1
        demand = 0
        for e in needed:
            if e in self.cache:
                self.touch(e, hit=True)
            else:
                demand += 1
                self.insert(e)
        pf = 0
        for e in prefetch:
            if e not in self.cache:
                pf += 1
                self.insert(e)
        return demand, pf

    def touch(self, e, hit):
        pass

    def insert(self, e):
        while self.full():
            self.evict()
        self.cache.add(e)
        self.touch(e, hit=False)

    def evict(self):
        raise NotImplementedError


class LRU(BaseCache):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.last = {}

    def touch(self, e, hit):
        self.last[e] = self.clock

    def evict(self):
        victim = min(self.cache, key=lambda e: self.last.get(e, -1))
        self.cache.discard(victim)
        self.last.pop(victim, None)


class LFU(BaseCache):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.count = {}

    def touch(self, e, hit):
        self.count[e] = self.count.get(e, 0) + 1

    def evict(self):
        victim = min(self.cache, key=lambda e: self.count.get(e, 0))
        self.cache.discard(victim)


class Static(BaseCache):
    """Cache pinned to top-C experts of a warmup prefix; never evicts/inserts."""
    def __init__(self, capacity, pinned):
        super().__init__(capacity)
        self.cache = set(list(pinned)[:capacity])

    def access(self, needed, prefetch=()):
        demand = sum(1 for e in needed if e not in self.cache)
        return demand, 0


class Belady(BaseCache):
    def __init__(self, capacity, trace_positions):
        super().__init__(capacity)
        self.pos = trace_positions          # unit -> sorted np.array of uses
        self.ptr = {u: 0 for u in trace_positions}

    def next_use(self, e):
        uses = self.pos.get(e)
        if uses is None:
            return np.inf
        i = self.ptr[e]
        while i < len(uses) and uses[i] < self.clock:
            i += 1
        self.ptr[e] = i
        return uses[i] if i < len(uses) else np.inf

    def evict(self):
        victim = max(self.cache, key=self.next_use)
        self.cache.discard(victim)


# ---------------------------------------------------------------- helpers
def build_units(experts, layer):
    """Return per-token required unit sets. layer='all' -> units=(layer,expert)."""
    L, T, k = experts.shape
    if layer == "all":
        E = int(experts.max()) + 1
        offs = (np.arange(L) * E)[:, None, None]
        units = (experts + offs).transpose(1, 0, 2).reshape(T, L * k)
        n_units = L * E
    else:
        li = int(layer)
        units = experts[li]                              # (T, k)
        n_units = int(experts.max()) + 1
    return units, n_units


def simulate(units, n_units, capacity, policy, warmup_frac=0.2,
             predictions=None, pred_start=0):
    T = len(units)
    if policy == "static":
        w = units[: int(T * warmup_frac)]
        freq = np.bincount(w.ravel(), minlength=n_units)
        cache = Static(capacity, np.argsort(-freq))
    elif policy == "belady":
        pos = {}
        for t in range(T):
            for e in units[t]:
                pos.setdefault(int(e), []).append(t)
        pos = {e: np.asarray(v) for e, v in pos.items()}
        cache = Belady(capacity, pos)
    elif policy == "lfu":
        cache = LFU(capacity)
    else:                                   # lru & probe share LRU eviction
        cache = LRU(capacity)

    demand = pf = 0
    for t in range(T):
        prefetch = ()
        if policy == "probe" and predictions is not None:
            i = t + 1 - pred_start          # prediction for token t+1
            if 0 <= i < len(predictions):
                prefetch = predictions[i]
        d, p = cache.access(units[t], prefetch)
        demand += d
        pf += p
    return demand / T, pf / T               # per-token averages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--layer", default="auto", help="int, 'auto' (middle) or 'all'")
    ap.add_argument("--capacities", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                    help="fractions of total units")
    ap.add_argument("--policies", nargs="+",
                    default=["lru", "lfu", "static", "belady"])
    ap.add_argument("--predictions", default=None,
                    help="results/predictions_<domain>.npz -> adds 'probe' policy")
    ap.add_argument("--expert_mb", type=float, default=6.0,
                    help="size of one expert's weights on storage (MB)")
    ap.add_argument("--ssd_gbps", type=float, default=3.0)
    ap.add_argument("--compute_ms", type=float, default=15.0,
                    help="per-token compute time if everything is resident")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = np.load(args.trace, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    experts = z["experts"]
    L = experts.shape[0]
    layer = (L // 2 if args.layer == "auto"
             else args.layer if args.layer == "all" else int(args.layer))
    units, n_units = build_units(experts, layer)
    domain = os.path.splitext(os.path.basename(args.trace))[0]
    print(f"{domain}: layer={layer} tokens={len(units)} units={n_units}")

    predictions, pred_start = None, 0
    if args.predictions:
        pz = np.load(args.predictions, allow_pickle=True)
        pmeta = json.loads(str(pz["meta"]))
        predictions, pred_start = pz["pred"], pmeta["start_token"]
        if "probe" not in args.policies:
            args.policies = list(args.policies) + ["probe"]
        print(f"  probe predictions: lag={pmeta['lag']} m={pmeta['pred_m']} "
              f"from token {pred_start}")

    load_ms = args.expert_mb / args.ssd_gbps           # 1 GB/s == 1 MB/ms
    rows = []
    for frac in args.capacities:
        cap = max(1, int(n_units * frac))
        for pol in args.policies:
            miss, pf = simulate(units, n_units, cap, pol,
                                predictions=predictions, pred_start=pred_start)
            tok_ms = args.compute_ms + miss * load_ms
            rows.append(dict(domain=domain, layer=str(layer), policy=pol,
                             cap_frac=frac, capacity=cap,
                             demand_miss_per_tok=miss, prefetch_per_tok=pf,
                             est_tok_per_s=1000.0 / tok_ms))
            print(f"  cap={frac:.0%} {pol:>7}: miss/tok={miss:6.3f} "
                  f"prefetch/tok={pf:5.2f} -> ~{1000.0 / tok_ms:6.1f} tok/s")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, f"cache_{domain}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    for pol, g in df.groupby("policy"):
        ax.plot(g["cap_frac"], g["demand_miss_per_tok"], marker="o", label=pol)
    ax.set_xlabel("cache capacity (fraction of experts)")
    ax.set_ylabel("demand misses per token")
    ax.set_title(f"Expert cache misses — {domain} (layer {layer})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"cache_{domain}.png"), dpi=150)
    print(f"Wrote cache_{domain}.csv / .png to {args.out}/")


if __name__ == "__main__":
    main()
