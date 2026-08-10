"""
Self-contained arm analyzer: read a run's exported router traces + val eval,
compute the pre-registered metrics (DESIGN.md) — per-domain PPL, temporal
locality P(reuse, lag-1) vs chance, and expert-cache miss/token under
LRU / static / belady at capacities {12.5, 25, 50}% of experts.

No dependency on moe-routing-lab so the experiment stands alone; the cache
model matches 04_cache_sim.py (demand-miss counting, prefetch overlapped).

  python analyze.py runs/b-main-s1            # print JSON
  python analyze.py runs/b-main-s1 --json out.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

DOMAINS = ["prose", "code", "math"]
CAPS = [0.125, 0.25, 0.5]


def load_trace(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    return z["experts"], meta            # (L, T, k), dict


def reuse_lag1(experts):
    """P(an expert chosen at t is reused at t+1), averaged over layers/tokens,
    vs chance = k / n_experts."""
    L, T, k = experts.shape
    hits = tot = 0
    for l in range(L):
        cur = experts[l]                 # (T, k)
        for t in range(T - 1):
            a = set(cur[t].tolist()); b = set(cur[t + 1].tolist())
            hits += len(a & b); tot += k
    return hits / max(tot, 1)


def miss_per_tok(experts, cap_frac, policy):
    """Demand misses per token summed over layers for one cache policy.
    Cache capacity = round(cap_frac * n_experts) per layer, independent caches."""
    L, T, k = experts.shape
    total_miss = 0.0
    for l in range(L):
        seq = experts[l]                 # (T, k)
        n_exp = int(seq.max()) + 1
        cap = max(1, int(round(cap_frac * max(n_exp, k))))
        if policy == "static":
            vals, counts = np.unique(seq, return_counts=True)
            resident = set(vals[np.argsort(-counts)[:cap]].tolist())
            for t in range(T):
                for e in seq[t]:
                    if e not in resident:
                        total_miss += 1
        elif policy == "lru":
            resident = []                # MRU at end
            for t in range(T):
                for e in seq[t]:
                    e = int(e)
                    if e in resident:
                        resident.remove(e)
                    else:
                        total_miss += 1
                        if len(resident) >= cap:
                            resident.pop(0)
                    resident.append(e)
        elif policy == "belady":
            # next-use index per (t, slot)
            nexus = {}
            positions = {}
            for t in range(T):
                for e in seq[t]:
                    positions.setdefault(int(e), []).append(t)
            resident = set()
            ptr = {e: 0 for e in positions}
            for t in range(T):
                for e in seq[t]:
                    e = int(e)
                    if e in resident:
                        continue
                    total_miss += 1
                    if len(resident) >= cap:
                        # evict resident expert whose next use is farthest
                        def next_use(x):
                            arr = positions[x]
                            i = ptr[x]
                            while i < len(arr) and arr[i] <= t:
                                i += 1
                            return arr[i] if i < len(arr) else math.inf
                        victim = max(resident, key=next_use)
                        resident.discard(victim)
                    resident.add(e)
                for e in seq[t]:
                    ptr[int(e)] += 1
    return total_miss / T


def analyze(run_dir):
    ev_path = os.path.join(run_dir, "final_eval.json")
    ev = json.load(open(ev_path)) if os.path.exists(ev_path) else {}
    out = {"run": os.path.basename(run_dir), "domains": {}}
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        out["config"] = json.load(open(cfg_path))
    for dom in DOMAINS:
        tp = os.path.join(run_dir, f"trace_{dom}.npz")
        if not os.path.exists(tp):
            continue
        experts, meta = load_trace(tp)
        L, _, k = experts.shape
        E = meta["num_experts"]
        loads = L * k                    # expert loads per token (all layers)
        rec = {"val_ppl": ev.get(dom, {}).get("val_ppl"),
               "reuse_lag1": round(reuse_lag1(experts), 4),
               "reuse_chance": round(k / E, 4),
               "miss": {}, "hit": {}}
        for cap in CAPS:
            rec["miss"][f"{cap}"] = {
                pol: round(miss_per_tok(experts, cap, pol), 4)
                for pol in ("lru", "static", "belady")}
            rec["hit"][f"{cap}"] = {
                pol: round(1 - rec["miss"][f"{cap}"][pol] / loads, 4)
                for pol in ("lru", "static", "belady")}
        out["domains"][dom] = rec
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = analyze(args.run_dir)
    txt = json.dumps(res, indent=1)
    print(txt)
    if args.json:
        open(args.json, "w").write(txt)
