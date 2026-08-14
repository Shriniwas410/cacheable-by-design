"""
Training-free cache-aware routing (idea from Skliar et al., cache-conditional
experts, TMLR 2025) applied to our already-trained 137M checkpoints.

At eval time only -- NO retraining -- we let the router deviate from its
top-k choice when a *cached* expert is within a relative tolerance tau of the
top expert's probability. Deviating to a resident expert avoids a load (a
cache miss). Sweeping tau traces a perplexity-vs-miss frontier that we compare
against the training-time locality frontier (arm B lambda-sweep).

Cache model is IDENTICAL to analyze.py: per-layer independent LRU cache of
capacity round(cap_frac * n_experts); a miss is a selected expert not resident;
miss/token = total misses summed over layers / tokens.

  python cacheeval.py --ckpt runs/a-main-s1/ckpt.pt --tag baseline \
      --caps 0.25 --taus 0,0.1,0.2,0.35,0.5,0.7 --tokens 12000 \
      --out runs/a-main-s1/cacheaware.json

Runs on CPU by design (the GPU is busy with the scale study; a 137M active
forward is cheap enough at a prototype token budget).
"""
import argparse, json, math, os
import numpy as np
import torch
import torch.nn.functional as F

from model import Config, StickyMoE, rope_cache

DOMAINS = ["prose", "code", "math"]


def reroute_token(p, resident_set, tau, k, order):
    """p: (E,) probs (np). order: experts by prob desc (np). resident_set: set.
    Returns list of k chosen experts, preferring cached experts within
    relative tolerance tau of the current best remaining prob."""
    chosen = []
    remaining = list(order)
    for _ in range(k):
        top = remaining[0]
        thr = (1.0 - tau) * p[top]
        pick = top
        for e in remaining:                 # prob-desc: first cached >= thr wins
            if e in resident_set and p[e] >= thr:
                pick = e
                break
        chosen.append(pick)
        remaining.remove(pick)
    return chosen


@torch.no_grad()
def eval_domain(model, tokens, tau, cap, seq, k, E, L, device):
    """Returns (sum_ce, n_positions, total_misses)."""
    cfg = model.cfg
    sum_ce = 0.0; n_pos = 0; misses = 0
    n = (len(tokens) - 1) // seq
    for w in range(n):
        s = w * seq
        idx = torch.from_numpy(tokens[s:s + seq].astype(np.int64))[None].to(device)
        tgt = torch.from_numpy(tokens[s + 1:s + seq + 1].astype(np.int64))[None].to(device)
        T = idx.shape[1]
        x = model.embed(idx)
        cos, sin = rope_cache(T, cfg.d_model // cfg.n_heads, device)
        caches = [[] for _ in range(L)]     # per-layer LRU (MRU at end)
        for li, blk in enumerate(model.blocks):
            x = x + blk.attn(blk.norm1(x), cos, sin)
            xn = blk.norm2(x)
            probs = F.softmax(blk.moe.router(xn).float(), dim=-1)[0].cpu().numpy()  # (T,E)
            order_all = np.argsort(-probs, axis=1)
            res = caches[li]; res_set = set(res)
            idxs = np.empty((T, k), dtype=np.int64)
            gts = np.empty((T, k), dtype=np.float32)
            for t in range(T):
                chosen = reroute_token(probs[t], res_set, tau, k, order_all[t])
                for e in chosen:            # LRU update + miss count (== analyze.py)
                    if e in res_set:
                        res.remove(e)
                    else:
                        misses += 1
                        if len(res) >= cap:
                            res_set.discard(res.pop(0))
                    res.append(e); res_set.add(e)
                idxs[t] = chosen
                pv = probs[t][chosen]; gts[t] = pv / pv.sum()
            flat = xn.reshape(-1, cfg.d_model)
            ti = torch.from_numpy(idxs).to(device)
            tg = torch.from_numpy(gts).to(device).to(flat.dtype)
            m = blk.moe._dispatch_loop(flat, ti, tg).view(1, T, cfg.d_model)
            x = x + m
        logits = model.lm_head(model.norm_f(x))
        ce = F.cross_entropy(logits[0].float(), tgt[0], reduction="sum")
        sum_ce += ce.item(); n_pos += T
    return sum_ce, n_pos, misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--caps", default="0.25")           # cap fractions
    ap.add_argument("--taus", default="0,0.1,0.2,0.35,0.5,0.7")
    ap.add_argument("--tokens", type=int, default=12000)  # per domain
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    device = "cpu"

    cfg = Config()                                        # 137M default
    model = StickyMoE(cfg).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd); model.eval()
    k, E, L = cfg.top_k, cfg.n_experts, cfg.n_layers

    data = {d: np.memmap(os.path.join(args.data_dir, f"val_{d}.bin"),
                         dtype=np.uint16)[:args.tokens] for d in DOMAINS}
    taus = [float(x) for x in args.taus.split(",")]
    caps = [float(x) for x in args.caps.split(",")]
    result = {"tag": args.tag, "ckpt": args.ckpt, "tokens_per_domain": args.tokens,
              "k": k, "E": E, "L": L, "points": []}
    for cap_frac in caps:
        cap = max(1, int(round(cap_frac * max(E, k))))
        for tau in taus:
            per_dom = {}; tot_ce = 0.0; tot_pos = 0; tot_miss = 0
            for d in DOMAINS:
                ce, npos, miss = eval_domain(model, data[d], tau, cap, args.seq, k, E, L, device)
                ppl = math.exp(ce / npos)
                mpt = miss / npos                          # miss/token (summed over layers)
                per_dom[d] = {"ppl": round(ppl, 3), "miss_per_tok": round(mpt, 4),
                              "hit": round(1 - mpt / (L * k), 4)}
                tot_ce += ce; tot_pos += npos; tot_miss += miss
            avg_ppl = math.exp(tot_ce / tot_pos)
            avg_mpt = tot_miss / tot_pos
            pt = {"cap_frac": cap_frac, "cap": cap, "tau": tau,
                  "avg_ppl": round(avg_ppl, 3), "avg_miss_per_tok": round(avg_mpt, 4),
                  "avg_hit": round(1 - avg_mpt / (L * k), 4), "per_domain": per_dom}
            result["points"].append(pt)
            print(json.dumps({kk: pt[kk] for kk in
                  ("cap", "tau", "avg_ppl", "avg_miss_per_tok", "avg_hit")}), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
