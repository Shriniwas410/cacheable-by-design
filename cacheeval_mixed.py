"""
NOVEL: domain-primed cache-aware routing on a MIXED multi-domain stream.

The realistic edge case neither StickyMoE (single-domain) nor cache-conditional
experts (single-domain) address: one serving stream whose segments switch domain
(code -> prose -> math ...). The LRU expert cache persists across the switch, so
at every boundary it holds the *previous* domain's experts -> a miss burst.

We interleave 1024-token windows round-robin across the 3 domains into one stream
with a persistent per-layer LRU cache, and compare:
  - plain    : cache-aware routing, no priming (cache carries stale experts across switch)
  - primed   : at each domain boundary, prefetch the domain's experts into the cache
               (arm-C: the domain's exclusive experts from domain_mask; overlapped =
               counted as prime-loads, not demand-miss stalls)
optionally with training-free tolerance tau. Demand miss/token = decode stalls.

  python cacheeval_mixed.py --ckpt runs/c-main-s1/ckpt.pt --cap 0.5 \
      --taus 0,0.3,0.5 --windows 12 --out runs/c-main-s1/mixed.json
"""
import argparse, json, math, os
import numpy as np, torch
import torch.nn.functional as F
from model import Config, StickyMoE, rope_cache
from cacheeval import reroute_token, cfg_for_ckpt

DOMAINS = ["prose", "code", "math"]


@torch.no_grad()
def run_stream(model, windows, tau, cap, prime, seq, device):
    """windows: list of (domain_id, token_np[seq+1]). Returns dict of metrics.
    Persistent per-layer LRU cache across the whole stream."""
    cfg = model.cfg; k, E, L = cfg.top_k, cfg.n_experts, cfg.n_layers
    dmask = model.domain_mask.cpu().numpy()          # (n_domains, E) 1=allowed
    excl = {d: [e for e in range(E) if dmask[d, e] > 0 and
                d * cfg.n_exclusive <= e < (d + 1) * cfg.n_exclusive]
            for d in range(cfg.n_domains)}            # domain -> its exclusive experts
    caches = [[] for _ in range(L)]; csets = [set() for _ in range(L)]
    demand_miss = 0; prime_load = 0; sum_ce = 0.0; n_pos = 0
    prev_dom = None
    for dom, toks in windows:
        idx = torch.from_numpy(toks[:seq].astype(np.int64))[None].to(device)
        tgt = torch.from_numpy(toks[1:seq + 1].astype(np.int64))[None].to(device)
        T = idx.shape[1]
        # ---- domain-prime at a boundary: prefetch domain experts (overlapped) ----
        if prime and dom != prev_dom:
            for li in range(L):
                res, rs = caches[li], csets[li]
                for e in excl[dom]:
                    if e not in rs:
                        prime_load += 1
                        if len(res) >= cap:
                            rs.discard(res.pop(0))
                        res.append(e); rs.add(e)
                    else:                              # refresh to MRU
                        res.remove(e); res.append(e)
        prev_dom = dom
        x = model.embed(idx)
        cos, sin = rope_cache(T, cfg.d_model // cfg.n_heads, device)
        for li, blk in enumerate(model.blocks):
            x = x + blk.attn(blk.norm1(x), cos, sin)
            xn = blk.norm2(x)
            probs = F.softmax(blk.moe.router(xn).float(), -1)[0].cpu().numpy()
            order = np.argsort(-probs, axis=1)
            res, rs = caches[li], csets[li]
            idxs = np.empty((T, k), np.int64); gts = np.empty((T, k), np.float32)
            for t in range(T):
                chosen = reroute_token(probs[t], rs, tau, k, order[t])
                for e in chosen:
                    if e in rs:
                        res.remove(e)
                    else:
                        demand_miss += 1
                        if len(res) >= cap:
                            rs.discard(res.pop(0))
                    res.append(e); rs.add(e)
                idxs[t] = chosen; pv = probs[t][chosen]; gts[t] = pv / pv.sum()
            flat = xn.reshape(-1, cfg.d_model)
            m = blk.moe._dispatch_loop(flat, torch.from_numpy(idxs).to(device),
                                       torch.from_numpy(gts).to(device).to(flat.dtype))
            x = x + m.view(1, T, cfg.d_model)
        logits = model.lm_head(model.norm_f(x))
        sum_ce += F.cross_entropy(logits[0].float(), tgt[0], reduction="sum").item()
        n_pos += T
    return {"tau": tau, "prime": prime, "ppl": round(math.exp(sum_ce / n_pos), 3),
            "demand_miss_per_tok": round(demand_miss / n_pos, 4),
            "prime_load_per_tok": round(prime_load / n_pos, 4),
            "demand_hit": round(1 - demand_miss / n_pos / (L * k), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cap", type=float, default=0.5)
    ap.add_argument("--taus", default="0,0.3,0.5")
    ap.add_argument("--windows", type=int, default=12)   # total windows in the stream
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count()); device = "cpu"
    cfg = cfg_for_ckpt(args.ckpt); model = StickyMoE(cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)); model.eval()
    cap = max(1, int(round(args.cap * cfg.n_experts)))

    mm = {d: np.memmap(os.path.join(args.data_dir, f"val_{d}.bin"), dtype=np.uint16)
          for d in DOMAINS}
    # round-robin 1024-token windows across domains into one stream
    windows = []
    per_dom_ptr = {d: 0 for d in range(3)}
    for w in range(args.windows):
        d = w % 3; s = per_dom_ptr[d]; per_dom_ptr[d] += args.seq
        windows.append((d, np.asarray(mm[DOMAINS[d]][s:s + args.seq + 1])))

    taus = [float(x) for x in args.taus.split(",")]
    res = {"ckpt": args.ckpt, "cap_frac": args.cap, "cap": cap,
           "windows": args.windows, "results": []}
    for tau in taus:
        for prime in (False, True):
            r = run_stream(model, windows, tau, cap, prime, args.seq, device)
            res["results"].append(r); print(json.dumps(r), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
