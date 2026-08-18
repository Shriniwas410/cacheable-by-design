"""
FEASIBILITY PROBE for lossless expert-load elision.

Question: what fraction of the 2nd (elidable) expert's LOADS are unnecessary for a
BIT-IDENTICAL next-token argmax? That is the ceiling on lossless load-elision with a
perfect (oracle) certificate. If ~0%, the idea is dead; if meaningful, worth a real
(bounded) certificate + a train-for-elision objective.

Method (top_k=2 model): for each layer L, run the model with layer L using ONLY its
top-1 expert (drop the top-2, renormalise gate to 1). Compare the final next-token
argmax to the full top-2 run. argmax preserved => that 2nd-expert load was elidable
at L for that token, losslessly. Also report gate(top2) and logit-margin stats
(how tight a real certificate must be), and compare baseline vs locality-trained.

  python probe_elision.py --ckpts a-main-s1,b-main-s1 --domain prose --windows 2
"""
import argparse, os
import numpy as np, torch
import torch.nn.functional as F
import model as M
from model import Config, StickyMoE, rope_cache

# --- monkeypatch MoE.forward to honour a per-layer k override (_fk) ---
def _moe_forward(self, x):
    B, T, C = x.shape
    logits = self.router(x)
    probs = F.softmax(logits.float(), dim=-1)
    k = getattr(self, "_fk", self.cfg.top_k) or self.cfg.top_k
    topv, topi = probs.topk(k, dim=-1)
    gates = topv / topv.sum(dim=-1, keepdim=True)
    out = self._dispatch_loop(x.reshape(-1, C), topi.reshape(-1, k),
                              gates.reshape(-1, k).to(x.dtype))
    return out.view(B, T, C), probs, topi
M.MoE.forward = _moe_forward


@torch.no_grad()
def logits_of(model, idx):
    cfg = model.cfg
    x = model.embed(idx)
    cos, sin = rope_cache(idx.shape[1], cfg.d_model // cfg.n_heads, idx.device)
    for blk in model.blocks:
        x, _, _ = blk(x, cos, sin)
    return model.lm_head(model.norm_f(x))          # (B,T,V)


@torch.no_grad()
def gate_top2_stats(model, idx):
    """Mean renormalised gate on the 2nd expert, per layer, + frac below 0.05."""
    cfg = model.cfg
    x = model.embed(idx)
    cos, sin = rope_cache(idx.shape[1], cfg.d_model // cfg.n_heads, idx.device)
    g2 = []
    for blk in model.blocks:
        xn = blk.norm2(x + blk.attn(blk.norm1(x), cos, sin))
        p = F.softmax(blk.moe.router(xn).float(), -1)
        tv, _ = p.topk(cfg.top_k, dim=-1)
        gg = (tv / tv.sum(-1, keepdim=True))[..., 1]     # 2nd-expert renorm gate
        g2.append(gg.flatten())
        x, _, _ = blk(x, cos, sin)
    return torch.stack(g2)                                # (L, N)


def run_model(tag, idx, device):
    cfg = Config()
    m = StickyMoE(cfg).to(device)
    m.load_state_dict(torch.load(f"runs/{tag}/ckpt.pt", map_location=device))
    m.eval()
    L = cfg.n_layers
    lf = logits_of(m, idx)
    full_arg = lf.argmax(-1)                              # (B,T)
    s = lf.float().sort(dim=-1, descending=True).values
    margin = (s[..., 0] - s[..., 1])                      # logit top1-top2 gap
    preserved = []
    for li in range(L):
        m.blocks[li].moe._fk = 1
        arg = logits_of(m, idx).argmax(-1)
        m.blocks[li].moe._fk = None
        preserved.append((arg == full_arg).float().mean().item())
    g2 = gate_top2_stats(m, idx)                          # (L,N)
    return {"tag": tag, "preserved_per_layer": [round(p, 3) for p in preserved],
            "mean_preserved": round(float(np.mean(preserved)), 3),
            "elidable_frac_all_loads": round(0.5 * float(np.mean(preserved)), 3),
            "gate_top2_mean": round(g2.mean().item(), 3),
            "gate_top2_frac_below_0.05": round((g2 < 0.05).float().mean().item(), 3),
            "logit_margin_median": round(margin.median().item(), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="a-main-s1,b-main-s1")
    ap.add_argument("--domain", default="prose")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count()); device = "cpu"
    mm = np.memmap(f"data/val_{args.domain}.bin", dtype=np.uint16)
    n = args.seq * args.windows
    idx = torch.from_numpy(mm[:n].astype(np.int64)).view(args.windows, args.seq).to(device)
    print(f"probe: {args.windows}x{args.seq}={n} {args.domain} tokens, top_k=2\n")
    for tag in args.ckpts.split(","):
        r = run_model(tag, idx, device)
        print(f"[{r['tag']}]  mean argmax-preserved (drop 2nd expert) = {r['mean_preserved']:.1%}"
              f"  => elidable {r['elidable_frac_all_loads']:.1%} of ALL expert loads (lossless ceiling)")
        print(f"   per-layer preserved: {r['preserved_per_layer']}")
        print(f"   gate(top2): mean {r['gate_top2_mean']}, frac<0.05 {r['gate_top2_frac_below_0.05']}"
              f" | logit margin median {r['logit_margin_median']}\n")


if __name__ == "__main__":
    main()
