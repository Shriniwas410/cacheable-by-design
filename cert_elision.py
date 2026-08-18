"""
(c) REALIZABLE lossless load-elision, resident gate certificate.

The probe gave the ORACLE ceiling (~45%). This measures what a REAL, load-free trigger
achieves: elide the top-2 expert whenever its renormalised gate g2 < eps (a purely
resident decision — no expert load needed to decide). Sweep eps and report, per model:
  - skip_rate: fraction of ALL expert loads elided (top-2 slots are 50% of loads)
  - argmax_agreement: fraction of positions whose next-token argmax is BIT-IDENTICAL to
    the full top-2 run  (100% = truly lossless)
  - ppl_delta: perplexity change vs full (0.00% = lossless)
The headline is: the largest eps at which agreement == 100% and its skip_rate.

  python cert_elision.py --ckpts a-main-s1,b-main-s1 --domain prose --windows 4
"""
import argparse, math, os
import numpy as np, torch
import torch.nn.functional as F
import model as M
from model import Config, StickyMoE, rope_cache


def _moe_forward(self, x):
    B, T, C = x.shape
    probs = F.softmax(self.router(x).float(), dim=-1)
    topv, topi = probs.topk(self.cfg.top_k, dim=-1)
    gates = topv / topv.sum(dim=-1, keepdim=True)
    eps = getattr(self, "_eps", 0.0)
    if eps and self.cfg.top_k >= 2:
        elide = gates[..., 1] < eps                       # per-token: 2nd expert negligible
        self._skipped += int(elide.sum())
        self._slots += int(elide.numel())
        g0 = torch.where(elide, torch.ones_like(gates[..., 0]), gates[..., 0])
        g1 = torch.where(elide, torch.zeros_like(gates[..., 1]), gates[..., 1])
        gates = torch.stack([g0, g1], dim=-1)             # elided tokens -> top-1 only
    out = self._dispatch_loop(x.reshape(-1, C), topi.reshape(-1, self.cfg.top_k),
                              gates.reshape(-1, self.cfg.top_k).to(x.dtype))
    return out.view(B, T, C), probs, topi
M.MoE.forward = _moe_forward


@torch.no_grad()
def logits_of(model, idx):
    cfg = model.cfg
    x = model.embed(idx)
    cos, sin = rope_cache(idx.shape[1], cfg.d_model // cfg.n_heads, idx.device)
    for blk in model.blocks:
        x, _, _ = blk(x, cos, sin)
    return model.lm_head(model.norm_f(x))


def ppl(logits, tgt):
    return math.exp(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                    tgt[:, 1:].reshape(-1)).item())


def run(tag, idx, epss, device):
    cfg = Config(); m = StickyMoE(cfg).to(device)
    m.load_state_dict(torch.load(f"runs/{tag}/ckpt.pt", map_location=device)); m.eval()
    for blk in m.blocks:
        blk.moe._eps = 0.0
    lf = logits_of(m, idx); af = lf.argmax(-1); pf = ppl(lf, idx)
    rows = []
    for eps in epss:
        for blk in m.blocks:
            blk.moe._eps = eps; blk.moe._skipped = 0; blk.moe._slots = 0
        le = logits_of(m, idx)
        skipped = sum(b.moe._skipped for b in m.blocks)
        slots = sum(b.moe._slots for b in m.blocks)
        skip_all = 0.5 * skipped / max(1, slots)          # top-2 slots are 50% of all loads
        agree = (le.argmax(-1) == af).float().mean().item()
        dppl = (ppl(le, idx) / pf - 1) * 100
        rows.append((eps, skip_all, agree, dppl))
        for blk in m.blocks:
            blk.moe._eps = 0.0
    return pf, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="a-main-s1,b-main-s1")
    ap.add_argument("--domain", default="prose")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--epss", default="0.02,0.05,0.1,0.2,0.3")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count()); device = "cpu"
    mm = np.memmap(f"data/val_{args.domain}.bin", dtype=np.uint16)
    n = args.seq * args.windows
    idx = torch.from_numpy(mm[:n].astype(np.int64)).view(args.windows, args.seq).to(device)
    epss = [float(x) for x in args.epss.split(",")]
    print(f"cert-elision: {args.windows}x{args.seq}={n} {args.domain} tokens\n")
    for tag in args.ckpts.split(","):
        pf, rows = run(tag, idx, epss, device)
        print(f"[{tag}]  full PPL={pf:.3f}")
        print(f"   {'eps':>5} {'skip%(all loads)':>16} {'argmax agree':>13} {'dPPL%':>8}  lossless?")
        for eps, sk, ag, dp in rows:
            tag_l = "LOSSLESS" if ag == 1.0 else ("near" if ag > 0.999 else "lossy")
            print(f"   {eps:>5} {sk*100:>15.1f}% {ag*100:>12.2f}% {dp:>+7.2f}  {tag_l}")
        print()


if __name__ == "__main__":
    main()
