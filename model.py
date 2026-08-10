"""
Minimal MoE decoder transformer for router-loss experiments (DESIGN.md).

Everything is deliberately plain PyTorch: the science lives in the three
router auxiliary losses, so the rest stays boring and inspectable.
  - balance : Switch-style load-balance loss (all arms)
  - locality: 1 - <p_t, p_{t-1}> adjacent-token routing consistency (arm B)
  - domain  : probability mass routed outside the token's allowed expert
              slice (arm C); allowed = domain's exclusive experts + shared
The forward returns router top-k indices per layer so evaluation can export
traces in the moe-routing-lab npz format.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 50257
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    n_experts: int = 16
    top_k: int = 2
    d_expert: int = 768
    max_seq: int = 1024
    n_domains: int = 3
    n_exclusive: int = 4   # experts exclusive to each domain (arm C)
    n_shared: int = 4      # experts shared across domains (arm C)


def rope_cache(seq, dim, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device) / dim))
    t = torch.arange(seq, device=device)
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, Dh)
    x1, x2 = x[..., ::2], x[..., 1::2]
    T = x.shape[2]
    c, s = cos[:T], sin[:T]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).reshape(B, T, C))


class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.w_gate = nn.Parameter(
            torch.empty(cfg.n_experts, cfg.d_model, cfg.d_expert))
        self.w_up = nn.Parameter(
            torch.empty(cfg.n_experts, cfg.d_model, cfg.d_expert))
        self.w_down = nn.Parameter(
            torch.empty(cfg.n_experts, cfg.d_expert, cfg.d_model))
        for w in (self.w_gate, self.w_up, self.w_down):
            nn.init.normal_(w, std=0.02)

    def forward(self, x):
        B, T, C = x.shape
        logits = self.router(x)                       # (B, T, E)
        probs = F.softmax(logits.float(), dim=-1)
        topv, topi = probs.topk(self.cfg.top_k, dim=-1)
        gates = topv / topv.sum(-1, keepdim=True)     # renormalize over top-k

        # loop dispatch wins on this GPU at this scale: padded-bmm (_dispatch)
        # measured 2.8x SLOWER under early-training imbalance (cap >> mean)
        out = self._dispatch_loop(x.reshape(-1, C),
                                  topi.reshape(-1, self.cfg.top_k),
                                  gates.reshape(-1, self.cfg.top_k).to(x.dtype))
        return out.view(B, T, C), probs, topi

    def _dispatch(self, flat, flat_i, flat_g):
        """Grouped (padded-bmm) expert dispatch. One bmm chain instead of
        E small matmuls; pad slot is a zero row so padding contributes 0."""
        E, k = self.cfg.n_experts, self.cfg.top_k
        N, C = flat.shape
        assign_e = flat_i.reshape(-1)                       # (N*k,)
        order = assign_e.argsort(stable=True)
        counts = torch.bincount(assign_e, minlength=E)      # (E,)
        cap = int(counts.max())
        # position of each sorted assignment within its expert's segment
        seg_start = torch.cumsum(counts, 0) - counts
        pos_in_seg = torch.arange(N * k, device=flat.device) - seg_start[assign_e[order]]
        # slot table: index N is the zero pad row
        slots = torch.full((E, cap), N, dtype=torch.long, device=flat.device)
        tok_of_assign = order // k
        slots[assign_e[order], pos_in_seg] = tok_of_assign
        padded = torch.cat([flat, flat.new_zeros(1, C)])[slots]      # (E, cap, C)
        a = F.silu(torch.bmm(padded, self.w_gate)) * torch.bmm(padded, self.w_up)
        y = torch.bmm(a, self.w_down)                                # (E, cap, C)
        gate_flat = flat_g.reshape(-1)[order]                        # sorted gates
        y_sel = y[assign_e[order], pos_in_seg] * gate_flat[:, None]
        out = torch.zeros_like(flat)
        out.index_add_(0, tok_of_assign, y_sel)
        return out

    def _dispatch_loop(self, flat, flat_i, flat_g):
        """Reference per-expert loop; kept for the parity unit test."""
        out = torch.zeros_like(flat)
        for e in range(self.cfg.n_experts):
            tok, slot = (flat_i == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            h = flat[tok]
            a = F.silu(h @ self.w_gate[e]) * (h @ self.w_up[e])
            out.index_add_(0, tok, (a @ self.w_down[e]) * flat_g[tok, slot, None])
        return out


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.moe = MoE(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        m, probs, topi = self.moe(self.norm2(x))
        return x + m, probs, topi


class StickyMoE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)     # default N(0,1) blows
        self.lm_head.weight = self.embed.weight          # up tied logits; tied
        self.register_buffer("domain_mask", self._domain_mask(), persistent=False)

    def _domain_mask(self):
        # (n_domains, n_experts) 1 = allowed. Exclusive slices then shared tail.
        cfg = self.cfg
        m = torch.zeros(cfg.n_domains, cfg.n_experts)
        for d in range(cfg.n_domains):
            m[d, d * cfg.n_exclusive:(d + 1) * cfg.n_exclusive] = 1.0
        m[:, cfg.n_domains * cfg.n_exclusive:
             cfg.n_domains * cfg.n_exclusive + cfg.n_shared] = 1.0
        return m

    def forward(self, idx, targets=None, domains=None, collect_topk=False):
        """idx (B,T) int64; domains (B,T) int64 or None; returns dict."""
        cfg = self.cfg
        x = self.embed(idx)
        cos, sin = rope_cache(idx.shape[1], cfg.d_model // cfg.n_heads, idx.device)
        balance = locality = domain_loss = x.new_zeros((), dtype=torch.float32)
        topks = []
        for blk in self.blocks:
            x, probs, topi = blk(x, cos, sin)
            if collect_topk:
                topks.append(topi)
            # Switch load-balance: E * sum_e f_e * P_e
            with torch.no_grad():
                onehot = F.one_hot(topi, cfg.n_experts).sum(2).float()  # (B,T,E)
            f = onehot.mean(dim=(0, 1)) / cfg.top_k
            P = probs.mean(dim=(0, 1))
            balance = balance + cfg.n_experts * (f * P).sum()
            # locality: adjacent-token routing consistency
            locality = locality + (1.0 - (probs[:, 1:] * probs[:, :-1]).sum(-1)).mean()
            if domains is not None:
                allowed = self.domain_mask[domains]                     # (B,T,E)
                domain_loss = domain_loss + (probs * (1.0 - allowed)).sum(-1).mean()
        n = len(self.blocks)
        out = {"balance": balance / n, "locality": locality / n,
               "domain": domain_loss / n}
        x = self.norm_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            out["lm_loss"] = F.cross_entropy(
                logits.view(-1, cfg.vocab_size).float(), targets.reshape(-1))
        if collect_topk:
            out["topk"] = torch.stack(topks)   # (L, B, T, k)
        return out
