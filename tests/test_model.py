"""Unit tests for the sticky-moe model and its three router losses (CPU)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from model import Config, StickyMoE

CFG = Config(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
             n_experts=8, top_k=2, d_expert=96, max_seq=64,
             n_domains=3, n_exclusive=2, n_shared=2)


def make(seed=0):
    torch.manual_seed(seed)
    return StickyMoE(CFG)


def test_forward_shapes_and_losses_finite():
    m = make()
    x = torch.randint(0, 256, (2, 32))
    d = torch.randint(0, 3, (2, 32))
    out = m(x, targets=x, domains=d, collect_topk=True)
    assert out["lm_loss"].isfinite()
    for k in ("balance", "locality", "domain"):
        assert out[k].isfinite() and out[k] >= 0
    assert out["topk"].shape == (2, 2, 32, 2)          # (L, B, T, k)
    assert out["topk"].max() < CFG.n_experts and out["topk"].min() >= 0


def test_losses_have_gradients():
    m = make()
    x = torch.randint(0, 256, (2, 16))
    d = torch.randint(0, 3, (2, 16))
    out = m(x, targets=x, domains=d)
    total = out["lm_loss"] + out["balance"] + out["locality"] + out["domain"]
    total.backward()
    g = m.blocks[0].moe.router.weight.grad
    assert g is not None and g.abs().sum() > 0


class ConstRouter(torch.nn.Module):
    """Always routes (near-one-hot) to a fixed expert, input-independent."""
    def __init__(self, expert, n_experts):
        super().__init__()
        self.expert, self.n_experts = expert, n_experts

    def forward(self, x):
        logits = x.new_full((*x.shape[:-1], self.n_experts), -100.0)
        logits[..., self.expert] = 100.0
        return logits


def _force_expert(m, expert):
    for blk in m.blocks:
        blk.moe.router = ConstRouter(expert, CFG.n_experts)


def test_locality_loss_zero_when_routing_constant():
    """Identical near-one-hot adjacent distributions give 1 - <p,p> ~ 0."""
    m = make()
    _force_expert(m, 0)
    x = torch.randint(0, 256, (1, 32))
    out = m(x)
    assert out["locality"].item() < 1e-3


def test_domain_loss_zero_when_routing_inside_slice():
    m = make()
    _force_expert(m, 0)                                # expert 0 = domain 0 slice
    x = torch.randint(0, 256, (1, 32))
    d = torch.zeros(1, 32, dtype=torch.long)           # all domain 0
    out = m(x, domains=d)
    assert out["domain"].item() < 1e-3
    d1 = torch.ones(1, 32, dtype=torch.long)           # domain 1: expert 0 outside
    out1 = m(x, domains=d1)
    assert out1["domain"].item() > 0.9


def test_domain_mask_layout():
    m = make()
    mask = m.domain_mask
    assert mask.shape == (3, 8)
    assert mask.sum(1).tolist() == [4.0, 4.0, 4.0]      # 2 exclusive + 2 shared
    assert (mask[:, 6:8] == 1).all()                    # shared tail
    assert mask[0, 0] == 1 and mask[1, 0] == 0          # exclusivity


def test_determinism():
    x = torch.randint(0, 256, (1, 16))
    o1 = make(7)(x, targets=x)["lm_loss"].item()
    o2 = make(7)(x, targets=x)["lm_loss"].item()
    assert o1 == o2


def test_grouped_dispatch_matches_loop():
    torch.manual_seed(3)
    m = make(3)
    moe = m.blocks[0].moe.double()
    flat = torch.randn(64, CFG.d_model, dtype=torch.float64)
    logits = moe.router(flat)
    topv, topi = torch.softmax(logits, -1).topk(CFG.top_k, dim=-1)
    gates = (topv / topv.sum(-1, keepdim=True))
    fast = moe._dispatch(flat, topi, gates)
    ref = moe._dispatch_loop(flat, topi, gates)
    assert torch.allclose(fast, ref, atol=1e-10), (fast - ref).abs().max()
