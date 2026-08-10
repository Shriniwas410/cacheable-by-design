"""Generate synthetic traces mimicking 01_log_routing.py output, with
controllable stickiness + domain bias + hidden states predictive of t+1."""
import json, os
import numpy as np

rng = np.random.default_rng(0)
L, T, E, k, d = 4, 6000, 64, 8, 64
os.makedirs("traces", exist_ok=True)

def make(domain, hot_offset, p_stick=0.75):
    hot = (np.arange(16) + hot_offset) % E          # domain-preferred experts
    experts = np.zeros((L, T, k), dtype=np.int16)
    for l in range(L):
        cur = rng.choice(E, size=k, replace=False)
        for t in range(T):
            keep = cur[rng.random(k) < p_stick]
            n_new = k - len(keep)
            pool = np.setdiff1d(np.concatenate([hot, rng.choice(E, 8)]), keep)
            new = rng.choice(pool, size=n_new, replace=False)
            cur = np.concatenate([keep, new])[:k]
            experts[l, t] = np.sort(cur)
    # hidden state at t encodes experts at t+1 (so probe beats persistence)
    emb = rng.standard_normal((E, d)).astype(np.float32)
    hidden = np.zeros((T, d), dtype=np.float16)
    for t in range(T):
        nxt = experts[L // 2, min(t + 1, T - 1)]
        hidden[t] = (emb[nxt].mean(0) + 0.35 * rng.standard_normal(d)).astype(np.float16)
    meta = json.dumps(dict(model="synthetic", top_k=k, num_experts=E,
                           n_moe_layers=L, hidden_layer=L // 2))
    np.savez_compressed(f"traces/{domain}.npz",
                        experts=experts, hidden=hidden, meta=meta)
    print("wrote", domain, experts.shape)

make("synth_code", hot_offset=0)
make("synth_wiki", hot_offset=32)
