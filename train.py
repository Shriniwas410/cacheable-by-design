"""
Trainer for the pre-registered arms (DESIGN.md).

  python train.py --arm A --tokens 50e6 --run-name a-sanity --seed 1
  python train.py --arm B --lambda-loc 0.05 --tokens 100e6 --run-name b-l05 --seed 1
  python train.py --arm C --tokens 300e6 --run-name c-main --seed 1
  python train.py --eval-only --run-name a-sanity   # val PPL + trace export

Data layout (from data.py): data/train.bin, data/train.dom (uint16 tokens,
uint8 domain ids, same length), data/val_<domain>.bin per domain.
Logs: runs/<name>/log.jsonl every 500 steps; config.json; ckpt.pt at end.
Trace export: runs/<name>/trace_<domain>.npz in moe-routing-lab format.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from model import Config, StickyMoE

DOMAINS = ["prose", "code", "math"]


def get_batch(tok_mm, dom_mm, bsz, seq, device, rng):
    ix = rng.integers(0, len(tok_mm) - seq - 1, size=bsz)
    x = torch.stack([torch.from_numpy(tok_mm[i:i + seq].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(tok_mm[i + 1:i + seq + 1].astype(np.int64)) for i in ix])
    d = torch.stack([torch.from_numpy(dom_mm[i:i + seq].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device), d.to(device)


@torch.no_grad()
def evaluate(model, data_dir, device, out_dir, seq=1024, val_tokens=500_000):
    """Per-domain val loss + router trace export (npz, lab format)."""
    model.eval()
    results = {}
    for di, dom in enumerate(DOMAINS):
        mm = np.memmap(os.path.join(data_dir, f"val_{dom}.bin"), dtype=np.uint16)
        n = min(val_tokens, len(mm) - 1)
        losses, traces = [], []
        for s in range(0, n - seq, seq):
            x = torch.from_numpy(mm[s:s + seq].astype(np.int64))[None].to(device)
            y = torch.from_numpy(mm[s + 1:s + seq + 1].astype(np.int64))[None].to(device)
            d = torch.full_like(x, di)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x, targets=y, domains=d, collect_topk=True)
            losses.append(out["lm_loss"].item())
            traces.append(out["topk"][:, 0].to(torch.int16).cpu().numpy())  # (L,T,k)
        experts = np.concatenate(traces, axis=1)
        meta = {"model": "sticky-moe", "top_k": experts.shape[2],
                "k": experts.shape[2], "num_experts": model.cfg.n_experts,
                "n_moe_layers": experts.shape[0], "source": "sticky-moe eval"}
        np.savez_compressed(os.path.join(out_dir, f"trace_{dom}.npz"),
                            experts=experts, meta=json.dumps(meta))
        results[dom] = {"val_loss": float(np.mean(losses)),
                        "val_ppl": float(math.exp(np.mean(losses)))}
    model.train()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C", "D"], default="A")
    ap.add_argument("--lambda-loc", type=float, default=0.0)
    ap.add_argument("--mu-dom", type=float, default=0.0)
    ap.add_argument("--alpha-balance", type=float, default=0.01)
    ap.add_argument("--tokens", type=float, default=50e6)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--eval-only", action="store_true")
    # scale-study knobs: override model size; 8-bit Adam to fit 8GB
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-experts", type=int, default=16)
    ap.add_argument("--d-expert", type=int, default=768)
    ap.add_argument("--adam8bit", action="store_true",
                    help="use bitsandbytes 8-bit AdamW (needed for >137M on 8GB)")
    args = ap.parse_args()

    # arm -> loss weights (explicit, recorded in config)
    if args.arm == "B" and args.lambda_loc == 0.0:
        args.lambda_loc = 0.05
    if args.arm in ("C", "D") and args.mu_dom == 0.0:
        args.mu_dom = 0.1
    if args.arm == "D" and args.lambda_loc == 0.0:
        args.lambda_loc = 0.05

    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)
    cfg = Config(d_model=args.d_model, n_layers=args.n_layers,
                 n_heads=max(1, args.d_model // 64),
                 n_experts=args.n_experts, d_expert=args.d_expert)
    model = StickyMoE(cfg).to(device)

    ckpt = os.path.join(run_dir, "ckpt.pt")
    if args.eval_only:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        res = evaluate(model, args.data_dir, device, run_dir, seq=args.seq)
        print(json.dumps(res, indent=1))
        return

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args) | {"params": sum(p.numel() for p in model.parameters())},
                  f, indent=1)

    tok_mm = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16)
    dom_mm = np.memmap(os.path.join(args.data_dir, "train.dom"), dtype=np.uint8)
    assert len(tok_mm) == len(dom_mm), "token/domain length mismatch"

    steps = int(args.tokens / (args.bsz * args.seq))
    if args.adam8bit:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr,
                                  weight_decay=0.1, betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                                betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.02)
    rng = np.random.default_rng(args.seed)
    log = open(os.path.join(run_dir, "log.jsonl"), "a")

    t0 = time.time()
    for step in range(steps):
        x, y, d = get_batch(tok_mm, dom_mm, args.bsz, args.seq, device, rng)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x, targets=y, domains=d if args.arm in ("C", "D") else None)
            loss = (out["lm_loss"]
                    + args.alpha_balance * out["balance"]
                    + args.lambda_loc * out["locality"]
                    + args.mu_dom * out["domain"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 500 == 0 or step == steps - 1:
            tps = (step + 1) * args.bsz * args.seq / (time.time() - t0)
            rec = {"step": step, "lm": round(out["lm_loss"].item(), 4),
                   "bal": round(out["balance"].item(), 4),
                   "loc": round(out["locality"].item(), 4),
                   "dom": round(out["domain"].item(), 4),
                   "tok_s": int(tps)}
            print(rec, flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()

    torch.save(model.state_dict(), ckpt)
    res = evaluate(model, args.data_dir, device, run_dir, seq=args.seq)
    with open(os.path.join(run_dir, "final_eval.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
