"""
Experiment 1 — Log MoE router decisions per token.

Runs a HuggingFace MoE model over domain corpora (data/<domain>/*.txt) and
records, for every token and every MoE layer, which top-k experts the router
selected. Optionally also saves one layer of hidden states (needed for the
look-ahead predictor in 03_probe.py).

Works with models that support `output_router_logits=True` in forward():
verified pattern: OLMoE, Mixtral, Qwen1.5/2-MoE. Others may need small tweaks.

Usage (defaults are sized for OLMoE-1B-7B, ~14 GB in bf16):
    python 01_log_routing.py --model allenai/OLMoE-1B-7B-0125 \
        --data_dir data --out_dir traces --max_tokens 50000 --save_hidden

Output per domain: traces/<domain>.npz with
    experts : int16 array (n_moe_layers, n_tokens, top_k)
    hidden  : float16 array (n_tokens, d_model)   [only if --save_hidden]
    meta    : JSON string (model id, k, num_experts, layer used for hidden)
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def iter_domains(data_dir):
    """Yield (domain_name, concatenated_text) for each subdirectory of txt files."""
    for path in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(path):
            continue
        texts = []
        for fp in sorted(glob.glob(os.path.join(path, "*.txt"))):
            with open(fp, encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
        if texts:
            yield os.path.basename(path), "\n\n".join(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0125")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="traces")
    ap.add_argument("--max_tokens", type=int, default=50000,
                    help="max tokens to trace per domain")
    ap.add_argument("--chunk_len", type=int, default=1024,
                    help="forward-pass sequence length")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--top_k", type=int, default=None,
                    help="experts per token; default = model config")
    ap.add_argument("--save_hidden", action="store_true",
                    help="also save hidden states (needed for 03_probe.py)")
    ap.add_argument("--hidden_layer", type=int, default=None,
                    help="which hidden_states index to save; default = middle")
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model,
                                        trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=args.device_map,
        trust_remote_code=args.trust_remote_code)
    model.eval()

    k = args.top_k or getattr(model.config, "num_experts_per_tok", None)
    if k is None:
        raise ValueError("Could not infer top_k from config; pass --top_k")
    n_experts = (getattr(model.config, "num_experts", None)
                 or getattr(model.config, "num_local_experts", None)
                 or getattr(model.config, "num_routed_experts", None))
    n_layers_cfg = model.config.num_hidden_layers
    hidden_layer = (args.hidden_layer if args.hidden_layer is not None
                    else n_layers_cfg // 2)
    print(f"top_k={k}  num_experts={n_experts}  layers={n_layers_cfg}  "
          f"hidden_layer={hidden_layer}")

    for domain, text in iter_domains(args.data_dir):
        print(f"\n=== domain: {domain} ===")
        ids = tok(text, return_tensors="pt").input_ids[0][: args.max_tokens]
        print(f"  {len(ids)} tokens")

        expert_chunks = []   # each: (n_moe_layers, T_chunk, k)
        hidden_chunks = []   # each: (T_chunk, d_model)

        for start in range(0, len(ids), args.chunk_len):
            chunk = ids[start: start + args.chunk_len]
            if len(chunk) < 8:
                break
            chunk = chunk.unsqueeze(0).to(model.device)
            with torch.no_grad():
                out = model(chunk,
                            output_router_logits=True,
                            output_hidden_states=args.save_hidden,
                            use_cache=False)

            layer_topk = []
            for lyr in out.router_logits:
                if lyr is None:          # non-MoE layer in hybrid models
                    continue
                logits2d = lyr.reshape(-1, lyr.shape[-1])          # (T, E)
                topk = torch.topk(logits2d, k, dim=-1).indices     # (T, k)
                layer_topk.append(topk.to(torch.int16).cpu().numpy())
            expert_chunks.append(np.stack(layer_topk))             # (L, T, k)

            if args.save_hidden:
                h = out.hidden_states[hidden_layer][0]             # (T, d)
                hidden_chunks.append(h.to(torch.float16).cpu().numpy())

            done = min(start + args.chunk_len, len(ids))
            print(f"  {done}/{len(ids)} tokens", end="\r")

        experts = np.concatenate(expert_chunks, axis=1)            # (L, T, k)
        meta = dict(model=args.model, top_k=int(k),
                    num_experts=int(n_experts) if n_experts else int(experts.max()) + 1,
                    n_moe_layers=int(experts.shape[0]),
                    hidden_layer=int(hidden_layer))
        save = dict(experts=experts, meta=json.dumps(meta))
        if args.save_hidden:
            save["hidden"] = np.concatenate(hidden_chunks, axis=0)
        out_path = os.path.join(args.out_dir, f"{domain}.npz")
        np.savez_compressed(out_path, **save)
        print(f"\n  saved {out_path}  experts{experts.shape}"
              + (f"  hidden{save['hidden'].shape}" if args.save_hidden else ""))


if __name__ == "__main__":
    main()
