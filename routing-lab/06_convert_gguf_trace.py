"""
Convert a llama-moe-trace binary stream into the lab's trace npz format.

Input records (repeated): int32 layer, int32 n_tokens, int32 k,
then n_tokens*k int32 expert ids -- one record per MoE layer per chunk,
emitted by the eval-callback tracer running a GGUF model unmodified.

Output: traces/<domain>.npz with experts (L, T, k) int16 + meta json,
identical in shape to what 01_log_routing.py produces, so 02/03/04 run
on it without changes.

Usage: python 06_convert_gguf_trace.py --bin trace_code.bin --out traces/code.npz \
           --model-id Qwen3-30B-A3B-Q4_K_M --num-experts 128
"""
import argparse
import json
import struct
from collections import defaultdict

import numpy as np


def read_records(path):
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                break
            layer, n_tokens, k = struct.unpack("<iii", head)
            data = np.frombuffer(f.read(4 * n_tokens * k), dtype="<i4")
            yield layer, data.reshape(n_tokens, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--num-experts", type=int, required=True)
    args = ap.parse_args()

    per_layer = defaultdict(list)
    for layer, ids in read_records(args.bin):
        per_layer[layer].append(ids)

    layers = sorted(per_layer)
    stacked = [np.concatenate(per_layer[l], axis=0) for l in layers]
    lengths = {s.shape[0] for s in stacked}
    assert len(lengths) == 1, f"unequal token counts per layer: {lengths}"
    experts = np.stack(stacked).astype(np.int16)          # (L, T, k)

    k = experts.shape[2]
    meta = {"model": args.model_id, "top_k": k, "k": k,
            "num_experts": args.num_experts,
            "n_moe_layers": len(layers),
            "source": "llama-moe-trace eval-callback (GGUF, weights untouched)"}
    np.savez_compressed(args.out, experts=experts, meta=json.dumps(meta))
    print(f"{args.out}: experts {experts.shape}, num_experts={args.num_experts}")


if __name__ == "__main__":
    main()
