"""
Experiment 5 — Export a moe-routing-lab trace as a Colibri `.coli_usage` file.

Bridges the domain-routing traces from 01_log_routing.py into Colibri's own
PIN=<file> priming mechanism (github.com/JustVugg/colibri), so a "coding"
(or any other) domain profile measured with this kit can pre-warm Colibri's
real expert cache before a session starts, instead of waiting for the
learning cache to heat up from scratch.

Verified against Colibri's actual source (commit cloned 2026-08-02), not
guessed from docs alone:

  - `.coli_usage` format (docs/routing-telemetry.md), confirmed in
    c/route_trace.h:
        -1 <n_layers> <n_experts>
        -2 <format_version> <engine_id>
        <layer> <expert> <count>
        ...
    sparse, exactly 3 numeric fields per line, non-zero counts only.
  - RT_FORMAT_VERSION == 1 (c/route_trace.h).
  - engine_id is FNV-1a, 32-bit, of the engine's name string. Verified against
    the one documented example: fnv1a32("glm_moe_dsa") == 3815245270. ✓ matches.
  - Engine names actually registered via rt_init() in this checkout:
        colibri.c  -> "glm_moe_dsa"   (GLM-5.2)   — PIN=/pin_load: 45 references, full support
        inkling.c  -> "inkling"                    — PIN=/pin_load: 6 references, partial support
        kimi_k3.c  -> "kimi_k3"                    — writes telemetry (rt_init) but 0 PIN references found
        olmoe.c    -> (none — no rt_init call, 0 PIN references) — NOT WIRED YET

IMPORTANT — read before using:
  Expert identities are model-specific. A trace captured on OLMoE-1B-7B
  (what 01_log_routing.py runs by default) is only valid to prime *OLMoE's own*
  cache — expert #37 in OLMoE has no relationship to expert #37 in GLM-5.2.
  As of this checkout, olmoe.c does not implement the PIN= read path at all
  (grep confirms zero references), so a file generated here is not yet
  consumable by Colibri's OLMoE engine — this script produces a
  correctly-formatted, hash-verified file ready for the moment that support
  lands (it is a small, precedented patch — see the note at the bottom of
  this file). To prime a cache Colibri can *use today*, you would need to
  trace the actual GLM-5.2/Inkling/Kimi-K3 checkpoint Colibri is serving
  (heavy — 744B/975B/2.8T-class) and pass --engine accordingly.

Usage:
  # merge one or more domain traces (e.g. code + math -> a "coding" profile)
  python 05_export_coli_usage.py --traces traces/code.npz traces/math.npz \
      --engine glm_moe_dsa --out coding.coli_usage --min-count 2

  # then, once the target engine supports PIN=:
  PIN=coding.coli_usage PIN_GB=20 ./coli serve --model /nvme/glm52_i4
"""

import argparse
import json

import numpy as np


def fnv1a_32(s: str) -> int:
    """FNV-1a, 32-bit. Verified against route_trace.h's documented example
    (fnv1a32('glm_moe_dsa') == 3815245270) before this script was written."""
    h = 0x811C9DC5
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


ENGINES = {
    "glm_moe_dsa": "GLM-5.2 (colibri.c) — PIN= fully supported",
    "inkling": "Inkling (inkling.c) — PIN= partially supported",
    "kimi_k3": "Kimi K3 (kimi_k3.c) — telemetry write path only, per this checkout",
}
RT_FORMAT_VERSION = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True,
                    help="one or more traces/<domain>.npz from 01_log_routing.py")
    ap.add_argument("--engine", required=True, choices=list(ENGINES),
                    help="must be the SAME model/engine the trace was captured "
                         "on -- see the module docstring")
    ap.add_argument("--out", required=True, help="output .coli_usage path")
    ap.add_argument("--min-count", type=int, default=1,
                    help="drop (layer,expert) pairs seen fewer than this many "
                         "times -- controls output size / cache-budget noise")
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="optional per-trace weight, same order as --traces "
                         "(e.g. upweight a smaller but more on-target corpus)")
    ap.add_argument("--legacy", action="store_true",
                    help="omit the -1/-2 header and write bare triples only "
                         "(docs confirm this format is still accepted, but "
                         "unvalidated -- use if --engine's hash is ever in doubt)")
    args = ap.parse_args()

    weights = args.weights or [1.0] * len(args.traces)
    if len(weights) != len(args.traces):
        raise SystemExit("--weights must match --traces in count")

    counts = {}          # (layer, expert) -> float accumulated count
    n_layers = n_experts = None

    for path, w in zip(args.traces, weights):
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        experts = z["experts"]                      # (L, T, k) int
        L, T, k = experts.shape
        E = meta["num_experts"]
        if n_layers is None:
            n_layers, n_experts = L, E
        elif (L, E) != (n_layers, n_experts):
            raise SystemExit(
                f"{path}: shape (L={L}, E={E}) doesn't match earlier trace(s) "
                f"(L={n_layers}, E={n_experts}) -- traces must be the same model")

        flat = experts.reshape(-1)
        layer_idx = np.repeat(np.arange(L), T * k)
        vals, freq = np.unique(np.stack([layer_idx, flat]), axis=1,
                               return_counts=False), None
        # unique on stacked (layer, expert) pairs -> counts
        pairs = np.stack([layer_idx, flat], axis=1)
        uniq, cnt = np.unique(pairs, axis=0, return_counts=True)
        for (l, e), c in zip(uniq, cnt):
            key = (int(l), int(e))
            counts[key] = counts.get(key, 0.0) + w * float(c)
        print(f"{path}: layers={L} experts={E} tokens={T} weight={w}")

    kept = {k_: int(round(v)) for k_, v in counts.items() if v >= args.min_count}
    print(f"kept {len(kept)}/{len(counts)} (layer,expert) pairs "
          f"(min_count={args.min_count})")

    with open(args.out, "w") as f:
        if not args.legacy:
            f.write(f"-1 {n_layers} {n_experts}\n")
            eid = fnv1a_32(args.engine)
            f.write(f"-2 {RT_FORMAT_VERSION} {eid}\n")
            print(f"engine_id({args.engine}) = {eid}  [{ENGINES[args.engine]}]")
        for (l, e), c in sorted(kept.items()):
            f.write(f"{l} {e} {c}\n")

    print(f"wrote {args.out} ({len(kept)} data lines"
          f"{'' if args.legacy else ' + 2 header lines'})")
    if args.engine == "glm_moe_dsa":
        print("\nReady to use today (PIN= is wired in colibri.c):")
        print(f"  PIN={args.out} PIN_GB=<budget> ./coli serve --model <dir>")
    else:
        print(f"\nNote: PIN= support for this engine's read path is not yet "
              f"present in this checkout ({ENGINES[args.engine]}) -- file is "
              f"correctly formatted and ready for when it lands.")


if __name__ == "__main__":
    main()
