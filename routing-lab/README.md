# moe-routing-lab

Characterize MoE expert-routing dynamics on real open models, train a cheap
look-ahead expert predictor, and quantify — via trace-driven cache simulation —
how much a prefetch-aware expert cache would beat LRU/LFU for memory-constrained
(edge/offloaded) MoE inference.

This is the low-cost empirical foundation for a "prefetch-aware MoE serving"
paper: no model training, one consumer GPU (or a 32 GB-RAM CPU box) is enough.

## Hypotheses

- **H1 — Stickiness.** Expert activations are temporally correlated:
  P(expert active at t+1 | active at t) is well above the independence
  baseline, and decays slowly with lag.
- **H2 — Small working sets.** Per domain, a minority of experts covers
  ≥95% of activations, so a partial cache is viable.
- **H3 — Predictability.** A linear probe on the hidden state at token t
  predicts the expert set at t+lag substantially better than persistence
  (repeat current experts) and frequency baselines — i.e., prefetch is
  learnable, not just "keep what you just used".
- **H4 — Cache payoff.** A predictor-driven prefetch policy closes a
  meaningful fraction of the gap between LRU and the Belady oracle at
  realistic capacities (20–50% of experts resident).

If H1/H2 hold strongly and H3 only weakly, that is *also* a finding
(persistence prefetch is near-optimal → simpler systems win).

## Hardware tiers

| Setup | Model to use | Notes |
|---|---|---|
| CPU only, 32 GB RAM | `allenai/OLMoE-1B-7B-0125` (bf16 ≈ 14 GB) | slow but fine for 50k tokens/domain |
| 1 GPU, 16–24 GB | OLMoE bf16, or `Qwen/Qwen1.5-MoE-A2.7B` | comfortable |
| Mac 32 GB+ | OLMoE via transformers (mps) | set `--device_map mps` |
| Bigger GPU | `deepseek-ai/DeepSeek-V2-Lite` (needs `--trust_remote_code`, verify router_logits output) | optional replication |

OLMoE: 64 experts, top-8, all layers MoE, fully open (weights, data, logs) —
the ideal instrument for this study.

## Run order

```bash
pip install -r requirements.txt

# 1. corpora (~100k tokens per domain; or drop your own .txt in data/<domain>/)
python 00_get_data.py --chars 400000

# 2. trace routing (the only step needing the model; hours on CPU, ~minutes on GPU)
python 01_log_routing.py --model allenai/OLMoE-1B-7B-0125 \
    --max_tokens 50000 --save_hidden

# 3. H1/H2: stickiness, working sets, domain specialization
python 02_stickiness.py

# 4. H3: look-ahead predictor vs baselines (per domain)
python 03_probe.py --trace traces/code.npz --lags 1 2 4 8

# 5. H4: cache simulation (per domain; add --layer all for whole-model units)
python 04_cache_sim.py --trace traces/code.npz \
    --predictions results/predictions_code.npz
```

Everything after step 2 is pure NumPy — you can copy `traces/` to a laptop
and iterate on analysis there.

## What each script measures

- **02_stickiness.py** → `results/stickiness.csv` (reuse prob & Jaccard vs lag,
  per layer/domain, with independence-chance column), `working_set.csv`
  (#experts for 90/95/99% coverage), `domain_similarity.csv` + heatmap.
- **03_probe.py** → recall@m of {probe, persistence, frequency} at lags
  1/2/4/8; saves per-token predicted expert sets for the simulator.
- **04_cache_sim.py** → demand-miss/token vs capacity for
  LRU / LFU / static / Belady-oracle / probe-prefetch, plus estimated
  tokens/sec from a parameterized latency model
  (`--expert_mb`, `--ssd_gbps`, `--compute_ms`).

## Self-test (no model needed)

`python _make_synthetic.py` writes two synthetic sticky traces; then run
steps 3–5 above on them to verify your install. Expected on synthetic data:
reuse ≈0.78 vs chance ≈0.26; probe beats persistence at lag 1; Belady < LRU
< LFU/static on misses. Note a real effect the simulator already exposes:
at small capacities, prefetching *pollutes* the cache (probe > LRU misses at
20% capacity, < LRU at 40%+). Capacity-aware prefetch throttling is an easy
novel contribution on top of this.

## Interpreting results → paper

Minimum publishable story (workshop @ MLSys/EuroMLSys/HotEdge-style):
1. First systematic characterization of routing temporal locality across
   domains on a fully open MoE (H1/H2 tables + figures).
2. Learnable look-ahead: probe beats persistence by X points at lag L (H3).
3. Trace-driven evidence that prediction-driven prefetch recovers Y% of
   oracle headroom at Z% capacity (H4) → motivates a real system.

Stretch (full paper): implement the winning policy inside llama.cpp's expert
offloading path (`--n-cpu-moe` / mmap experts) or a Python offloader, and
report wall-clock tokens/sec on one consumer device.

## Related work to position against (read before writing)

EdgeMoE (arXiv:2308.14352) · PowerInfer-2 (2406.06282) · HOBBIT (2411.01433)
· MoE-Infinity (2401.14361) · AdapMoE (2408.10284) · Pre-gated MoE
(2308.12066) · SwapMoE (2308.15030) · DALI (2602.03495) · ReMoE (2605.27081).
Gap you occupy: they engineer caches/prefetchers; none provides a
domain-stratified *measurement study* of routing dynamics or asks whether
look-ahead is learnable with a probe — and none ties predictor quality to
oracle-gap closure.

## Extending to ideas #4 and #8

The simulator's cache-and-predict skeleton is reusable:
- **#4 (knowledge-on-flash):** replace "experts" with kNN-LM datastore shards;
  same miss-rate-vs-capacity analysis.
- **#8 (capability router):** replace "experts" with LoRA adapters and tokens
  with queries; the Static/LRU/predictor comparison becomes an
  adapter-eviction study.

## Bridging to Colibri (github.com/JustVugg/colibri)

`05_export_coli_usage.py` converts a trace from step 2 into Colibri's own
`.coli_usage` priming format (`PIN=<file>`), verified line-for-line against
their actual source (`c/route_trace.h`): format version, the FNV-1a32
engine-id hash (checked against their documented example), and which engines
currently implement the PIN= read path at all (grepped this checkout: full
support in `colibri.c`/GLM-5.2, partial in `inkling.c`, none yet in
`olmoe.c`/`kimi_k3.c`'s read side).

**Read this before using it:** expert identities are model-specific. A trace
is only a valid priming file for the *same* model it was captured on — an
OLMoE trace cannot prime GLM-5.2's cache. Since `olmoe.c` doesn't implement
`PIN=` yet, this script currently produces a correctly-formatted file with no
engine to consume it; it's ready for the day that lands (a small, precedented
patch — `olmoe.c` needs the same five `rt_*` calls `colibri.c` already makes,
per `docs/routing-telemetry.md`'s "Using it from a new engine" section). If
you trace GLM-5.2 itself instead (heavy — needs the real checkpoint), the
output is usable today via `PIN=<file> PIN_GB=<n> ./coli serve`.

```bash
python 05_export_coli_usage.py --traces traces/code.npz traces/math.npz \
    --engine glm_moe_dsa --out coding.coli_usage --min-count 2
```

## Repo layout

```
00_get_data.py           fetch small public domain corpora
01_log_routing.py        run MoE model, log expert choices (+hidden states)
02_stickiness.py         H1/H2 analysis
03_probe.py              H3 look-ahead predictor (NumPy Adam, no torch needed)
04_cache_sim.py          H4 cache policies + latency model
05_export_coli_usage.py  bridge to Colibri's real PIN=<file> mechanism
data/ traces/ results/
```
