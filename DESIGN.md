# sticky-moe: Can MoE routers be trained for cache locality at no quality cost?

**Pre-registration — written and locked 2026-08-05, before any training run.**
Results will be reported against these criteria whether they confirm or refute.

## Motivation (measured, not assumed)

Serving massive MoE models from disk on small machines is bandwidth-bound by
*bytes of expert weights touched per token*. On 2026-08-05 we measured
Qwen3-30B-A3B routing with a zero-surgery GGUF tracer (llama-moe-trace, 32k
tokens, 4 domains): temporal locality P(reuse, lag-1) = 0.444 vs 0.223 chance
(2.0x), 95% working set = 52.5% of experts, LRU hit 65.9% at a 13.4% cache
budget, Belady oracle 79.1%. Code-domain experts near-orthogonal to other
domains (similarity 0.11–0.16). All of that locality is *accidental* — no
production router is trained for it. Hypothesis: training for it directly
buys much higher cacheability at negligible quality cost, which converts
directly into disk-streamed serving speed on small machines.

## Research questions

- **RQ1 (locality loss):** Does adding a temporal-consistency term to router
  training reduce expert-cache miss rate at equal language-modeling quality?
- **RQ2 (domain partitioning):** Does domain-structured routing enable
  static placement (pin one domain slice, serve at full speed) at small
  quality cost?

## Arms

All arms: identical model, data, token order, optimizer, schedule, steps.
Only router auxiliary losses differ.

- **A — baseline:** standard Switch-style load-balance loss (weight α=0.01).
- **B — locality:** A + λ·L_loc, L_loc = mean over adjacent token pairs of
  (1 − ⟨p_t, p_{t−1}⟩), p = softmax router distribution, per MoE layer.
  λ sweep: {0.01, 0.05, 0.2} short runs, best λ gets full run.
- **C — domain-partitioned:** A + μ·L_dom. Each domain d gets 4 exclusive
  experts + 4 shared (of E=16); L_dom = probability mass routed outside the
  allowed set for the token's domain. μ = 0.1. Load balance computed within
  the allowed set.
- **D (conditional):** B+C combined — only if both B and C individually pass.

## Model & budget

- Decoder transformer, d_model 384, 8 layers, 6 heads, RoPE, RMSNorm,
  SwiGLU experts: E=16 per layer, top-k=2, expert d_ff 768.
  ~137M total params, ~38M active/token. GPT-2 BPE (50257), tied embeddings.
- Hardware: single RTX 3070 8GB (bf16 autocast, AdamW).
- Token budgets: sanity 50M; λ-triage 100M/arm; main runs 300M/arm.
  Main arms A and best-B get 2 seeds; C gets 1 seed (2nd if it passes).
- Data: 3 domains — prose (WikiText-103), code (permissively-licensed Python
  sample), math (OpenWebMath sample); ≥100M tokens/domain available,
  interleaved document-level with domain tags carried through. Held-out
  per-domain validation of 2M tokens each. SHA256 of every prepared shard
  recorded in data/MANIFEST.

## Metrics (all computed by the existing measurement stack)

Router picks during validation are exported in moe-routing-lab npz format
(experts (L,T,k) + meta), then evaluated by the *same* 02_stickiness and
04_cache_sim code that measured Qwen3 — one instrument, toy-to-frontier.

1. Validation perplexity, overall + per domain.
2. P(reuse, lag-1) vs chance; working set @95%.
3. Cache-sim demand miss/token at capacities {12.5%, 25%, 50%} (=2/4/8 of
   16 experts), policies LRU + static + belady.
4. Modeled disk-streamed tok/s (04's latency model, constants stated).

## Pre-registered success criteria

- **RQ1 pass:** ≥30% relative reduction in LRU miss/token at 25% capacity,
  with validation PPL within +1% of arm A (same tokens, same seed protocol).
- **RQ2 pass:** ≥90% static-pin hit rate within-domain at 50% capacity
  (= one domain slice: 4 exclusive + 4 shared of 16), with overall PPL
  within +2% of arm A.
- **Honest-failure clauses:** if locality loss collapses experts (any expert
  <1% marginal usage across a 10M-token window) that run is reported as a
  failure mode, not tuned away silently. If PPL cost exceeds bounds at all
  λ, RQ1 is reported refuted at this scale.

## Threats to validity (acknowledged up front)

- 137M-scale routers may behave differently from 235B-scale ones; results
  are laws-at-small-scale evidence, not frontier proof.
- Undertrained models (300M tokens ≪ Chinchilla-optimal) — comparisons stay
  fair because budgets are identical across arms; absolute PPL is not the
  claim, deltas are.
- Single hardware/precision configuration (bf16, one GPU).

## Amendments (logged before any arm's results were unblinded)

- **2026-08-06 (before sanity completion):** measured throughput on the RTX
  3070 is 7,751 tok/s (per-expert loop dispatch; a padded-bmm grouped path
  measured 2.8x *slower* under early-training imbalance and was reverted —
  parity test retained). To keep wall-clock feasible, token budgets are
  amended: λ-triage 100M → 50M per run, main arms 300M → 200M per run.
  Identical budgets across all arms, so comparisons remain fair; statistical
  power is somewhat reduced. Success criteria unchanged.

## Provenance & reproducibility

- Every run: fixed seeds recorded in runs/<name>/config.json; git-hashable
  code; data manifest hashes; loss curves + router health logged every 500
  steps to runs/<name>/log.jsonl. No result reported without its config.
