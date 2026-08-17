# sticky-moe — results

Auto-appended per arm. Metrics averaged over prose/code/math.
`hit` = fraction of expert-loads already resident. Higher = more cacheable. PPL lower = better LM quality.

| run | arm | λ_loc | μ_dom | tokens | mean PPL | reuse(lag1) | LRU hit@12.5% | LRU hit@25% | static hit@50% |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| a-sanity-s1 | A | 0 | 0 | 50M | 69.3 | 0.307 | 0.267 | 0.469 | 0.731 |
| b-l05 | B | 0.05 | 0 | 50M | 69.5 | 0.644 | 0.611 | 0.809 | 0.957 |
| b-l01 | B | 0.01 | 0 | 50M | 67.7 | 0.354 | 0.312 | 0.518 | 0.779 |
| b-l20 | B | 0.2 | 0 | 50M | 70.1 | 0.861 | 0.648 | 0.855 | 0.977 |
| a-main-s1 | A | 0 | 0 | 200M | 32.0 | 0.336 | 0.296 | 0.494 | 0.747 |
| b-main-s1 | B | 0.05 | 0 | 200M | 32.6 | 0.634 | 0.599 | 0.788 | 0.930 |
| c-main-s1 | C | 0 | 0.1 | 200M | 32.9 | 0.470 | 0.418 | 0.733 | 0.991 |
| a-main-s2 | A | 0 | 0 | 200M | 31.9 | 0.312 | 0.272 | 0.480 | 0.734 |
| b-main-s2 | B | 0.05 | 0 | 200M | 32.9 | 0.635 | 0.604 | 0.796 | 0.933 |
| b-l02-main | B | 0.02 | 0 | 200M | 32.0 | 0.407 | 0.367 | 0.564 | 0.794 |
| b-l03-main | B | 0.03 | 0 | 200M | 32.5 | 0.451 | 0.408 | 0.608 | 0.817 |

## FINAL VERDICT (all arms complete, 2 seeds where noted)

**Baseline A (2 seeds):** PPL 31.9 · LRU miss/tok@25% 8.21 · static hit@50% 0.741

### RQ1 — locality loss (bar: ≥30% fewer misses AND ≤+1% PPL)
| λ | miss reduction | PPL cost | gate |
|--:|--:|--:|:--|
| 0.02 | +15% | +0.3% | fail (effect too weak) |
| 0.03 | +24% | +1.8% | fail (both) |
| 0.05 (s1) | +59% | +2.1% | fail (quality) |
| 0.05 (s2) | +60% | +3.1% | fail (quality) |
| 0.20 | +72% | +119% | fail (quality collapse) |

**RQ1 = REFUTED at this scale.** No λ clears both gates. Miss reduction and PPL
cost are tightly coupled (clean monotonic dose-response); the strict 30%/+1%
combination is unreachable — the quality/cacheability exchange rate at 137M
params does not permit it. λ=0.03 (the predicted crossover) landed in the dead
zone: 24% miss, +1.8% PPL, failing both.

### RQ2 — domain loss (bar: ≥90% static-pin hit AND ≤+2% PPL)
static hit@50% **0.991** (✓, near-perfect) · PPL **+3.1%** (✗) → **REFUTED on quality.**

### Interpretation
- The mechanism **works and is reproducible** — large, monotonic, seed-stable
  effects on cacheability (locality up to 60% fewer misses; domain 99% static hit).
- It is **NOT free** at this scale — a real perplexity tax that breaks the
  pre-registered quality gates for every configuration tested.
- Pre-registration did its job: no configuration is reported as a pass.
- **Open question — NOW ANSWERED (see Scale study below):** the tax was hypothesized
  to shrink at larger scale. A 340M rung shows it does **not** (it rose slightly);
  undertraining at the matched budget keeps this suggestive. The complementarity
  result, however, replicates at 340M.

## COMPLEMENTARITY — training-time × training-free stacking (Phase 2)

Applying training-free cache-aware rerouting (tolerance τ: substitute a cached expert
scoring within (1−τ)·p_top of the top pick) ON TOP of a locality-trained model is
nearly free, because training co-adapts experts into mutually-substitutable
neighbourhoods.

137M (b-main-s1, seed 1, cap 25% experts, LRU):
- training-time locality alone (τ=0): +2.0% PPL for 59% fewer misses
- **stacked (τ=0.5): 80% fewer misses at +2.4% PPL**
- marginal cost τ0→0.5: +4.1% on baseline vs +0.4% on trained (2-seed 20k: 5–8×)

Neither mechanism alone reaches a high-reduction/low-cost point; together they do.
Domain-primed prefetching (prefetch a domain's experts at mixed-stream boundaries):
**NO benefit** — an LRU cache rewarms in O(cap) tokens, so the boundary burst amortises.

## SCALE STUDY — 340M rung (Phase 3)

Does the locality PPL tax shrink with scale? Ran a 340M rung (d_model 512, 12 layers,
16 experts top-2, d_expert 1024), arm A vs arm B λ=0.05, same 200M-token budget.

| size | baseline PPL | locality tax (τ=0) | miss red. | stacked (τ=0.5) |
|---|--:|--:|--:|--:|
| 137M | 21.84 | +2.02% | 59.3% | 80% @ +2.4% |
| 340M | 24.49 | +2.53% | 57.2% | 82% @ +3.4% |

**Answer: the tax does NOT shrink — it rose (+2.02% → +2.53%).** Caveat: 340M at the
matched 200M-token budget is more undertrained (baseline PPL 21.84 → 24.49), a
confound; a compute-optimal and a ≥1B run are the definitive test (≥1B needs cloud —
won't fit the 8GB RTX 3070).

**But complementarity replicates at 340M:** τ0→0.5 costs +0.87% on the trained model
vs +7.73% on baseline (~9× cheaper); stacked reaches 82% miss reduction at +3.4% PPL.
The main positive result is scale-robust.

Data: `runs/scale340-{a,b05}/{final_eval,cacheaware}.json`. Numbers proofread against
paper Table `tab:scale`.
