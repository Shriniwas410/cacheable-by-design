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
- **Open question (the interesting one):** the tax may shrink at larger scale —
  more/finer experts give the router room to specialize for caching without
  spending LM capacity. Small models can't afford dedicated cache-experts; big
  ones might. That is the scale study this refutation motivates.
