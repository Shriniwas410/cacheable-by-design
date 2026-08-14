# Reference discovery — candidate papers (2026-08-14)

Main-thread WebSearch sweep (16 queries; the 4-angle subagent sweep is still blocked on the
session limit). **These are UNVERIFIED search snippets, not citations.** Per repo rule, none
is added until its PDF is downloaded and grounded (`cite-verify`). `CITED` = already in
references.bib. Tier = fit to *our* paper (train-time locality tax + train-free tolerance
rerouting + stacking + substitutability mechanism + domain-priming negative).

⚠️ **Name collisions to avoid:**
- "ReMoE" is TWO papers: **2605.27081** = *Boosting Expert Reuse thru Router Fine-Tuning*
  (the one we want) vs **2412.14711** = *Fully Differentiable MoE with ReLU Routing* (ICLR'25,
  unrelated).
- "ST-MoE": our `zoph2022stmoe` (stable/transferable) ≠ a "spatio-temporal MoE" prefetch paper.

## TIER 1 — near-novelty-threat / must position against
| id / link | one-line | status |
|---|---|---|
| [Oracle-MoE](https://openreview.net/forum?id=wn6WHREK9k) (ICML'25, [mlr v267](https://proceedings.mlr.press/v267/zhou25b.html)) | **Trains** locality-preserving routing in an "oracle space" (attention-derived) to cut inter-token expert-activation variation; GPT-2 **200M/350M/790M/2B** ladder, SOTA speed at fixed memory w/o task-perf loss. | NEW — closest competitor to our thesis **and** our scale study |
| [ReMoE (reuse)](https://arxiv.org/abs/2605.27081) 2605.27081 | Post-hoc router fine-tune: temporal-locality loss + Trust-KL anchor → **+26% expert reuse**, perf held. The "post-hoc FT" method StickyMoE Pareto-dominates. | NEW |

## TIER 2 — strengthen a specific claim
| id / link | one-line | why it fits |
|---|---|---|
| [In-depth Analysis of Caching & Prefetching in MoE Offloading](https://arxiv.org/pdf/2511.05814) 2511.05814 | Systematic study; LRU/LFU each capture only temporal OR long-tail; **Score-Aware (MRS)** replacement using router scores beats LRU. | our miss metric uses plain LRU — shows LRU is beatable; honesty of `analyze.py` |
| [Merge, Then Compress](https://arxiv.org/html/2310.01334) 2310.01334 | SMoE experts are redundant; group by routing/output similarity and merge. | direct evidence for our **substitutable-neighbourhoods** mechanism |
| [DERN: Dropping Experts, Recombining Neurons](https://arxiv.org/abs/2509.10377) 2509.10377 | Retraining-free expert prune+recombine using router stats; >5% over prior at 50% sparsity. | substitutability / redundancy evidence |
| [Cross-Layer Gate expert prediction](https://arxiv.org/html/2502.12224v1) 2502.12224 | Next-layer top-1 expert **96%** predictable from adjacent-layer gate inputs. | the prefetch/predict alternative to routing-for-locality; expert choice is correlated |
| [Spatio-Temporal Expert Prefetching](https://arxiv.org/html/2606.15453v1) 2606.15453 | Cross-layer + cross-token expert correlation (χ² p<0.01) → prefetch. | evidences temporal locality exists to exploit |
| [Two-Stage Domain-Aware Expert Offloading](https://ieeexplore.ieee.org/document/11397596/) (IEEE) / [DAOP](https://arxiv.org/pdf/2501.10375) 2501.10375 | Preload a request's **domain** experts at prefill, then locality-aware decode prefetch. | the positive-claim domain line our **domain-priming negative** contrasts with |
| [Three Phases of Expert Routing](https://arxiv.org/html/2604.04230v1) 2604.04230 | Load balance evolves in training: router loosens balance as experts differentiate. | mechanism: experts **co-adapt**, matches our substitutability story |

## TIER 3 — optional context / umbrella cites
| id / link | one-line |
|---|---|
| [MELINOE](https://arxiv.org/pdf/2602.11192) 2602.11192 | Fine-tuning enables memory-efficient MoE inference (post-hoc axis, w/ ReMoE) |
| [Survey: MoE Inference Optimization](https://arxiv.org/pdf/2412.14219) 2412.14219 (ACM CSUR) | umbrella survey of MoE inference-efficiency techniques |
| [Guiding the Experts: Semantic Priors](https://arxiv.org/abs/2505.18586) 2505.18586 | spatially-aware aux loss aligning expert activation to semantics |
| [SoftMoE differentiable routing](https://arxiv.org/abs/2606.17952) 2606.17952 | soft top-k relaxation, fewer active experts, AR-compatible |

## Also surfaced (NEW, low fit — logged for completeness, not recommended)
Merging/pruning: SHAPE 2606.09886 · Sub-MoE 2506.23266 · PuzzleMoE 2511.04805 · Diversifying-Expert-Knowledge 2407.09590 · LightMoE 2603.12645 · Prune+Distill-to-Dense 2605.28207 · Compressed-Experts 2503.00634 · Router-Calibration 2603.02217 ·
Caching/systems: SliceMoE 2512.12990 · Diff-MoE (SC'25) · HybriMoE 2504.05897 · FineMoE 2502.05370 · DALI 2602.03495 · CommitMoE (AAAI) · DuoServe-MoE 2509.07379 · Patterns-behind-Chaos 2510.05497 · CPU-GPU-collab 2512.16473 · Edge-latency-placement 2508.12851 · DynExpertQuant 2511.15015 · Adaptive-Expert-Split 2509.08342 ·
Prefetch/spec: SpecPrefetch 2607.24787 · MoE-SpeQ 2511.14102 · Speculating-Experts 2603.19289 · SP-MoE 2510.10302 · SpecMD 2602.03921 · DraftExpert 2607.24434 · Pre-Attention-Expert-Pred (ETH) ·
Routing/specialization: Dirichlet-Prior-Shaping 2510.01185 · Advancing-Expert-Specialization 2505.22323 · Probing-Semantic-Routing 2502.10928 · Load-Balancing-w/-Similarity (OpenReview FNuvMnGAm8) · ReMoE-ReLU 2412.14711 (⚠️collision) ·
On-device/NPU: Apple-Silicon-NPU-MoE 2604.18788 · Fast-On-device-NPU 2407.05858 · Comprehensive-MoE-Survey 2503.07137
