# References Summary

Verified reference list for the paper on running massive MoE LLMs on memory-constrained edge hardware
(bandwidth wall, router telemetry, Path-Mapped Serving, locality/domain auxiliary-loss router negative result).

- **Total entries:** 53
- **Full-read (intro + method/results, beyond abstract):** 53 / 53 (yes)
- **Transport:** arXiv HTML (`arxiv.org/html`) or ar5iv full text, read via Claude-in-Chrome and curl-to-disk+grep of the *same* HTML (no PDF-to-markdown, no summarizer in the loop). Identifiers + venues taken from the arXiv abstract page.
- Every entry in `references.bib` carries a `% verified_finding:` line quoting a specific non-abstract claim/number.

| citekey | title | year | why relevant | verified |
|---|---|---|---|---|
| shazeer2017moe | Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer | 2017 | Origin of sparsely-gated MoE; conditional compute vs bytes-touched | yes |
| lepikhin2021gshard | GShard: Scaling Giant Models with Conditional Computation | 2020 | Top-2 gating, expert capacity/aux-loss that later serving must respect | yes |
| fedus2022switch | Switch Transformers | 2022 | Top-1 routing, capacity factor, selective precision — core MoE baseline | yes |
| du2022glam | GLaM: Efficient Scaling of LMs with MoE | 2022 | Active-vs-total params: 96.6B/1.2T active, half GPT-3 inference FLOPs | yes |
| zoph2022stmoe | ST-MoE: Stable and Transferable Sparse Expert Models | 2022 | Router z-loss; expert specialization analysis (domain structure) | yes |
| zhou2022expertchoice | MoE with Expert Choice Routing | 2022 | Variable experts/token, perfect load balance — routing-structure baseline | yes |
| dai2024deepseekmoe | DeepSeekMoE (fine-grained + shared experts) | 2024 | Fine-grained experts + shared expert; specialization = orthogonality | yes |
| jiang2024mixtral | Mixtral of Experts | 2024 | Sec.5 temporal locality of expert routing "leveraged for caching" — key thesis cite | yes |
| deepseekai2024deepseekv3 | DeepSeek-V3 Technical Report | 2024 | 671B/37B active, aux-loss-free load balancing — frontier MoE at scale | yes |
| yang2025qwen3 | Qwen3 Technical Report | 2025 | 235B-A22B MoE + thinking budget (adaptive inference compute) | yes |
| muennighoff2024olmoe | OLMoE: Open Mixture-of-Experts LMs | 2024 | Routing saturates early; experts rarely co-activated; domain/vocab specialization | yes |
| deepseekai2024deepseekv2 | DeepSeek-V2 | 2024 | MLA cuts KV cache 93.3% — memory-bandwidth relevance | yes |
| wei2024skyworkmoe | Skywork-MoE training techniques | 2024 | Gating logit normalization for expert diversity (routing shaping) | yes |
| fedus2022review | A Review of Sparse Expert Models | 2022 | Taxonomy of routing algorithms (token-choice vs global assignment) | yes |
| lieber2024jamba | Jamba: Hybrid Transformer-Mamba LM | 2024 | 52B/12B-active hybrid MoE fitting a single 80GB GPU at 256K ctx | yes |
| song2024powerinfer | PowerInfer: Serving on a Consumer GPU | 2024 | Hot/cold neuron placement by activation frequency (locality/caching) | yes |
| xue2024powerinfer2 | PowerInfer-2: Inference on a Smartphone | 2024 | 47B LLM on a phone @11.68 tok/s — edge memory wall | yes |
| yi2023edgemoe | EdgeMoE: Sparse LLMs on Mobile | 2023 | Expert buffer + per-expert bitwidth; experts dominate IO cost | yes |
| xue2024moeinfinity | MoE-Infinity: Sparsity-Aware Expert Cache | 2024 | Expert-activation traces guide prefetch/eviction — telemetry-driven serving | yes |
| kamahori2025fiddler | Fiddler: CPU-GPU Orchestration for MoE | 2025 | Compute cold experts on CPU; activations << weights (bytes argument) | yes |
| hwang2023pregated | Pre-gated MoE (algorithm-system co-design) | 2023 | Predict next-layer experts to prefetch — cache-friendly routing | yes |
| du2024sidamoe | SiDA-MoE: Data-Aware Serving | 2024 | Hash predictor for expert offload; 3.93x tput, 80% memory saving | yes |
| eliseev2023mixtraloffload | Fast MoE Inference with Offloading | 2023 | LRU expert cache + speculative prefetch; Mixtral @2-3 tok/s on T4 | yes |
| cao2024moelightning | MoE-Lightning: Memory-constrained GPUs | 2024 | Roofline/HBM-aware CPU-GPU-IO pipeline; 10.3x on a 16GB T4 | yes |
| zhong2024adapmoe | AdapMoE: Adaptive Expert Gating | 2024 | Activate 25% fewer experts => fewer bytes fetched | yes |
| tang2024hobbit | HOBBIT: Mixed-Precision Expert Offloading | 2024 | Token/layer/sequence-level expert loading+prefetch+caching hierarchy | yes |
| song2024promoe | ProMoE: Proactive Caching | 2024 | Learned predictor prefetches experts ahead of use (2.2x) | yes |
| xu2025moegen | MoE-Gen: Module-Based Batching | 2025 | Single-GPU high-throughput MoE offload (DeepSeek-V2 236B on 24GB) | yes |
| kong2024swapmoe | SwapMoE: Tunable Memory Budget | 2023 | Tunable hot-expert working set; memory/accuracy tradeoff | yes |
| sheng2023flexgen | FlexGen: High-Throughput Single-GPU Inference | 2023 | Canonical memory-bound offload roofline; OPT-175B on 16GB GPU | yes |
| aminabadi2022deepspeed | DeepSpeed Inference | 2022 | Heterogeneous CPU+NVMe+GPU inference for oversized models | yes |
| pope2022scaling | Efficiently Scaling Transformer Inference | 2022 | Roofline/MFU analysis; 29ms/token, memory-bandwidth vs compute | yes |
| frantar2023gptq | GPTQ Post-Training Quantization | 2023 | 3-4 bit weights => fewer bytes/token; 175B on one GPU | yes |
| lin2024awq | AWQ: Activation-aware Weight Quantization | 2024 | Protect 1% salient channels; edge-friendly low-bit weights | yes |
| xiao2023smoothquant | SmoothQuant W8A8 | 2023 | Migrate activation outliers to weights; 2x memory reduction | yes |
| liu2025spinquant | SpinQuant: Learned Rotations | 2025 | 4-bit W+A+KV within 2.9 pts of FP — aggressive byte reduction | yes |
| wang2023bitnet | BitNet: 1-bit Transformers | 2023 | BitLinear; 1-bit weights cut memory/energy (edge relevance) | yes |
| ma2024bitnet158 | BitNet b1.58 (ternary) | 2024 | Ternary weights match FP16 quality — minimal bytes/param | yes |
| liu2025paretoq | ParetoQ: Extreme Low-bit Scaling Laws | 2025 | Unified 1/1.58/2/3/4-bit comparison; 2-3 bit transition | yes |
| dettmers2023qlora | QLoRA: Finetuning Quantized LLMs | 2023 | 4-bit NF4 finetuning; enables adapting edge models cheaply | yes |
| dettmers2022llmint8 | LLM.int8() | 2022 | Outlier-aware 8-bit matmul; foundational weight quantization | yes |
| xiao2024streamingllm | StreamingLLM (Attention Sinks) | 2024 | Attention-sink KV eviction; bounds KV memory for long streams | yes |
| zhang2023h2o | H2O: Heavy-Hitter Oracle | 2023 | Heavy-hitter KV eviction; up to 29x throughput vs offload baselines | yes |
| xiao2024duoattention | DuoAttention (retrieval/streaming heads) | 2024 | Per-head KV budgeting; 2.55x long-context memory reduction | yes |
| leviathan2023speculative | Fast Inference via Speculative Decoding | 2023 | Draft+verify to cut decode steps under memory-bound decode | yes |
| cai2024medusa | Medusa (multiple decoding heads) | 2024 | Draft-model-free multi-head speculation; 2.2-2.8x | yes |
| li2024eagle | EAGLE (feature-level drafting) | 2024 | Feature-uncertainty drafting; 2.7-3x lossless speedup | yes |
| li2024eagle2 | EAGLE-2 (dynamic draft trees) | 2024 | Context-aware dynamic draft tree; 3-4x | yes |
| li2025eagle3 | EAGLE-3 (training-time test) | 2025 | Direct token prediction + multi-layer fusion; up to 6.5x | yes |
| miao2024specinfer | SpecInfer (tree-based speculation) | 2024 | Tree speculation cuts weight memory accesses; helps offloaded decode | yes |
| borzunov2023petals | Petals: Collaborative Inference | 2022 | Swarm splitting beats offload for BLOOM-176B on consumer GPUs | yes |
| hu2022pipeedge | Pipeline Parallelism on Heterogeneous Edge | 2021 | Optimal transformer pipeline partition across edge devices | yes |
| zhou2024survey | Survey on Efficient Inference for LLMs | 2024 | Roofline argument that decode/attention is memory-bound | yes |
