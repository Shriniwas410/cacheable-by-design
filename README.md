# sticky-moe

**Can a Mixture-of-Experts router be *trained* to be cache-friendly, at no accuracy cost?**
A pre-registered study — and an honest negative result — plus the systems measurements
that motivated it. All experiments run on a single commodity GPU (RTX 3070, 8 GB).

📄 Paper: [`paper/paper.pdf`](paper/paper.pdf) · Pre-registration: [`DESIGN.md`](DESIGN.md) ·
Results: [`RESULTS.md`](RESULTS.md)

## TL;DR

Serving a 235B MoE on a home PC is **memory-bandwidth-bound**: each token streams ~12 GB of
expert weights, and on consumer hardware most experts sit on a 2.4 GB/s SSD. We measured the
wall, built a router-telemetry tool to see the routing structure inside it, and tested whether
that structure can be trained to be more cacheable.

**Finding:** it can — locality training cuts expert-cache misses up to **60%** and domain
training reaches a **99%** static-pin hit rate — **but every configuration fails a
pre-registered ≤1% perplexity gate.** Cacheability and quality are tightly coupled at 137M
scale; no loss weight threads the joint bar. The mechanism works; it is not free at this
scale. Whether the tax shrinks with scale is the open question.

## Repository layout

| Path | What |
|---|---|
| `DESIGN.md` | The frozen pre-registration (hypotheses, arms, metrics, pass/fail bars) |
| `model.py` | 137M MoE transformer + the three router losses (balance / locality / domain) |
| `train.py` | Arm runner; trains, evaluates, exports router traces |
| `analyze.py` | Self-contained metrics: perplexity, reuse, LRU/LFU/static/Belady cache sim |
| `summarize.py` | Builds `RESULTS.md` and the pre-registered verdict |
| `data.py` | 3-domain corpus builder (prose/code/math), SHA-256 manifested |
| `cacheeval.py`, `cacheeval_mixed.py` | Training-free cache-aware τ-rerouting eval (CPU); mixed-domain stream + domain-primed variant |
| `frontier.py`, `frontier2.py` | Complementarity frontier plots (ΔPPL vs cache-miss reduction), provenance-guarded |
| `provenance.py` | Run-provenance guard: `assert_comparable()` refuses to overlay runs with mismatched tokens/params/seed |
| `probe_elision.py`, `cert_elision.py` | Lossless expert-load-elision feasibility probe + resident-gate realizable-rate experiment |
| `run_experiment.sh`, `finish_experiment.sh`, `run_lambda_refine.sh`, `run_scale340.sh`, `run_scale_eval.sh` | Orchestrators for the arm sequence and the 137M→340M scale study |
| `tests/` | 7 unit tests (loss math, gradients, determinism, dispatch parity) |
| `runs/` | Per-run `config.json`, `metrics.json`, `final_eval.json`, logs, and router traces (`.npz`). **Checkpoints and the corpus are excluded** — see below |
| `routing-lab/` | `moe-routing-lab`: measures routing on real GGUF models; `06_convert_gguf_trace.py` bridges the tracer output into the analysis pipeline |
| `llama-moe-trace/` | `moe-trace.cpp` — router-telemetry addition to llama.cpp's eval-callback (weights untouched) |
| `paper/` | LaTeX source, verified bibliography, and reference-verification tooling (`verify_refs.py`, `ingest_refpdfs.py`, `build_ctx.py`, `source_relink.py`, `verify_workflow.js`, `VERIFICATION.md`) |

## Reproduce

```bash
pip install torch numpy tiktoken datasets
python data.py --tokens-per-domain 100e6 --val-tokens 2e6   # builds data/ (excluded from git)
python train.py --arm A --tokens 200e6 --run-name a-main --seed 1
python train.py --arm B --tokens 200e6 --run-name b-main --seed 1 --lambda-loc 0.05
python train.py --arm C --tokens 200e6 --run-name c-main --seed 1
python analyze.py runs/b-main --json runs/b-main/metrics.json
python summarize.py --verdict
python -m pytest tests/ -q
```

Complementarity (training-time × training-free stacking) and the 137M→340M scale rung:

```bash
# training-free cache-aware τ-sweep on a trained checkpoint, then the frontier figure
python cacheeval.py --ckpt runs/b-main-s1/ckpt.pt --tag stack --out runs/b-main-s1/cacheaware.json
python frontier2.py                    # complementarity figure (guarded by provenance.py)
# scale study (adam8bit, fits 8 GB): train both 340M arms, then the tax comparison
bash run_scale340.sh                   # arm A + arm B (λ=0.05) at 200M tokens
bash run_scale_eval.sh                 # τ-sweep both 340M arms + print 137M-vs-340M locality-tax table
```

The `llama-moe-trace` tool is a small addition to
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `examples/eval-callback` — see
[`llama-moe-trace/README.md`](llama-moe-trace/README.md).

## What is NOT in the repo (and why)

Model checkpoints (`*.pt`, ~524 MB each), GGUF weights, and the 300M-token training corpus
are excluded (`.gitignore`) — they are large and fully regenerable from the released code +
the SHA-256 data manifest. The small router traces (`.npz`) and all run configs/metrics are
included, so every number in the paper is reproducible.

## Paper numbers → runs

The main results (paper Table 5 / `RESULTS.md`) come from `runs/a-main-s{1,2}`,
`runs/b-main-s{1,2}`, `runs/b-l0{2,3}-main`, and `runs/c-main-s1`. The λ dose-response (paper
Fig. 6) is those B-arm runs at λ ∈ {0.02, 0.03, 0.05, 0.20}. The frontier-model routing
profile (paper §4, Qwen3-30B) comes from `routing-lab/` + `llama-moe-trace`.

The complementarity result (paper §8.1) and the scale table (`tab:scale`) come from the
`cacheaware.json` under `runs/a-main-s1`, `runs/b-main-s1` (137M) and `runs/scale340-a`,
`runs/scale340-b05` (340M). The near-lossless elision numbers come from `cert_elision.py`
and `probe_elision.py` on the same checkpoints.

## Citation

Paper: [arXiv:2608.18261](https://arxiv.org/abs/2608.18261). Author: Shriniwas Ramesh
Suram ([ORCID 0009-0009-0452-9407](https://orcid.org/0009-0009-0452-9407)).

```bibtex
@misc{suram2026cacheable,
  title        = {Cacheable by Design? Training Mixture-of-Experts Routers for
                  Locality Against the Edge Memory-Bandwidth Wall: A Pre-Registered
                  Negative Result, with a Systems Measurement Study},
  author       = {Suram, Shriniwas Ramesh},
  year         = {2026},
  eprint       = {2608.18261},
  archivePrefix= {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/2608.18261}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
