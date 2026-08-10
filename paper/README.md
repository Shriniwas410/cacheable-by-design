# Paper draft — build & status

**`paper.tex`** — full draft. Self-contained figures (TikZ diagrams + pgfplots
charts); no external image files needed.

## Compile
```
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```
Needs a TeX distribution with `pgfplots`, `tikz`, `natbib`, `booktabs`,
`hyperref` (TeX Live / MiKTeX default).

## Contents
- **Tables:** memory hierarchy · 235B serving + batching collapse · Qwen3-30B
  routing profile · cross-domain similarity · main results (all arms).
- **Figures:** MoE layer (TikZ) · bandwidth wall + batching collapse (pgfplots)
  · cache-hit vs budget (pgfplots) · sticky-moe objective (TikZ) · λ
  dose-response dual-axis (pgfplots).
- **Sections:** intro · background · the bandwidth wall · measuring routing ·
  Path-Mapped Serving · sticky-moe method · results · discussion · limitations
  · related work · conclusion.

## References — reconciliation step (pending)
`references.bib` is produced by a separate full-PDF-verified reference pass
(Claude-in-Chrome, ≥50 entries, each read beyond the abstract). When it lands:
1. Check every `\cite{key}` in `paper.tex` resolves to a verified entry in
   `references.bib`. The paper uses keys like `shazeer2017moe`, `fedus2022switch`,
   `powerinfer2`, `moeinfinity`, `hobbit`, `edgemoe`, `adapmoe`, `deepseekmoe`,
   `deepseekv3`, `qwen3`, `banpick`, `flexgen`, `powerinfer`, `gptq`, `awq`,
   `bitnet158`, `streamingllm`, `leviathan`, `eagle`, `lepikhin2020gshard`.
2. For any cited paper the verifier could **not** confirm from a full PDF,
   either remove the `\cite` or swap to a verified alternative — do **not** ship
   an unverified citation.
3. `references_summary.md` lists what was verified.

## Provenance
Every number traces to a run in `../sticky-moe/` (RESULTS.md, run configs,
seeds, data hashes) or a measurement in `../SESSION-LOG-2026-08.md` /
`../colibri-mesh/docs/`. Author/affiliation block in `paper.tex` is a
placeholder to fill.
