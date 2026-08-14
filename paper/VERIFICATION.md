# Reference & result verification record

Purpose: make every cited claim and every reported number **traceable to a durable local
source** and **reproducible by a reviewer**. Written after two real defects shipped —
a misquoted concurrent paper (StickyMoE) and a pilot run plotted as a canonical one.

Date of this pass: 2026-08-14.

## What is now durable (was ephemeral)
- The 56 reference PDFs previously lived only in a session-scoped `/tmp` scratchpad. They
  are now in `paper/refpdfs/*.pdf` with sha256 in `refpdfs/manifest.json` and extracted
  text in `refpdfs/txt/`. Rebuild: `python ingest_refpdfs.py --src <dir>`.
- The three load-bearing concurrent-work PDFs were **re-downloaded from arXiv and
  sha-compared** to the local copies — all MATCH, i.e. the local corpus is the published
  artifact:
  - `kayyam2026sticky` 2607.08780 — sha MATCH
  - `eldr2026` 2607.00466 — sha MATCH
  - `cachecond2024` 2412.00099 — sha MATCH

## Deterministic gate — `python verify_refs.py`
Re-runnable. Asserts per entry: local PDF exists + sha256 == manifest; author has no
placeholder (`{X Authors}`/`TBD`); every load-bearing number in `verified_finding` appears
in the PDF text; title matches page 1; `source:` names a local pdf.

Result: **0 HARD failures, 2 SOFT (both benign)**
- `kayyam2026sticky`: number `137` not in PDF — correct; it is a cross-reference to *our*
  137M model inside the note, not a StickyMoE number.
- `cachecond2024`: number `2025.` not in PDF — a publication year, not a claim.

Proven to bite (adversarial): on the EdgeMoE text `2.78 → not found`, `2.77 → found`. The
checker would have caught the original `2.78x`-vs-`2.77x` error deterministically.

Scope of the number check (so the gate isn't read as tighter than it is): it checks numbers
with ≥2 digits, or a decimal point, or a `%`/`×` unit — single digits without a unit are
skipped (else `GPT-3`, `v4`, `top-2` are noise), and it is a substring-presence test, not a
sign test. It cannot see a claim's **direction** ("reduces by 5"); that is caught only by
the semantic grounding below, which was done for the 3 load-bearing entries, not all 56.

## Semantic grounding (direction + magnitude) of load-bearing entries
Read from each paper's own full-text PDF excerpt; direction confirmed (the class the number
check cannot catch):
| entry | claim (direction-critical) | PDF confirms |
|---|---|---|
| kayyam2026sticky (StickyMoE) | perplexity **improves** −4.1%/−0.9%, switch ↓59%, misses ↓3.92× | ✓ "improving perplexity on the medium model … cache misses up to 3.92×" |
| eldr2026 (ELDR) | median TPOT ↓5.9–13.9%, outputs unchanged | ✓ verbatim in abstract; authors match |
| cachecond2024 (Cache-Cond. Experts) | miss ↓>50% at 0.1–3% PPL; beats oracle bound; TMLR 2025 | ✓ verbatim; Skliar/Qualcomm; 2412.00099v2, TMLR Jun 2025 |

The original StickyMoE defect (quoted as a PPL *degradation*) is fixed in `references.bib`
and `paper.tex`; the corrected direction is what creates the honest single-domain-positive
vs multi-domain-tax tension the paper now discusses.

## Result provenance — new mechanical guard caught a live defect
`sticky-moe/provenance.py :: assert_comparable(dirs)` refuses to overlay runs that differ in
`tokens/params/seq`. It now gates `frontier.py` and `frontier2.py`. On first run it flagged:
- **b-l20 (λ=0.20) is a 50M-token PILOT**, not a 200M run — yet it was a row in `tab:main`
  captioned "Main results (200M tokens/arm)" with PPL 70.1. **Removed that row** (b-l20 is
  still shown, correctly labeled 50M, in `tab:allruns`). `fig:dose` and `tab:allruns` were
  already correct (they use only 200M points / carry a tokens column).

Token ground truth (from each run's `config.json`):
- 200M canonical: a-main-s1/s2, b-l02-main, b-l03-main, b-main-s1/s2, c-main-s1
- 50M pilots: a-sanity-s1, b-l01, b-l05, b-l20

## Discovery sweep — 8 references added (2026-08-14), all PDF-grounded
A 16-query main-thread WebSearch sweep (candidates in `refpdfs/candidates.md`) found
same-mechanism work we had missed. 8 were downloaded, grounded from their own PDFs, and
added to `references.bib` + cited in `paper.tex` (bib now 64; `verify_refs.py` 0 HARD;
recompiles clean, no undefined citations):
- **zhou2025oraclemoe** (Oracle-MoE, ICML'25) — trains locality-preserving routing, GPT-2
  200M–2B ladder; closest prior to our thesis + our scale study. Non-arXiv → verified via
  `localpdf` (sha 76c847c5898a); `verify_refs.py` extended to check non-arXiv entries.
- **zhu2026remoe** (2605.27081) — post-hoc router FT, +26% reuse. ⚠️ not the ReLU-ReMoE.
- **li2024mcsmoe** (2310.01334, ICLR'24) — merge redundant experts, 80% mem/20% FLOPs cut →
  substitutability evidence for our mechanism.
- **fang2025fate** (2502.12224) — cross-layer gate prediction, 99% hit, 4.5×/1.9× speedup.
- **zhang2025daop** (2501.10375) — data/domain-aware offloading (8.20×) → contrast for our
  domain-priming negative.
- **mouzouni2026threephases** (2604.04230) — experts specialise over training → co-adaptation.
- **raje2026melinoe** (2602.11192) — post-hoc FT, 3× fewer transfers.
- **liu2025moesurvey** (2412.14219, ACM CSUR) — inference-optimisation survey.
Dropped after reading the PDF: 2511.05814 (a student analysis report, weak source).

## Reproduce
```
cd paper
python ingest_refpdfs.py --src <pdf-dir> --redownload kayyam2026sticky,eldr2026,cachecond2024
python verify_refs.py            # expect 0 HARD
cd ../sticky-moe
python provenance.py runs/a-main-s1 runs/b-main-s1   # comparable → prints table
python frontier2.py              # guarded; refuses if a pilot sneaks in
```

## Pending (blocked, not skipped)
The multi-agent sweep `paper/verify_workflow.js` (semantic grounding of the other 53
background citations + a discovery sweep for missed concurrent work, by mechanism name /
citation graph / concurrent window) was launched but every agent errored on the account
**session limit** (resets 11pm ET). Re-run on reset:
`Workflow({scriptPath:"paper/verify_workflow.js", resumeFromRunId:"wf_cf5ceb1d-473", args:{...}})`.
The deterministic gate + manual grounding of the load-bearing entries already stand
independently of that sweep.
