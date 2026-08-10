# Publication plan — sticky-moe / edge-MoE bandwidth-wall paper

**Author:** Shriniwas Ramesh Suram · ORCID 0009-0009-0452-9407 · University of Cumberlands
**Paper:** *Cacheable by Design? Training MoE Routers for Locality Against the Edge Memory-Bandwidth Wall — A Pre-Registered Negative Result.*

---

## 0. Honest venue assessment (read this first)

What the paper is: a **methodologically clean, pre-registered negative result** plus a
**real systems-measurement contribution** (the bandwidth-wall quantification, the
`llama-moe-trace` instrument, the Qwen3 routing profile, Path-Mapped Serving analysis).
What it is *not* (yet): a large-scale result. It is one 137M-parameter configuration on a
single consumer GPU.

Implication for targeting — be realistic, avoid wasted submission cycles:

| Venue tier | Fit for THIS draft | Note |
|---|---|---|
| **arXiv preprint** | ✅ do immediately | priority + citeable + ORCID/DOI |
| **TMLR** (journal) | ✅ **best "top" target** | reviews on *soundness*, explicitly welcomes negative & reproducibility results, rolling deadline — a top-reputation ML venue that fits this paper as-is |
| **Workshops** (ICBINB, Efficient-ML, MoE, on-device, ML-for-Systems) | ✅ strong, near-term | non-archival → keeps main-venue option open; great visibility |
| **MLSys / EuroMLSys / EuroSys** | ⚠️ possible (systems framing) | the wall + `llama-moe-trace` + serving analysis is the systems hook; short/workshop track realistic now, main track stronger with a deployed engine |
| **AAAI / NeurIPS / ICML / ICLR / EMNLP main track** | ⚠️ hard as-is | competitive main tracks rarely take a single-scale negative result; **do this after the scale study** (137M→1B→7B, ~\$500 cloud) which turns the refutation into a scaling claim |

**Recommended primary path: arXiv now → TMLR submission → a workshop for visibility.**
**Recommended stretch path: run the scale study → resubmit strengthened version to MLSys or AAAI.**
Do not gate the preprint on the scale study; do gate the top-conference main-track attempt on it.

---

## 1. Pre-submission checklist (finish before any submission)

- [ ] Fill final author block (done: name/ORCID/affiliation). Decide whether to also credit
      the employer or keep "independent."
- [ ] **Release code + data** (Phase 2) and add the repo URL + archival DOI to the paper's
      Reproducibility paragraph.
- [ ] Add a short **Ethics / Broader Impact** paragraph (efficiency research; no human
      subjects; note energy/accessibility angle) — required by most ML venues.
- [ ] Add a **Reproducibility checklist** (many venues supply one; TMLR/NeurIPS style).
- [ ] Produce an **anonymized build** for double-blind venues: comment out `\author{}` +
      the ORCID `\thanks`, strip the repo URL to "link withheld for review." Keep a
      one-line `\newif` toggle so one file serves both.
- [ ] Proofread the 5 verified over-claim flags the reference agent was mid-checking
      (resume it after the session limit resets) — ensure each `verified_finding` matches
      the cited paper exactly.
- [ ] Consider adding the **135M→ scale-study result** if you run it before submitting to a
      conference (not needed for arXiv/TMLR/workshop).

---

## 2. Phase 1 — arXiv preprint (this week)

Purpose: timestamp priority, get a citeable DOI, surface on your ORCID.

Online steps:
1. **Account:** log in / register at arxiv.org. First-time cs.LG submitters may need an
   **endorsement** — arXiv will prompt; a colleague with cs.LG papers endorses you, or your
   `.edu` (ucumberlands) address may auto-qualify. Start this early; it can take days.
2. **Link ORCID:** arxiv.org → Account → "Change your ORCID iD" → authorize
   `0009-0009-0452-9407`. This makes new submissions auto-appear on your ORCID record.
3. **Prepare source:** arXiv compiles TeX itself. Upload `paper.tex` + `references.bib`
   (figures are inline TikZ/pgfplots — no image files). Include `paper.bbl` too (arXiv
   sometimes needs the pre-built `.bbl`; generate it with `tectonic --keep-intermediates`).
4. **Submit:** New Submission → primary category **cs.LG**, cross-list **cs.DC**
   (distributed/systems) and optionally **cs.AR**. Title, abstract, comments ("10 pages,
   pre-registered negative result; code at <repo>").
5. **After acceptance:** arXiv issues an arXiv ID + DOI (`10.48550/arXiv.XXXX`). Verify it
   appears under Works on your ORCID.

---

## 3. Phase 2 — Code & data release (this week, parallel to Phase 1)

Reviewers weight reproducibility heavily; you already have the material.

1. **GitHub repo** — one public repo bundling: `sticky-moe/` (DESIGN.md pre-registration,
   model/train/analyze/summarize, tests, RESULTS.md, run configs), `moe-routing-lab/`
   (incl. `06_convert_gguf_trace.py`), the `llama-moe-trace` patch, and a README that maps
   every paper number to its run. MIT/Apache-2.0 license.
2. **Archive on Zenodo** — link your GitHub to Zenodo, cut a release → Zenodo mints a
   **DOI**. In Zenodo settings, add your ORCID so the dataset/software appears on your
   record.
3. **Reference the DOIs** in the paper's Reproducibility paragraph (repo + Zenodo DOI).

---

## 4. Phase 3 — Workshop (near-term visibility, non-archival)

Best-fit workshops (verify each year's CFP + exact deadline — cycles below are typical, not
guaranteed):
- **ICBINB — "I Can't Believe It's Not Better"** (NeurIPS/ICLR workshop): purpose-built for
  rigorous negative/surprising results. *The* natural home for this paper.
- **ENLSP** (Efficient Natural Language & Speech Processing, NeurIPS): efficiency/MoE fit.
- **ML for Systems** (NeurIPS) or **EuroMLSys** (with EuroSys, spring): the systems angle.
- **On-device / Edge ML** workshops (various).

Online steps:
1. Create an **OpenReview** account; link your ORCID under Profile.
2. Find the workshop's OpenReview venue page from its CFP; note page limit (usually 4–8 pp)
   and whether archival (prefer **non-archival** so you can still submit the full paper to a
   journal/conference).
3. Trim `paper.tex` to the page limit (drop some related-work prose, keep the core result +
   one systems section) → submit PDF.

---

## 5. Phase 4 — Main venue

### 5a. TMLR (recommended, do after arXiv + code release)
Transactions on Machine Learning Research — OpenReview-hosted journal, **rolling
submission**, decisions on **technical soundness** rather than novelty/impact, explicitly
open to negative results. Strong reputation; the best realistic "top" home for this paper.
1. OpenReview account (ORCID linked).
2. TMLR venue → New Submission; follow the TMLR LaTeX style (light reformat of `paper.tex`).
3. Not double-blind in the usual sense (author-optional); include the code/DOIs.
4. Expect an Action Editor + reviewers; certifications ("Reproducibility", "Featured")
   possible on acceptance. On acceptance TMLR provides a DOI → appears on ORCID.

### 5b. Conference main track (after the scale study strengthens the claim)
With the 137M→1B→7B ladder result added, target one of:
- **MLSys** (systems framing — the wall + serving + tracer is a clean MLSys story). Typical
  abstract deadline ~fall; verify.
- **AAAI** (broad AI; typical abstract deadline ~August, full ~September — verify the 2027
  cycle). Two-column AAAI style; 7–8 pp.
- **EMNLP / ACL** efficient-methods track (if you lean the framing toward LM efficiency).
Steps are the same shape: OpenReview/CMT account → ORCID linked → anonymized PDF in the
venue's style → submit by the (verified) deadline.

---

## 6. ORCID integration — do it at every step

Your ORCID (`0009-0009-0452-9407`) should auto-collect each output:
1. **arXiv** → Account → link ORCID (Phase 1.2). New preprints post automatically.
2. **Zenodo** → Settings → link ORCID. Software/data DOI posts automatically.
3. **OpenReview** → Profile → add ORCID. TMLR/workshop acceptances associate.
4. **Crossref/DataCite DOIs** (journal, arXiv) → most publishers push to ORCID if you
   authorize the "trusted party" prompt once.
5. Fallback: any output not auto-added → ORCID → Works → Add manually by DOI.

---

## 7. Suggested timeline

| When | Action |
|---|---|
| Week 1 | Finish checklist §1; GitHub + Zenodo release (§3); start arXiv endorsement (§2.1) |
| Week 1–2 | arXiv preprint live (§2); ORCID shows it |
| Week 2–3 | Submit to a workshop (§4) matched to the nearest CFP |
| Week 2–4 | Format + submit to **TMLR** (§5a) |
| Weeks 3–8 | (Optional, ~\$500 cloud) run the **scale study**; add results |
| Next conf cycle | Strengthened version → **MLSys / AAAI** (§5b) |

---

## 8. One caution

This is an honest negative result. Its credibility *is* its value — reviewers at TMLR and
ICBINB will reward the pre-registration and the reproducibility, and punish any overclaim.
Keep the framing exactly as written: the mechanism works, it is not free at this scale, and
the strict criteria are not met. Do not soften "refuted" into "promising" for a submission.
