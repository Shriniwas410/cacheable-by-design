"""
Append one arm's result row to RESULTS.md, and (when arms exist) evaluate the
pre-registered success criteria against arm A. Reads runs/<name>/metrics.json
produced by analyze.py.

  python summarize.py --append b-l05     # append row for that run
  python summarize.py --verdict          # write the criteria verdict block
"""
import argparse
import glob
import json
import os

RESULTS = "RESULTS.md"
BASELINE = "a-sanity-s1"     # arm A reference (updated to a-main-s1 when present)


def load(run):
    p = f"runs/{run}/metrics.json"
    return json.load(open(p)) if os.path.exists(p) else None


def mean_ppl(m):
    v = [d["val_ppl"] for d in m["domains"].values() if d.get("val_ppl")]
    return sum(v) / len(v) if v else None


def mean_metric(m, cap, pol, kind="hit"):
    v = [d[kind][cap][pol] for d in m["domains"].values()]
    return sum(v) / len(v)


def append_row(run):
    m = load(run)
    if not m:
        print(f"no metrics for {run}"); return
    cfg = m.get("config", {})
    arm = cfg.get("arm", "?"); lam = cfg.get("lambda_loc", 0); mu = cfg.get("mu_dom", 0)
    toks = cfg.get("tokens", 0)
    ppl = mean_ppl(m)
    reuse = sum(d["reuse_lag1"] for d in m["domains"].values()) / len(m["domains"])
    lru125 = mean_metric(m, "0.125", "lru")
    lru25 = mean_metric(m, "0.25", "lru")
    stat50 = mean_metric(m, "0.5", "static")
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a") as f:
        if new:
            f.write("# sticky-moe — results\n\n"
                    "Auto-appended per arm. Metrics averaged over prose/code/math.\n"
                    "`hit` = fraction of expert-loads already resident. Higher = more "
                    "cacheable. PPL lower = better LM quality.\n\n"
                    "| run | arm | λ_loc | μ_dom | tokens | mean PPL | reuse(lag1) | "
                    "LRU hit@12.5% | LRU hit@25% | static hit@50% |\n"
                    "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|\n")
        f.write(f"| {run} | {arm} | {lam:g} | {mu:g} | {toks/1e6:g}M | "
                f"{ppl:.1f} | {reuse:.3f} | {lru125:.3f} | {lru25:.3f} | {stat50:.3f} |\n")
    print(f"appended {run}: PPL {ppl:.1f} reuse {reuse:.3f} LRU@25% {lru25:.3f}")


def verdict():
    base = load(BASELINE) if load(BASELINE) else None
    # prefer a-main-s1 as baseline if present
    if load("a-main-s1"):
        globals()["BASELINE"] = "a-main-s1"; base = load("a-main-s1")
    if not base:
        print("no baseline"); return
    base_ppl = mean_ppl(base)
    base_miss25 = mean_metric(base, "0.25", "lru", "miss")
    lines = ["\n## Pre-registered verdict (vs arm A baseline)\n",
             f"Baseline `{BASELINE}`: mean PPL {base_ppl:.1f}, "
             f"LRU miss/token@25% = {base_miss25:.3f}\n"]
    # RQ1: best B arm
    for run in sorted(glob.glob("runs/b-*")):
        r = os.path.basename(run); m = load(r)
        if not m:
            continue
        ppl = mean_ppl(m); miss25 = mean_metric(m, "0.25", "lru", "miss")
        red = (base_miss25 - miss25) / base_miss25 * 100
        ppl_cost = (ppl - base_ppl) / base_ppl * 100
        rq1 = "PASS" if (red >= 30 and ppl_cost <= 1) else "fail"
        lines.append(f"- **RQ1 {r}**: miss reduction {red:+.1f}%, PPL cost "
                     f"{ppl_cost:+.1f}% → **{rq1}** (bar: ≥30% & ≤+1%)\n")
    # RQ2: C arm, static-pin within-domain @50%
    for run in sorted(glob.glob("runs/c-*")):
        r = os.path.basename(run); m = load(r)
        if not m:
            continue
        ppl = mean_ppl(m); hit50 = mean_metric(m, "0.5", "static")
        ppl_cost = (ppl - base_ppl) / base_ppl * 100
        rq2 = "PASS" if (hit50 >= 0.90 and ppl_cost <= 2) else "fail"
        lines.append(f"- **RQ2 {r}**: static hit@50% {hit50:.3f}, PPL cost "
                     f"{ppl_cost:+.1f}% → **{rq2}** (bar: ≥0.90 & ≤+2%)\n")
    with open(RESULTS, "a") as f:
        f.writelines(lines)
    print("".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--append")
    ap.add_argument("--verdict", action="store_true")
    args = ap.parse_args()
    if args.append:
        append_row(args.append)
    if args.verdict:
        verdict()
