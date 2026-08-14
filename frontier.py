"""
Overlay the TRAINING-FREE cache-aware frontier (baseline arm-A model, tau sweep)
against TRAINING-TIME locality points (locality-trained checkpoints at tau=0),
all on one axis: perplexity vs LRU expert-cache misses/token, cap = 25% of experts.

Reference point = baseline arm-A at tau=0 (no intervention at all). Every point's
%dPPL and %miss-reduction is expressed relative to it, so the training-free curve
and the training-time markers are directly comparable.

  python frontier.py     # prints table, writes runs/frontier_cacheaware.png + .json
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "runs"
FREE = f"{RUN}/a-main-s1/cacheaware.json"                 # training-free sweep (baseline model)
TRAINED = {                                               # training-time: label -> path
    "b-l02 (lam=.02)":  f"{RUN}/b-l02-main/cacheaware.json",
    "b-l03 (lam=.03)":  f"{RUN}/b-l03-main/cacheaware.json",
    "b-main (lam=.05)": f"{RUN}/b-main-s1/cacheaware.json",   # canonical 200M lambda=.05
    "b-l20 (lam=.20)":  f"{RUN}/b-l20/cacheaware.json",
}
HEADLINE_KEY = "b-main (lam=.05)"


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    from provenance import assert_comparable
    # GUARD: refuse to overlay series that don't share tokens/params/seq (the b-l05 pilot
    # was a 50M-token run once plotted as the canonical 200M lambda=.05 — never again).
    dirs = [os.path.dirname(FREE)] + [os.path.dirname(p) for p in TRAINED.values()
                                      if os.path.exists(p)]
    assert_comparable(dirs)

    free = load(FREE)
    assert free, f"missing {FREE}"
    pts = sorted(free["points"], key=lambda p: p["tau"])
    ref = next(p for p in pts if p["tau"] == 0.0)
    ppl0, miss0 = ref["avg_ppl"], ref["avg_miss_per_tok"]

    def rel(p):
        return (round((p["avg_ppl"] / ppl0 - 1) * 100, 2),
                round((1 - p["avg_miss_per_tok"] / miss0) * 100, 1))

    print(f"reference (baseline arm-A, tau=0): PPL={ppl0}  miss/tok={miss0}  "
          f"hit={ref['avg_hit']}\n")
    print("TRAINING-FREE cache-aware routing (baseline model, no retraining):")
    print(f"  {'tau':>5} {'PPL':>8} {'dPPL%':>7} {'miss/tok':>9} {'missRed%':>8} {'hit':>6}")
    rows = []
    for p in pts:
        dppl, dmiss = rel(p)
        rows.append({"method": "training-free", "tau": p["tau"], "ppl": p["avg_ppl"],
                     "dppl_pct": dppl, "miss": p["avg_miss_per_tok"],
                     "miss_red_pct": dmiss, "hit": p["avg_hit"]})
        print(f"  {p['tau']:>5} {p['avg_ppl']:>8} {dppl:>7} {p['avg_miss_per_tok']:>9} "
              f"{dmiss:>8} {p['avg_hit']:>6}")

    tr_points = []
    print("\nTRAINING-TIME locality (trained checkpoints, tau=0 = normal routing):")
    for name, path in TRAINED.items():
        d = load(path)
        if not d:
            continue
        p0 = next((q for q in d["points"] if q["tau"] == 0.0), None)
        if not p0:
            continue
        dppl, dmiss = rel(p0)
        tr_points.append((name, p0["avg_miss_per_tok"], p0["avg_ppl"], dppl, dmiss))
        print(f"  {name:12s} PPL={p0['avg_ppl']:>7}  dPPL%={dppl:>6}  "
              f"miss/tok={p0['avg_miss_per_tok']:>7}  missRed%={dmiss:>6}")

    # headline: cheapest training-free point that beats b-l05's miss reduction
    bl05 = next((t for t in tr_points if t[0] == HEADLINE_KEY), None)
    if bl05:
        target_miss = bl05[1]
        cand = [r for r in rows if r["miss"] <= target_miss]
        if cand:
            best = min(cand, key=lambda r: r["dppl_pct"])
            print(f"\nHEADLINE: training-free reaches b-l05's miss level "
                  f"({target_miss:.2f}/tok) at tau={best['tau']} for {best['dppl_pct']:+.2f}% PPL "
                  f"vs training-time's {bl05[3]:+.2f}% PPL "
                  f"-> training-free is {'CHEAPER' if best['dppl_pct'] < bl05[3] else 'costlier'}.")

    # plot
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    xs = [r["miss"] for r in rows]; ys = [r["ppl"] for r in rows]
    ax.plot(xs, ys, "-o", color="#2266cc", label="training-free (baseline, $\\tau$ sweep)")
    for r in rows:
        ax.annotate(f"$\\tau$={r['tau']}", (r["miss"], r["ppl"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    for name, mx, py, dppl, dmiss in tr_points:
        ax.scatter([mx], [py], marker="s", s=60, zorder=5,
                   label=f"training-time {name} ($\\tau$=0)")
    ax.scatter([miss0], [ppl0], marker="*", s=180, color="k", zorder=6, label="baseline (no intervention)")
    ax.set_xlabel("LRU expert-cache misses / token  (cap = 25% of experts)")
    ax.set_ylabel("perplexity (multi-domain avg)")
    ax.set_title("Training-free cache-aware routing vs training-time locality (137M)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = f"{RUN}/frontier_cacheaware.png"
    fig.savefig(out_png, dpi=140)
    json.dump({"reference": ref, "training_free": rows, "training_time": tr_points},
              open(f"{RUN}/frontier_cacheaware.json", "w"), indent=1)
    print(f"\nwrote {out_png} + runs/frontier_cacheaware.json")


if __name__ == "__main__":
    main()
