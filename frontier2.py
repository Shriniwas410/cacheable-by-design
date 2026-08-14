"""
The complementarity figure: training-time locality and training-free cache-aware
routing STACK. Plots dPPL% vs miss-reduction%, all relative to the true baseline
(arm-A, tau=0). Three series:
  - training-free alone  (baseline model, tau sweep)          [blue]
  - STACKED              (b-main l=.05 model, tau sweep)       [green]
  - training-time alone  (trained checkpoints, tau=0)         [squares]
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "runs"
BASE_PPL, BASE_MISS = 21.841, 8.3251                       # a-main tau=0 reference

from provenance import assert_comparable
# GUARD: every run overlaid below must share tokens/params/seq (b-l05 pilot guard).
assert_comparable([f"{RUN}/a-main-s1", f"{RUN}/b-main-s1", f"{RUN}/b-l02-main",
                   f"{RUN}/b-l03-main", f"{RUN}/b-l20"])

def pts(path):
    d = json.load(open(path)); ps = sorted(d["points"], key=lambda p: p["tau"])
    return [(p["tau"], (1 - p["avg_miss_per_tok"] / BASE_MISS) * 100,
             (p["avg_ppl"] / BASE_PPL - 1) * 100) for p in ps]

free = pts(f"{RUN}/a-main-s1/cacheaware.json")
stack = pts(f"{RUN}/b-main-s1/cacheaware.json")
tt = []
for lab, pth in [("lam=.02", "b-l02-main"), ("lam=.03", "b-l03-main"),
                 ("lam=.05", "b-main-s1"), ("lam=.20", "b-l20")]:
    p0 = sorted(json.load(open(f"{RUN}/{pth}/cacheaware.json"))["points"],
                key=lambda p: p["tau"])[0]
    tt.append((lab, (1 - p0["avg_miss_per_tok"] / BASE_MISS) * 100,
               (p0["avg_ppl"] / BASE_PPL - 1) * 100))

fig, ax = plt.subplots(figsize=(6.6, 4.6))
for series, col, mk, name in [(free, "#2266cc", "o", "training-free alone (baseline model)"),
                              (stack, "#1a9850", "^", "STACKED: train-time l=.05 + train-free tau")]:
    xs = [s[1] for s in series]; ys = [s[2] for s in series]
    ax.plot(xs, ys, "-", marker=mk, color=col, label=name, zorder=4)
    for tau, x, y in series:
        if tau in (0.3, 0.5, 0.7):
            ax.annotate(f"t={tau}", (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
for lab, x, y in tt:
    ax.scatter([x], [y], marker="s", s=55, color="#d73027", zorder=5)
    ax.annotate(lab, (x, y), fontsize=7, xytext=(4, -8), textcoords="offset points", color="#d73027")
ax.scatter([tt[0][1]], [tt[0][2]], marker="s", s=55, color="#d73027", label="training-time alone (tau=0)")
ax.axhline(1.0, ls="--", color="gray", lw=1)
ax.text(2, 1.4, "strict <=1% PPL gate", fontsize=7, color="gray")
ax.set_ylim(-1, 30)
ax.set_xlabel("expert-cache miss reduction vs baseline (%)  [cap = 25% experts, LRU]")
ax.set_ylabel("perplexity cost vs baseline (%)")
ax.set_title("Locality is cheapest when TRAINED and REROUTED together (137M, multi-domain)")
ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{RUN}/complementarity.png", dpi=140)
print("wrote runs/complementarity.png")
# print the crossover summary
print("\nAt fixed +~2.4% PPL budget:")
print(f"  training-time alone: {tt[2][1]:.1f}% miss reduction")
print(f"  STACKED (+tau=0.5):  {stack[3][1]:.1f}% miss reduction  (dPPL {stack[3][2]:+.2f}%)")
print(f"  training-free alone at that PPL: ~50% (tau=0.5, dPPL {free[3][2]:+.1f}%)")
