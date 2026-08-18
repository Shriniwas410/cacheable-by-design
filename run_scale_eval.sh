#!/bin/bash
# QUEUED scale-study evaluation — run ONLY after both 340M arms finish
# (runs/SCALE340.FINISHED). Tests the paper's open RQ: does the locality PPL tax
# of a fixed cacheability gain shrink from 137M to 340M?
# CPU-only (torch threads); safe to run once training has released the GPU.
set -e
cd /mnt/c/Users/shrin/Desktop/AI/sticky-moe

A=runs/scale340-a          # baseline (lambda=0)
B=runs/scale340-b05        # locality (lambda=0.05)
TAUS="0,0.1,0.2,0.35,0.5,0.7"

echo "=== provenance guard: A and B must share tokens/params/seq ==="
python3 -c "from provenance import assert_comparable; assert_comparable(['$A','$B']); print('OK comparable')"

echo "=== cacheeval arm A (340M baseline) ==="
python3 cacheeval.py --ckpt $A/ckpt.pt --tag scale340a --caps 0.25 --taus $TAUS \
    --tokens 12000 --out $A/cacheaware.json

echo "=== cacheeval arm B (340M locality lambda=0.05) ==="
python3 cacheeval.py --ckpt $B/ckpt.pt --tag scale340b05 --caps 0.25 --taus $TAUS \
    --tokens 12000 --out $B/cacheaware.json

echo "=== SCALE TAX COMPARISON: locality PPL tax + miss reduction, 137M vs 340M ==="
python3 - <<'PY'
import json
def tau0(run):
    p = sorted(json.load(open(f"runs/{run}/cacheaware.json"))["points"], key=lambda x: x["tau"])
    return next(x for x in p if x["tau"] == 0.0)
def gap(a_run, b_run, label):
    a, b = tau0(a_run), tau0(b_run)
    dppl = (b["avg_ppl"]/a["avg_ppl"] - 1) * 100
    dmiss = (1 - b["avg_miss_per_tok"]/a["avg_miss_per_tok"]) * 100
    print(f"  {label:6s}  A_ppl={a['avg_ppl']:.2f} B_ppl={b['avg_ppl']:.2f}  "
          f"locality tax={dppl:+.2f}% PPL  miss reduction={dmiss:+.1f}%")
    return dppl, dmiss
print("Training-time locality (lambda=0.05) tax at tau=0, arm A vs arm B:")
try:    g137 = gap("a-main-s1", "b-main-s1", "137M")
except Exception as e: print("  137M: (need runs/*/cacheaware.json)", e); g137=None
g340 = gap("scale340-a", "scale340-b05", "340M")
if g137:
    verdict = "SHRINKS (supports hypothesis)" if g340[0] < g137[0] else "does NOT shrink"
    print(f"\n  => locality PPL tax {g137[0]:+.2f}% (137M) -> {g340[0]:+.2f}% (340M): {verdict}")
print("\n  CAVEAT: 340M at the fixed 200M-token budget is more undertrained than 137M")
print("  (fewer tokens/param); read the tax trend, not absolute PPL.")
PY
echo "=== DONE. Wrote $A/cacheaware.json + $B/cacheaware.json ==="
