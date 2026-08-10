#!/bin/bash
# Autonomous pre-registered experiment (DESIGN.md, amended budgets).
# Each arm: train (auto-evals + exports traces) -> analyze -> append RESULTS row.
# Ordered so the first A-vs-B signal lands early. Failures are logged, not fatal.
cd /mnt/c/Users/shrin/Desktop/AI/sticky-moe
run(){  # name arm tokens seed extra...
  local name=$1 arm=$2 toks=$3 seed=$4; shift 4
  echo "=== [$(date +%H:%M)] $name (arm $arm, ${toks} tok, seed $seed) ==="
  python3 train.py --arm "$arm" --tokens "$toks" --run-name "$name" --seed "$seed" "$@" \
    && python3 analyze.py "runs/$name" --json "runs/$name/metrics.json" >/dev/null \
    && python3 summarize.py --append "$name" \
    || echo "!! $name FAILED"
}

# analyze the already-trained sanity run as the 50M arm-A reference
python3 analyze.py runs/a-sanity-s1 --json runs/a-sanity-s1/metrics.json >/dev/null
python3 summarize.py --append a-sanity-s1

# --- lambda triage for locality loss (50M each, amended) ---
run b-l05  B 50e6 1 --lambda-loc 0.05
run b-l01  B 50e6 1 --lambda-loc 0.01
run b-l20  B 50e6 1 --lambda-loc 0.20

# pick best lambda: max reuse gain with PPL within +1% of a-sanity
BEST=$(python3 - <<'PY'
import json,glob,os
def load(r):
    p=f"runs/{r}/metrics.json"; return json.load(open(p)) if os.path.exists(p) else None
def mppl(m):
    v=[d["val_ppl"] for d in m["domains"].values() if d.get("val_ppl")]; return sum(v)/len(v)
def miss25(m):
    return sum(d["miss"]["0.25"]["lru"] for d in m["domains"].values())/len(m["domains"])
base=load("a-sanity-s1"); bppl=mppl(base)
best=None;bestscore=1e9
for lam,r in [("0.05","b-l05"),("0.01","b-l01"),("0.20","b-l20")]:
    m=load(r)
    if not m: continue
    if (mppl(m)-bppl)/bppl>0.01: continue      # PPL gate
    if miss25(m)<bestscore: bestscore=miss25(m); best=lam
print(best or "0.05")
PY
)
echo "=== best lambda = $BEST ==="

# --- main arms at 200M (amended) ---
run a-main-s1 A 200e6 1
run b-main-s1 B 200e6 1 --lambda-loc "$BEST"
run c-main-s1 C 200e6 1
# second seeds on A and best-B
run a-main-s2 A 200e6 2
run b-main-s2 B 200e6 2 --lambda-loc "$BEST"

python3 summarize.py --verdict
echo "=== EXPERIMENT COMPLETE ==="
