#!/bin/bash
# Follow-up (pre-registered λ-tuning): find the locality weight that keeps the
# large miss reduction while satisfying the strict RQ1 quality gate (≤+1% PPL).
# Brackets the knee at λ=0.02 and 0.03, 200M tokens, seed 1, vs the existing
# a-main baseline. Waits for the GPU to free (orchestrator's last arm) first.
cd /mnt/c/Users/shrin/Desktop/AI/sticky-moe

echo "=== waiting for GPU (orchestrator to finish) ==="
while pgrep -f "run_experiment.sh" >/dev/null || pgrep -f "train.py" >/dev/null; do
  sleep 60
done
echo "=== GPU free at $(date +%H:%M) — starting λ refinement ==="

run(){  # name lambda
  local name=$1 lam=$2
  echo "=== [$(date +%H:%M)] $name (arm B, λ=$lam, 200M, seed 1) ==="
  python3 train.py --arm B --tokens 200e6 --run-name "$name" --seed 1 --lambda-loc "$lam" \
    && python3 analyze.py "runs/$name" --json "runs/$name/metrics.json" >/dev/null \
    && python3 summarize.py --append "$name" \
    || echo "!! $name FAILED"
}

run b-l02-main 0.02
run b-l03-main 0.03

# RQ1 check for the refinement arms vs a-main baseline (2 seeds)
python3 - <<'PY'
import json
def L(r): return json.load(open(f"runs/{r}/metrics.json"))
def mppl(m):
    v=[d["val_ppl"] for d in m["domains"].values() if d.get("val_ppl")]; return sum(v)/len(v)
def miss(m): return sum(d["miss"]["0.25"]["lru"] for d in m["domains"].values())/len(m["domains"])
A=[L("a-main-s1"),L("a-main-s2")]
bppl=sum(mppl(a) for a in A)/2; bmiss=sum(miss(a) for a in A)/2
print(f"\n=== λ-REFINEMENT VERDICT (baseline PPL {bppl:.1f}, miss/tok@25% {bmiss:.3f}) ===")
best=None
for lam,r in [("0.02","b-l02-main"),("0.03","b-l03-main"),("0.05","b-main-s1")]:
    try: m=L(r)
    except: continue
    red=(bmiss-miss(m))/bmiss*100; pc=(mppl(m)-bppl)/bppl*100
    ok = red>=30 and pc<=1
    print(f"λ={lam}: miss reduction {red:+.0f}%  PPL {mppl(m):.1f} ({pc:+.1f}%)  RQ1 {'PASS' if ok else 'fail'}")
    if ok and (best is None or red>best[1]): best=(lam,red)
print(f">>> best λ satisfying strict RQ1: {best[0] if best else 'NONE — gentler λ or accept trade needed'}"
      + (f" ({best[1]:+.0f}% miss reduction)" if best else ""))
PY
echo "=== REFINEMENT COMPLETE ==="
