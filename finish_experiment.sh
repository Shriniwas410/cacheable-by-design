#!/bin/bash
# Finish the interrupted experiment, DETACHED (setsid) so it survives the
# Claude Code process exiting. Redo b-main-s2 (was lost at 92%, no checkpoint),
# write the A/B/C verdict, then the λ-refinement (0.02, 0.03) + its verdict.
cd /mnt/c/Users/shrin/Desktop/AI/sticky-moe
export HF_HUB_ENABLE_HF_TRANSFER=1

run(){  # name arm lambda
  local name=$1 arm=$2 lam=$3
  echo "=== [$(date +%m-%d\ %H:%M)] $name (arm $arm, λ=$lam, 200M, seed varies) ==="
  local seedflag="--seed 1"; [ "$name" = "b-main-s2" ] && seedflag="--seed 2"
  python3 train.py --arm "$arm" --tokens 200e6 --run-name "$name" $seedflag --lambda-loc "$lam" \
    && python3 analyze.py "runs/$name" --json "runs/$name/metrics.json" >/dev/null \
    && python3 summarize.py --append "$name" \
    || { echo "!! $name FAILED"; return 1; }
}

# 1. finish the A/B/C set (second B seed)
run b-main-s2 B 0.05

# 2. λ-refinement
run b-l02-main B 0.02
run b-l03-main B 0.03

# 3. verdicts
echo "=== FINAL VERDICTS ==="
python3 - <<'PY'
import json
def L(r):
    try: return json.load(open(f"runs/{r}/metrics.json"))
    except: return None
def mppl(m):
    v=[d["val_ppl"] for d in m["domains"].values() if d.get("val_ppl")]; return sum(v)/len(v)
def miss(m): return sum(d["miss"]["0.25"]["lru"] for d in m["domains"].values())/len(m["domains"])
def hit50(m): return sum(d["hit"]["0.5"]["static"] for d in m["domains"].values())/len(m["domains"])
A=[L("a-main-s1"),L("a-main-s2")]
bppl=sum(mppl(a) for a in A)/2; bmiss=sum(miss(a) for a in A)/2
bhit=sum(hit50(a) for a in A)/2
print(f"BASELINE A (2 seeds): PPL {bppl:.1f}  miss/tok@25% {bmiss:.3f}  static hit@50% {bhit:.3f}\n")
print("RQ1 — locality (miss reduction >=30% AND PPL <=+1%):")
for lam,r in [("0.02","b-l02-main"),("0.03","b-l03-main"),("0.05","b-main-s1"),("0.05","b-main-s2"),("0.20","b-l20")]:
    m=L(r)
    if not m: continue
    red=(bmiss-miss(m))/bmiss*100; pc=(mppl(m)-bppl)/bppl*100
    print(f"  λ={lam} {r:12}: miss {red:+.0f}%  PPL {mppl(m):.1f} ({pc:+.1f}%)  -> {'PASS' if red>=30 and pc<=1 else 'fail'}")
print("\nRQ2 — domain (static hit@50% >=0.90 AND PPL <=+2%):")
for r in ["c-main-s1"]:
    m=L(r)
    if not m: continue
    pc=(mppl(m)-bppl)/bppl*100
    print(f"  {r}: static hit@50% {hit50(m):.3f}  PPL {mppl(m):.1f} ({pc:+.1f}%)  -> {'PASS' if hit50(m)>=0.90 and pc<=2 else 'fail'}")
PY
echo "=== FINISH_EXPERIMENT COMPLETE $(date) ==="
touch runs/FINISHED.flag
