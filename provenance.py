"""
Result provenance + a hard guard against the b-l05 mixup: a 50M-token PILOT was once
plotted as if it were the canonical 200M lambda=.05 run, flipping a headline. Every run
already writes config.json (tokens/seed/lambda/params); nothing ever ASSERTED that two
series overlaid on one axis shared a token budget. This module does.

  from provenance import load_prov, assert_comparable
  assert_comparable(["runs/a-main-s1", "runs/b-main-s1", "runs/b-l20"])   # raises if not comparable
"""
import hashlib, json, os

# fields that MUST be equal for two runs to sit on the same comparison axis
# (the independent variables arm / lambda_loc are deliberately NOT here)
COMPARE_KEYS = ("tokens", "params", "seq")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prov(run_dir, with_ckpt_sha=True):
    """Return a provenance dict for a run, caching it to <run_dir>/provenance.json.
    Combines config.json + ckpt sha256 + final eval ppl."""
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"{run_dir}: no config.json — cannot establish provenance")
    cfg = json.load(open(cfg_path))
    prov = {"run_dir": run_dir,
            "run_name": cfg.get("run_name", os.path.basename(run_dir.rstrip("/"))),
            "arm": cfg.get("arm"), "lambda_loc": cfg.get("lambda_loc"),
            "mu_dom": cfg.get("mu_dom"), "tokens": cfg.get("tokens"),
            "seq": cfg.get("seq"), "seed": cfg.get("seed"), "params": cfg.get("params")}
    ck = os.path.join(run_dir, "ckpt.pt")
    if with_ckpt_sha and os.path.exists(ck):
        cache = os.path.join(run_dir, "provenance.json")
        old = json.load(open(cache)) if os.path.exists(cache) else {}
        if old.get("ckpt_bytes") == os.path.getsize(ck) and old.get("ckpt_sha256"):
            prov["ckpt_sha256"] = old["ckpt_sha256"]
        else:
            prov["ckpt_sha256"] = _sha256(ck)
        prov["ckpt_bytes"] = os.path.getsize(ck)
    fe = os.path.join(run_dir, "final_eval.json")
    if os.path.exists(fe):
        ev = json.load(open(fe))
        prov["final_ppl"] = {d: round(v["val_ppl"], 3) for d, v in ev.items()
                             if isinstance(v, dict) and "val_ppl" in v}
    json.dump(prov, open(os.path.join(run_dir, "provenance.json"), "w"), indent=1)
    return prov


def assert_comparable(run_dirs, keys=COMPARE_KEYS):
    """Raise if the runs do not all share the COMPARE_KEYS values. This is the guard
    that would have stopped the pilot-vs-canonical plot."""
    provs = [load_prov(d, with_ckpt_sha=False) for d in run_dirs]
    for k in keys:
        vals = {p["run_name"]: p.get(k) for p in provs}
        uniq = set(vals.values())
        if len(uniq) > 1:
            detail = ", ".join(f"{n}={v}" for n, v in vals.items())
            raise ValueError(
                f"REFUSING to plot: runs differ in '{k}' -> {detail}. "
                f"Series on one axis must share {keys}. "
                f"(This is the b-l05 pilot-vs-canonical guard.)")
    return provs


if __name__ == "__main__":
    import sys
    dirs = sys.argv[1:] or ["runs/a-main-s1", "runs/b-main-s1"]
    for p in assert_comparable(dirs):
        print(f"{p['run_name']:14s} tokens={p['tokens']:.0f} params={p['params']} "
              f"lam={p['lambda_loc']} sha={p.get('ckpt_sha256','-')[:12]} ppl={p.get('final_ppl')}")
    print("OK comparable")
