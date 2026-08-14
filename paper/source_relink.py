"""
Re-point every `% source:` line at the local PDF corpus (id + sha256 prefix), keeping
the original remote-read note for audit. Idempotent. Run after ingest_refpdfs.py.
"""
import json, os, re
import verify_refs as v

HERE = os.path.dirname(os.path.abspath(__file__))
man = {e["eprint"]: e for e in json.load(open(os.path.join(HERE, "refpdfs", "manifest.json")))["entries"]
       if e.get("sha256")}
lines = open(v.BIB, encoding="utf-8").read().splitlines()

# collect comment-line indices per following entry
out = list(lines)
comment_idxs = []
n = 0
for i, ln in enumerate(lines):
    s = ln.lstrip()
    if s.startswith("%"):
        comment_idxs.append(i)
        continue
    m = re.match(r"\s*@\w+\s*\{\s*([^,]+),", ln)
    if m:
        # find eprint of this entry
        body = "\n".join(lines[i:i + 40])
        em = re.search(r"eprint\s*=\s*\{([^}]+)\}", body)
        aid = em.group(1).strip() if em else ""
        sha = man.get(aid, {}).get("sha256", "")[:12]
        # rewrite the source line among the comment block just above
        for ci in comment_idxs:
            cs = lines[ci].lstrip()
            if re.match(r"%\s*source\s*:", cs, re.I):
                indent = lines[ci][: len(lines[ci]) - len(cs)]
                orig = re.sub(r"^%\s*source\s*:\s*", "", cs, flags=re.I).strip()
                if aid and sha and "refpdfs/" not in orig:
                    out[ci] = f"{indent}% source: local pdf refpdfs/{aid}.pdf sha256:{sha}; orig: {orig}"
                    n += 1
        comment_idxs = []
    elif ln.strip() == "":
        comment_idxs = []
    else:
        comment_idxs = []

open(v.BIB, "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"relinked {n} source lines to local pdfs")
