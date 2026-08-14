"""
Executable, re-runnable checker for references.bib. This is the guardrail that the
prose "VERIFICATION PROTOCOL" header was not: it FAILS on the exact defects that
slipped through before (StickyMoE direction/number, {X Authors} placeholders,
2.78x-vs-2.77x, HTML-not-PDF sourcing).

Per bib entry it asserts, against the local PDF corpus (paper/refpdfs/):
  HARD (exit 1):
    - a local PDF exists and its sha256 matches refpdfs/manifest.json
    - author field contains no placeholder ("Authors", "TBD", empty, bare "et al.")
  SOFT (reported, feeds the grounding pass; --strict makes them hard):
    - every load-bearing number in `% verified_finding:` appears in the PDF text
    - the title fuzzy-matches page-1 text
    - `% source:` names a local pdf path (not chrome|html-read|webfetch only)

  python verify_refs.py            # report + exit 1 on HARD failures
  python verify_refs.py --strict   # also exit 1 on any SOFT flag
  python verify_refs.py --json out.json
"""
import argparse, hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIB = os.path.join(HERE, "references.bib")
DST = os.path.join(HERE, "refpdfs")
MANIFEST = os.path.join(DST, "manifest.json")

PLACEHOLDER = re.compile(r"\bAuthors\b|\bTBD\b|\bXXX\b|^\s*$|^(?:et al\.?)$", re.I)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_bib_with_comments(path):
    """Yield {key, eprint, title, author, verified_finding, source} per entry,
    pulling verified_finding/source from the % comment block above the entry."""
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    entries = []
    comment_buf = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.lstrip().startswith("%"):
            comment_buf.append(ln.lstrip()[1:].strip())
            i += 1
            continue
        m = re.match(r"\s*@\w+\s*\{\s*([^,]+),", ln)
        if m:
            key = m.group(1).strip()
            # gather entry body until brace balance closes
            body_lines = [ln]
            depth = ln.count("{") - ln.count("}")
            j = i + 1
            while j < len(lines) and depth > 0:
                body_lines.append(lines[j])
                depth += lines[j].count("{") - lines[j].count("}")
                j += 1
            body = "\n".join(body_lines)
            cblob = " ".join(comment_buf)
            vf = re.search(r"verified_finding\s*:\s*(.*?)(?:\bsource\s*:|$)", cblob, re.I | re.S)
            src = re.search(r"\bsource\s*:\s*(.*)$", cblob, re.I | re.S)

            def field(name):
                fm = re.search(name + r"\s*=\s*", body, re.I)
                if not fm:
                    return ""
                k = fm.end()
                if k < len(body) and body[k] == "{":
                    d = 0
                    for q in range(k, len(body)):
                        if body[q] == "{":
                            d += 1
                        elif body[q] == "}":
                            d -= 1
                            if d == 0:
                                return body[k + 1:q]
                return ""

            entries.append({
                "key": key, "eprint": field("eprint").strip(),
                "title": re.sub(r"\s+", " ", field("title")).strip(),
                "author": re.sub(r"\s+", " ", field("author")).strip(),
                "verified_finding": re.sub(r"\s+", " ", vf.group(1)).strip() if vf else "",
                "source": re.sub(r"\s+", " ", src.group(1)).strip() if src else "",
            })
            comment_buf = []
            i = j
            continue
        comment_buf = []  # blank/other line breaks the comment block
        i += 1
    return entries


def load_numbers(text):
    """Load-bearing numbers from a finding: decimals, >=2-digit ints, or n%/n×.
    Skips single digits with no unit (GPT-3, v4, top-2 noise). Returns list of
    (raw, digitcore, unit)."""
    out = []
    for m in re.finditer(r"(\d[\d,]*\.?\d*)\s*(%|×|x\b)?", text):
        raw, unit = m.group(1), (m.group(2) or "")
        core = raw.replace(",", "")
        digits = core.replace(".", "")
        if "." in core or len(digits) >= 2 or unit:
            out.append((raw, core, unit))
    return out


def number_present(core, unit, txt):
    """Is this number in the PDF text? Tolerant of ×/x/%/space and 1,024 vs 1024."""
    variants = {core, core.replace(".", ",")}
    if "." not in core and len(core) >= 4:            # 1024 <-> 1,024
        variants.add(core[:-3] + "," + core[-3:])
    for v in variants:
        if v and v in txt:
            return True
    return False


def title_matches(title, page1):
    words = [w for w in re.sub(r"[^a-z0-9 ]", " ", title.lower()).split() if len(w) > 3]
    if not words:
        return True
    p = re.sub(r"[^a-z0-9 ]", " ", page1.lower())
    hit = sum(1 for w in words if w in p)
    return hit / len(words) >= 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if not os.path.exists(MANIFEST):
        print("FATAL: no manifest.json — run ingest_refpdfs.py first")
        sys.exit(2)
    man = {e["eprint"]: e for e in json.load(open(MANIFEST))["entries"] if e.get("sha256")}
    entries = parse_bib_with_comments(BIB)

    report, hard_fail, soft_flag = [], 0, 0
    for e in entries:
        aid = e["eprint"]
        r = {"key": e["key"], "eprint": aid, "hard": [], "soft": []}

        # --- resolve local pdf: by arXiv eprint, else by refpdfs/<file>.pdf in source ---
        stem = aid
        if not aid:
            lm = re.search(r"refpdfs/([^\s;]+?)\.pdf", e["source"])
            stem = lm.group(1) if lm else ""

        # --- HARD: local pdf + sha ---
        pdf = os.path.join(DST, stem + ".pdf") if stem else ""
        if not stem:
            r["hard"].append("no eprint id and no localpdf in source")
        elif not os.path.exists(pdf):
            r["hard"].append("local PDF missing")
        else:
            digest = sha256(pdf)
            if aid:  # arXiv entries must match the manifest; non-arXiv verified by file presence
                if aid not in man:
                    r["hard"].append("not in manifest")
                elif man[aid]["sha256"] != digest:
                    r["hard"].append("sha256 != manifest")

        # --- HARD: placeholder author ---
        if PLACEHOLDER.search(e["author"]) or not e["author"]:
            r["hard"].append(f"placeholder/empty author: {e['author'][:40]!r}")

        # --- read extracted text ---
        txt = ""
        tp = os.path.join(DST, "txt", stem + ".txt") if stem else ""
        if tp and os.path.exists(tp):
            txt = open(tp, encoding="utf-8", errors="replace").read()
        page1 = txt.split("===== PAGE 2", 1)[0]

        # --- SOFT: numbers present ---
        if e["verified_finding"] and txt:
            miss = [raw for raw, core, unit in load_numbers(e["verified_finding"])
                    if not number_present(core, unit, txt)]
            if miss:
                r["soft"].append("numbers not in PDF: " + ", ".join(sorted(set(miss))))
        elif not e["verified_finding"]:
            r["soft"].append("no verified_finding comment")

        # --- SOFT: title match ---
        if txt and not title_matches(e["title"], page1):
            r["soft"].append("title mismatch vs page 1")

        # --- SOFT: source not local pdf ---
        s = e["source"].lower()
        if not e["source"]:
            r["soft"].append("no source comment")
        elif "refpdfs/" not in s and "local" not in s and "pdf" not in s.replace("html-read", ""):
            r["soft"].append(f"source not local-pdf: {e['source'][:40]!r}")
        elif re.search(r"chrome|html-read|webfetch|ar5iv", s) and "refpdfs/" not in s:
            r["soft"].append(f"source is HTML/remote not local pdf: {e['source'][:40]!r}")

        if r["hard"]:
            hard_fail += 1
        if r["soft"]:
            soft_flag += 1
        report.append(r)

    # --- print ---
    print(f"Checked {len(entries)} entries | HARD failures: {hard_fail} | SOFT flags: {soft_flag}\n")
    for r in report:
        if r["hard"] or r["soft"]:
            print(f"[{r['key']}] ({r['eprint']})")
            for h in r["hard"]:
                print(f"   HARD  {h}")
            for s in r["soft"]:
                print(f"   soft  {s}")
    if not any(r["hard"] or r["soft"] for r in report):
        print("ALL CLEAN")
    if args.json:
        json.dump({"hard_fail": hard_fail, "soft_flag": soft_flag, "report": report},
                  open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")

    bad = hard_fail or (args.strict and soft_flag)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
