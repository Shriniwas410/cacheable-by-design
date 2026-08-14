"""
Ingest reference PDFs into the paper repo as durable, traceable evidence.

Copies every reference PDF out of the (ephemeral, session-scoped) scratchpad into
paper/refpdfs/, computes a sha256 for each, extracts full text to refpdfs/txt/,
and writes refpdfs/manifest.json mapping bib-key <-> arXiv id <-> file <-> sha256.

Idempotent: re-running re-copies only missing/changed files and rewrites the manifest.

  python ingest_refpdfs.py --src <scratchpad/refpdfs> [--redownload key1,key2]
"""
import argparse, hashlib, json, os, re, shutil, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BIB = os.path.join(HERE, "references.bib")
DST = os.path.join(HERE, "refpdfs")
TXT = os.path.join(DST, "txt")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_bib(path):
    """Return list of {key, eprint, title, author} for every @entry."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    entries = []
    for m in re.finditer(r"@\w+\s*\{\s*([^,]+),", txt):
        key = m.group(1).strip()
        body = txt[m.end():]
        nxt = re.search(r"\n@\w+\s*\{", body)
        body = body[: nxt.start()] if nxt else body

        def field(name):
            fm = re.search(name + r"\s*=\s*", body, re.I)
            if not fm:
                return ""
            i = fm.end()
            if body[i] == "{":
                depth = 0
                for j in range(i, len(body)):
                    if body[j] == "{":
                        depth += 1
                    elif body[j] == "}":
                        depth -= 1
                        if depth == 0:
                            return body[i + 1:j]
            return ""

        entries.append({"key": key, "eprint": field("eprint").strip(),
                        "title": re.sub(r"\s+", " ", field("title")).strip(),
                        "author": re.sub(r"\s+", " ", field("author")).strip()})
    return entries


def extract_text(pdf, out):
    import fitz
    doc = fitz.open(pdf)
    parts = []
    for pno in range(doc.page_count):
        parts.append(f"\n===== PAGE {pno + 1} =====\n" + doc[pno].get_text("text"))
    doc.close()
    open(out, "w", encoding="utf-8").write("".join(parts))
    return len(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir of <arxivid>.pdf files")
    ap.add_argument("--redownload", default="", help="comma bib-keys to re-fetch from arXiv")
    args = ap.parse_args()
    os.makedirs(TXT, exist_ok=True)

    entries = parse_bib(BIB)
    redl = set(x for x in args.redownload.split(",") if x)
    manifest = {"generated_by": "ingest_refpdfs.py", "count": 0, "entries": []}
    missing, ok = [], 0

    for e in entries:
        aid = e["eprint"]
        if not aid:
            manifest["entries"].append({**e, "status": "NO_EPRINT"})
            missing.append(e["key"] + " (no eprint)")
            continue
        dst_pdf = os.path.join(DST, aid + ".pdf")
        src_pdf = os.path.join(args.src, aid + ".pdf")

        # optional integrity re-download of load-bearing PDFs
        if e["key"] in redl:
            url = f"https://arxiv.org/pdf/{aid}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                data = urllib.request.urlopen(req, timeout=60).read()
                redl_path = os.path.join(DST, aid + ".redl.pdf")
                open(redl_path, "wb").write(data)
                e["redownload_sha256"] = sha256(redl_path)
            except Exception as ex:
                e["redownload_error"] = str(ex)

        if not os.path.exists(dst_pdf):
            if os.path.exists(src_pdf):
                shutil.copy2(src_pdf, dst_pdf)
            else:
                manifest["entries"].append({**e, "status": "PDF_MISSING"})
                missing.append(f"{e['key']} ({aid})")
                continue
        digest = sha256(dst_pdf)
        txt_path = os.path.join(TXT, aid + ".txt")
        pages = extract_text(dst_pdf, txt_path)
        row = {**e, "pdf": f"refpdfs/{aid}.pdf", "txt": f"refpdfs/txt/{aid}.txt",
               "sha256": digest, "bytes": os.path.getsize(dst_pdf), "pages": pages,
               "status": "OK"}
        if "redownload_sha256" in e:
            row["redownload_matches"] = (e["redownload_sha256"] == digest)
        manifest["entries"].append(row)
        ok += 1

    manifest["count"] = ok
    json.dump(manifest, open(os.path.join(DST, "manifest.json"), "w"), indent=1)
    print(f"OK {ok}/{len(entries)} entries have local PDF+txt+sha256")
    if missing:
        print("MISSING:", *missing, sep="\n  ")
    # redownload integrity report
    for r in manifest["entries"]:
        if "redownload_matches" in r:
            tag = "MATCH" if r["redownload_matches"] else "DIFFERS"
            print(f"  redownload {r['key']} ({r['eprint']}): sha {tag}")
        if "redownload_error" in r:
            print(f"  redownload {r['key']}: ERROR {r['redownload_error']}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
