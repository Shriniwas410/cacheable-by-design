"""
Build compact per-entry context slices for the grounding pass, so each subagent
reads ~3KB (windows around the finding's numbers + the abstract) instead of a full
paper. Writes refpdfs/ctx/<key>.txt and refpdfs/cited.json (ids/titles cited).
"""
import json, os, re
import verify_refs as v

HERE = os.path.dirname(os.path.abspath(__file__))
CTX = os.path.join(HERE, "refpdfs", "ctx")
os.makedirs(CTX, exist_ok=True)

WIN = 8          # lines of context each side of a number hit
CAP = 3800       # char cap per ctx file


def page_of(lines, i):
    for j in range(i, -1, -1):
        m = re.match(r"===== PAGE (\d+)", lines[j])
        if m:
            return m.group(1)
    return "?"


def build(entry, txt):
    lines = txt.splitlines()
    nums = [core for _, core, _ in v.load_numbers(entry["verified_finding"])]
    picked = []            # (start,end,page)
    for core in nums:
        for i, ln in enumerate(lines):
            if core in ln.replace(",", ""):
                picked.append((max(0, i - WIN), min(len(lines), i + WIN + 1), page_of(lines, i)))
    # abstract: first ~25 lines after page 1 marker
    picked.append((0, 26, "1"))
    picked.sort()
    merged = []
    for s, e, pg in picked:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e), merged[-1][2])
        else:
            merged.append((s, e, pg))
    out, total = [], 0
    for s, e, pg in merged:
        block = f"[~p{pg}] " + " ".join(lines[s:e]).strip()
        block = re.sub(r"\s+", " ", block)[:1400]
        if total + len(block) > CAP:
            block = block[: CAP - total]
        out.append(block)
        total += len(block)
        if total >= CAP:
            break
    return "\n---\n".join(out)


def main():
    entries = v.parse_bib_with_comments(v.BIB)
    cited = []
    for e in entries:
        aid = e["eprint"]
        cited.append({"key": e["key"], "eprint": aid, "title": e["title"]})
        tp = os.path.join(HERE, "refpdfs", "txt", aid + ".txt")
        if not (aid and os.path.exists(tp)):
            continue
        txt = open(tp, encoding="utf-8", errors="replace").read()
        ctx = build(e, txt)
        with open(os.path.join(CTX, e["key"] + ".txt"), "w", encoding="utf-8") as f:
            f.write(f"KEY: {e['key']}  arXiv:{aid}\n")
            f.write(f"CLAIM (verified_finding): {e['verified_finding']}\n\n")
            f.write("PDF EXCERPTS (page-tagged, around the claim's numbers + abstract):\n")
            f.write(ctx)
    json.dump(cited, open(os.path.join(HERE, "refpdfs", "cited.json"), "w"), indent=1)
    print(f"wrote {len(entries)} ctx files + cited.json")


if __name__ == "__main__":
    main()
