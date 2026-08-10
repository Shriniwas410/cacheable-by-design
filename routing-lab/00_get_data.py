"""
Fetch small domain corpora for routing analysis (~400k chars each by default).

Domains: general (WikiText-103), code (CodeSearchNet Python),
         medical (PubMed abstracts), math (GSM8K).

All public, no auth. If a download fails, just drop your own .txt files into
data/<domain>/ instead — 01_log_routing.py only needs plain text.

Usage: python 00_get_data.py --chars 400000
"""

import argparse
import os

from datasets import load_dataset


def dump(texts, domain, out_dir, max_chars):
    os.makedirs(os.path.join(out_dir, domain), exist_ok=True)
    buf, total = [], 0
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue
        buf.append(t)
        total += len(t)
        if total >= max_chars:
            break
    path = os.path.join(out_dir, domain, "corpus.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(buf))
    print(f"  {domain}: {total} chars -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--chars", type=int, default=400_000,
                    help="max characters per domain (~4 chars/token)")
    args = ap.parse_args()

    jobs = [
        ("general",
         lambda: (r["text"] for r in load_dataset(
             "wikitext", "wikitext-103-raw-v1", split="train",
             streaming=True))),
        ("code",
         lambda: (r["whole_func_string"] for r in load_dataset(
             "code_search_net", "python", split="train",
             streaming=True, trust_remote_code=True))),
        ("medical",
         lambda: (r["article"] for r in load_dataset(
             "ccdv/pubmed-summarization", "document", split="train",
             streaming=True, trust_remote_code=True))),
        ("math",
         lambda: (r["question"] + "\n" + r["answer"] for r in load_dataset(
             "gsm8k", "main", split="train", streaming=True))),
    ]

    for domain, get in jobs:
        print(f"Fetching {domain} ...")
        try:
            dump(get(), domain, args.out_dir, args.chars)
        except Exception as exc:            # noqa: BLE001
            print(f"  !! {domain} failed ({exc}). "
                  f"Drop your own .txt files in {args.out_dir}/{domain}/")


if __name__ == "__main__":
    main()
