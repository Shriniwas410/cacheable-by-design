"""
Build the 3-domain corpus (DESIGN.md): prose / code / math.

Streams from HuggingFace datasets, tokenizes with GPT-2 BPE (tiktoken),
writes uint16 token shards + parallel uint8 domain ids, records SHA256 in
data/MANIFEST. Interleaving is document-level so domain tags are honest.

  python data.py --tokens-per-domain 100e6 --val-tokens 2e6
"""
import argparse
import hashlib
import os

import numpy as np
import tiktoken
from datasets import load_dataset

SOURCES = {
    "prose": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
    "code": ("codeparrot/codeparrot-clean", None, "content"),
    "math": ("open-web-math/open-web-math", None, "text"),
}
DOMAIN_ID = {"prose": 0, "code": 1, "math": 2}


def stream_docs(name, config, field):
    if name == "bigcode/the-stack-smol":
        ds = load_dataset(name, data_dir=config, split="train", streaming=True)
    else:
        ds = load_dataset(name, config, split="train", streaming=True)
    for row in ds:
        text = row.get(field) or ""
        if len(text) > 200:
            yield text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens-per-domain", type=float, default=100e6)
    ap.add_argument("--val-tokens", type=float, default=2e6)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    train_toks, train_doms = [], []
    manifest = []
    for dom, (name, config, field) in SOURCES.items():
        want = int(args.tokens_per_domain + args.val_tokens)
        got, buf = 0, []
        print(f"=== {dom}: streaming {name} until {want:,} tokens")
        for text in stream_docs(name, config, field):
            ids = enc.encode_ordinary(text) + [eot]
            buf.extend(ids)
            got += len(ids)
            if got >= want:
                break
        assert got >= want, f"{dom}: only {got:,} tokens available"
        val = np.array(buf[:int(args.val_tokens)], dtype=np.uint16)
        trn = buf[int(args.val_tokens):want]
        vpath = os.path.join(args.out, f"val_{dom}.bin")
        val.tofile(vpath)
        manifest.append((f"val_{dom}.bin", len(val),
                         hashlib.sha256(val.tobytes()).hexdigest()))
        train_toks.append(np.array(trn, dtype=np.uint16))
        train_doms.append(np.full(len(trn), DOMAIN_ID[dom], dtype=np.uint8))
        print(f"  {dom}: train {len(trn):,}  val {len(val):,}")

    # document-level interleave is preserved inside each domain block; blocks
    # are concatenated then shuffled at the *chunk* level (1M tokens) so
    # training batches mix domains without splitting documents mid-sequence
    # more than chunk boundaries already do.
    chunk = 1_000_000
    pieces = []
    for toks, doms in zip(train_toks, train_doms):
        for s in range(0, len(toks), chunk):
            pieces.append((toks[s:s + chunk], doms[s:s + chunk]))
    rng = np.random.default_rng(0)
    rng.shuffle(pieces)
    all_toks = np.concatenate([p[0] for p in pieces])
    all_doms = np.concatenate([p[1] for p in pieces])

    tpath, dpath = os.path.join(args.out, "train.bin"), os.path.join(args.out, "train.dom")
    all_toks.tofile(tpath)
    all_doms.tofile(dpath)
    manifest.append(("train.bin", len(all_toks),
                     hashlib.sha256(all_toks.tobytes()).hexdigest()))
    manifest.append(("train.dom", len(all_doms),
                     hashlib.sha256(all_doms.tobytes()).hexdigest()))

    with open(os.path.join(args.out, "MANIFEST"), "w") as f:
        for name_, n, h in manifest:
            f.write(f"{name_}\t{n}\t{h}\n")
    print(f"train total {len(all_toks):,} tokens; manifest written")


if __name__ == "__main__":
    main()
