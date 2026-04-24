"""
Extract Sentences with Rare POS Tags for Targeted Training
=========================================================
从 SegPOS 语料中抽取含稀少标签（adj, n.prop, rare v.*）的句子，
构建专项训练集，用于二次 fine-tune 或对比学习。

Usage:
    .venv/bin/python scripts/extract_rare_tags.py
"""

import os, sys, json, random, glob
from pathlib import Path
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────

SEGPOS_BASE   = Path(__file__).parent.parent / "data" / "segpos_extracted"
OUTPUT_DIR    = Path(__file__).parent.parent / "data"  / "corpus" / "rare_tags_dataset"
RARE_TAG_MERGE = {
    # keep merged labels for matching
    "n.v.invar":       "v.invar",
    "n.v.past":        "v.past",
    "n.v.pres":        "v.pres",
    "n.v.fut":         "v.fut",
    "n.v.neg":         "v.neg",
    "n.v.cop":         "v.cop",
    "n.v.imp":         "v.imp",
    "n.v.aux":         "v.aux",
    "n.v.fut.n.v.pres":"v.fut.v.pres",
    "n.v.fut.n.v.past":"v.fut.v.past",
    "n.v.past.n.v.pres":"v.past.v.pres",
    "adv.intense":     "adj",
    "adv.dir":         "adj",
    "adv.temp":        "adj",
    "adv.mim":         "adj",
    "adv.proclausal":  "adj",
}

# Tags to extract (with minimum support thresholds)
RARE_TARGETS = {
    "adj":             1,      # any adj
    "n.prop":          1,      # proper nouns
    "n.v.invar":       1,
    "n.v.past":        1,
    "n.v.pres":        1,
    "n.v.fut":         1,
    "v.past.v.pres":   1,      # mixed tense (hard case)
    "v.fut.v.pres":    1,
    "v.fut.v.past":    1,
    "numeral":         1,      # numerals
    "skt":             1,      # Sanskrit transliterations
}

# Tibetan validity check
import re
TIBETAN_CONSONANT_RE = re.compile(r'[\u0F40-\u0F69\u0F71-\u0F87]')

def simplify_tag(raw: str) -> str:
    return RARE_TAG_MERGE.get(raw, raw)

def is_valid_tibetan(word: str) -> bool:
    if not word: return False
    if re.search(r'[A-Za-z0-9]', word): return False
    return bool(TIBETAN_CONSONANT_RE.search(word))

def parse_segpos_line(line: str):
    line = re.sub(r'\s*<utt>\s*$', '', line.strip())
    if not line:
        return []
    pairs = []
    for chunk in line.split():
        chunk = chunk.strip()
        if not chunk or chunk in ('<utt>', '<utt'):
            continue
        if '/' in chunk:
            parts = chunk.rsplit('/', 1)
            word = parts[0]
            raw_tag = parts[1] if len(parts) == 2 else "xxx"
        else:
            word = chunk
            raw_tag = "xxx"
        if not word or word in ('p1','p2','p3','p4','p5'):
            continue
        tag = simplify_tag(raw_tag)
        if tag == "numeral" and not is_valid_tibetan(word):
            continue
        pairs.append((word, tag))
    return pairs

# ── Extraction ───────────────────────────────────────────────────────────────

def main():
    import time
    t0 = time.time()

    print("=" * 60)
    print("  Rare Tags Extraction")
    print("=" * 60)

    pos_files = glob.glob(str(SEGPOS_BASE / "**" / "pos" / "*.txt"), recursive=True)
    pos_files = [f for f in pos_files if "__MACOSX" not in f]

    coll_files = {}
    for f in pos_files:
        coll = Path(f).parent.parent.name
        coll_files.setdefault(coll, []).append(f)

    # Collect sentences containing rare tags
    rare_sents = []       # (collection, line_str, pairs, rare_tag_counts)
    all_sents  = []       # all sentences (for sampling negatives)
    tag_counter = Counter()

    for coll_name, files in sorted(coll_files.items()):
        for f in files:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    pairs = parse_segpos_line(line)
                    if not pairs:
                        continue

                    # Count rare tags in this sentence
                    rare_in_sent = Counter()
                    for word, tag in pairs:
                        if tag in RARE_TARGETS:
                            rare_in_sent[tag] += 1

                    if rare_in_sent:
                        rare_sents.append((coll_name, line.strip(), pairs, rare_in_sent))
                        for t, c in rare_in_sent.items():
                            tag_counter[t] += c
                    all_sents.append((coll_name, line.strip(), pairs))

    print(f"\n[1] Extraction results:")
    print(f"    Total sentences:      {len(all_sents):,}")
    print(f"    Sentences w/ rare tags: {len(rare_sents):,}")
    print(f"\n[2] Rare tag distribution:")
    for tag, count in sorted(tag_counter.items(), key=lambda x: -x[1]):
        merge_src = [k for k, v in RARE_TAG_MERGE.items() if v == tag]
        src_str = f" (from {merge_src})" if merge_src else ""
        print(f"    {tag:<25s}: {count:>8,} occurrences{src_str}")

    # Save raw sentences with rare tags
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rare_jsonl = OUTPUT_DIR / "rare_sentences.jsonl"
    with open(rare_jsonl, "w", encoding="utf-8") as f:
        for coll, line_str, pairs, rc in rare_sents:
            # Also include surrounding context sentences from same collection
            # (for CRF chain learning)
            json.dump({
                "collection": coll,
                "text": line_str,
                "pairs": pairs,
                "rare_tags": dict(rc),
                "n_tokens": len(pairs),
            }, f, ensure_ascii=False)
            f.write("\n")
    print(f"\n[3] Saved {len(rare_sents):,} rare sentences → {rare_jsonl}")

    # Save statistics
    stats = {
        "total_sents": len(all_sents),
        "rare_sents": len(rare_sents),
        "tag_counts": dict(tag_counter),
        "rare_targets": RARE_TARGETS,
    }
    with open(OUTPUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[4] Statistics saved → {OUTPUT_DIR / 'stats.json'}")
    print(f"\n  Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
