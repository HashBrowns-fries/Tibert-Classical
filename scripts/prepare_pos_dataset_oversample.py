"""
Build Oversampled POS Dataset for Hard Tags
==========================================
Oversample sentences containing hard tags (adj, n.prop, rare v.*)
by 5× so the model sees them more frequently during training.

Usage:
    .venv/bin/python scripts/prepare_pos_dataset_oversample.py
"""

import os, sys, re, json, random, glob
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset

# ── Config ────────────────────────────────────────────────────────────────────

SEGPOS_BASE = Path(__file__).parent.parent / "data" / "segpos_extracted"
SPM_MODEL   = Path(__file__).parent.parent / "model" / "TiBERT-classical-spm-80k" / "final_model"
OUTPUT_DIR  = Path(__file__).parent.parent / "data"  / "corpus" / "pos_dataset_over"

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
DEV_RATIO   = 0.1
TEST_RATIO  = 0.1

# Oversample sentences containing these tags
OVERRARE_TAGS  = {"adj", "n.prop"}
MINRARE_TAGS   = {"v.past.v.pres", "v.fut.v.pres", "v.fut.v.past", "skt"}
OVERSAMPLE_FACTOR = 3
MINRARE_FACTOR    = 2

# Rare tag merging (same as prepare_pos_dataset.py)
RARE_TAG_MERGE = {
    "n.v.invar":        "v.invar",
    "n.v.past":         "v.past",
    "n.v.pres":         "v.pres",
    "n.v.fut":          "v.fut",
    "n.v.neg":          "v.neg",
    "n.v.cop":          "v.cop",
    "n.v.imp":          "v.imp",
    "n.v.aux":          "v.aux",
    "n.v.fut.n.v.pres":"v.fut.v.pres",
    "n.v.fut.n.v.past":"v.fut.v.past",
    "n.v.past.n.v.pres":"v.past.v.pres",
    "adv.intense":       "adj",
    "adv.dir":           "adj",
    "adv.temp":          "adj",
    "adv.mim":           "adj",
    "adv.proclausal":    "adj",
}

TIBETAN_CONSONANT_RE = re.compile(r'[\u0F40-\u0F69\u0F71-\u0F87]')

LABEL_TO_ID: dict = {"O": 0}
LABEL_COUNTER: Counter = Counter()

def simplify_tag(raw_tag: str) -> str:
    return RARE_TAG_MERGE.get(raw_tag, raw_tag)

def _ensure_label(tag: str):
    if tag not in LABEL_TO_ID:
        LABEL_TO_ID[tag] = len(LABEL_TO_ID)
    LABEL_COUNTER[tag] += 1

def is_valid_tibetan_word(word: str) -> bool:
    if not word: return False
    if re.search(r'[A-Za-z0-9]', word): return False
    return bool(TIBETAN_CONSONANT_RE.search(word))

def parse_segpos_line(line: str):
    line = re.sub(r'\s*<utt>\s*$', '', line.strip())
    if not line: return []
    pairs = []
    for chunk in line.split():
        chunk = chunk.strip()
        if not chunk or chunk in ('<utt>', '<utt'): continue
        if '/' in chunk:
            parts = chunk.rsplit('/', 1)
            word = parts[0]; raw_tag = parts[1] if len(parts)==2 else "xxx"
        else:
            word = chunk; raw_tag = "xxx"
        if not word or word in ('p1','p2','p3','p4','p5'): continue
        tag = simplify_tag(raw_tag)
        if tag == "numeral" and not is_valid_tibetan_word(word): continue
        pairs.append((word, tag))
    return pairs

def spm_tokenize_word(word):
    if word == "་": return ["་"]
    if word == "།": return ["།"]
    if word.endswith("་"): return [word[:-1], "་"]
    if word.endswith("།"): return [word[:-1], "།"]
    return [word]

def align_word_tags_to_tokens(word_tags):
    tokens, labels = [], []
    for word, tag in word_tags:
        for sub in spm_tokenize_word(word):
            tokens.append(sub); labels.append(tag)
    return tokens, labels

# ── Tokenizer ─────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))
from continued_pretrain import ClassicalTibetanTokenizer

def load_tokenizer():
    return ClassicalTibetanTokenizer(spm_model_file=str(SPM_MODEL / "spm.model"))

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import time
    t0 = time.time()
    print("=" * 60)
    print("  POS Dataset — Oversampled for Hard Tags")
    print("=" * 60)

    tokenizer = load_tokenizer()
    print(f"  Vocab size: {tokenizer.vocab_size}")

    pos_files = glob.glob(str(SEGPOS_BASE / "**" / "pos" / "*.txt"), recursive=True)
    pos_files = [f for f in pos_files if "__MACOSX" not in f]

    coll_files = {}
    for f in pos_files:
        coll = Path(f).parent.parent.name
        coll_files.setdefault(coll, []).append(f)

    all_records = []
    oversample_counts = Counter()

    for coll_name, files in sorted(coll_files.items()):
        coll_sents = 0
        for f in files:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    pairs = parse_segpos_line(line)
                    if not pairs or len(pairs) < 2: continue
                    tokens, labels = align_word_tags_to_tokens(pairs)
                    if not tokens or len(tokens) < 2: continue

                    tags_in_sent = set(labels)
                    factor = 1
                    if   tags_in_sent & OVERRARE_TAGS: factor = OVERSAMPLE_FACTOR
                    elif tags_in_sent & MINRARE_TAGS:   factor = MINRARE_FACTOR

                    for lbl in labels: _ensure_label(lbl)
                    input_ids = [tokenizer.bos_token_id] \
                              + [tokenizer._convert_token_to_id(tok) for tok in tokens] \
                              + [tokenizer.eos_token_id]
                    label_ids  = [-100] + [LABEL_TO_ID.get(l, 0) for l in labels] + [-100]

                    rec = {"input_ids": input_ids, "labels": label_ids}
                    all_records.append(rec)
                    for _ in range(factor - 1):
                        all_records.append(rec)
                        oversample_counts[factor] += 1
                    coll_sents += 1
        print(f"  {coll_name}: {coll_sents:,} sentences", flush=True)

    ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
    NUM_LABELS = len(LABEL_TO_ID)

    total_extra = sum(oversample_counts.values())
    print(f"\nTotal records (after oversampling): {len(all_records):,}")
    print(f"  Standard (1×): {len(all_records) - total_extra:,}")
    print(f"  Oversampled  (5×): {oversample_counts.get(OVERSAMPLE_FACTOR, 0):,} sentences × {OVERSAMPLE_FACTOR} = {oversample_counts.get(OVERSAMPLE_FACTOR, 0) * (OVERSAMPLE_FACTOR-1):,} extra")
    print(f"  MinRare      (3×): {oversample_counts.get(MINRARE_FACTOR, 0):,} sentences × {MINRARE_FACTOR} = {oversample_counts.get(MINRARE_FACTOR, 0) * (MINRARE_FACTOR-1):,} extra")

    print(f"\n{NUM_LABELS} labels:")
    for lid, lbl in sorted(ID_TO_LABEL.items()):
        print(f"  {lid:3d}: {lbl:25s}  count={LABEL_COUNTER.get(lbl,0):>12,}")

    print("\n[Shuffling...]")
    random.seed(RANDOM_SEED)
    random.shuffle(all_records)
    n = len(all_records)
    n_train = int(n * TRAIN_RATIO)
    n_dev   = int(n * DEV_RATIO)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, start, end in [
        ("train", 0, n_train),
        ("dev",   n_train, n_train + n_dev),
        ("test",  n_train + n_dev, n),
    ]:
        split_data = all_records[start:end]
        if not split_data: continue
        max_len = min(max(len(r["input_ids"]) for r in split_data), 512)
        n_s = len(split_data)
        print(f"  Saving {split_name}: {n_s:,} × {max_len} ...", flush=True)

        ids_arr = np.full((n_s, max_len), tokenizer.pad_token_id, dtype=np.int32)
        labs_arr = np.full((n_s, max_len), -100, dtype=np.int32)
        for i, r in enumerate(split_data):
            ids_arr[i, :min(len(r["input_ids"]), max_len)] = r["input_ids"][:max_len]
            labs_arr[i, :min(len(r["labels"]),  max_len)] = r["labels"][:max_len]
            if i > 0 and i % 500000 == 0:
                print(f"    {i:,} / {n_s:,} ...", flush=True)

        np.save(OUTPUT_DIR / f"{split_name}_input_ids.npy", ids_arr)
        del ids_arr, labs_arr   # free memory immediately
        import gc; gc.collect()

        # Reload for labels (to save memory: do ids and labs separately)
        ids_arr2 = np.load(OUTPUT_DIR / f"{split_name}_input_ids.npy", mmap_mode="r")
        labs_arr2 = np.full((n_s, max_len), -100, dtype=np.int32)
        for i, r in enumerate(split_data):
            labs_arr2[i, :min(len(r["labels"]), max_len)] = r["labels"][:max_len]
        np.save(OUTPUT_DIR / f"{split_name}_labels.npy", labs_arr2)
        del ids_arr2, labs_arr2, split_data
        import gc; gc.collect()
        print(f"    Done: {n_s:,} records  ({time.time()-t0:.0f}s)")

    with open(OUTPUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label_to_id": LABEL_TO_ID, "id_to_label": ID_TO_LABEL, "num_labels": NUM_LABELS}, f, ensure_ascii=False)

    print(f"\nTotal: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
