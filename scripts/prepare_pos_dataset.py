"""
Prepare POS Dataset from SegPOS Files (Full Raw Labels)
======================================================
Parses the full SegPOS corpus and aligns with the SPM tokenizer.
No label simplification — all 91 raw tags preserved.
"""

import os, sys, re, json, random, glob
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────────────────────────

SEGPOS_BASE = Path(__file__).parent.parent / "data" / "segpos_extracted"
SPM_MODEL   = Path(__file__).parent.parent / "model" / "TiBERT-classical-spm-500k" / "final_model"
OUTPUT_DIR  = Path(__file__).parent.parent / "data" / "corpus" / "pos_dataset"
LABEL_MAP_FILE = OUTPUT_DIR / "label_map.json"

MAX_SENTS_PER_COLLECTION = None   # 全量数据
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
DEV_RATIO   = 0.1
TEST_RATIO  = 0.1

# ── Discover all labels from full corpus ─────────────────────────────────────
print("Discovering all labels from corpus...", flush=True)
_pos_files = glob.glob(str(SEGPOS_BASE / "**" / "pos" / "*.txt"), recursive=True)
_pos_files = [f for f in _pos_files if "__MACOSX" not in f]
print(f"  Found {len(_pos_files)} SegPOS files", flush=True)

_all_tags = Counter()
for _f in _pos_files:
    with open(_f, encoding="utf-8", errors="ignore") as _fh:
        for _line in _fh:
            _line = re.sub(r'\s*<utt>\s*$', '', _line.strip())
            for _chunk in _line.split():
                _chunk = _chunk.strip()
                if '/' in _chunk:
                    _all_tags[_chunk.rsplit('/', 1)[1]] += 1

LABEL_TO_ID = {"O": 0}
_offset = 1
for _tag, _ in _all_tags.most_common():
    if _tag not in LABEL_TO_ID:
        LABEL_TO_ID[_tag] = _offset
        _offset += 1

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
NUM_LABELS = len(LABEL_TO_ID)
print(f"\n{NUM_LABELS} raw labels:")
for lid, lbl in sorted(ID_TO_LABEL.items()):
    print(f"  {lid:3d}: {lbl}")
print(flush=True)

# ── Tokenizer ────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))
from continued_pretrain import ClassicalTibetanTokenizer

def load_tokenizer():
    return ClassicalTibetanTokenizer(spm_model_file=str(SPM_MODEL / "spm.model"))

# ── SegPOS Parser ────────────────────────────────────────────────────────────

def simplify_tag(raw_tag: str) -> int:
    """Raw tag → label ID (1-to-1, no simplification)."""
    return LABEL_TO_ID.get(raw_tag, LABEL_TO_ID["O"])

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
        pairs.append((word, simplify_tag(raw_tag)))
    return pairs

def spm_tokenize_word(word: str):
    if word == "་": return ["་"]
    if word == "།": return ["།"]
    if word.endswith("་"): return [word[:-1], "་"]
    if word.endswith("།"): return [word[:-1], "།"]
    return [word]

def align_word_tags_to_tokens(word_tags, tokenizer):
    tokens, labels = [], []
    for word, label_id in word_tags:
        for sub in spm_tokenize_word(word):
            tokens.append(sub)
            labels.append(label_id)
    return tokens, labels

def encode_tokens(tokens, labels, tokenizer):
    ids = [tokenizer._convert_token_to_id(tok) for tok in tokens]
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    return [bos] + ids + [eos], [-100] + labels + [-100]

# ── Dataset ──────────────────────────────────────────────────────────────────

class PosDataset(Dataset):
    def __init__(self, split: str, data_dir: Path = OUTPUT_DIR):
        self.input_ids = np.load(data_dir / f"{split}_input_ids.npy", mmap_mode="r")
        self.labels    = np.load(data_dir / f"{split}_labels.npy",     mmap_mode="r")
        self.size = len(self.input_ids)
    def __len__(self): return self.size
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "labels":    torch.tensor(self.labels[idx],     dtype=torch.long),
        }

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import time
    t0 = time.time()
    print("=" * 60)
    print("  SegPOS Dataset — Full Raw Labels (Streaming)")
    print("=" * 60)

    print("\n[1] Loading tokenizer...")
    tokenizer = load_tokenizer()
    print(f"  Vocab size: {tokenizer.vocab_size}")

    print("\n[2] Collecting all records (streaming pass)...")
    pos_files = glob.glob(str(SEGPOS_BASE / "**" / "pos" / "*.txt"), recursive=True)
    pos_files = [f for f in pos_files if "__MACOSX" not in f]

    coll_files = {}
    for f in pos_files:
        coll = Path(f).parent.parent.name
        coll_files.setdefault(coll, []).append(f)

    # First pass: collect all records (no limit)
    all_records = []
    tag_stats = Counter()
    skipped = 0
    n_files = 0

    for coll_name, files in sorted(coll_files.items()):
        coll_sents = 0
        for f in files:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    pairs = parse_segpos_line(line)
                    if not pairs or len(pairs) < 2:
                        skipped += 1
                        continue
                    tokens, labels = align_word_tags_to_tokens(pairs, tokenizer)
                    if not tokens or len(tokens) < 2:
                        skipped += 1
                        continue
                    input_ids, label_ids = encode_tokens(tokens, labels, tokenizer)
                    all_records.append({
                        "input_ids": input_ids,
                        "labels": label_ids,
                    })
                    for lid in labels:
                        tag_stats[ID_TO_LABEL[lid]] += 1
                    coll_sents += 1
            n_files += 1
        print(f"  {coll_name}: {coll_sents} sentences", flush=True)

    print(f"\nTotal records: {len(all_records):,}  (skipped {skipped:,})  [{time.time()-t0:.0f}s]")
    print("\nTop-25 labels:")
    for tag, count in tag_stats.most_common(25):
        print(f"  {tag:25s}: {count:>12,d}")

    print("\n[3] Shuffling...", flush=True)
    random.seed(RANDOM_SEED)
    random.shuffle(all_records)
    n = len(all_records)
    n_train = int(n * TRAIN_RATIO)
    n_dev   = int(n * DEV_RATIO)
    splits = {
        "train": (all_records[:n_train], "input_ids", "labels"),
        "dev":   (all_records[n_train:n_train + n_dev], "input_ids", "labels"),
        "test":  (all_records[n_train + n_dev:], "input_ids", "labels"),
    }
    print(f"  train={n_train:,}, dev={n_dev:,}, test={n - n_train - n_dev:,}")

    print("\n[4] Saving numpy arrays...", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, (data, ids_key, labs_key) in splits.items():
        if not data:
            print(f"  Skipping {split_name} (empty)")
            continue
        # Compute max_len
        max_len = max(len(r[ids_key]) for r in data)
        max_len = min(max_len, 512)
        n = len(data)
        print(f"  Saving {split_name}: {n:,} × {max_len} ...", flush=True)
        ids_arr = np.full((n, max_len), tokenizer.pad_token_id, dtype=np.int32)
        labs_arr = np.full((n, max_len), -100, dtype=np.int32)
        for i, r in enumerate(data):
            ids_arr[i, :min(len(r[ids_key]), max_len)] = r[ids_key][:max_len]
            labs_arr[i, :min(len(r[labs_key]), max_len)] = r[labs_key][:max_len]
        np.save(OUTPUT_DIR / f"{split_name}_input_ids.npy", ids_arr)
        np.save(OUTPUT_DIR / f"{split_name}_labels.npy",     labs_arr)
        print(f"    Saved {split_name}: {ids_arr.shape}  ({time.time()-t0:.0f}s elapsed)")

    # Metadata (first 100k of train only)
    print("\n[5] Saving metadata...", flush=True)
    for split_name, (data, _, _) in splits.items():
        limit = 100_000 if split_name == "train" else 20_000
        with open(OUTPUT_DIR / f"{split_name}_meta.jsonl", "w", encoding="utf-8") as f:
            for r in data[:limit]:
                json.dump({"tokens": r["tokens"] if "tokens" in r else [], "collection": ""}, f, ensure_ascii=False)
                f.write("\n")

    # Label map
    with open(LABEL_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump({"label_to_id": LABEL_TO_ID, "id_to_label": ID_TO_LABEL, "num_labels": NUM_LABELS}, f, ensure_ascii=False)

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print(f"Done! Total records: {len(all_records):,}")

if __name__ == "__main__":
    main()
