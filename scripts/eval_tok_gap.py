"""
评估推理时分词策略对 POS 准确率的影响。
对 SegPOS 句子运行评估，计算 Accuracy / Weighted F1 / Macro F1。

训练时的分词方式（来自 prepare_pos_dataset.py）：
  每个 SegPOS word 经过 spm_tokenize_word(word):
    word == "་"        → ["་"]
    word == "།"        → ["།"]
    word.endswith("་") → [word[:-1], "་"]
    word.endswith("།") → [word[:-1], "།"]
    else               → [word]   (SPM subword within word, word-level longest-match)

三种推理策略：
  GT         — 逐 word 应用 spm_tokenize_word（与训练完全一致）
  Greedy     — 拼接全句音节后做 full-text longest-match（跨 shad 边界）
  Corrected  — 按 shad 切分段，段内逐 word 做 spm_tokenize_word

用法：
  .venv/bin/python scripts/eval_tok_gap.py [--max-sents N]
"""
import argparse, json, re, sys, torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from continued_pretrain import ClassicalTibetanTokenizer

MODEL_DIR   = Path(__file__).parent.parent / "model" / "TiBERT-classical-spm-500k" / "final_model"
SPM_MODEL   = MODEL_DIR / "spm.model"
CKPT_PATH   = Path(__file__).parent.parent / "model" / "pos_classifier" / "crf_supcon" / "best_model.pt"
LABEL_MAP   = CKPT_PATH.parent / "test_results.json"
SEGPOS_DIR  = Path(__file__).parent.parent / "data" / "segpos_extracted"


# ── Model ─────────────────────────────────────────────────────────────────────

class PosTagger(torch.nn.Module):
    def __init__(self, vocab_size=32007, num_labels=77, max_len=512):
        super().__init__()
        from transformers import BertConfig, BertModel
        self.bert = BertModel(BertConfig(
            vocab_size=vocab_size, hidden_size=768, num_hidden_layers=12,
            num_attention_heads=12, intermediate_size=3072,
            max_position_embeddings=max_len, pad_token_id=0,
        ))
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.last_hidden_state))


# ── SPM tokenization per word (mirrors prepare_pos_dataset.py) ───────────────

def spm_tokenize_word(word, tokenizer):
    """Exactly replicates prepare_pos_dataset.py spm_tokenize_word()."""
    if word == "་":
        return ["་"]
    if word == "།":
        return ["།"]
    if word.endswith("་"):
        root = word[:-1]
        # Longest-match on root within this word
        return _spm_longest_word(root, tokenizer) + ["་"]
    if word.endswith("།"):
        root = word[:-1]
        return _spm_longest_word(root, tokenizer) + ["།"]
    return _spm_longest_word(word, tokenizer)


def _spm_longest_word(text, tokenizer):
    """Longest-match within a single word (no shad crossing)."""
    unk = tokenizer._token2id.get("[UNK]")
    tokens, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch in ("་", "།", "༔"):
            tokens.append(ch); i += 1; continue
        best, max_len = None, min(10, len(text) - i)
        for n in range(max_len, 0, -1):
            sub = text[i:i+n]
            if any(c in ("་", "།", "༔") for c in sub):
                continue
            if tokenizer._convert_token_to_id(sub) != unk:
                best = sub; break
        tokens.append(best if best else text[i])
        i += len(best) if best else 1
    return tokens


# ── Three strategies ──────────────────────────────────────────────────────────

def tokenize_gt(parsed_words, tokenizer):
    """
    GT: each SegPOS word → spm_tokenize_word (same as training).
    Returns list of (token, tag) aligned with model predictions.
    """
    tokens, labels = [], []
    for word, tag in parsed_words:
        subtoks = spm_tokenize_word(word, tokenizer)
        for sub in subtoks:
            tokens.append(sub)
            labels.append(tag)
    return tokens, labels


def tokenize_greedy(raw_sylls, tokenizer):
    """
    Greedy: join all syllables, full-text longest-match.
    Labels:  shad/tsek → 'punc', stripped syllable → first non-sep prediction.
    """
    text = "".join(raw_sylls)
    raw_toks = _spm_longest_full(text, tokenizer)
    # Assign labels: stripped syllable → first token in that syllable
    stripped_sylls = [s for s in raw_sylls if s not in ("་", "།", "༔")]
    return raw_toks, stripped_sylls


def tokenize_corrected(raw_sylls, tokenizer):
    """
    Corrected: split on shad → within each segment, apply word-level SPM.
    """
    text = "".join(raw_sylls)
    # Split on shad/tsek separators
    parts = re.split(r'(་|།|༔)', text)
    tokens = []
    for part in parts:
        if not part:
            continue
        if part in ("་", "།", "༔"):
            tokens.append(part)
        else:
            # Longest-match within this bare segment
            seg_toks = _spm_longest_full(part, tokenizer)
            tokens.extend(seg_toks)
    stripped_sylls = [s for s in raw_sylls if s not in ("་", "།", "༔")]
    return tokens, stripped_sylls


def _spm_longest_full(text, tokenizer):
    """Full-text longest-match (used by Greedy)."""
    unk = tokenizer._token2id.get("[UNK]")
    tokens, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch in ("་", "།", "༔"):
            tokens.append(ch); i += 1; continue
        best, max_len = None, min(10, len(text) - i)
        for n in range(max_len, 0, -1):
            sub = text[i:i+n]
            if any(c in ("་", "།", "༔") for c in sub):
                continue
            if tokenizer._convert_token_to_id(sub) != unk:
                best = sub; break
        tokens.append(best if best else text[i])
        i += len(best) if best else 1
    return tokens


# ── Align model predictions → gold labels ────────────────────────────────────

def align_tokens_to_syllables(syllables, spm_tokens, preds):
    """
    Map SPM predictions back to stripped syllables.
    For GT: stripped syllables and tokens are 1-to-1 (direct return).
    For Greedy/Corrected: multi-char Tibetan words may fragment into
    multiple tokens. We scan tokens sequentially and greedily match each
    stripped syllable to the first token that starts within it.
    Returns list of predicted label IDs (one per stripped syllable).
    """
    stripped_sylls = [s for s in syllables if s not in ("་", "།", "༔")]
    n = len(stripped_sylls)

    # Build non-sep token sequence
    seq = [(t, p) for t, p in zip(spm_tokens, preds) if t not in ("་", "།", "༔")]

    # Direct 1-to-1 when counts match (GT case)
    if len(seq) == n:
        return [p for _, p in seq]

    aligned = []
    tok_i = 0
    for syll in stripped_sylls:
        if tok_i >= len(seq):
            aligned.append(preds[-1] if preds else 0)
            continue

        tok, pred = seq[tok_i]

        if tok == syll:
            # Exact match
            aligned.append(pred); tok_i += 1
        elif syll.startswith(tok) or tok in syll:
            # tok is inside or at start of syllable → use tok's prediction
            aligned.append(pred); tok_i += 1
        else:
            # Mismatch: look ahead to find the first token that belongs to this syllable
            found = False
            for look in range(tok_i + 1, min(tok_i + 8, len(seq))):
                look_tok, look_pred = seq[look]
                if look_tok == syll or syll.startswith(look_tok) or look_tok in syll:
                    aligned.append(look_pred); tok_i = look + 1; found = True; break
            if not found:
                aligned.append(pred); tok_i += 1

    while len(aligned) < n:
        aligned.append(preds[-1] if preds else 0)

    return aligned[:n]


# ── SegPOS parser (mirrors prepare_pos_dataset.py) ───────────────────────────

SIMPLE_TAGS = {
    "n.v.invar": "n.v.invar", "n.v.past": "n.v.past",
    "n.v.pres": "n.v.pres", "n.v.fut": "n.v.fut",
    "n.v.fut.n.v.pres": "n.v.fut.n.v.pres", "n.v.past.n.v.pres": "n.v.past.n.v.pres",
    "n.v.fut.n.v.past": "n.v.fut.n.v.past",
    "v.past.v.pres": "v.past.v.pres",
    "v.fut.v.pres": "v.fut.v.pres", "v.fut.v.past": "v.fut.v.past",
}

def simplify_tag(t):
    return SIMPLE_TAGS.get(t, t)


def parse_segpos_line(line):
    """Parse SegPOS line → [(word, tag)] matching prepare_pos_dataset.py."""
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
        if not word or word in ('p1', 'p2', 'p3', 'p4', 'p5'):
            continue
        pairs.append((word, simplify_tag(raw_tag)))
    return pairs


def parse_segpos_sylls(line):
    """Parse SegPOS line → (raw_sylls, gold_labels) matching spm_tokenize_word output."""
    pairs = parse_segpos_line(line)
    sylls, labs = [], []
    for word, tag in pairs:
        if word == "་":
            sylls.append(word); labs.append("punc")
        elif word == "།":
            sylls.append(word); labs.append("punc")
        elif word.endswith("་"):
            sylls.append(word[:-1]); labs.append(tag)
            sylls.append("་"); labs.append("punc")
        elif word.endswith("།"):
            sylls.append(word[:-1]); labs.append(tag)
            sylls.append("།"); labs.append("punc")
        else:
            sylls.append(word); labs.append(tag)
    return sylls, labs


# ── Model inference ──────────────────────────────────────────────────────────

def predict_tokens(tokens, tokenizer, model, device):
    """Run model on token list → list of predicted label IDs (one per token)."""
    ids = [tokenizer.bos_token_id] + [tokenizer._convert_token_to_id(t) for t in tokens] + [tokenizer.eos_token_id]
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(ids_t, attention_mask=(ids_t != 0).long())
        preds = logits.argmax(dim=-1).squeeze(0).cpu().tolist()[1:-1]
    return preds


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred_strs, gold_strs):
    total = len(gold_strs)
    correct = sum(1 for p, g in zip(pred_strs, gold_strs) if p == g)
    acc = correct / total if total > 0 else 0

    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    for p_str, g_str in zip(pred_strs, gold_strs):
        stats[g_str]["support"] += 1
        if p_str == g_str:
            stats[g_str]["tp"] += 1
        else:
            stats[g_str]["fn"] += 1
            stats[p_str]["fp"] += 1

    results = []
    for label, s in stats.items():
        p = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] > 0 else 0
        r = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] > 0 else 0
        f = 2 * p * r / (p + r) if p + r > 0 else 0
        results.append({"label": label, "precision": p, "recall": r, "f1": f, "support": s["support"]})

    total_support = sum(r["support"] for r in results)
    wf1 = sum(r["f1"] * r["support"] for r in results) / max(total_support, 1)
    mf1 = sum(r["f1"] for r in results) / max(len(results), 1)
    return acc, wf1, mf1


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sents", type=int, default=None)
    ap.add_argument("--strategies", type=str, default="all")
    args = ap.parse_args()

    strat_list = ["GT", "Greedy", "Corrected"] if args.strategies == "all" else args.strategies.split(",")

    print("Loading tokenizer...")
    tokenizer = ClassicalTibetanTokenizer(str(SPM_MODEL))

    print("Loading label map...")
    with open(LABEL_MAP, encoding="utf-8") as f:
        lm = json.load(f)
    id2label = {}
    for k, v in lm.get("label_stats", {}).items():
        id2label[int(k)] = v["label"]
    id2label[0] = "O"  # padding / O tag, not UNK

    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PosTagger(vocab_size=32007, num_labels=77, max_len=512)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    clean = {}
    for k, v in state.items():
        k = k.replace("module.", "").replace("bert.", "")
        clean[k] = v
    bert_keys = {k: v for k, v in clean.items()
                 if k.startswith("embeddings") or k.startswith("encoder") or k.startswith("pooler")}
    model.bert.load_state_dict(bert_keys, strict=False)
    cls_keys = {k: v for k, v in clean.items() if "classifier" in k}
    cls_clean = {k.replace("classifier.", ""): v for k, v in cls_keys.items()}
    model.classifier.load_state_dict(cls_clean, strict=True)
    model.to(device); model.eval()
    print(f"  Device: {device}")

    print("\nLoading SegPOS sentences...")
    import glob
    sentences = []
    for pos_file in glob.glob(str(SEGPOS_DIR / "**" / "pos" / "*.txt"), recursive=True):
        if "__MACOSX" in pos_file:
            continue
        with open(pos_file, encoding="utf-8", errors="ignore") as f:
            for line in f:
                parsed = parse_segpos_line(line)
                sylls, labs = parse_segpos_sylls(line)
                if len(parsed) >= 2:  # at least 2 words
                    sentences.append((parsed, sylls, labs))
                if args.max_sents and len(sentences) >= args.max_sents:
                    break
        if args.max_sents and len(sentences) >= args.max_sents:
            break
    print(f"  {len(sentences):,} sentences loaded")

    all_results = {}
    for strat in strat_list:
        print(f"\n--- {strat} ---")
        all_pred_strs, all_lab_strs = [], []

        for parsed, sylls, labs in tqdm(sentences, desc=strat):
            stripped_sylls = [s for s in sylls if s not in ("་", "།", "༔")]
            # Build gold labels for stripped syllables: each non-sep word gets its tag
            gold_strs_list = []
            for s, l in zip(sylls, labs):
                if s not in ("་", "།", "༔"):
                    gold_strs_list.append(l)

            if strat == "GT":
                # GT: word-level SPM (same as training)
                tokens, gold_labels = tokenize_gt(parsed, tokenizer)
                preds = predict_tokens(tokens, tokenizer, model, device)
                # tokens and gold_labels are 1-to-1; strip seps for comparison
                nonsep_preds = [(t, p) for t, p in zip(tokens, preds) if t not in ("་", "།", "༔")]
                nonsep_golds = [(t, l) for t, l in zip(tokens, gold_labels) if t not in ("་", "།", "༔")]
                pred_ids = [p for _, p in nonsep_preds]
                pred_strs = [id2label.get(p, f"UNK_{p}") for p in pred_ids]
                gold_strs = [l for _, l in nonsep_golds]

            elif strat == "Greedy":
                tokens, _ = tokenize_greedy(sylls, tokenizer)
                preds = predict_tokens(tokens, tokenizer, model, device)
                preds_aligned = align_tokens_to_syllables(sylls, tokens, preds)
                pred_strs = [id2label.get(p, f"UNK_{p}") for p in preds_aligned]
                gold_strs = gold_strs_list[:len(pred_strs)]

            else:  # Corrected
                tokens, _ = tokenize_corrected(sylls, tokenizer)
                preds = predict_tokens(tokens, tokenizer, model, device)
                preds_aligned = align_tokens_to_syllables(sylls, tokens, preds)
                pred_strs = [id2label.get(p, f"UNK_{p}") for p in preds_aligned]
                gold_strs = gold_strs_list[:len(pred_strs)]

            all_pred_strs.extend(pred_strs)
            all_lab_strs.extend(gold_strs)

        acc, wf1, mf1 = compute_metrics(all_pred_strs, all_lab_strs)
        all_results[strat] = {"accuracy": acc, "weighted_f1": wf1, "macro_f1": mf1}
        print(f"  Acc={acc:.4f}  WF1={wf1:.4f}  MF1={mf1:.4f}")

    print("\n" + "="*60)
    print(f"{'Strategy':<12} {'Accuracy':>9} {'W-F1':>9} {'M-F1':>9}")
    print("-"*60)
    for strat, m in all_results.items():
        print(f"{strat:<12} {m['accuracy']:>8.2%} {m['weighted_f1']:>8.2%} {m['macro_f1']:>8.2%}")
    print("="*60)

    out_path = CKPT_PATH.parent / "tok_gap_eval.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
