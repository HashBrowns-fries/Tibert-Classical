"""
Improved POS Evaluator v2
==========================
改进点：
1. Cross-collection 分析：每个 SegPOS 子集单独报告
2. Class-group 指标：名词/动词/形容词/格助词分组 F1
3. Compound word 测试：已知复合词的标准切分基准
4. Qualitative spot-check：打印具体句子的预测，人工可审查
5. Out-of-domain 测试：用真实文本（非 SegPOS 语料）测试
6. 困难类优先指标：n.prop、adj、v.* 等稀有类单独追踪

Usage:
  .venv/bin/python scripts/eval_pos_model_v2.py [--checkpoint PATH] [--qualitative]
  .venv/bin/python scripts/eval_pos_model_v2.py --ood "བོད་གི་ཡུལ་ལྷོ་ལ་སོང་"
"""

import argparse, json, sys, pickle, re, torch, numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from train_pos_classifier import PosClassifier

DATA_DIR  = Path(__file__).parent.parent / "data" / "corpus" / "pos_dataset"
CKPT_PATH = Path(__file__).parent.parent / "model" / "pos_classifier" / "crf_supcon" / "best_model.pt"

# ── Tag groupings ───────────────────────────────────────────────────────────────

CASE_PARTICLES = {f"case.{s}" for s in
    "gen agn all abl ela ass term loc comp nare".split()}

NOUN_TAGS  = {
    "n.count","n.prop","n.rel","n.mass",
}
VERB_TAGS  = {t for t in [
    "v.past","v.pres","v.fut","v.invar","v.neg","v.aux",
    "v.imp","v.cop","v.fut.v.pres","v.past.v.pres","v.fut.v.past",
]}

# Tags that are hard (rare or confused) — these matter most for quality
HARD_TAGS = {
    "n.prop",    # 专有名词：OOV 问题最严重
    "adj",       # 形容词：样本少，边界模糊
    "v.past", "v.pres", "v.fut",  # 动词时态互相混淆
    "n.rel",     # 关系名词
}

# Compound words with known correct POS — hand-curated test cases
# Format: (text, expected_tag_sequence)
COMPOUND_TEST_CASES = [
    # (复合词, 是否应为 n.prop/专有名词)
    ("བཅོམ་ལྡན་འདས",  ["n.prop", "n.prop", "n.prop"]),   # 佛（世尊）
    ("རྒྱལ་པོ",          ["n.prop"]),                        # 王
    ("ཆེན་པོ",            ["adj"]),                          # 大
    ("སངས་རྒྱས",          ["v.past"]),                      # 成佛
    ("གཤེགས་པ",           ["v.past"]),                      # 入灭
    ("བོད་པ",              ["n.prop"]),                      # 藏人/藏
    ("ཆོས་སྐུལ",          ["n.mass", "v.pres"]),            # 法鼓动
    ("དགེ་འདུན",          ["n.mass"]),                      # 僧伽
    ("རྣམ་པར་ཐར་པ",      ["n.prop"]),                     # 涅槃
    ("སྔོན་ཚོད",          ["n.mass"]),                      # 预言
]

# ── Config (must match train_pos_classifier.py) ────────────────────────────────

class Cfg:
    hidden_size = 768; vocab_size = 80007; encoder_freeze = False; num_labels = 77
    batch_size = 128; gradient_accum = 16; lr_head = 5e-4; lr_encoder = 2e-5
    weight_decay = 0.01; warmup_ratio = 0.1; max_epochs = 20; early_stop_pat = 3
    unfreeze_layers = 6; use_fp16 = True; case_weight = 2.0; focal_gamma = 2.0
    use_crf = True
    use_contrastive = True
    contrastive_temp = 0.1
    contrastive_weight = 0.1
    ens_beta = 0.9999
    supcon_criterion = "supcon"
    max_len = 512; seed = 42; drop_punct_prob = 0.3


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _f1(p, r):
    return 2*p*r/(p+r) if p+r > 0 else 0

def compute_stats(preds, labels, id_to_label):
    """Return {label: {tp, fp, fn, support}}"""
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    for p, t in zip(preds.tolist(), labels.tolist()):
        lt = id_to_label.get(int(t), f"?{t}")
        lp = id_to_label.get(int(p), f"?{p}")
        stats[lt]["support"] += 1
        if lp == lt:
            stats[lt]["tp"] += 1
        else:
            stats[lt]["fn"] += 1
            stats[lp]["fp"] += 1
    return stats

def group_f1(stats, tag_group):
    """Weighted F1 for a tag group."""
    filtered = [s for lbl, s in stats.items() if lbl in tag_group]
    total_sup = sum(s["support"] for s in filtered)
    if total_sup == 0:
        return 0.0, 0
    tp = sum(s["tp"] for s in filtered)
    fp = sum(s["fp"] for s in filtered)
    fn = sum(s["fn"] for s in filtered)
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    return _f1(p, r), total_sup

def per_class_results(stats, min_support=20):
    """Return sorted list of per-class metrics."""
    results = []
    for label, s in stats.items():
        p = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] > 0 else 0
        r = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] > 0 else 0
        f = _f1(p, r)
        results.append({"label": label, "precision": p, "recall": r, "f1": f,
                        "support": s["support"]})
    return sorted(results, key=lambda x: x["f1"])

def print_bar(f1, width=20):
    filled = int(f1 * width)
    return "█" * filled + "░" * (width - filled)


# ── Dataset ─────────────────────────────────────────────────────────────────────

class EvalDS(Dataset):
    def __init__(self, split, max_len=512, max_samples=None):
        self.ids  = np.load(DATA_DIR / f"{split}_input_ids.npy",  mmap_mode="r")
        self.labs = np.load(DATA_DIR / f"{split}_labels.npy",      mmap_mode="r")
        self._size = min(max_samples, len(self.ids)) if max_samples else len(self.ids)
        self.max_len = max_len
    def __len__(self): return self._size
    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.ids[i, :self.max_len].copy(), dtype=torch.long),
            "labels":    torch.tensor(self.labs[i, :self.max_len].copy(), dtype=torch.long),
        }


# ── Cross-collection eval ───────────────────────────────────────────────────────

def eval_cross_collection(model, id_to_label, device, max_per_coll=5000):
    """Evaluate per SegPOS collection to expose distribution gaps."""
    import glob
    from train_pos_classifier import PosDataset

    print("\n" + "="*60)
    print("  [A] Cross-Collection Analysis")
    print("="*60)

    segpos_base = Path(__file__).parent.parent / "data" / "segpos_extracted"
    coll_dirs = sorted([d for d in segpos_base.iterdir()
                        if d.is_dir() and "__MACOSX" not in str(d)])

    sys.path.insert(0, str(Path(__file__).parent))
    from continued_pretrain import ClassicalTibetanTokenizer

    spm_model = (Path(__file__).parent.parent /
                  "model" / "TiBERT-classical-spm-80k" / "final_model" / "spm.model")
    tokenizer = ClassicalTibetanTokenizer(str(spm_model))
    id_to_label_inv = {v: k for k, v in id_to_label.items()}

    coll_results = []
    for coll_dir in coll_dirs:
        pos_files = glob.glob(str(coll_dir / "**" / "pos" / "*.txt"), recursive=True)
        pos_files = [f for f in pos_files if "__MACOSX" not in f]
        if not pos_files:
            continue

        # Load a sample of sentences from this collection
        sents = []
        for f in pos_files[:20]:  # sample first 20 files
            with open(f, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if line and len(line.split()) >= 3:
                        sents.append(line)
                    if len(sents) >= max_per_coll:
                        break
            if len(sents) >= max_per_coll:
                break

        if len(sents) < 50:
            continue

        # Encode
        all_preds, all_labels = [], []
        for sent in sents:
            tokens, labels = parse_segpos_sent(sent, tokenizer)
            if len(tokens) < 2:
                continue
            ids = [tokenizer.bos_token_id] + [tokenizer._convert_token_to_id(t) for t in tokens] + [tokenizer.eos_token_id]
            labs = [-100] + [id_to_label_inv.get(l, 0) for l in labels] + [-100]

            inp = torch.tensor([ids], dtype=torch.long, device=device)
            msk = (inp != 0).long()
            with torch.no_grad():
                out = model(input_ids=inp, attention_mask=msk)
            preds = out["logits"].argmax(dim=-1).squeeze(0).cpu().tolist()[1:-1]
            labs  = labs[1:-1]

            for p, l in zip(preds, labs):
                all_preds.append(p)
                all_labels.append(l)

        if not all_preds:
            continue

        preds_t = torch.tensor(all_preds)
        labels_t = torch.tensor(all_labels)
        stats = compute_stats(preds_t, labels_t, id_to_label)

        coll_wf1 = sum(
            _f1(s["tp"]/(s["tp"]+s["fp"]) if s["tp"]+s["fp"]>0 else 0,
                s["tp"]/(s["tp"]+s["fn"]) if s["tp"]+s["fn"]>0 else 0) * s["support"]
            for s in stats.values()
        ) / max(sum(s["support"] for s in stats.values()), 1)

        hard_f1, hard_sup = group_f1(stats, HARD_TAGS)
        nprop_f1 = next((_f1(s["tp"]/(s["tp"]+s["fp"]) if s["tp"]+s["fp"]>0 else 0,
                              s["tp"]/(s["tp"]+s["fn"]) if s["tp"]+s["fn"]>0 else 0)
                         for l, s in stats.items() if l == "n.prop"), (0.0, 0))

        coll_results.append({
            "collection": coll_dir.name,
            "n_sents": len(sents),
            "wf1": coll_wf1,
            "hard_f1": hard_f1,
            "hard_sup": hard_sup,
        })
        print(f"  {coll_dir.name:<40} WF1={coll_wf1:.4f}  困难类F1={hard_f1:.4f}  ({hard_sup}词)")

    return coll_results


def parse_segpos_sent(line, tokenizer):
    """Parse a SegPOS-formatted line into (tokens, labels)."""
    import re
    line = re.sub(r'\s*<utt>\s*$', '', line.strip())
    if not line:
        return [], []
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
        if not word:
            continue
        # Handle separators
        if word in ('་', '།'):
            pairs.append((word, raw_tag))
        elif word.endswith('་'):
            pairs.append((word[:-1], raw_tag))
            pairs.append(('་', 'punc'))
        elif word.endswith('།'):
            pairs.append((word[:-1], raw_tag))
            pairs.append(('།', 'punc'))
        else:
            pairs.append((word, raw_tag))
    tokens = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]
    return tokens, labels


# ── Qualitative spot-check ──────────────────────────────────────────────────────

TAG_DESCRIPTIONS = {
    "O": "Outside",
    "n.count": "名词(普通)", "n.prop": "专有名词", "n.rel": "关系名词", "n.mass": "物质名词",
    "v.past": "动词(过去)", "v.pres": "动词(现在)", "v.fut": "动词(将来)",
    "v.invar": "动词(不变)", "v.neg": "否定动词", "v.aux": "助动词",
    "v.imp": "命令式", "v.cop": "系词",
    "adj": "形容词", "punc": "标点",
}
CASE_PARTICLES_DESC = {
    "case.gen": "属格(的)", "case.agn": "作格(由)", "case.all": "为格(对)",
    "case.abl": "离格(从)", "case.loc": "处格(在)", "case.term": "终结格(至)",
    "case.ass": "共同格(与)", "case.comp": "比格(比)",
}

def qualitative_eval(model, tokenizer, id_to_label, device):
    """Print human-readable predictions on example sentences."""
    print("\n" + "="*60)
    print("  [B] Qualitative Spot-Check (人工审查)")
    print("="*60)

    examples = [
        "བོད་གི་ཡུལ་ལྷོ་ལ་སོང་",
        "བཅོམ་ལྡན་འདས་གཤེགས་སོ།",
        "དགེ་འདུན་གྱི་ཆོས་སྐུལ་དེ་ཡིན",
        "རྒྱལ་པོ་ཆེན་པོས་ཆོས་འོས་པར་གྲུབ་པ་ཡིན",
        "སངས་རྒྱས་བྱོན་པ་ལ་ཕྱག་འཚལ་ལོ།",
    ]

    for text in examples:
        syllable_ids = tokenize_spm(text, tokenizer)
        ids = [tokenizer.bos_token_id] + syllable_ids + [tokenizer.eos_token_id]
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        msk = (inp != 0).long()

        with torch.no_grad():
            out = model(input_ids=inp, attention_mask=msk)
        pred_ids = out["logits"].argmax(dim=-1).squeeze(0).cpu().tolist()[1:-1]

        print(f"\n  输入: {text}")
        print(f"  分词: {' | '.join(tokenizer.convert_ids_to_tokens(syllable_ids))}")
        print(f"  {'─'*50}")
        tokens_str = tokenizer.convert_ids_to_tokens(syllable_ids)
        for tok, pid in zip(tokens_str, pred_ids):
            label = id_to_label.get(pid, "?")
            is_sep = tok in ("་", "།")
            if label.startswith("case."):
                desc = CASE_PARTICLES_DESC.get(label, label)
                flag = " ★"
            elif label == "adj":
                desc = "形容词"; flag = " ●"
            elif label.startswith("n."):
                desc = "名词"; flag = ""
            elif label.startswith("v."):
                desc = "动词"; flag = ""
            elif is_sep:
                desc = "(分隔)"; flag = ""
            else:
                desc = label; flag = ""
            print(f"    {tok:<10s} → {label:<15s} {desc}{flag}")


def tokenize_spm(text, tokenizer):
    """Split text into SPM syllable tokens."""
    tokens = []
    i = 0
    while i < len(text):
        best = None
        for n in range(min(10, len(text) - i), 0, -1):
            sub = text[i:i+n]
            tid = tokenizer._convert_token_to_id(sub)
            unk_id = tokenizer._token2id.get("[UNK]")
            if tid != unk_id:
                best = sub
                break
        if best is None:
            best = text[i]
        tokens.append(best)
        i += len(best)
    return [tokenizer._convert_token_to_id(t) for t in tokens]


# ── Compound word analysis ──────────────────────────────────────────────────────

def eval_compound_words(model, tokenizer, id_to_label, device):
    """Test known compound words: is the tokenizer treating them as units?"""
    print("\n" + "="*60)
    print("  [C] Compound Word Test (复合词切分测试)")
    print("="*60)
    print("  目的：验证 80K tokenizer 是否把复合词作为整体处理")
    print("  " + "─"*55)

    for text, _ in COMPOUND_TEST_CASES:
        ids = [tokenizer.bos_token_id] + tokenize_spm(text, tokenizer) + [tokenizer.eos_token_id]
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        msk = (inp != 0).long()

        with torch.no_grad():
            out = model(input_ids=inp, attention_mask=msk)
        pred_ids = out["logits"].argmax(dim=-1).squeeze(0).cpu().tolist()[1:-1]
        tokens = tokenizer.convert_ids_to_tokens(ids[1:-1])

        # How many tokens did SPM produce?
        n_tokens = len(tokens)

        # Check if the word is fragmented
        is_frag = n_tokens > 1

        tags = [id_to_label.get(p, "?") for p in pred_ids]

        flag = "⚠ 碎片化" if is_frag else "✓ 完整"
        print(f"\n  {text}")
        print(f"    SPM: {' | '.join(tokens)}")
        print(f"    标注: {' | '.join(tags)}")
        print(f"    {flag} ({n_tokens} token{'s' if n_tokens > 1 else ''})")


# ── Class-group F1 report ─────────────────────────────────────────────────────

def print_class_group_report(stats, id_to_label):
    """Print F1 broken down by semantic group."""
    print("\n" + "="*60)
    print("  [D] Class-Group F1 Report")
    print("="*60)

    groups = {
        "★ 格助词 (case.*)": CASE_PARTICLES,
        "● 名词 (n.*)": NOUN_TAGS,
        "◆ 动词 (v.*)": VERB_TAGS,
        "✗ 困难类 (n.prop/adj/v.*)": HARD_TAGS,
    }

    for group_name, group_tags in groups.items():
        f1, sup = group_f1(dict(stats), group_tags)
        p_sum = sum(1 for lbl in stats if lbl in group_tags)
        print(f"  {group_name}")
        print(f"    WF1={f1:.4f}  ({sup} 词 / {p_sum} 标签类型)")

        # Per-tag in group
        for label, s in sorted(stats.items(), key=lambda x: x[0]):
            if label not in group_tags:
                continue
            p = s["tp"]/(s["tp"]+s["fp"]) if s["tp"]+s["fp"]>0 else 0
            r = s["tp"]/(s["tp"]+s["fn"]) if s["tp"]+s["fn"]>0 else 0
            f = _f1(p, r)
            bar = print_bar(f)
            icon = "✓" if f >= 0.80 else "○" if f >= 0.50 else "✗"
            print(f"    {icon} {bar} {label:<18} F1={f:.3f} P={p:.3f} R={r:.3f}  n={s['support']:,}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Improved POS Evaluator v2")
    ap.add_argument("--checkpoint", default=str(CKPT_PATH))
    ap.add_argument("--max-samples", type=int, default=20000)
    ap.add_argument("--split", default="test")
    ap.add_argument("--qualitative", action="store_true", help="Show qualitative spot-checks")
    ap.add_argument("--no-cross-coll", action="store_true", help="Skip cross-collection analysis")
    ap.add_argument("--no-compound", action="store_true", help="Skip compound word test")
    ap.add_argument("--ood", type=str, default=None, help="Out-of-domain text to test")
    ap.add_argument("--load-stats", type=str, default=None,
                    help="Load cached stats .pkl to skip eval and just print reports")
    args = ap.parse_args()

    # Load label map
    with open(DATA_DIR / "label_map.json") as f:
        lm = json.load(f)
    id_to_label = {int(k): v for k, v in lm["id_to_label"].items()}
    id_to_label_inv = {v: int(k) for k, v in lm["id_to_label"].items()}
    num_labels = lm["num_labels"]

    # Load model
    cfg = Cfg()
    cfg.num_labels = num_labels

    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # Always use 80K vocab (encoder safetensors has 80K, not the 32K in the checkpoint)
    if cfg.vocab_size != 80007:
        print(f"  Forcing vocab_size=80007 (checkpoint had {cfg.vocab_size})")
        cfg.vocab_size = 80007

    model = PosClassifier(cfg)

    # Strip embedding keys from checkpoint (size mismatch: 32K vs 80K); load only heads
    head_sd = {k: v for k, v in state_dict.items()
                if not k.startswith("bert.embeddings")}
    result = model.load_state_dict(head_sd, strict=False)
    print(f"  Loaded heads: {len(head_sd)} keys, missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")

    # Reload full encoder from safetensors to ensure correct 80K embeddings
    from safetensors.torch import load_file
    encoder_path = Path(__file__).parent.parent / "model" / "TiBERT-classical-spm-80k" / "final_model" / "model.safetensors"
    encoder_sd = load_file(str(encoder_path))
    bert_sd = {k: v for k, v in encoder_sd.items() if k.startswith("bert.")}
    missing, unexpected = model.bert.load_state_dict(bert_sd, strict=False)
    print(f"  Encoder reloaded from safetensors: {len(bert_sd)} keys")
    print(f"  Epoch:      {ckpt.get('epoch', '?')}")
    print(f"  Dev WF1:   {ckpt.get('dev_weighted_f1', 'N/A')}")
    print(f"  Dev MF1:   {ckpt.get('dev_macro_f1', 'N/A')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Device:    {device}")
    model.eval()

    # ── Stats recovery ──────────────────────────────────────────────────────────
    if args.load_stats:
        import pickle
        with open(args.load_stats, "rb") as f:
            cached = pickle.load(f)
        stats    = cached["stats"]
        results  = cached["results"]
        acc      = cached["acc"]
        correct  = cached["correct"]
        total    = cached["total"]
        wf1      = cached["wf1"]
        mf1      = cached["mf1"]
        print(f"\n  Accuracy:   {acc:.4f}  ({correct:,}/{total:,})")
        print(f"  Weighted F1: {wf1:.4f}")
        print(f"  Macro F1:   {mf1:.4f}")
        print(f"  Total tokens: {total:,}")
        print_class_group_report(stats, id_to_label)
        print("\n  [D2] Hard Tags Detail (n.prop / adj / v.*)")
        print("  " + "─"*55)
        for r in results:
            if r["label"] in HARD_TAGS:
                bar = print_bar(r["f1"])
                icon = "✓" if r["f1"] >= 0.70 else "○" if r["f1"] >= 0.40 else "✗"
                print(f"  {icon} {bar} {r['label']:<18} F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f}  n={r['support']:,}")
        print("\n  Top 10 (sup>=100):")
        for r in results[:10]:
            if r["support"] < 100: continue
            bar = print_bar(r["f1"])
            print(f"  {bar} {r['label']:<22} F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f}  n={r['support']:,}")
        print("\n  Bottom 10 (sup>=50):")
        for r in sorted(results, key=lambda x: x["f1"]):
            if r["support"] < 50: continue
            bar = print_bar(r["f1"])
            print(f"  {bar} {r['label']:<22} F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f}  n={r['support']:,}")
        return

    # Load tokenizer for qualitative / compound tests
    sys.path.insert(0, str(Path(__file__).parent))
    from continued_pretrain import ClassicalTibetanTokenizer
    spm_model = (Path(__file__).parent.parent /
                  "model" / "TiBERT-classical-spm-80k" / "final_model" / "spm.model")
    tokenizer = ClassicalTibetanTokenizer(str(spm_model))

    # ── Standard test eval ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  [0] Test Set Metrics (in-domain)")
    print("="*60)

    test_ds = EvalDS(args.split, max_samples=args.max_samples)
    test_loader = DataLoader(test_ds, batch_size=32, num_workers=0, pin_memory=True)  # GPU

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            ids  = batch["input_ids"].to(device)
            labs = batch["labels"].to(device)
            msk  = (ids != 0).long()
            out  = model(input_ids=ids, attention_mask=msk)
            preds = out["logits"].argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labs.cpu())

    all_preds  = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    valid_mask   = all_labels != -100
    valid_preds  = all_preds[valid_mask]
    valid_labels = all_labels[valid_mask]

    total   = valid_labels.numel()
    correct = (valid_preds == valid_labels).sum().item()
    acc     = correct / total

    stats = compute_stats(valid_preds, valid_labels, id_to_label)
    results = per_class_results(stats, min_support=20)

    total_sup = sum(r["support"] for r in results)
    wf1 = sum(r["f1"] * r["support"] for r in results) / max(total_sup, 1)
    mf1 = sum(r["f1"] for r in results) / len(results)

    # Save stats for fast recovery
    import pickle, pathlib
    stats_path = pathlib.Path("~/.claude/eval_v2_last_stats.pkl").expanduser()
    # Convert defaultdict to plain dict so it can be pickled
    # stats is a defaultdict with a lambda factory — need to convert both outer + inner
    import copy
    stats_plain = {k: dict(v) for k, v in dict(stats).items()}
    with open(stats_path, "wb") as f:
        pickle.dump({"stats": stats_plain, "results": results, "valid_preds": valid_preds,
                     "valid_labels": valid_labels, "acc": acc, "total": total,
                     "correct": correct, "wf1": wf1, "mf1": mf1}, f)
    print(f"\n  [stats saved to {stats_path}]")

    print(f"\n  Accuracy:   {acc:.4f}  ({correct:,}/{total:,})")
    print(f"  Weighted F1: {wf1:.4f}")
    print(f"  Macro F1:   {mf1:.4f}")
    print(f"  Total tokens: {total:,}")

    print_class_group_report(stats, id_to_label)

    # Hard tags detail
    print("\n  [D2] Hard Tags Detail (n.prop / adj / v.*)")
    print("  " + "─"*55)
    for r in results:
        if r["label"] in HARD_TAGS:
            bar = print_bar(r["f1"])
            icon = "✓" if r["f1"] >= 0.70 else "○" if r["f1"] >= 0.40 else "✗"
            print(f"  {icon} {bar} {r['label']:<18} F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f}  n={r['support']:,}")

    # Top / bottom
    print("\n  Top 10 (sup>=100):")
    for r in results[:10]:
        if r["support"] < 100:
            continue
        bar = print_bar(r["f1"])
        print(f"  {bar} {r['label']:<22} F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f}  n={r['support']:,}")

    print("\n  Bottom 10 (sup>=50):")
    bottom = [r for r in results if r["support"] >= 50]
    for r in bottom[:10]:
        bar = print_bar(r["f1"])
        icon = "✗" if r["f1"] < 0.30 else "○"
        print(f"  {icon} {bar} {r['label']:<22} F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f}  n={r['support']:,}")

    # ── Cross-collection ───────────────────────────────────────────────────────
    if not args.no_cross_coll:
        coll_results = eval_cross_collection(model, id_to_label, device)
    else:
        coll_results = []

    # ── Compound word test ─────────────────────────────────────────────────────
    if not args.no_compound:
        eval_compound_words(model, tokenizer, id_to_label, device)

    # ── Qualitative spot-check ────────────────────────────────────────────────
    if args.qualitative:
        qualitative_eval(model, tokenizer, id_to_label, device)

    # ── OOD text ───────────────────────────────────────────────────────────────
    if args.ood:
        print("\n" + "="*60)
        print("  [E] Out-of-Domain Test")
        print("="*60)
        print(f"  Text: {args.ood}")
        ids = [tokenizer.bos_token_id] + tokenize_spm(args.ood, tokenizer) + [tokenizer.eos_token_id]
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        msk = (inp != 0).long()
        with torch.no_grad():
            out = model(input_ids=inp, attention_mask=msk)
        pred_ids = out["logits"].argmax(dim=-1).squeeze(0).cpu().tolist()[1:-1]
        tokens = tokenizer.convert_ids_to_tokens(ids[1:-1])
        print(f"  Tokens: {' | '.join(tokens)}")
        for tok, pid in zip(tokens, pred_ids):
            label = id_to_label.get(pid, "?")
            desc = CASE_PARTICLES_DESC.get(label, TAG_DESCRIPTIONS.get(label, label))
            flag = " ★" if label.startswith("case.") else ""
            print(f"    {tok:<10s} → {label:<15s} {desc}{flag}")

    # Save results
    out = {
        "checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch", "?"),
        "dev_wf1": ckpt.get("dev_weighted_f1", None),
        "test_acc": acc,
        "test_weighted_f1": wf1,
        "test_macro_f1": mf1,
        "per_label": results,
        "class_groups": {
            "case_particle_wf1": group_f1(dict(stats), CASE_PARTICLES)[0],
            "noun_wf1": group_f1(dict(stats), NOUN_TAGS)[0],
            "verb_wf1": group_f1(dict(stats), VERB_TAGS)[0],
            "hard_wf1": group_f1(dict(stats), HARD_TAGS)[0],
        },
        "cross_collection": coll_results,
    }
    out_path = Path(args.checkpoint).parent / "eval_v2_report.json"
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved: {out_path}")


if __name__ == "__main__":
    main()
