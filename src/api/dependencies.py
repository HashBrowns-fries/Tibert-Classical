"""
FastAPI 依赖项：TiBERT tokenizer + POS 模型加载与缓存。

服务器启动时一次性加载，后续请求复用。
"""

from __future__ import annotations
import json
import torch
import torch.nn as nn
from functools import lru_cache
from pathlib import Path
from typing import Any


# ── Paths ──────────────────────────────────────────────────────────────────────

MODEL_DIR   = Path(__file__).parent.parent.parent / "model" / "TiBERT-classical-spm-500k" / "final_model"
SPM_MODEL   = MODEL_DIR / "spm.model"
CHECKPOINT  = Path(__file__).parent.parent.parent / "model" / "pos_classifier" / "crf_supcon" / "best_model.pt"
# test_results.json has the correct label IDs from the training run
LABEL_MAP   = Path(__file__).parent.parent.parent / "model" / "pos_classifier" / "crf_supcon" / "test_results.json"

# ── Case particle metadata ─────────────────────────────────────────────────────

CASE_PARTICLE_NAMES: dict[str, tuple[str, str]] = {
    "case.gen":  ("属格", "表示所属关系，相当于汉语的「的」或英语的 'of'"),
    "case.agn":  ("作格", "表示施事/工具，相当于「由…(做)」或英语 'by'"),
    "case.all":  ("为格", "表示间接宾语/目的，相当于「对 / 给 / 为」或英语 'to/for'"),
    "case.abl":  ("离格", "表示离开/起点，相当于「从 / 离」或英语 'from/away'"),
    "case.ela":  ("从格", "表示从某处出发，相当于「从…起」或英语 'from'"),
    "case.ass":  ("共同格", "表示共同/伴随，相当于「与 / 和 / 同」或英语 'with/together'"),
    "case.term": ("终结格", "表示到达点/方向，相当于「至 / 向」或英语 'to/until'"),
    "case.loc":  ("处格", "表示场所/位置，相当于「在」或英语 'at/in'"),
    "case.comp": ("比格", "表示比较，相当于「比」或英语 'than'"),
    "case.nare": ("连格", "表示连接/序列"),
    "case.odd":  ("方格", "表示方向/方面"),
    "case.sem":  ("对格", "表示对待/关系"),
    "case.rung": ("从格变体", "表示从由"),
    "case.fin":  ("终助格", "表示结束"),
    "case.cont": ("连持续格", "表示持续"),
    "case.imp":  ("命令格", "表示命令/要求"),
    "case.impf": ("未完成命令格", "表示未完成命令"),
    "case.ques": ("疑问格", "表示疑问"),
}

POS_NAMES: dict[str, str] = {
    # 名词
    "n.count": "普通名词",
    "n.prop":  "专有名词",
    "n.rel":   "关系名词",
    "n.mass":  "物质名词",
    # 动词
    "v.past":   "过去时动词",
    "v.pres":   "现在时动词",
    "v.fut":    "将来时动词",
    "v.invar":  "不变词动词",
    "v.neg":    "否定动词",
    "v.aux":    "助动词",
    "v.imp":    "命令式动词",
    "v.cop":    "系词（是/为）",
    # 名动复合词
    "n.v.invar":    "名动复合词（不变）",
    "n.v.cop":      "名动复合词（系词）",
    "n.v.fut":      "名动复合词（将来）",
    "n.v.past":     "名动复合词（过去）",
    "n.v.pres":     "名动复合词（现在）",
    "n.v.aux":      "名动复合词（助动）",
    "n.v.neg":      "名动复合词（否定）",
    "n.v.imp":      "名动复合词（命令）",
    "n.v.fut.n.v.past":     "名动复合词（将来-过去）",
    "n.v.fut.n.v.pres":    "名动复合词（将来-现在）",
    "n.v.past.n.v.pres":   "名动复合词（过去-现在）",
    # 其他词类
    "adj":       "形容词",
    "neg":       "否定词",
    "punc":      "标点符号",
    "skt":       "梵语音译词",
    # 副词
    "adv.dir":       "方向副词",
    "adv.intense":   "程度副词",
    "adv.mim":       "拟声副词",
    "adv.proclausal":"句子副词",
    "adv.temp":      "时间副词",
    # 连接动词
    "cv.abl":    "离格连接动词",
    "cv.agn":    "作格连接动词",
    "cv.all":    "为格连接动词",
    "cv.are":    "方向连接动词",
    "cv.ass":    "共同格连接动词",
    "cv.comp":   "比较连接动词",
    "cv.cont":   "持续连接动词",
    "cv.ela":    "从格连接动词",
    "cv.fin":    "终结连接动词",
    "cv.gen":    "属格连接动词",
    "cv.imp":    "命令连接动词",
    "cv.impf":   "未完成连接动词",
    "cv.loc":    "处格连接动词",
    "cv.nare":   "连格连接动词",
    "cv.odd":    "方格连接动词",
    "cv.ques":   "疑问连接动词",
    "cv.rung":   "从格变体连接动词",
    "cv.sem":    "对格连接动词",
    "cv.term":   "终结连接动词",
    # 小品词
    "cl.focus":  "焦点小品词",
    "cl.quot":   "引用小品词",
    # 限定词
    "d.dem":     "指示限定词",
    "d.det":     "限定词",
    "d.emph":    "强调限定词",
    "d.indef":   "不定限定词",
    "d.plural":  "复数限定词",
    "d.tsam":    "等比限定词（ཏེ་སྟེ་）",
    # 数词
    "num.card":  "基数词",
    "num.ord":   "序数词",
    "numeral":   "数词",
    # 代词/介词
    "p.indef":   "不定代词",
    "p.interrog": "疑问代词",
    "p.pers":    "人称代词",
    "p.refl":    "反身代词",
    # 其他
    "interj":    "感叹词",
    "line.num":  "行号",
    "page.num":  "页码",
    "dunno":     "未识别词",
    "O":         "未分类",
    # 格助词（简短名称）
    "case.gen":  "属格助词",
    "case.agn":  "作格助词",
    "case.all":  "为格助词",
    "case.abl":  "离格助词",
    "case.ela":  "从格助词",
    "case.ass":  "共同格助词",
    "case.term": "终结格助词",
    "case.loc":  "处格助词",
    "case.comp": "比格助词",
    "case.nare": "连格助词",
    "case.odd":  "方格助词",
    "case.sem":  "对格助词",
    "case.rung": "从格变体助词",
    "case.fin":  "终助格",
    "case.cont": "连持续格",
    "case.imp":  "命令格",
    "case.impf": "未完命令格",
    "case.ques": "疑问格",
}


# ── Tokenizer ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_tokenizer() -> Any:
    """加载 SentencePiece tokenizer（启动时加载，缓存复用）。"""
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent.parent.parent / "scripts"))
    from continued_pretrain import ClassicalTibetanTokenizer
    return ClassicalTibetanTokenizer(spm_model_file=str(SPM_MODEL))


# ── Label map ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_label_map() -> dict[int, str]:
    """
    加载标签映射表：从 test_results.json 读取（与 checkpoint 训练时一致）。
    test_results.json 中 label_stats 的 key 就是训练时的标签 ID。
    """
    with open(LABEL_MAP, encoding="utf-8") as f:
        lm = json.load(f)
    # test_results.json: { "label_stats": { "1": {"label": "punc", ...}, "2": {...}, ... } }
    stats = lm.get("label_stats", {})
    id_to_label = {}
    for k, v in stats.items():
        id_to_label[int(k)] = v["label"]
    # 补全 checkpoint 中有但 test_results 中未出现的标签（ID 0 和少数稀有标签）
    for i in range(77):
        if i not in id_to_label:
            id_to_label[i] = f"UNK_{i}"
    return id_to_label


# ── POS Model ─────────────────────────────────────────────────────────────────

class PosTagger(nn.Module):
    """
    TiBERT encoder + POS classification head.
    架构与 crf_supcon/best_model.pt 训练时完全一致。
    """

    def __init__(self, vocab_size: int = 32007, num_labels: int = 77, max_len: int = 512):
        super().__init__()
        from transformers import BertConfig, BertModel
        self.bert = BertModel(BertConfig(
            vocab_size=vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=max_len,
            pad_token_id=0,
        ))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.last_hidden_state))


@lru_cache(maxsize=1)
def get_pos_model() -> tuple[PosTagger, str]:
    """
    加载 POS 分类模型。
    返回 (model, device)。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[POS Model] Loading TiBERT POS tagger on {device}...")

    model = PosTagger(vocab_size=32007, num_labels=77, max_len=512)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state = ckpt["model_state"]

    # Strip "module." (DataParallel) and "bert." (module prefix) from keys
    clean = {}
    for k, v in state.items():
        k = k.replace("module.", "").replace("bert.", "")
        clean[k] = v

    # Load BERT encoder weights
    bert_keys = {k: v for k, v in clean.items()
                 if k.startswith("embeddings") or k.startswith("encoder") or k.startswith("pooler")}
    missing_bert, unexpected_bert = model.bert.load_state_dict(bert_keys, strict=False)
    if missing_bert:
        print(f"  [POS] BERT missing keys: {len(missing_bert)} (CRF/contrastive head, expected)")
    if unexpected_bert:
        print(f"  [POS] BERT unexpected keys: {len(unexpected_bert)}")

    # Load classifier weights
    cls_keys = {k: v for k, v in clean.items() if "classifier" in k}
    cls_clean = {k.replace("classifier.", ""): v for k, v in cls_keys.items()}
    missing_cls, unexpected_cls = model.classifier.load_state_dict(cls_clean, strict=True)
    print(f"  [POS] Classifier loaded (missing={missing_cls})")
    print(f"  [POS] Epoch {ckpt.get('epoch', '?')} | dev_acc={ckpt.get('dev_acc', 'N/A'):.4f}")

    model.to(device)
    model.eval()
    print(f"[POS Model] Ready on {device}")
    return model, device


# ── Shad-aware tokenization (matching training pipeline) ────────────────────────

def spm_tokenize(text: str, tokenizer) -> list[str]:
    """
    将藏文文本切分为 token 列表。

    训练时的 spm_tokenize_word() 行为：
      ① 去掉词尾的 ་/། (shad markers)
      ②bare syllable 用 SentencePiece longest-match 编码
      ③shad marker 单独作为一个 token

    推理时必须完全复现这个行为，否则有 ~30pp 的 tokenization gap。
    """
    tokens = []
    unk = tokenizer._token2id.get("[UNK]")
    i = 0
    while i < len(text):
        ch = text[i]
        # Shad markers are always their own token
        if ch in ("་", "།", "༔"):
            tokens.append(ch)
            i += 1
            continue
        # Longest-match within the current syllable (don't cross shad boundaries)
        best = None
        max_len = min(10, len(text) - i)
        for n in range(max_len, 0, -1):
            sub = text[i:i+n]
            # Stop if sub contains a shad marker mid-span
            if any(c in ("་", "།", "༔") for c in sub):
                continue
            if tokenizer._convert_token_to_id(sub) != unk:
                best = sub
                break
        if best is None:
            best = text[i]  # fallback: single char
        tokens.append(best)
        i += len(best)
    return tokens


# ── POS tagging logic ──────────────────────────────────────────────────────────

def tag_text(text: str) -> tuple[list[dict], dict, str]:
    """
    对一段藏文文本进行 POS 标注。

    Returns:
        tokens: list of dicts with token, pos, pos_zh, is_case_particle, case_name, case_desc
        stats: dict with noun/verb/case_particle/syllable counts
        syllables_str: formatted syllable string for display
    """
    tokenizer = get_tokenizer()
    model, device = get_pos_model()
    id_to_label = get_label_map()

    syllables = spm_tokenize(text, tokenizer)

    ids = [tokenizer.bos_token_id]
    for s in syllables:
        ids.append(tokenizer._convert_token_to_id(s))
    ids.append(tokenizer.eos_token_id)

    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    attn = (ids_t != 0).long()

    with torch.no_grad():
        logits = model(ids_t, attention_mask=attn)
        preds = logits.argmax(dim=-1).squeeze(0).cpu().tolist()

    tokens_out = []
    noun_count = verb_count = case_count = 0

    for s, lid in zip(syllables, preds[1:-1]):
        label = id_to_label.get(lid, "O")
        is_sep = s in ("་", "།")

        if is_sep:
            # Separator markers → punc, no case analysis
            tokens_out.append({
                "token": s,
                "pos": "punc",
                "pos_zh": "分隔符",
                "is_case_particle": False,
                "case_name": None,
                "case_desc": None,
            })
            continue

        case_meta = CASE_PARTICLE_NAMES.get(label)
        is_case = case_meta is not None

        if is_case:
            case_count += 1
        elif label.startswith("n.") or label.startswith("n.v."):
            noun_count += 1
        elif label.startswith("v."):
            verb_count += 1

        tokens_out.append({
            "token": s,
            "pos": label,
            "pos_zh": POS_NAMES.get(label, label),
            "is_case_particle": is_case,
            "case_name": case_meta[0] if case_meta else None,
            "case_desc": case_meta[1] if case_meta else None,
        })

    bare_syllables = [s for s in syllables if s not in ("་", "།", "༔")]
    syllables_str = " · ".join(bare_syllables)
    stats = {
        "nouns": noun_count,
        "verbs": verb_count,
        "case_particles": case_count,
        "syllable_count": len(bare_syllables),
    }

    return tokens_out, stats, syllables_str


# ── Corpus stats ──────────────────────────────────────────────────────────────

def get_corpus_stats() -> dict:
    """返回语料库统计信息。"""
    import json
    from pathlib import Path

    pos_dataset_dir = Path(__file__).parent.parent.parent / "data" / "corpus" / "pos_dataset"

    collections = []
    total_sents = 0
    meta_file = pos_dataset_dir / "test_meta.jsonl"
    if meta_file.exists():
        coll_counts: dict = {}
        try:
            with open(meta_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        coll = obj.get("collection", "unknown")
                        coll_counts[coll] = coll_counts.get(coll, 0) + 1
                        total_sents += 1
                    except Exception:
                        pass
        except FileNotFoundError:
            pass
        collections = [{"name": k, "count": v} for k, v in coll_counts.items()]

    # POS dataset stats from test_results.json
    pos_stats = {}
    test_results = LABEL_MAP
    if test_results.exists():
        try:
            with open(test_results, encoding="utf-8") as f:
                tr = json.load(f)
            n_test = tr.get("num_test_examples", 0)
            if n_test:
                pos_stats["test"] = {"sentences": n_test}
        except Exception:
            pass

    return {
        "total_sentences": total_sents,
        "total_collections": len(collections),
        "collections": collections,
        "pos_dataset_stats": pos_stats,
    }
