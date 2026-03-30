"""
FastAPI 依赖项：模型加载与缓存。

服务器启动时一次性加载 TiBERT tokenizer + POS model 到 GPU，
后续请求复用，避免重复加载。
"""

from __future__ import annotations
import json
import torch
import torch.nn as nn
from functools import lru_cache
from pathlib import Path
from typing import Any


# ── Paths ──────────────────────────────────────────────────────────────────────

# 使用包含 SPM tokenizer 的 fine-tuned 模型（不是原始 TiBERT）
MODEL_DIR   = Path(__file__).parent.parent.parent / "model" / "TiBERT-classical-spm-500k" / "final_model"
SPM_MODEL   = MODEL_DIR / "spm.model"
CHECKPOINT  = Path(__file__).parent.parent.parent / "model" / "pos_classifier" / "best_model.pt"
LABEL_MAP   = Path(__file__).parent.parent.parent / "data"  / "corpus" / "pos_dataset" / "label_map.json"

# ── Case particle metadata ─────────────────────────────────────────────────────

CASE_PARTICLE_NAMES = {
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
}

POS_NAMES = {
    "O":         "未分类",
    "n.count":   "普通名词",
    "n.prop":    "专有名词",
    "n.rel":     "关系名词",
    "n.mass":    "物质名词",
    "v.past":    "过去时动词",
    "v.pres":    "现在时动词",
    "v.fut":     "将来时动词",
    "v.invar":   "不变词动词",
    "v.neg":     "否定动词",
    "v.aux":     "助动词",
    "v.imp":     "命令式动词",
    "v.cop":     "系词（是/为）",
    "v.*.past":  "动词过去时词根",
    "v.*.pres":  "动词现在时词根",
    "v.*.fut":   "动词将来时词根",
    "n.v.*":     "名动复合词",
    "adj":       "形容词",
    "cv.*":      "副动词",
    "cl.*":      "类标记",
    "d.*":       "限定词/冠词",
    "num.*":     "数词",
    "adv.*":     "副词",
    "punc":      "标点符号",
    "neg":       "否定词",
    "skt":       "梵语音译词",
    # 格助词
    "case.gen":  "属格",
    "case.agn":  "作格",
    "case.all":  "为格",
    "case.abl":  "离格",
    "case.ela":  "从格",
    "case.ass":  "共同格",
    "case.term": "终结格",
    "case.loc":  "处格",
    "case.comp": "比格",
    "case.nare": "连格",
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
def get_label_map() -> dict:
    """加载标签映射表。"""
    with open(LABEL_MAP, encoding="utf-8") as f:
        lm = json.load(f)
    return {int(k): v for k, v in lm["id_to_label"].items()}


# ── POS Model ─────────────────────────────────────────────────────────────────

class PosTagger(nn.Module):
    def __init__(self, num_labels=36):
        super().__init__()
        from transformers import BertConfig, BertModel
        self.bert = BertModel(BertConfig(
            vocab_size=32007, hidden_size=768, num_hidden_layers=12,
            num_attention_heads=12, intermediate_size=3072,
            max_position_embeddings=512, pad_token_id=0,
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
    id_to_label = get_label_map()

    model = PosTagger(num_labels=len(id_to_label))
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model, device


# ── POS tagging logic ─────────────────────────────────────────────────────────

def spm_tokenize(text: str, tokenizer) -> list[str]:
    """
    将藏文文本切分为 token 列表。

    策略：先按 ་/། 天然分词符号切分句子（藏文正字法保证每个音节以辅音丛开头），
    每个音节用 tokenizer.encode() 处理——整词在词表则返回整词，
    不在则被 SPM 切成 subword 碎片。
    ་/། 本身作为独立 token 保留。
    """
    # 1. 先按 ་/། 天然切分
    parts: list[str] = []
    i = 0
    start = 0
    while i < len(text):
        ch = text[i]
        if ch in ("་", "།"):
            if start < i:
                parts.append(text[start:i])
            parts.append(ch)
            i += 1
            start = i
        else:
            i += 1
    if start < len(text):
        parts.append(text[start:])

    # 2. 对每个非符号部分做 tokenizer encode
    result: list[str] = []
    for part in parts:
        if part in ("་", "།"):
            result.append(part)
            continue
        # 藏文字符范围：U+0F40–U+0FBC
        if not any(0x0F40 <= ord(c) <= 0xFBC for c in part):
            continue
        # tokenizer.encode() 会把整词切成 subword pieces
        tokens = tokenizer._tokenize(part)
        result.extend(tokens)
    return result


def tag_text(text: str) -> tuple[list[dict], dict]:
    """
    对一段藏文文本进行 POS 标注。

    Returns:
        tokens: list of TokenResponse-compatible dicts
        stats: dict with noun/verb/case_particle counts
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
    attn  = (ids_t != 0).long()

    with torch.no_grad():
        logits = model(ids_t, attention_mask=attn)
        logits = logits["logits"] if isinstance(logits, dict) else logits
        preds  = logits.argmax(dim=-1).squeeze(0).cpu().tolist()

    tokens_out = []
    noun_count = verb_count = case_count = 0

    for s, lid in zip(syllables, preds[1:-1]):
        label = id_to_label.get(lid, "O")
        is_sep = s in ("་", "།")
        case_meta = CASE_PARTICLE_NAMES.get(label)
        is_case = (not is_sep) and (case_meta is not None)

        if is_case:
            case_count += 1
        elif label.startswith("n."):
            noun_count += 1
        elif label.startswith("v."):
            verb_count += 1

        # 后处理：་/། 强制标为 punc（训练数据中这类标注有噪声）
        if is_sep:
            label = "punc"
            is_case = False

        tokens_out.append({
            "token": s,
            "pos": label,
            "pos_zh": POS_NAMES.get(label, label),
            "is_case_particle": is_case,
            "case_name": case_meta[0] if case_meta else None,
            "case_desc": case_meta[1] if case_meta else None,
        })

    syllables_str = " · ".join(s for s in syllables if s not in ("་", "།"))
    stats = {
        "nouns": noun_count,
        "verbs": verb_count,
        "case_particles": case_count,
        "syllable_count": len([s for s in syllables if s not in ("་", "།")]),
    }

    return tokens_out, stats, syllables_str


def get_corpus_stats() -> dict:
    """返回语料库统计信息。"""
    import json
    from pathlib import Path

    pos_dataset_dir = Path(__file__).parent.parent.parent / "data" / "corpus" / "pos_dataset"

    # 语料库统计（如果有 metadata）
    collections = []
    total_sents = 0
    meta_file = pos_dataset_dir / "test_meta.jsonl"
    if meta_file.exists():
        coll_counts: dict = {}
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
        collections = [{"name": k, "count": v} for k, v in coll_counts.items()]

    # POS dataset stats
    pos_stats = {}
    for split in ["train", "dev", "test"]:
        npy = pos_dataset_dir / f"{split}_input_ids.npy"
        if npy.exists():
            import numpy as np
            arr = np.load(npy, mmap_mode="r")
            pos_stats[split] = {"sentences": len(arr), "max_length": int(arr.shape[1])}

    return {
        "total_sentences": total_sents,
        "total_collections": len(collections),
        "collections": collections,
        "pos_dataset_stats": pos_stats,
    }
