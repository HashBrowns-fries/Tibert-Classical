"""
Classical Tibetan Natural Language Analysis
==========================================
输入一句藏文 → POS标注 + 自然语言语法解释

Pipeline:
    藏文输入
    → TiBERT POS标注器 (格助词识别)
    → Qwen LLM 自然语言解释
    → 结构化输出

Usage:
    export DASHSCOPE_API_KEY="your-key"
    .venv/bin/python scripts/analyze_tibetan.py "བོད་གི་ཡུལ་ལྷོ་ལ་སོང་"
"""

import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal

sys.path.insert(0, str(Path(__file__).parent))
from continued_pretrain import ClassicalTibetanTokenizer
from transformers import BertConfig, BertModel

# ── Paths ──────────────────────────────────────────────────────────────────────

MODEL_DIR  = Path(__file__).parent.parent / "model" / "TiBERT-classical-spm-500k" / "final_model"
CHECKPOINT = Path(__file__).parent.parent / "model" / "pos_classifier" / "best_model.pt"
LABEL_MAP  = Path(__file__).parent.parent / "data"  / "corpus" / "pos_dataset" / "label_map.json"

# ── Case particle metadata ────────────────────────────────────────────────────

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
    "n.count":  "普通名词",
    "n.prop":    "专有名词",
    "n.rel":    "关系名词",
    "n.mass":   "物质名词",
    "v.past":   "过去时动词",
    "v.pres":   "现在时动词",
    "v.fut":    "将来时动词",
    "v.invar":  "不变词动词",
    "v.neg":    "否定动词",
    "v.aux":    "助动词",
    "v.imp":    "命令式动词",
    "v.cop":    "系词（是/为）",
    "v.*.past": "动词过去时词根",
    "v.*.pres": "动词现在时词根",
    "v.*.fut":  "动词将来时词根",
    "n.v.*":    "名动复合词",
    "adj":      "形容词",
    "cv.*":     "连接动词",
    "cl.*":     "类标记",
    "d.*":      "限定词/冠词",
    "num.*":    "数词",
    "adv.*":    "副词",
    "punc":     "标点符号",
    "neg":      "否定词",
    "skt":      "梵语音译词",
    "O":        "未分类",
}


# ── POS Model ────────────────────────────────────────────────────────────────

class PosTagger(nn.Module):
    def __init__(self, num_labels=36):
        super().__init__()
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


@dataclass
class TaggedToken:
    token: str
    label: str
    label_zh: str
    is_case_particle: bool
    case_name: str
    case_desc: str


# ── LLM Prompt Builder ────────────────────────────────────────────────────────

def build_llm_prompt(text: str, tagged: list[TaggedToken]) -> str:
    """Build a rich prompt for Qwen with POS-tagged data."""

    # Build token table
    case_particles_found = []
    token_rows = []
    for t in tagged:
        if t.token in ("་", "།"):
            continue  # skip separator markers
        pos_zh = t.label_zh
        note = ""
        if t.is_case_particle:
            note = f"【{t.case_name}】{t.case_desc}"
            case_particles_found.append(f"{t.token}（{t.case_name}: {t.case_desc}）")
        token_rows.append(f"  - {t.token}: {pos_zh}{' ' + note if note else ''}")

    tokens_block = "\n".join(token_rows) if token_rows else "  （无有效音节）"

    # Case particle summary
    case_block = ""
    if case_particles_found:
        case_block = (
            "\n\n本句中的格助词及其功能：\n"
            + "\n".join(f"  • {cp}" for cp in case_particles_found)
        )

    # Overall structure hint
    nouns = [t for t in tagged if t.label.startswith("n.") and t.token not in ("་", "།")]
    verbs = [t for t in tagged if t.label.startswith("v.") and t.token not in ("་", "།")]
    adj   = [t for t in tagged if t.label == "adj" and t.token not in ("་", "།")]
    structure = []
    for t in tagged:
        if t.token in ("་", "།"):
            continue
        if t.is_case_particle:
            structure.append(f"[{t.case_name}助词]")
        elif t.label.startswith("n."):
            structure.append(f"[名词:{t.token}]")
        elif t.label.startswith("v."):
            structure.append(f"[动词:{t.token}]")
        elif t.label == "adj":
            structure.append(f"[形容词:{t.token}]")
        elif t.label == "neg":
            structure.append(f"[否定词:{t.token}]")
        elif t.label == "punc":
            pass
        else:
            structure.append(f"[{t.token}]")

    prompt = f"""请分析以下古典藏文，提供详细的自然语言语法解释。

【原文】
{text}

【词性标注结果】（由机器自动标注，仅供参考）
{tokens_block}
{case_block}

【句法结构提示】
""" + " ".join(structure) + f"""

请用中文（白话文）给出完整的语法分析，包括：
1. 句子的基本含义（整句翻译）
2. 每个词的含义
3. 格助词的功能（属格/作格/为格等）
4. 句子的语法结构分析
5. 特殊语法现象说明（如有）
"""
    return prompt


# ── Grammar Analyzer ──────────────────────────────────────────────────────────

def call_qwen_llm(prompt: str, api_key: str) -> str:
    """Call Qwen via DashScope."""
    import dashscope
    from dashscope import Generation
    from dashscope.api_entities.dashscope_response import Message

    dashscope.api_key = api_key

    SYSTEM = (
        "你是一位专业的古典藏文（Classical Tibetan）语言学家，精通藏文语法、"
        "词性标注和格助词分析。请用中文（白话文）详细分析用户提供的古典藏文句子。"
    )

    response = Generation.call(
        model="qwen-plus",
        messages=[
            Message(role="system", content=SYSTEM),
            Message(role="user", content=prompt),
        ],
        temperature=0.2,
        top_p=0.9,
        api_key=api_key,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Qwen API error: {response.code} — {getattr(response, 'message', '')}"
        )
    return response.output.text


# ── Full Pipeline ─────────────────────────────────────────────────────────────

class ClassicalTibetanAnalyzer:
    """端到端古典藏文分析器：POS标注 + LLM解释"""

    def __init__(self, device: str = None, api_key: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = ClassicalTibetanTokenizer(
            spm_model_file=str(MODEL_DIR / "spm.model")
        )

        # Load label map
        with open(LABEL_MAP, encoding="utf-8") as f:
            lm = json.load(f)
        self.id_to_label = {int(k): v for k, v in lm["id_to_label"].items()}

        # Load POS model
        self.model = PosTagger(num_labels=len(self.id_to_label))
        ckpt = torch.load(CHECKPOINT, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        self.model.to(self.device)
        self.model.eval()

        # API key
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not self.api_key:
            print("⚠️  未设置 DASHSCOPE_API_KEY，LLM解释功能不可用（仅POS标注）")

    def spm_tokenize(self, text: str) -> list[str]:
        """Split text into syllables."""
        tokens = []
        i = 0
        while i < len(text):
            best = None
            for n in range(min(10, len(text) - i), 0, -1):
                sub = text[i:i+n]
                tid = self.tokenizer._convert_token_to_id(sub)
                unk = self.tokenizer._token2id.get("[UNK]")
                if tid != unk:
                    best = sub
                    break
            if best is None:
                best = text[i]
            tokens.append(best)
            i += len(best)
        return tokens

    def tag(self, text: str) -> list[TaggedToken]:
        """POS tag a text string."""
        syllables = self.spm_tokenize(text)

        ids = [self.tokenizer.bos_token_id]
        for s in syllables:
            ids.append(self.tokenizer._convert_token_to_id(s))
        ids.append(self.tokenizer.eos_token_id)

        ids_t = torch.tensor([ids], dtype=torch.long, device=self.device)
        attn  = (ids_t != 0).long()

        with torch.no_grad():
            logits = self.model(ids_t, attention_mask=attn)
            preds  = logits.argmax(dim=-1).squeeze(0).cpu().tolist()

        results = []
        for s, lid in zip(syllables, preds[1:-1]):
            label = self.id_to_label.get(lid, "O")
            case_meta = CASE_PARTICLE_NAMES.get(label)
            is_sep = s in ("་", "།")
            is_case = (not is_sep) and (case_meta is not None)
            results.append(TaggedToken(
                token=s,
                label=label,
                label_zh=POS_NAMES.get(label, label),
                is_case_particle=is_case,
                case_name=case_meta[0] if case_meta else "",
                case_desc=case_meta[1] if case_meta else "",
            ))
        return results

    def analyze(self, text: str, use_llm: bool = True) -> dict:
        """
        Full analysis: POS tag + optional LLM explanation.

        Returns dict with keys:
            - original: original text
            - syllables: list of syllables
            - tagged: list of TaggedToken
            - llm_explanation: natural language explanation (if use_llm=True and API key available)
        """
        tagged = self.tag(text)
        syllables = [t.token for t in tagged if t.token not in ("་", "།")]

        result = {
            "original": text,
            "syllables": " · ".join(syllables),
            "tagged": tagged,
            "llm_explanation": None,
        }

        if use_llm and self.api_key:
            try:
                prompt = build_llm_prompt(text, tagged)
                explanation = call_qwen_llm(prompt, self.api_key)
                result["llm_explanation"] = explanation
            except Exception as e:
                result["llm_explanation"] = f"⚠️ LLM调用失败: {e}"

        return result


# ── Output Formatter ──────────────────────────────────────────────────────────

def print_analysis(result: dict):
    """Pretty-print the analysis result."""
    tagged = result["tagged"]
    syllables = [t for t in tagged if t.token not in ("་", "།")]

    print()
    print("═" * 66)
    print(f"  原文：{result['original']}")
    print("═" * 66)

    # Token table
    print()
    print("  ┌─ 词性标注（★ = 格助词）")
    print("  │")
    case_particles = []
    for t in tagged:
        if t.token in ("་", "།"):
            print(f"  │   {t.token:<6s} （音节分隔符 tsheg）")
            continue
        flag = " ★" if t.is_case_particle else ""
        case = f" → {t.case_name}：{t.case_desc}" if t.is_case_particle else ""
        print(f"  │  {t.token:<6s}  {t.label_zh:<12s}{flag}{case}")
        if t.is_case_particle:
            case_particles.append(t)

    # Case particle summary
    if case_particles:
        print("  │")
        print("  │  ★ 格助词一览：")
        for cp in case_particles:
            print(f"  │    འབྱིན → {cp.token}：{cp.case_name}（{cp.case_desc}）")
    print("  └─")

    # LLM explanation
    if result["llm_explanation"]:
        print()
        print("  ┌─ 自然语言解释（LLM）")
        print("  │")
        for line in result["llm_explanation"].strip().split("\n"):
            line = line.strip()
            if line:
                print(f"  │  {line}")
        print("  └─")
    elif syllables:
        print()
        print("  ⚠️  设置 DASHSCOPE_API_KEY 环境变量以获取自然语言解释")
        print("      export DASHSCOPE_API_KEY='your-key'")

    print()
    print("═" * 66)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="古典藏文自然语言分析")
    parser.add_argument("text", nargs="?", default="བོད་གི་ཡུལ་ལྷོ་ལ་སོང་",
                        help="要分析的古典藏文")
    parser.add_argument("--no-llm", action="store_true", help="跳过LLM解释（仅POS标注）")
    parser.add_argument("--api-key", default=None, help="DashScope API Key（也可通过DASHSCOPE_API_KEY环境变量设置）")
    args = parser.parse_args()

    analyzer = ClassicalTibetanAnalyzer(api_key=args.api_key)
    result = analyzer.analyze(args.text, use_llm=not args.no_llm)
    print_analysis(result)


if __name__ == "__main__":
    main()
