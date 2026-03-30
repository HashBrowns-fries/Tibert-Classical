"""
Fine-tune Qwen2.5-3B-Instruct for Classical Tibetan Grammar Analysis
====================================================================
Uses QLoRA (4-bit) to fine-tune on SegPOS data converted to instruction format.

Training data pipeline:
    SegPOS sentence → POS tagger → dictionary lookup → instruction format → train

Usage:
    # 1. Generate training data (takes ~30 min for 50k samples)
    python scripts/finetune_qwen_pos.py --generate_data --num_samples 50000

    # 2. Fine-tune with QLoRA
    python scripts/finetune_qwen_pos.py --train --data_file data/corpus/grammar_analysis_train.jsonl

Architecture:
    Qwen2.5-3B-Instruct + QLoRA (4-bit, r=16, alpha=32)
    Target modules: q_proj, k_proj, v_proj, o_proj
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from itertools import islice

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR    = PROJECT_ROOT / "model" / "Qwen2.5-3B-Instruct"
DATA_DIR     = PROJECT_ROOT / "data"  / "corpus"
OUTPUT_DIR   = PROJECT_ROOT / "model" / "qwen-grammar-3b"
SEGPOS_BASE  = PROJECT_ROOT / "data" / "segpos_extracted"

# ── POS Tag Chinese Names ───────────────────────────────────────────────────────

POS_NAMES_ZH = {
    "O": "未分类",
    "n.count": "普通名词",
    "n.prop": "专有名词",
    "n.rel": "关系名词",
    "n.mass": "物质名词",
    "n.v.invar": "名动不变词",
    "n.v.past": "名动过去时",
    "n.v.pres": "名动现在时",
    "n.v.fut": "名动将来时",
    "n.v.fut.n.v.pres": "名动将现复合",
    "n.v.fut.n.v.past": "名动将过复合",
    "n.v.past.n.v.pres": "名动过现复合",
    "n.v.neg": "名动否定式",
    "n.v.aux": "名动助词式",
    "n.v.cop": "名动系词式",
    "n.v.imp": "名动命令式",
    "v.past": "过去时动词",
    "v.pres": "现在时动词",
    "v.fut": "将来时动词",
    "v.invar": "不变词动词",
    "v.neg": "否定动词",
    "v.aux": "助动词",
    "v.imp": "命令式动词",
    "v.cop": "系词",
    "v.fut.v.pres": "将现复合动词",
    "v.fut.v.past": "将过复合动词",
    "v.past.v.pres": "过现复合动词",
    "adj": "形容词",
    "case.gen": "属格助词",
    "case.agn": "作格助词",
    "case.all": "为格助词",
    "case.abl": "离格助词",
    "case.ela": "从格助词",
    "case.ass": "共同格助词",
    "case.term": "终结格助词",
    "case.loc": "处格助词",
    "case.comp": "比格助词",
    "case.nare": "连格助词",
    "case.fin": "终助词",
    "case.impf": "未完成体助词",
    "case.odd": "奇数助词",
    "case.cont": "持续助词",
    "case.sem": "语义格助词",
    "cv.fin": "副动叙述式",
    "cv.sem": "副动语义式",
    "cv.term": "副动终结式",
    "cv.loc": "副动处格式",
    "cv.gen": "副动属格式",
    "cv.all": "副动为格式",
    "cv.ass": "副动共同式",
    "cv.agn": "副动作格式",
    "cv.imp": "副动命令式",
    "cv.abl": "副动离格式",
    "cv.ela": "副动从格式",
    "cv.impf": "副动未完式",
    "cv.comp": "副动比格式",
    "cv.odd": "副动奇数式",
    "cv.rung": "副动连用式",
    "cv.ques": "副动疑问式",
    "cv.nare": "副动连格式",
    "cv.cont": "副动持续式",
    "cv.are": "副动方式",
    "cl.focus": "焦点标记",
    "cl.quot": "引用标记",
    "d.dem": "指示代词",
    "d.plural": "复数代词",
    "d.det": "限定代词",
    "d.emph": "强调代词",
    "d.indef": "不定代词",
    "d.tsam": "泛指代词",
    "punc": "标点符号",
    "neg": "否定词",
    "skt": "梵语音译",
    "dunno": "未知",
    "p.interrog": "疑问代词",
    "p.pers": "人称代词",
    "p.refl": "反身代词",
    "p.indef": "不定代词",
    "num.card": "基数词",
    "num.ord": "序数词",
    "numeral": "数词",
    "adv.temp": "时间副词",
    "adv.proclausal": "句子副词",
    "adv.intense": "程度副词",
    "adv.dir": "方向副词",
    "adv.mim": "拟声副词",
    "interj": "感叹词",
    "line.num": "行号",
    "page.num": "页码",
}


def format_pos_token(token: str, pos: str) -> str:
    """Format a single token with its POS tag for display."""
    if token in ("་", "།"):
        return f"- {token} [punc·分词符号]"
    zh = POS_NAMES_ZH.get(pos, pos)
    return f"- {token} [{pos}·{zh}]"


def build_translation_example(tokens: list) -> str:
    """Build a simple translation hint from POS tags (placeholder)."""
    words = []
    for t, p in tokens:
        if t in ("་", "།"):
            continue
        if p.startswith("case."):
            words.append(f"[{POS_NAMES_ZH.get(p, p)}]")
        elif p.startswith("n."):
            words.append("[名]")
        elif p.startswith("v."):
            words.append("[动]")
        elif p == "adj":
            words.append("[形]")
        elif p == "neg":
            words.append("[否]")
        else:
            words.append(t)
    return " ".join(words)


def build_analysis_output(tokens: list, dict_lookup: dict = None) -> str:
    """Build the target analysis text from POS tokens."""
    lines = ["## 词语解析"]
    for t, p in tokens:
        if t in ("་", "།"):
            continue
        zh = POS_NAMES_ZH.get(p, p)

        # Get dictionary definition if available
        dict_def = ""
        if dict_lookup and t in dict_lookup:
            defs = dict_lookup[t].get("entries", [])
            if defs:
                # Take first definition, first line, max 80 chars
                defn = defs[0].get("definition", "")
                defn = defn.split("\n")[0][:80]
                if defn:
                    dict_def = f"：「{defn}」"

        lines.append(f"  - {t}[{zh}]{dict_def}")

    # Add syntactic structure hint
    structure_parts = []
    for t, p in tokens:
        if t in ("་", "།"):
            continue
        if p.startswith("case."):
            structure_parts.append(f"[{POS_NAMES_ZH.get(p, p)}]")
        elif p.startswith("n."):
            structure_parts.append(f"[名]{t}")
        elif p.startswith("v."):
            structure_parts.append(f"[动]{t}")
        elif p == "adj":
            structure_parts.append(f"[形]{t}")
        elif p == "neg":
            structure_parts.append(f"[否]{t}")
        else:
            structure_parts.append(t)

    lines.append("")
    lines.append("## 句法结构")
    lines.append(" ".join(structure_parts))

    return "\n".join(lines)


# ── Data Generation ─────────────────────────────────────────────────────────────

def parse_segpos_line(line: str) -> list:
    """Parse a SegPOS line into (word, tag) pairs."""
    import re
    line = re.sub(r'\s*<utt>\s*$', '', line.strip())
    if not line:
        return []
    pairs = []
    for chunk in line.split():
        chunk = chunk.strip()
        if not chunk or chunk in ('<utt>',):
            continue
        if '/' in chunk:
            parts = chunk.rsplit('/', 1)
            word = parts[0]
            tag = parts[1] if len(parts) == 2 else "O"
        else:
            word = chunk
            tag = "O"
        if not word or word in ('p1','p2','p3','p4','p5'):
            continue
        pairs.append((word, tag))
    return pairs


def spm_tokenize_word(word: str) -> list:
    """Split a word at ་/། boundaries."""
    if word == "་": return ["་"]
    if word == "།": return ["།"]
    if word.endswith("་"): return [word[:-1], "་"]
    if word.endswith("།"): return [word[:-1], "།"]
    return [word]


def generate_instruction_sample(
    sentence: str,
    tokens: list[tuple],
    dict_lookup: dict = None,
    max_tokens: int = 40,
) -> dict:
    """
    Convert a SegPOS-annotated sentence into an instruction-tuning sample.

    Returns a dict with 'messages' (ChatML format) for Qwen2.5.
    """
    # Filter out punc tokens for cleaner display (optional: keep them)
    content_tokens = [(t, p) for t, p in tokens if t not in ("་", "།")]

    # Build POS annotation lines
    pos_lines = []
    for t, p in tokens:
        pos_lines.append(format_pos_token(t, p))
    pos_block = "\n".join(pos_lines) if pos_lines else "（无有效音节）"

    # Build dictionary block
    if dict_lookup:
        dict_lines = []
        for t, p in content_tokens:
            if t in dict_lookup:
                entries = dict_lookup[t].get("entries", [])
                if entries:
                    defn = entries[0].get("definition", "").split("\n")[0][:80]
                    if defn:
                        dict_lines.append(f"- {t}：{defn}")
        dict_block = "\n".join(dict_lines) if dict_lines else "（无词典释义）"
    else:
        dict_block = "（无词典释义）"

    # Build assistant output
    analysis = build_analysis_output(tokens, dict_lookup)

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个古典藏文语法学家。根据POS词性标注和词典释义，"
                "给出简洁、结构化的语法分析。输出格式：\n"
                "## 词语解析\n"
                "- <词>[<词性>·<中文>]：<词典释义>\n"
                "## 句法结构\n"
                "<简要结构描述>"
            ),
        },
        {
            "role": "user",
            "content": (
                f"分析以下古典藏文：\n\n"
                f"藏文原文：{sentence}\n\n"
                f"POS词性标注：\n{pos_block}\n\n"
                f"词典释义：\n{dict_block}"
            ),
        },
        {
            "role": "assistant",
            "content": analysis,
        },
    ]

    return {"messages": messages}


def generate_training_data(
    num_samples: int = 50000,
    output_file: str = None,
    seed: int = 42,
) -> Path:
    """Generate instruction-tuning data from SegPOS corpus."""
    import glob
    import re

    random.seed(seed)

    if output_file is None:
        output_file = DATA_DIR / f"grammar_analysis_train_{num_samples}.jsonl"

    print(f"Generating {num_samples:,} training samples from SegPOS...")

    # Collect all pos files
    pos_files = glob.glob(str(SEGPOS_BASE / "**" / "pos" / "*.txt"), recursive=True)
    pos_files = [f for f in pos_files if "__MACOSX" not in f]
    random.shuffle(pos_files)

    # Load tokenizer for POS inference
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from continued_pretrain import ClassicalTibetanTokenizer
    from src.api.dependencies import get_pos_model, get_label_map

    print("Loading TiBERT tokenizer and POS model...")
    tok_path = PROJECT_ROOT / "model" / "TiBERT-classical-spm-500k" / "final_model" / "spm.model"
    tokenizer = ClassicalTibetanTokenizer(spm_model_file=str(tok_path))

    # Try to load POS model (optional — if checkpoint exists)
    pos_model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        pos_model, _ = get_pos_model()
        pos_model.to(device)
        pos_model.eval()
        id_to_label = get_label_map()
        print(f"  POS model loaded (device={device})")
    except Exception as e:
        print(f"  POS model not available ({e}), using SegPOS gold labels")

    # Try to load dictionary lookup
    dict_lookup_func = None
    try:
        from src.dict import lookup_word
        dict_lookup_func = lookup_word
        print("  Dictionary lookup available")
    except Exception as e:
        print(f"  Dictionary not available ({e})")

    samples_written = 0
    sentences_seen = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for fpath in tqdm(pos_files, desc="Scanning files"):
            if samples_written >= num_samples:
                break

            with open(fpath, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if samples_written >= num_samples:
                        break

                    # Parse gold labels from SegPOS
                    gold_pairs = parse_segpos_line(line)
                    if not gold_pairs:
                        continue

                    # Filter: only keep sentences with enough content
                    content = [(t, p) for t, p in gold_pairs if t not in ("་", "།")]
                    if len(content) < 2 or len(content) > max_tokens:
                        continue

                    sentences_seen += 1

                    # Reconstruct full sentence
                    sentence = "".join(w for w, _ in gold_pairs)

                    # Get dictionary lookup for content words
                    dl = None
                    if dict_lookup_func and random.random() < 0.3:  # 30% include dict
                        dl = {}
                        for t, p in content[:5]:  # max 5 words
                            try:
                                dl[t] = dict_lookup_func(t, None, False)
                            except Exception:
                                pass

                    # Generate sample
                    sample = generate_instruction_sample(sentence, gold_pairs, dl)

                    # Format as ChatML text
                    text = tokenizer.apply_chat_template(
                        sample["messages"],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    out_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    samples_written += 1

                    if samples_written % 5000 == 0:
                        print(f"  Written: {samples_written:,} / {num_samples:,}")

    print(f"\nDone! Wrote {samples_written:,} samples to {output_file}")
    print(f"Scanned {sentences_seen:,} sentences (from {len(pos_files)} files)")
    return output_file


# ── Dataset ────────────────────────────────────────────────────────────────────

class GrammarDataset(Dataset):
    """Loads JSONL instruction-tuning data."""

    def __init__(self, file_path: str, tokenizer, max_length: int = 1024):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Count lines
        with open(file_path, encoding="utf-8") as f:
            self.size = sum(1 for _ in f)

        print(f"  Dataset: {self.size:,} samples from {file_path}")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        with open(self.file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == idx:
                    obj = json.loads(line)
                    text = obj["text"]

                    # Tokenize — only labels for language modeling
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors=None,
                    )
                    return {
                        "input_ids": encoding["input_ids"],
                        "attention_mask": encoding.get("attention_mask", [1] * len(encoding["input_ids"])),
                        "labels": encoding["input_ids"],
                    }
        raise IndexError(idx)


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    data_file: str,
    output_dir: str = None,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_seq_length: int = 1024,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    save_steps: int = 500,
    seed: int = 42,
):
    """Fine-tune Qwen2.5-3B with QLoRA."""
    set_seed(seed)

    if output_dir is None:
        output_dir = str(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Qwen2.5-3B Grammar Fine-tuning (QLoRA)")
    print(f"{'='*60}")

    # ── Load tokenizer ──────────────────────────────────────────────────────────
    print(f"\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # ── Load model with QLoRA ─────────────────────────────────────────────────
    print(f"\n[2] Loading Qwen2.5-3B with 4-bit QLoRA...")
    bnb_config = {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
    }
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        device_map="auto",
        trust_remote_code=True,
        quantization_config=type(
            __import__('transformers').modeling_utils.QuantizationConfig
        )(**bnb_config),
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA config ─────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Dataset ────────────────────────────────────────────────────────────────
    print(f"\n[3] Loading dataset...")
    dataset = GrammarDataset(data_file, tokenizer, max_length=max_seq_length)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_seq_length)

    # ── Training arguments ─────────────────────────────────────────────────────
    steps_per_epoch = len(dataset) // (per_device_train_batch_size * gradient_accumulation_steps)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=True,
        tf32=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        seed=seed,
        optim="paged_adamw_8bit",
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[4] Starting training...")
    print(f"  Epochs: {num_train_epochs}")
    print(f"  Batch size: {per_device_train_batch_size} × {gradient_accumulation_steps} = "
          f"{per_device_train_batch_size * gradient_accumulation_steps}")
    print(f"  Steps: {steps_per_epoch * num_train_epochs:,}")
    print(f"  LR: {learning_rate}")
    print(f"  LoRA: r={lora_r}, alpha={lora_alpha}")
    print(f"  Max sequence: {max_seq_length}")
    print(f"  Output: {output_dir}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    # ── Save final model ───────────────────────────────────────────────────────
    final_path = Path(output_dir) / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\n  Saved to {final_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen for Tibetan grammar")
    parser.add_argument("--generate_data", action="store_true",
                        help="Generate training data from SegPOS")
    parser.add_argument("--train", action="store_true",
                        help="Run training")
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of training samples to generate")
    parser.add_argument("--data_file", type=str,
                        help="Path to training JSONL file")
    parser.add_argument("--output_dir", type=str,
                        help="Output directory for model")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_tokens", type=int, default=40,
                        help="Max content tokens per sample")
    args = parser.parse_args()

    if args.generate_data:
        generate_training_data(
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
        )

    elif args.train:
        if not args.data_file:
            raise ValueError("--data_file required for training")
        train(
            data_file=args.data_file,
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accum,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            max_seq_length=args.max_length,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
