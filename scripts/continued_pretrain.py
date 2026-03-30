"""
TiBERT Continued Pre-training on Classical Tibetan Corpus
TiBERT继续预训练 - 使用古典藏文佛典语料

用法:
    # 继续预训练TiBERT (使用原生vocab)
    python continued_pretrain.py --corpus ../data/corpus/extracted/combined.json

    # 使用更大的batch size和多GPU
    python continued_pretrain.py --batch_size 32 --epochs 5
"""

import os
import json
import random
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    BertConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
    PreTrainedTokenizer,
)
from tqdm import tqdm
import sentencepiece as spm


# ──────────────────────────────────────────────────────────────────────────────
#  ClassicalTibetanTokenizer
#  A PreTrainedTokenizer wrapper around the trained SentencePiece model.
#  Adds BERT special tokens on top of the SPM vocabulary.
# ──────────────────────────────────────────────────────────────────────────────
# BERT specials are inserted at the beginning (ids 0-4)

class ClassicalTibetanTokenizer(PreTrainedTokenizer):
    """
    Wraps a trained SentencePiece model for classical Tibetan as a
    PreTrainedTokenizer so it works directly with BertForMaskedLM.

    ID layout:
      0  [PAD]
      1  [UNK]
      2  [CLS]  ← also maps to SPM bos_id=2
      3  [SEP]  ← also maps to SPM eos_id=3
      4  [MASK]
      5+ SPM tokens

    The tsheg (་) and shad (།) are kept as standalone tokens by SPM,
    which is ideal for Tibetan morphological analysis.
    """

    def __init__(
        self,
        spm_model_file: str,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        bos_token: str = "[CLS]",
        eos_token: str = "[SEP]",
        mask_token: str = "[MASK]",
        model_max_length: int = 512,
        **kwargs,
    ):
        # Load the trained SPM model
        self.spm = spm.SentencePieceProcessor()
        self.spm.Load(spm_model_file)
        self._spm_model_path = spm_model_file  # remember path for save

        # Build the full vocabulary: BERT specials + SPM pieces
        # SPM pieces start from its own offset; we prepend specials.
        self._spm_vocab_size = self.spm.get_piece_size()
        self._spm_offset = 5  # BERT specials occupy 0-4

        # Build token-to-id and id-to-token maps
        self._token2id: Dict[str, int] = {}
        self._id2token: Dict[int, str] = {}

        # BERT special tokens first (ids 0-4)
        for i, tok in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]):
            self._token2id[tok] = i
            self._id2token[i] = tok

        # Tibetan punctuation as special tokens (ids 5-6)
        # These are UNK in the SPM model but grammatically essential
        TIBETAN_SPECIAL = [("།", "[SHAD]"), ("་", "[TSEG]")]
        for new_id, (tib_char, name) in enumerate(TIBETAN_SPECIAL, start=5):
            self._token2id[tib_char] = new_id
            self._id2token[new_id] = tib_char
        self._spm_offset = 5 + len(TIBETAN_SPECIAL)  # = 7

        for spm_id in range(self._spm_vocab_size):
            piece = self.spm.id_to_piece(spm_id)
            new_id = spm_id + self._spm_offset
            self._token2id[piece] = new_id
            self._id2token[new_id] = piece

        # Init base class FIRST (sets up _special_tokens_map etc.)
        super().__init__(
            model_max_length=model_max_length,
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            **kwargs,
        )

        # Override the id attributes with our SPM-aligned values
        self._pad_token_id = self._token2id["[PAD]"]
        self._unk_token_id = self._token2id["[UNK]"]
        self._bos_token_id = self._token2id["[CLS]"]
        self._eos_token_id = self._token2id["[SEP]"]
        self._mask_token_id = self._token2id["[MASK]"]

    @property
    def vocab_size(self) -> int:
        return self._spm_offset + self._spm_vocab_size

    # ── Core encoding methods ───────────────────────────────────────────────

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """SPM subword tokenization with U+2581 prefix stripping."""
        pieces = self.spm.encode(text, out_type=str)
        result = []
        for p in pieces:
            if p == "\u2581":
                continue  # skip standalone U+2581 (whitespace)
            elif p.startswith("\u2581"):
                result.append(p[1:])  # strip U+2581 prefix
            else:
                result.append(p)
        return result

    def _convert_token_to_id(self, token: str) -> int:
        if token.startswith("\u2581"):
            stripped = token[1:]
            # If stripping leaves content, use stripped form (normal subword)
            # If stripping leaves nothing (token was just U+2581), keep full token
            token = stripped if stripped else token
        if not token:
            return self._token2id["[PAD]"]
        return self._token2id.get(token, self._token2id["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        unk_id = self._token2id.get("[UNK]", 1)
        return self._id2token.get(index, self._id2token.get(unk_id, "[UNK]"))
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Join SPM pieces back into Tibetan text.

        Since _tokenize already strips ▁ prefixes, tokens here are plain Tibetan syllables.
        We insert ་ (tsheg) between consecutive consonant pieces to approximate syllable structure,
        but the safest approach is simple concatenation.
        """
        return "".join(tokens)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._token2id)

    # ── Encode methods (required by DataCollatorForLanguageModeling) ────────

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        **kwargs,
    ):
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        return self.convert_tokens_to_ids(tokens)

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs) -> str:
        if skip_special_tokens:
            tokens = [
                self._id2token[i]
                for i in token_ids
                if i not in (self.pad_token_id, self.bos_token_id,
                             self.eos_token_id, self.mask_token_id)
            ]
        else:
            tokens = [self._id2token.get(i, self._token2id["[UNK]"]) for i in token_ids]
        return self.convert_tokens_to_string(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        """Save tokenizer files (vocab + SPM model)."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save vocab.json
        prefix = f"{filename_prefix}" if filename_prefix else ""
        vocab_file = save_dir / f"{prefix}vocab.json"
        import json as _json
        with open(vocab_file, "w", encoding="utf-8") as f:
            _json.dump(self._token2id, f, ensure_ascii=False, indent=2)

        # Copy SPM model
        spm_file = save_dir / f"{prefix}spm.model"
        import shutil
        shutil.copy(self._spm_model_path, spm_file)

        return (str(vocab_file), str(spm_file))


class ClassicalTibetanDataset(Dataset):
    """古典藏文数据集"""

    def __init__(self, corpus_json: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        """
        初始化数据集

        Args:
            corpus_json: 语料库JSON文件路径
            tokenizer: BERT分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loading corpus from {corpus_json}...")
        with open(corpus_json, 'r', encoding='utf-8') as f:
            corpus = json.load(f)

        # 按文档分句
        self.texts = []
        self._load_texts(corpus)

        print(f"Loaded {len(self.texts):,} texts/documents")

    def _load_texts(self, corpus: dict):
        """加载语料文本"""
        for collection, documents in corpus.items():
            for doc_id, paragraphs in documents.items():
                for para in paragraphs:
                    # 按shad(།)分句
                    sentences = para.split('\u0f0d')  # །
                    for sent in sentences:
                        sent = sent.strip()
                        if len(sent) > 10:  # 过滤短句
                            self.texts.append(sent)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]

        # tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


class TokenizedDataset(Dataset):
    """
    预分词缓存数据集（numpy mmap 模式）— 将 tokenize 结果存为 .npy 文件，
    全部通过内存映射访问，不占用 RAM，支持多进程并行训练。
    """

    def __init__(self, cache_dir: str, max_samples: Optional[int] = None):
        import numpy as np
        ids_path = Path(cache_dir) / "input_ids.npy"
        mask_path = Path(cache_dir) / "attention_mask.npy"

        # 尝试加载 mmap（不占用 RAM）
        self._ids_mmap = np.load(ids_path, mmap_mode='r')
        self._mask_mmap = np.load(mask_path, mmap_mode='r')

        self._num_samples = min(
            len(self._ids_mmap),
            max_samples if max_samples is not None else len(self._ids_mmap)
        )
        print(f"  Memory-mapped {self._num_samples:,} samples from {cache_dir}")

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            'input_ids': torch.from_numpy(self._ids_mmap[idx].copy()),
            'attention_mask': torch.from_numpy(self._mask_mmap[idx].copy()),
        }


def build_tokenized_cache(
    corpus_json: str,
    tokenizer: PreTrainedTokenizer,
    output_path: str,
    max_length: int = 256,
    max_samples: Optional[int] = None,
    batch_size: int = 1000,
    desc: str = "Tokenizing"
) -> str:
    """
    将语料预分词并缓存到磁盘，避免每个 epoch 重复 tokenize。

    Args:
        corpus_json: 语料 JSON 路径
        tokenizer: 分词器
        output_path: 缓存文件保存路径 (.pkl)
        max_length: 最大序列长度
        max_samples: 最大样本数（None = 全部）
        batch_size: 每批 tokenize 的样本数
        desc: 进度条描述

    Returns:
        缓存文件路径
    """
    npy_dir = Path(output_path)
    ids_file = npy_dir / "input_ids.npy"
    if ids_file.exists():
        print(f"  Cache already exists at {npy_dir}, skipping tokenization.")
        return str(npy_dir)

    # 加载语料
    print(f"  Loading corpus from {corpus_json} ...")
    with open(corpus_json, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    texts = []
    for collection, documents in corpus.items():
        for doc_id, paragraphs in documents.items():
            for para in paragraphs:
                sentences = para.split('\u0f0d')
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) > 10:
                        texts.append(sent)

    if max_samples is not None:
        texts = texts[:max_samples]

    print(f"  Tokenizing {len(texts):,} sentences ...")

    all_input_ids = []
    all_attention_mask = []

    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i:i + batch_size]
        encodings = tokenizer(
            batch,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        all_input_ids.append(encodings['input_ids'])
        all_attention_mask.append(encodings['attention_mask'])

    all_input_ids = np.concatenate(all_input_ids, axis=0)
    all_attention_mask = np.concatenate(all_attention_mask, axis=0)

    # 保存为 numpy（每个 .npy 可被 mmap）
    npy_dir.mkdir(parents=True, exist_ok=True)
    ids_file = npy_dir / "input_ids.npy"
    mask_file = npy_dir / "attention_mask.npy"

    np.save(ids_file, all_input_ids.astype(np.int64))
    np.save(mask_file, all_attention_mask.astype(np.int64))

    print(f"  NPY cache saved: {ids_file} ({all_input_ids.shape}), {mask_file} ({all_attention_mask.shape})")
    return str(npy_dir)


def create_continued_pretrain_dataset(
    corpus_json: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    sample_ratio: float = 0.3
) -> ClassicalTibetanDataset:
    """
    创建继续预训练数据集

    Args:
        corpus_json: 语料库JSON文件路径
        tokenizer: BERT分词器
        max_length: 最大序列长度
        sample_ratio: 采样比例（用于快速测试）

    Returns:
        ClassicalTibetanDataset实例
    """
    return ClassicalTibetanDataset(corpus_json, tokenizer, max_length)


def expand_tokenizer_with_spm(
    tibert_dir: str,
    spm_vocab_file: str,
    output_dir: str
) -> Tuple[BertTokenizerFast, int]:
    """
    将SentencePiece词汇表中的新token添加到TiBERT分词器

    Args:
        tibert_dir: TiBERT模型目录
        spm_vocab_file: SentencePiece词汇表文件
        output_dir: 输出目录

    Returns:
        (tokenizer, num_added_tokens) - 更新后的tokenizer和新添加的token数量
    """
    print("\n" + "=" * 50)
    print("Expanding TiBERT tokenizer with SPM vocabulary...")
    print("=" * 50)

    # 加载原始TiBERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(tibert_dir)
    original_vocab_size = len(tokenizer)

    # 读取TiBERT词汇
    tibert_vocab_path = Path(tibert_dir) / "vocab.txt"
    with open(tibert_vocab_path, 'r', encoding='utf-8') as f:
        tibert_tokens = set(line.strip().split('\t')[0] for line in f)

    print(f"TiBERT vocab size: {len(tibert_tokens)}")

    # 读取SPM词汇
    with open(spm_vocab_file, 'r', encoding='utf-8') as f:
        spm_tokens = set(line.strip().split('\t')[0] for line in f)

    print(f"SPM vocab size: {len(spm_tokens)}")

    # 找出新token (在SPM中但不在TiBERT中的)
    new_tokens = spm_tokens - tibert_tokens

    # 过滤掉太短的token和特殊token
    new_tokens = {
        t for t in new_tokens
        if len(t) >= 2 and not t.startswith('<') and not t.startswith('▁')
    }

    print(f"New tokens to add: {len(new_tokens)}")

    # 显示一些新token示例
    sample_new = sorted(new_tokens)[:20]
    print(f"Sample new tokens: {sample_new}")

    # 添加新tokens
    if new_tokens:
        num_added = tokenizer.add_tokens(list(new_tokens))
        print(f"Added {num_added} tokens to tokenizer")
    else:
        num_added = 0

    # 保存扩展后的tokenizer
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_path))

    # 更新模型的embeddings以适应新的vocab_size
    print(f"Original vocab size: {original_vocab_size}")
    print(f"New vocab size: {len(tokenizer)}")

    print("=" * 50)
    return tokenizer, num_added


def resize_model_embeddings(model, new_vocab_size, device: str = 'cuda'):
    """
    调整模型的embedding层以适应新的vocab大小

    Args:
        model: BERT模型
        new_vocab_size: 新的vocab大小
        device: 设备
    """
    model.resize_token_embeddings(new_vocab_size)
    print(f"Resized model embeddings to {new_vocab_size}")


def evaluate_model(model, tokenizer, test_texts: List[str], device: str = 'cuda') -> Dict:
    """
    评估模型在古典藏文上的MLM表现

    Args:
        model: TiBERT模型
        tokenizer: 分词器
        test_texts: 测试文本列表
        device: 设备

    Returns:
        评估结果字典
    """
    model.eval()

    total_loss = 0
    total_tokens = 0

    # 经典的古典藏文测试句
    classic_samples = [
        "༄༅། །ཤེས་རབ་ཀྱི་ཕ་རོལ་ཏུ་ཕྱིན་པ།",  # 般若波罗蜜多心经
        "༄༅༅། །ཉི་ཁྲི་ཁ་པ་བཞུགས་སོ།",  # 金刚经开头
        "སངས་རྒྱས་ཆོས་ཐམས་ཅད་མཁྱེན་པ་དང་།",  # 佛智
        "བདེ་གཤེགས་ཆོས་ཀྱི་གཟུགས་མཆོང་།",  # 佛陀
    ]

    print("\n" + "=" * 50)
    print("Evaluating model on classical Tibetan texts...")
    print("=" * 50)

    with torch.no_grad():
        for text in classic_samples:
            # tokenize
            inputs = tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 计算 MLM loss（将 input_ids 作为 labels）
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                labels=inputs['input_ids'],
                return_dict=True
            )
            loss = outputs.loss.item()

            num_tokens = inputs['input_ids'].numel()
            total_loss += loss * num_tokens
            total_tokens += num_tokens

            print(f"  Text: {text[:40]}...")
            print(f"  Loss: {loss:.4f}")

    avg_loss = total_loss / total_tokens if total_tokens > 0 else total_loss
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    results = {
        'avg_loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens
    }

    print(f"\n  Average Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.4f}")

    model.train()
    return results


def load_tibert_for_continued_pretrain(
    model_dir: str = None,
    device: str = None
) -> Tuple[BertForMaskedLM, BertTokenizerFast]:
    """
    加载TiBERT模型用于继续预训练

    Args:
        model_dir: TiBERT模型目录
        device: 设备 ('cuda' or 'cpu')

    Returns:
        (model, tokenizer)元组
    """
    if model_dir is None:
        model_dir = Path(__file__).parent / "TiBERT"
    else:
        model_dir = Path(model_dir)

    # 确定设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载tokenizer - 使用TiBERT自带的vocab (使用Fast版本以加速)
    tokenizer = BertTokenizerFast.from_pretrained(str(model_dir))
    print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")

    # 加载模型
    model = BertForMaskedLM.from_pretrained(str(model_dir))
    model.gradient_checkpointing_enable()  # 减少激活内存占用
    model.to(device)
    print(f"Loaded TiBERT model")
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Num layers: {model.config.num_hidden_layers}")
    print(f"  Num attention heads: {model.config.num_attention_heads}")

    return model, tokenizer


def train_tibert_continued(
    model,
    train_dataset,
    tokenizer,
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-5,
    warmup_steps: int = 1000,
    save_steps: int = 5000,
    logging_steps: int = 500,
    seed: int = 42
):
    """
    继续预训练TiBERT

    Args:
        model: TiBERT模型
        train_dataset: 训练数据集
        tokenizer: 分词器
        output_dir: 输出目录
        num_train_epochs: 训练轮数
        per_device_train_batch_size: 批大小
        learning_rate: 学习率
        warmup_steps: 预热步数
        save_steps: 保存步数
        logging_steps: 日志步数
        seed: 随机种子
    """
    set_seed(seed)

    # 数据整理器 - 用于MLM (需要tokenizer来正确处理padding)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  # 15%的token被mask
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=3,
        prediction_loss_only=True,
        logging_first_step=True,
        dataloader_num_workers=0,
        seed=seed,
        fp16=torch.cuda.is_available(),
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    print("\n" + "=" * 60)
    print("Starting Continued Pre-training...")
    print("=" * 60)
    print(f"  Num train examples: {len(train_dataset):,}")
    print(f"  Epochs: {num_train_epochs}")
    print(f"  Batch size: {per_device_train_batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  MLM probability: 0.15")
    print(f"  Output dir: {output_dir}")
    print("=" * 60 + "\n")

    # 开始训练
    trainer.train()

    return trainer


def main():
    import argparse

    # 自动检测项目根目录（向上两级）
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent

    parser = argparse.ArgumentParser(description='TiBERT Continued Pre-training')
    parser.add_argument('--corpus', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'corpus' / 'extracted' / 'combined.json'),
                        help='Combined corpus JSON file')
    parser.add_argument('--model_dir', type=str,
                        default=str(PROJECT_ROOT / 'model' / 'TiBERT'),
                        help='TiBERT model directory')
    parser.add_argument('--output_dir', type=str,
                        default=str(PROJECT_ROOT / 'model' / 'TiBERT-classical'),
                        help='Output directory for continued pre-trained model')
    parser.add_argument('--spm_vocab', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'corpus' / 'spm' / 'tibetan_sp.vocab'),
                        help='SentencePiece vocab file for vocabulary expansion')
    parser.add_argument('--expand_vocab', action='store_true',
                        help='Expand TiBERT vocabulary with SPM tokens')
    parser.add_argument('--use_spm_tokenizer', action='store_true',
                        help='Use the trained SPM tokenizer (classical Tibetan-optimized, 0%% UNK)')
    parser.add_argument('--spm_model', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'corpus' / 'spm' / 'tibetan_sp.model'),
                        help='Path to trained SentencePiece model file')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps (effective_batch = batch_size * grad_accum * n_gpus)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Warmup steps')
    parser.add_argument('--save_steps', type=int, default=5000,
                        help='Save steps')
    parser.add_argument('--logging_steps', type=int, default=500,
                        help='Logging steps')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help='Sampling ratio for quick testing (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--eval_before_train', action='store_true',
                        help='Evaluate model before training')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training (only expand vocab and save)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of training samples (default: all)')
    parser.add_argument('--use_cache', action='store_true', default=True,
                        help='Use pre-tokenized cache (default: True)')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable pre-tokenized cache')
    parser.add_argument('--build_cache', action='store_true',
                        help='Only build tokenized cache, then exit')

    args = parser.parse_args()

    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TiBERT Continued Pre-training")
    print("古典藏文TiBERT继续预训练")
    print("=" * 60)

    # 确定设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 加载模型和tokenizer
    print("\n[Step 1] Loading model and tokenizer...")

    if args.use_spm_tokenizer:
        # ── 使用训练好的SPM分词器（古典藏文优化，0%% UNK）───────────────
        print(f"  Loading classical Tibetan SPM tokenizer: {args.spm_model}")
        tokenizer = ClassicalTibetanTokenizer(
            spm_model_file=args.spm_model,
            model_max_length=args.max_length,
        )
        print(f"  SPM tokenizer vocab size: {tokenizer.vocab_size:,}")

        # 加载原始TiBERT模型并resize embeddings以匹配新vocab
        print(f"  Loading TiBERT base model from: {args.model_dir}")
        model = BertForMaskedLM.from_pretrained(args.model_dir)
        old_vocab = model.config.vocab_size
        model.resize_token_embeddings(tokenizer.vocab_size)
        print(f"  Resized embeddings: {old_vocab:,} → {tokenizer.vocab_size:,}")
        model.gradient_checkpointing_enable()
        model.to(device)

    elif args.expand_vocab:
        # 扩展词汇表
        tokenizer, num_added = expand_tokenizer_with_spm(
            args.model_dir,
            args.spm_vocab,
            str(output_dir / "expanded_tokenizer")
        )

        # 加载模型并resize embeddings
        model = BertForMaskedLM.from_pretrained(args.model_dir)
        model.gradient_checkpointing_enable()
        if num_added > 0:
            resize_model_embeddings(model, len(tokenizer), device)
        model.to(device)

        # 更新model_dir以使用扩展后的tokenizer
        args.model_dir = str(output_dir / "expanded_tokenizer")
    else:
        model, tokenizer = load_tibert_for_continued_pretrain(args.model_dir, device)

    # 2. 评估训练前的模型 (可选)
    if args.eval_before_train:
        print("\n[Step 2] Evaluating before training...")
        evaluate_model(model, tokenizer, [], device)

    # 3. 创建数据集
    print("\n[Step 3] Creating training dataset...")
    use_cache = not args.no_cache
    cache_dir = output_dir / "tokenized_cache"

    # --build_cache: 仅构建缓存后退出（不需要模型）
    if args.build_cache:
        print("Building tokenized cache ...")
        if args.use_spm_tokenizer:
            tokenizer_cache = ClassicalTibetanTokenizer(
                spm_model_file=args.spm_model,
                model_max_length=args.max_length,
            )
        else:
            from transformers import BertTokenizerFast
            tokenizer_cache = BertTokenizerFast.from_pretrained(str(Path(args.model_dir)))
        build_tokenized_cache(
            corpus_json=args.corpus,
            tokenizer=tokenizer_cache,
            output_path=str(cache_dir),
            max_length=args.max_length,
            max_samples=args.max_samples,
            desc="Building cache"
        )
        print("Cache build complete!")
        return

    npy_ids = cache_dir / "input_ids.npy"
    if use_cache and npy_ids.exists():
        print("  Using pre-tokenized cache (mmap) ...")
        train_dataset = TokenizedDataset(str(cache_dir), max_samples=args.max_samples)
    else:
        if use_cache:
            print("  Building tokenized cache (one-time, saves to disk) ...")
            build_tokenized_cache(
                corpus_json=args.corpus,
                tokenizer=tokenizer,
                output_path=str(cache_dir),
                max_length=args.max_length,
                max_samples=args.max_samples,
            )
        train_dataset = ClassicalTibetanDataset(
            corpus_json=args.corpus,
            tokenizer=tokenizer,
            max_length=args.max_length
        )
        if args.max_samples is not None:
            train_dataset.texts = train_dataset.texts[:args.max_samples]
            print(f"  Limited to {args.max_samples:,} samples for training")
        train_dataset = ClassicalTibetanDataset(
            corpus_json=args.corpus,
            tokenizer=tokenizer,
            max_length=args.max_length
        )
        if args.max_samples is not None:
            train_dataset.texts = train_dataset.texts[:args.max_samples]
            print(f"  Limited to {args.max_samples:,} samples for training")

    if args.skip_train:
        print("\n[Skip] Training skipped as requested.")
        # 只保存tokenizer
        tokenizer.save_pretrained(str(output_dir / "tokenizer"))
        print(f"Tokenizer saved to: {output_dir / 'tokenizer'}")
        return

    # 4. 继续预训练
    print("\n[Step 4] Training...")
    trainer = train_tibert_continued(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        seed=args.seed
    )

    # 5. 保存模型
    print("\n[Step 5] Saving model...")
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    print(f"Saved to: {final_model_path}")

    # 6. 评估训练后的模型
    print("\n[Step 6] Evaluating after training...")
    evaluate_model(model, tokenizer, [], device)

    print("\n" + "=" * 60)
    print("Continued Pre-training Complete!")
    print("=" * 60)
    print(f"\nTo use the continued pre-trained model:")
    print(f"  Model: {final_model_path}")
    print(f"  Tokenizer: {output_dir / 'tokenizer' if not args.expand_vocab else output_dir / 'expanded_tokenizer'}")


if __name__ == "__main__":
    main()