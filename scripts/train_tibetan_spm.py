"""
Tibetan SentencePiece Tokenizer Training
从头训练藏文分词器 - 基于古典藏文佛典语料
"""

import os
import re
import json
from pathlib import Path
import sentencepiece as spm


class TibetanSPMTrainer:
    """藏文SentencePiece分词器训练器"""

    def __init__(self, corpus_dir: str, output_dir: str):
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_corpus(self, combined_json: str = None) -> list:
        """加载语料库"""
        if combined_json is None:
            combined_json = self.corpus_dir / "extracted" / "combined.json"

        with open(combined_json, 'r', encoding='utf-8') as f:
            corpus = json.load(f)

        sentences = []
        total_chars = 0

        for collection, documents in corpus.items():
            for doc_id, paragraphs in documents.items():
                for para in paragraphs:
                    # 按shad(།)分句
                    sents = self.split_sentences(para)
                    sentences.extend(sents)
                    total_chars += len(para)

        print(f"Loaded {len(sentences):,} sentences")
        print(f"Total characters: {total_chars:,}")
        return sentences

    def split_sentences(self, text: str) -> list:
        """按藏文shad(།)分句"""
        # 保留tsheg用于训练
        sentences = text.split('\u0f0d')  # །
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]

    def prepare_training_file(self, sentences: list, filename: str = "train.txt"):
        """生成SentencePiece训练文件"""
        output_path = self.output_dir / filename

        # 过滤有效藏文字符
        def is_valid_tibetan(text: str) -> bool:
            tibetan_count = sum(1 for c in text if 0x0f40 <= ord(c) <= 0x0fbc)
            return tibetan_count > len(text) * 0.3

        valid_sentences = [s for s in sentences if is_valid_tibetan(s)]

        with open(output_path, 'w', encoding='utf-8') as f:
            for sent in valid_sentences:
                f.write(sent + '\n')

        print(f"Training file: {output_path}")
        print(f"Valid sentences: {len(valid_sentences):,} / {len(sentences):,}")
        return output_path

    def train(self, train_file: str,
              vocab_size: int = 32000,
              model_type: str = "unigram",
              character_coverage: float = 0.9995,
              model_name: str = "tibetan_sp"):
        """
        训练SentencePiece模型

        Args:
            train_file: 训练文件路径
            vocab_size: 词表大小 (8000, 16000, 32000)
            model_type: 模型类型 (unigram, bpe, char)
            character_coverage: 字符覆盖率
            model_name: 输出模型名前缀
        """
        print(f"\nTraining SentencePiece model...")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Model type: {model_type}")
        print(f"  Character coverage: {character_coverage}")

        # 训练参数
        spm.SentencePieceTrainer.train(
            input=str(train_file),
            model_prefix=str(self.output_dir / model_name),
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            pad_id=0,          # padding
            unk_id=1,          # unknown
            bos_id=2,          # begin of sequence
            eos_id=3,          # end of sequence
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<s>',
            eos_piece='</s>',
        )

        model_file = self.output_dir / f"{model_name}.model"
        vocab_file = self.output_dir / f"{model_name}.vocab"

        print(f"\nTraining completed!")
        print(f"Model: {model_file}")
        print(f"Vocab: {vocab_file}")

        return str(model_file)

    def load_and_analyze(self, model_file: str):
        """加载模型并分析"""
        sp = spm.SentencePieceProcessor()
        sp.load(model_file)

        vocab_size = sp.get_piece_size()
        print(f"\nModel analysis:")
        print(f"  Vocabulary size: {vocab_size:,}")

        # 统计各种类型的token
        tibetan_tokens = 0
        special_tokens = 0
        latin_tokens = 0

        for i in range(vocab_size):
            piece = sp.id_to_piece(i)
            if piece.startswith('<'):
                special_tokens += 1
            elif re.match(r'^[\u0f40-\u0fbc]+$', piece):
                tibetan_tokens += 1
            else:
                latin_tokens += 1

        print(f"  Tibetan tokens: {tibetan_tokens:,}")
        print(f"  Special tokens: {special_tokens:,}")
        print(f"  Other tokens: {latin_tokens:,}")

        return sp

    def encode_sample(self, model_file: str, sample_texts: list):
        """对样本进行编码测试"""
        sp = spm.SentencePieceProcessor()
        sp.load(model_file)

        print(f"\nEncoding samples:")
        for text in sample_texts[:3]:
            pieces = sp.encode(text, out_type=str)
            ids = sp.encode(text, out_type=int)
            print(f"  Text: {text[:50]}...")
            print(f"  Pieces: {pieces[:10]}...")
            print(f"  IDs: {ids[:10]}...")
            print()


def merge_vocabularies(tibert_vocab: str, spm_vocab: str, output_file: str):
    """
    合并TiBERT词汇表和SentencePiece词汇表

    Args:
        tibert_vocab: TiBERT vocab.txt路径
        spm_vocab: SentencePiece vocab.txt路径
        output_file: 输出合并后的词汇表路径
    """
    print("\nMerging vocabularies...")

    # 读取TiBERT词汇表
    with open(tibert_vocab, 'r', encoding='utf-8') as f:
        tibert_words = set(line.strip() for line in f)

    # 读取SPM词汇表
    with open(spm_vocab, 'r', encoding='utf-8') as f:
        spm_words = set(line.split('\t')[0] for line in f)

    # 合并
    merged = tibert_words | spm_words

    print(f"  TiBERT vocab: {len(tibert_words):,}")
    print(f"  SPM vocab: {len(spm_words):,}")
    print(f"  Merged vocab: {len(merged):,}")
    print(f"  New tokens: {len(merged - tibert_words):,}")

    # 保存合并后的词汇表
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in sorted(merged):
            f.write(word + '\n')

    print(f"  Saved to: {output_file}")
    return merged


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Tibetan SentencePiece Training')
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    parser.add_argument('--corpus', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'corpus' / 'extracted' / 'combined.json'),
                        help='Combined corpus JSON file')
    parser.add_argument('--output', type=str,
                        default=str(Path(__file__).parent),
                        help='Output directory')
    parser.add_argument('--vocab_size', type=int, default=32000,
                        help='Vocabulary size (8000, 16000, 32000)')
    parser.add_argument('--model_type', type=str, default='unigram',
                        choices=['unigram', 'bpe', 'char'],
                        help='SentencePiece model type')
    parser.add_argument('--tibert_vocab', type=str,
                        default=str(PROJECT_ROOT / 'model' / 'TiBERT' / 'vocab.txt'),
                        help='TiBERT vocabulary file')
    parser.add_argument('--merge', action='store_true',
                        help='Merge with TiBERT vocabulary')

    args = parser.parse_args()

    trainer = TibetanSPMTrainer(
        corpus_dir=args.output,
        output_dir=args.output
    )

    # 1. 加载语料
    print("=" * 60)
    print("STEP 1: Loading corpus...")
    sentences = trainer.load_corpus(combined_json=args.corpus)

    # 2. 生成训练文件
    print("\n" + "=" * 60)
    print("STEP 2: Preparing training file...")
    train_file = trainer.prepare_training_file(sentences)

    # 3. 训练模型
    print("\n" + "=" * 60)
    print("STEP 3: Training SentencePiece model...")
    model_file = trainer.train(
        train_file=train_file,
        vocab_size=args.vocab_size,
        model_type=args.model_type
    )

    # 4. 分析模型
    print("\n" + "=" * 60)
    print("STEP 4: Analyzing model...")
    sp = trainer.load_and_analyze(model_file)

    # 5. 编码测试
    print("\n" + "=" * 60)
    print("STEP 5: Encoding test...")
    sample = ["༄༅། །ཤེས་རབ་ཀྱི་ཕ་རོལ་ཏུ་ཕྱིན་པ།",
              "༄༅༅། །ཉི་ཁྲི་ཁ་པ་བཞུགས་སོ།"]
    trainer.encode_sample(model_file, sample)

    # 6. 合并词汇表（可选）
    if args.merge:
        print("\n" + "=" * 60)
        print("STEP 6: Merging vocabularies...")
        spm_vocab = str(Path(args.output) / "tibetan_sp.vocab")
        merged_vocab = str(Path(args.output) / "merged_vocab.txt")
        merge_vocabularies(args.tibert_vocab, spm_vocab, merged_vocab)

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
