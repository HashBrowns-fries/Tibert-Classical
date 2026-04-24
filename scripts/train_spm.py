"""
Tibetan SentencePiece Tokenizer Training
使用 SentencePiece 对藏文语料进行分词训练
"""

import os
import re
import json
from pathlib import Path
import sentencepiece as spm


class TibetanSentencePieceTrainer:
    """藏文SentencePiece分词器训练器"""

    def __init__(self, corpus_dir: str, output_dir: str):
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_corpus(self, combined_json: str) -> list:
        """加载语料库"""
        with open(combined_json, 'r', encoding='utf-8') as f:
            corpus = json.load(f)

        sentences = []
        for collection, documents in corpus.items():
            for doc_id, paragraphs in documents.items():
                for para in paragraphs:
                    # 按shad(།)分句
                    sentences.extend(self.split_sentences(para))

        print(f"Loaded {len(sentences)} sentences")
        return sentences

    def split_sentences(self, text: str) -> list:
        """按藏文shad(།)分句"""
        # 移除tsheg(་)进行分句
        text_no_tsheg = text.replace('\u0f0b', ' ')
        sentences = text_no_tsheg.split('\u0f0d')  # །
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]

    def prepare_training_file(self, sentences: list, filename: str = "train.txt"):
        """生成SentencePiece训练文件"""
        output_path = self.output_dir / filename

        # 过滤有效藏文字符
        def is_tibetan(text: str) -> bool:
            tibetan_count = sum(1 for c in text if 0x0f40 <= ord(c) <= 0x0fbc)
            return tibetan_count > len(text) * 0.5

        valid_sentences = [s for s in sentences if is_tibetan(s)]

        with open(output_path, 'w', encoding='utf-8') as f:
            for sent in valid_sentences:
                f.write(sent + '\n')

        print(f"Training file saved: {output_path}")
        print(f"Valid sentences: {len(valid_sentences)} / {len(sentences)}")
        return output_path

    def train(self, train_file: str,
              vocab_size: int = 8000,
              model_type: str = "unigram",
              character_coverage: float = 0.9995,
              model_name: str = "tibetan"):
        """
        训练SentencePiece模型

        Args:
            train_file: 训练文件路径
            vocab_size: 词表大小 (8000, 16000, 32000)
            model_type: 模型类型 (unigram, bpe, char, word)
            character_coverage: 字符覆盖率
            model_name: 输出模型名前缀
        """
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
        print(f"Model file: {model_file}")
        print(f"Vocab file: {vocab_file}")

        return str(model_file)

    def encode_corpus(self, combined_json: str, model_file: str, output_file: str):
        """
        使用训练好的模型对语料进行编码
        """
        # 加载模型
        sp = spm.SentencePieceProcessor()
        sp.load(model_file)

        # 加载语料
        with open(combined_json, 'r', encoding='utf-8') as f:
            corpus = json.load(f)

        encoded_corpus = {}
        total_tokens = 0

        for collection, documents in corpus.items():
            encoded_corpus[collection] = {}
            for doc_id, paragraphs in documents.items():
                encoded_paragraphs = []
                for para in paragraphs:
                    # 编码
                    ids = sp.encode(para, out_type=int)
                    pieces = sp.encode(para, out_type=str)
                    encoded_paragraphs.append({
                        'ids': ids,
                        'pieces': pieces
                    })
                    total_tokens += len(ids)
                encoded_corpus[collection][doc_id] = encoded_paragraphs

        # 保存编码后的语料
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(encoded_corpus, f, ensure_ascii=False, indent=2)

        print(f"\nEncoding completed!")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Encoded corpus saved: {output_path}")

        # 统计词表
        vocab_size = sp.get_piece_size()
        print(f"Vocabulary size: {vocab_size}")

        return encoded_corpus


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
    parser.add_argument('--vocab_size', type=int, default=8000,
                        help='Vocabulary size')
    parser.add_argument('--model_type', type=str, default='unigram',
                        choices=['unigram', 'bpe', 'char', 'word'],
                        help='SentencePiece model type')
    parser.add_argument('--character_coverage', type=float, default=0.9995,
                        help='Character coverage (0.9995-1.0, 1.0=all chars)')
    parser.add_argument('--train_only', action='store_true',
                        help='Only train, skip encoding')

    args = parser.parse_args()

    trainer = TibetanSentencePieceTrainer(
        corpus_dir=args.output,
        output_dir=args.output
    )

    # 1. 加载语料
    print("=" * 50)
    print("Step 1: Loading corpus...")
    sentences = trainer.load_corpus(args.corpus)

    # 2. 生成训练文件
    print("\n" + "=" * 50)
    print("Step 2: Preparing training file...")
    train_file = trainer.prepare_training_file(sentences)

    # 3. 训练模型
    print("\n" + "=" * 50)
    print("Step 3: Training SentencePiece model...")
    model_file = trainer.train(
        train_file=train_file,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
    )

    # 4. 编码语料 (可选)
    if not args.train_only:
        print("\n" + "=" * 50)
        print("Step 4: Encoding corpus...")
        trainer.encode_corpus(args.corpus, model_file, "encoded_corpus.json")

    print("\n" + "=" * 50)
    print("All done!")


if __name__ == "__main__":
    main()
