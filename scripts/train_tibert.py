"""
TiBERT微调训练脚本

从combined.json流式读取语料，对TiBERT进行MLM预训练或下游任务微调
"""

import sys
import json
import torch
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TIBERT_MODEL_NAME


class TibetanCorpusDataset(Dataset):
    """古典藏文语料数据集"""

    def __init__(self, corpus_path: str, model_path: str, max_length: int = 128):
        self.max_length = max_length
        self.sentences = []

        # 加载tokenizer
        vocab_path = Path(model_path) / "vocab.txt"
        vocab = self._load_vocab(vocab_path)
        self.tokenizer = BertTokenizer(vocab=vocab)
        self.tokenizer.do_lower_case = False

        self._load_corpus(corpus_path)

    def _load_corpus(self, corpus_path: str):
        """流式加载语料"""
        print(f"Loading corpus from {corpus_path}...")

        with open(corpus_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 遍历所有文档的所有句子
        for source, documents in data.items():
            for doc_id, sentences in documents.items():
                for sentence in sentences:
                    # 过滤太短的句子
                    if len(sentence) >= 10:
                        self.sentences.append(sentence)

        print(f"Loaded {len(self.sentences)} sentences")

    def _load_vocab(self, vocab_file):
        """Load vocabulary from vocab.txt file"""
        vocab = OrderedDict()
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.rstrip('\r\n')
                vocab[token] = idx
        return vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]
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


class MLMTrainer:
    """MLM训练器"""

    def __init__(
        self,
        model_path: str,
        train_dataset: Dataset,
        output_dir: str,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        epochs: int = 3,
        device: str = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载tokenizer
        print("Loading tokenizer...")
        vocab_path = Path(model_path) / "vocab.txt"
        vocab = self._load_vocab(vocab_path)
        self.tokenizer = BertTokenizer(vocab=vocab)
        self.tokenizer.do_lower_case = False

        # 加载模型
        print("Loading model...")
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.train()

        # 准备数据
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        self.epochs = epochs
        self.batch_size = batch_size

    def _load_vocab(self, vocab_file):
        vocab = OrderedDict()
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.rstrip('\r\n')
                vocab[token] = idx
        return vocab

    def mask_tokens(self, input_ids, attention_mask):
        """随机mask tokens"""
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, 0.15)

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in input_ids
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix.masked_fill_(input_ids == self.tokenizer.pad_token_id, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return input_ids, labels

    def train_step(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        # Mask tokens
        input_ids, labels = self.mask_tokens(input_ids, attention_mask)

        outputs = self.model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # 简单的MLM head
        predictions = torch.nn.functional.linear(
            sequence_output,
            self.model.embeddings.word_embeddings.weight
        )

        loss = torch.nn.CrossEntropyLoss()(
            predictions.view(-1, self.tokenizer.vocab_size),
            labels.view(-1)
        )

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def train(self):
        """训练循环"""
        print(f"\nStarting training for {self.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Output dir: {self.output_dir}")

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")

            for batch in pbar:
                loss = self.train_step(batch)
                total_loss += loss
                num_batches += 1

                pbar.set_postfix({'loss': f'{loss:.4f}'})

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}")

            # 保存检查点
            checkpoint_path = self.output_dir / f"checkpoint-epoch-{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # 保存最终模型
        final_path = self.output_dir / "final_model.pt"
        torch.save(self.model.state_dict(), final_path)
        print(f"Training complete! Final model saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train TiBERT on Tibetan corpus")
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/corpus/extracted/combined.json",
        help="Path to combined.json corpus file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to TiBERT model (default: from config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples for quick testing"
    )

    args = parser.parse_args()

    model_path = args.model or TIBERT_MODEL_NAME

    print("=" * 60)
    print("TiBERT Training Script")
    print("=" * 60)
    print(f"Corpus: {args.corpus}")
    print(f"Model: {model_path}")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)

    # 创建数据集
    dataset = TibetanCorpusDataset(
        args.corpus,
        model_path=model_path,
        max_length=args.max_length
    )

    if args.max_samples:
        dataset.sentences = dataset.sentences[:args.max_samples]
        print(f"Limited to {args.max_samples} samples for testing")

    # 创建训练器并开始训练
    trainer = MLMTrainer(
        model_path=model_path,
        train_dataset=dataset,
        output_dir=args.output,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )

    trainer.train()


if __name__ == "__main__":
    main()
