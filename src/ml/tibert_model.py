"""
TiBERT Model Module

Loads and interfaces with the TiBERT base model for Tibetan language understanding.
"""

from typing import Optional, List, Dict
from collections import OrderedDict
import torch
from transformers import AutoModel, BertTokenizer, PreTrainedModel, PreTrainedTokenizer
import os


class TibertModel:
    """TiBERT基座模型封装"""

    def __init__(
        self,
        model_name: str = "McGill-NLP/Tibert-base",
        device: Optional[str] = None
    ):
        """
        Initialize TiBERT model.

        Args:
            model_name: HuggingFace模型名称或本地路径
            device: 计算设备 (cuda/cpu)
        """
        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer: PreTrainedTokenizer = None
        self.model: PreTrainedModel = None
        self._load_model()

    def _load_model(self):
        """Load TiBERT tokenizer and model"""
        print(f"Loading TiBERT model: {self.model_name}")

        # Load vocab manually to handle Tibetan characters correctly
        vocab_path = os.path.join(self.model_name, "vocab.txt")
        vocab = self._load_vocab(vocab_path)
        print(f"Loaded vocab size: {len(vocab)}")

        # Initialize tokenizer with custom vocab
        self.tokenizer = BertTokenizer(vocab=vocab)
        self.tokenizer.do_lower_case = False

        # Load model
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"TiBERT loaded on {self.device}")

    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        """Load vocabulary from vocab.txt file"""
        vocab = OrderedDict()
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.rstrip('\r\n')
                vocab[token] = idx
        return vocab

    def encode(
        self,
        text: str,
        return_tensors: str = "pt",
        add_special_tokens: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode Tibetan text to token IDs and attention mask.

        Args:
            text: 输入藏文文本
            return_tensors: 返回tensor类型 ('pt', 'np', None)
            add_special_tokens: 是否添加特殊token ([CLS], [SEP])

        Returns:
            包含input_ids和attention_mask的字典
        """
        return self.tokenizer(
            text,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
            padding=True,
            truncation=True,
            max_length=512
        )

    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """
        Decode token IDs back to text.

        Args:
            token_ids: token ID序列

        Returns:
            解码后的文本列表
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def get_embeddings(
        self,
        text: str,
        pooling_strategy: str = "mean"
    ) -> torch.Tensor:
        """
        Get contextualized embeddings for input text.

        Args:
            text: 输入藏文文本
            pooling_strategy: 池化策略 (mean, cls, max)

        Returns:
            文本的embedding向量 [hidden_size]
        """
        inputs = self.encode(text)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]

        if pooling_strategy == "cls":
            embeddings = hidden_states[:, 0, :]
        elif pooling_strategy == "max":
            embeddings = hidden_states.max(dim=1).values
        else:  # mean
            attention_mask = inputs["attention_mask"]
            embeddings = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1)
            embeddings = embeddings / attention_mask.sum(dim=1, keepdim=True)

        return embeddings.squeeze(0)

    def batch_encode(
        self,
        texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Batch encode multiple texts.

        Args:
            texts: 文本列表

        Returns:
            批量编码结果
        """
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

    def get_batch_embeddings(
        self,
        texts: List[str],
        pooling_strategy: str = "mean"
    ) -> torch.Tensor:
        """
        Get embeddings for batch of texts.

        Args:
            texts: 文本列表
            pooling_strategy: 池化策略

        Returns:
            embeddings矩阵 [batch_size, hidden_size]
        """
        inputs = self.batch_encode(texts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state

        attention_mask = inputs["attention_mask"]

        if pooling_strategy == "cls":
            embeddings = hidden_states[:, 0, :]
        elif pooling_strategy == "max":
            embeddings = hidden_states.max(dim=1).values
        else:
            embeddings = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1)
            embeddings = embeddings / attention_mask.sum(dim=1, keepdim=True)

        return embeddings

    @property
    def hidden_size(self) -> int:
        """Get model hidden size"""
        return self.model.config.hidden_size

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.model.config.vocab_size


def main():
    """Test TiBERT model loading"""
    print("Testing TiBERT model...")

    tibert = TibertModel()

    # Test single text
    text = "བོད་སྐད"
    embedding = tibert.get_embeddings(text)
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")

    # Test batch
    texts = ["བོད་སྐད", "དབྱིན་སྐད"]
    embeddings = tibert.get_batch_embeddings(texts)
    print(f"Batch embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
