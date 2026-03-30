"""
Tibetan Tokenizer using SentencePiece
藏文分词器 - 基于SentencePiece
"""

import sentencepiece as spm
from pathlib import Path
from typing import List, Tuple, Optional


class TibetanTokenizer:
    """藏文分词器"""

    def __init__(self, model_path: str):
        """
        初始化分词器

        Args:
            model_path: SentencePiece模型文件路径 (.model)
        """
        self.model_path = Path(model_path)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_path))

        # 获取词表大小
        self.vocab_size = self.sp.get_piece_size()
        print(f"Loaded SentencePiece model: {model_path}")
        print(f"Vocabulary size: {self.vocab_size}")

    def tokenize(self, text: str) -> List[str]:
        """
        分词

        Args:
            text: 输入藏文文本

        Returns:
            分词结果列表
        """
        pieces = self.sp.encode(text, out_type=str)
        return pieces

    def tokenize_to_ids(self, text: str) -> List[int]:
        """
        分词并转换为ID

        Args:
            text: 输入藏文文本

        Returns:
            token IDs列表
        """
        ids = self.sp.encode(text, out_type=int)
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        将IDs解码为文本

        Args:
            ids: token IDs列表

        Returns:
            解码后的文本
        """
        return self.sp.decode(ids)

    def decode_pieces(self, pieces: List[str]) -> str:
        """
        将分词结果解码为文本

        Args:
            pieces: 分词结果列表

        Returns:
            解码后的文本
        """
        return self.sp.decode_pieces(pieces)

    def get_vocab_size(self) -> int:
        """获取词表大小"""
        return self.vocab_size

    def is_unknown(self, piece: str) -> bool:
        """检查是否为未知词"""
        return self.sp.is_unknown(self.sp.piece_to_id(piece))

    def piece_to_id(self, piece: str) -> int:
        """将piece转换为ID"""
        return self.sp.piece_to_id(piece)

    def id_to_piece(self, id: int) -> str:
        """将ID转换为piece"""
        return self.sp.id_to_piece(id)


def load_tokenizer(model_dir: str = None, model_name: str = "tibetan") -> TibetanTokenizer:
    """
    加载分词器的便捷函数

    Args:
        model_dir: 模型目录
        model_name: 模型名称

    Returns:
        TibetanTokenizer实例
    """
    if model_dir is None:
        model_dir = Path(__file__).parent / "spm"
    else:
        model_dir = Path(model_dir)

    model_path = model_dir / f"{model_name}.model"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return TibetanTokenizer(str(model_path))


# 使用示例
if __name__ == "__main__":
    import sys

    # 测试分词器
    model_path = sys.argv[1] if len(sys.argv) > 1 else None

    if model_path:
        tokenizer = TibetanTokenizer(model_path)

        # 测试文本
        test_texts = [
            "༄༅། །ཤེས་རབ་ཀྱི་ཕ་རོལ་ཏུ་ཕྱིན་པ།",
            "༄༅༅། །ཉི་ཁྲི་ཁ་པ་བཞུགས་སོ། །",
        ]

        for text in test_texts:
            print(f"\nInput:  {text}")
            print(f"Tokens: {tokenizer.tokenize(text)}")
            print(f"IDs:    {tokenizer.tokenize_to_ids(text)}")
            print(f"Decode: {tokenizer.decode(tokenizer.tokenize_to_ids(text))}")
    else:
        print("Usage: python tokenizer.py <model_path>")
        print("Example: python tokenizer.py spm/tibetan.model")
