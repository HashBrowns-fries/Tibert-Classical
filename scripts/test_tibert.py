"""
TiBERT推理测试脚本
验证TiBERT模型能否正常加载和推理
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.tibert_model import TibertModel
from config import TIBERT_MODEL_NAME


def test_model_loading():
    """测试模型加载"""
    print("=" * 50)
    print("1. 测试模型加载")
    print("=" * 50)

    try:
        tibert = TibertModel(model_name=TIBERT_MODEL_NAME)
        print(f"✓ 模型加载成功")
        print(f"  - Hidden size: {tibert.hidden_size}")
        print(f"  - Vocab size: {tibert.vocab_size}")
        return tibert
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return None


def test_tokenization(tibert):
    """测试分词"""
    print("\n" + "=" * 50)
    print("2. 测试分词")
    print("=" * 50)

    test_texts = [
        "བོད་སྐད",
        "བོད་སྐད་དང་དབྱིན་སྐད་ཚལ་མཛོད",
        "གི་རྣམས་ལ་ནས་བོད"
    ]

    for text in test_texts:
        inputs = tibert.encode(text)
        tokens = tibert.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        print(f"\n输入: {text}")
        print(f"Tokens: {tokens}")


def test_embeddings(tibert):
    """测试embedding提取"""
    print("\n" + "=" * 50)
    print("3. 测试Embedding提取")
    print("=" * 50)

    text = "བོད་སྐད"
    embedding = tibert.get_embeddings(text, pooling_strategy="mean")
    print(f"输入: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 5 dims): {embedding[:5].tolist()}")


def test_batch(tibert):
    """测试批量处理"""
    print("\n" + "=" * 50)
    print("4. 测试批量处理")
    print("=" * 50)

    texts = [
        "བོད་སྐད",
        "དབྱིན་སྐད",
        "ཚལ་མཛོད"
    ]

    embeddings = tibert.get_batch_embeddings(texts, pooling_strategy="mean")
    print(f"输入文本数: {len(texts)}")
    print(f"输出embedding shape: {embeddings.shape}")


def main():
    print("TiBERT 推理测试")
    print("=" * 50)

    # 1. 测试模型加载
    tibert = test_model_loading()
    if tibert is None:
        return

    # 2. 测试分词
    test_tokenization(tibert)

    # 3. 测试embedding
    test_embeddings(tibert)

    # 4. 测试批量
    test_batch(tibert)

    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
