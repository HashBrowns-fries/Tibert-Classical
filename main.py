"""
古典藏文解析辅助学习系统 - Main Entry Point

Architecture:
- TiBERT: 基座模型，提供藏文理解能力
- Qwen LLM: 分析引擎，分词/词性/语法解释
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import QWEN_API_KEY, QWEN_MODEL, TIBERT_MODEL_NAME


def print_header():
    """Print welcome header."""
    print("=" * 60)
    print("  古典藏文解析辅助学习系统")
    print("  Classical Tibetan Parser - Learning Assistant")
    print("=" * 60)


def check_dependencies():
    """Check installed dependencies."""
    print("\n[Dependency Check]")

    # Transformers
    try:
        import transformers
        print(f"  ✓ transformers {transformers.__version__}")
    except ImportError:
        print("  ✗ transformers not installed")

    # PyTorch
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError:
        print("  ✗ PyTorch not installed")

    # Qwen API
    if QWEN_API_KEY:
        print("  ✓ Qwen API key configured")
    else:
        print("  ✗ Qwen API key not set (set DASHSCOPE_API_KEY env var)")


def demo_tibert():
    """Demo TiBERT model loading."""
    print("\n[TiBERT Model Demo]")

    try:
        from transformers import AutoModel, AutoTokenizer

        print(f"  Loading model: {TIBERT_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(TIBERT_MODEL_NAME)
        model = AutoModel.from_pretrained(TIBERT_MODEL_NAME)

        # Test encoding
        text = "བོད་སྐད"
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)

        print(f"  Input: {text}")
        print(f"  Embedding shape: {outputs.last_hidden_state.shape}")
        print("  ✓ TiBERT loaded successfully")
    except Exception as e:
        print(f"  ✗ Failed to load TiBERT: {e}")


def demo_llm():
    """Demo Qwen LLM API."""
    print("\n[Qwen LLM Demo]")

    if not QWEN_API_KEY:
        print("  ✗ Qwen API key not configured")
        return

    try:
        import dashscope
        from dashscope import Generation

        dashscope.api_key = QWEN_API_KEY

        prompt = "请将以下古典藏文分词：བོད་སྐད་དང་དབྱིན་སྐད་ཚལ་མཛོད"
        response = Generation.call(
            model=QWEN_MODEL,
            prompt=prompt
        )

        if response.status_code == 200:
            print(f"  Response: {response.output.text[:100]}...")
            print("  ✓ Qwen API working")
        else:
            print(f"  ✗ API error: {response.code}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def main():
    """Main entry point."""
    print_header()
    check_dependencies()

    print("\n" + "-" * 60)
    print("Demos:")
    demo_tibert()
    demo_llm()

    print("\n" + "=" * 60)
    print("System ready! Run 'streamlit run app/main.py' for UI")
    print("=" * 60)


if __name__ == "__main__":
    main()
