# 古典藏文解析辅助学习系统

Classical Tibetan Parser - Learning Assistant

基于TiBERT基座模型 + Qwen LLM，实现古典藏文的智能解析、语法分析、助词解读，帮助用户学习和理解古典藏文。

## 核心架构

```
输入古典藏文文本
    ↓
TiBERT (基座模型) ──→ 提供藏文语义理解能力
    ↓
Qwen LLM ──→ 分词、词性标注、格助词分析、语法解释
    ↓
输出：分词结果 | 词性标注 | 格助词解读 | 语法分析
```

## 功能特性

- **智能分词**: LLM驱动的精确分词
- **词性标注**: 名词、动词、助词等自动标注
- **格助词解读**: 属格、具格、为格、离格等语法功能解释
- **语法分析**: 整句语法结构分析
- **学习辅助**: 助词用法详解、例句展示

## 项目结构

```
classical_tibetan_parser/
├── config.py                 # 项目配置
├── pyproject.toml            # Python依赖
│
├── src/                      # 核心源代码
│   ├── __init__.py
│   ├── ml/                   # ML模块
│   │   ├── tibert_model.py  # TiBERT模型
│   │   └── trainer.py        # 微调训练
│   └── api/                  # API层
│       └── grammar_api.py    # 语法分析API (LLM驱动)
│
├── app/                      # Streamlit前端
│   ├── main.py
│   └── pages/
│       ├── analyzer.py       # 文本分析
│       ├── learner.py        # 学习辅助
│       └── corpus.py         # 语料管理
│
├── scripts/                  # 工具脚本
│   ├── prepare_data.py      # 数据预处理
│   └── train_tibert.py      # TiBERT微调
│
├── data/                     # 数据
│   └── corpus/              # 语料库
│
├── database/                 # 藏文文献库
│
└── models/                  # 模型存储
```

## 安装

```bash
pip install -e .
# 或
uv sync
```

### 依赖

- Python >= 3.10
- PyTorch >= 2.0
- Transformers
- Streamlit
- dashscope (Qwen API)

## 快速开始

### 1. 配置API

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

### 2. 分析文本

```python
from src.api.grammar_api import GrammarAnalyzer

analyzer = GrammarAnalyzer()
result = analyzer.analyze("བོད་སྐད་དང་དབྱིན་སྐད་ཚལ་མཛོད")

print(result.tokens)       # 分词结果
print(result.pos_tags)     # 词性标注
print(result.grammar)      # 语法分析
```

### 3. 启动前端

```bash
streamlit run app/main.py
```

### 4. 训练模型

```bash
python scripts/train_tibert.py --data data/corpus/annotated
```

## 数据库

包含以下藏文文献：
- eKangyur (甘珠尔)
- DharmaDownload
- DrikungChetsang
- GuruLamaWorks
- KarmaDelek
- OCR 2017
- PalriParkhang
- Shechen
- TulkuSangag
- UCB-OCR

## 开发计划

详见 [开发计划文档](./开发计划文档.md)
