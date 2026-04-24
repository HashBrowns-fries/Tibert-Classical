# TibSplit — 古典藏文形态学解析系统

**TibSplit** 是一款基于 BERT-CRF 的古典藏文形态学解析框架，结合领域适配编码器、类别平衡损失与诊断评估，系统揭示细粒度 POS 标注中的类别不平衡问题。

> 论文：[TibSplit：诊断古典藏文形态学解析中的类别不平衡](#)（12 页，xelatex 编译）

---

## 核心功能

| 功能 | 描述 |
|------|------|
| **POS 标注** | TiBERT-classical-spm-500k 编码器 + BERT-CRF，77 类细粒度标签 |
| **格助词分析** | 属格/作格/为格/离格/从格/共同格/终结格/处格，自动识别与解释 |
| **Gemma 分词** | gemma-2-mitra-it 驱动藏文音节切分（་/། 分隔符） |
| **RAG 问答** | ChromaDB 向量检索 + gemma 生成，藏文佛典语料库问答 |
| **SRS 学习** | SM-2 算法助词间隔重复复习，gemma 评分练习 |
| **词典查询** | 540k+ 条目 SQLite 词典，音节级查找 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      React 前端 (Vite + Tailwind v4)        │
│  AnalyzerPage · LearnerPage · LookupPage · RagPage        │
└────────────────────────┬──────────────────────────────────┘
                         │ HTTP / API
┌────────────────────────▼──────────────────────────────────┐
│                   FastAPI 后端                              │
│                                                          │
│  server.py          rag_server.py                        │
│  · /pos             · /rag (RAG 问答)                    │
│  · /analyze         · /segment (分词)                    │
│  · /learn/*         · /lookup (词典)                     │
│  · /corpus/stats    · /rag/stats                         │
└──┬──────────────────────────┬──────────────────────────────┘
   │ vLLM                    │ ChromaDB                    │
▼  gemma-2-mitra-it      ▼  sentence-transformers       ▼
   (GPU 0)                  embedder                      TiBERT-classical-spm-500k
                                                         POS 分类器
```

---

## 项目结构

```
Tibert-Classical/
├── src/api/
│   ├── server.py          # 主 API：POS / analyze / learn
│   ├── rag_server.py     # RAG API：/rag / segment / lookup
│   ├── rag.py            # ChromaDB 索引构建
│   ├── models.py         # Pydantic 请求/响应模型
│   └── dependencies.py   # 模型加载（tokenizer / POS classifier）
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── AnalyzerPage.tsx   # 藏文分析器
│   │   │   ├── LearnerPage.tsx     # 助词学习 + SRS
│   │   │   ├── LookupPage.tsx      # 词典查询
│   │   │   └── RagPage.tsx        # RAG 佛典问答
│   │   ├── hooks/
│   │   │   ├── useAnalysis.ts     # 分析 API hook
│   │   │   ├── useLearner.ts      # SRS + 练习 hook
│   │   │   ├── useLookup.ts       # 词典查询 hook
│   │   │   └── useRag.ts         # RAG hook
│   │   └── index.css             # Tailwind v4 暗色主题
│   └── package.json
│
├── scripts/
│   ├── train_pos_classifier.py     # BERT-CRF POS 训练
│   ├── continued_pretrain.py       # TiBERT 领域适配 MLM
│   ├── prepare_pos_dataset.py      # SegPOS → token-classification
│   ├── run_pos_inference.py       # POS 推理脚本
│   ├── eval_pos_model_v2.py       # 逐类别 F1 分析
│   ├── build_rag_index.py         # ChromaDB 索引构建
│   └── learner_corpus_analysis.py # 语料库统计分析
│
├── model/
│   ├── TiBERT-classical-spm-500k/ # MLM 微调编码器
│   └── pos_classifier/best_model.pt # BERT-CRF 分类器（36 类）
│
├── paper.tex    # 中文论文，xelatex 编译
└── paper.pdf   # 编译输出（12 页）
```

---

## 安装

```bash
pip install -e .
uv sync
```

### 依赖

- Python >= 3.10
- PyTorch >= 2.0
- Transformers, Accelerate
- vLLM（GPU 推理）
- sentence-transformers（RAG embedder）
- ChromaDB
- FastAPI + uvicorn
- React 18 + Vite + Tailwind v4

---

## 快速开始

### 后端 API

```bash
# 主服务（POS / analyze / learn）
.venv/bin/python -m src.api.server

# RAG 服务（独立，GPU 0）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m src.api.rag_server
```

### 前端

```bash
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173`。

### CLI

```bash
tibert analyze "བོད་གི་ཡུལ་ལྷོ་ལ་སོང་"
# POS 标注 + 语法解释

tibert pos "བོད་གི་ཡུལ་ལྷོ་ལ་སོང་"
# 仅 POS 标注（毫秒级）

tibert serve --port 8000
# 启动 REST API
```

---

## 模型

| 模型 | 描述 |
|------|------|
| **TiBERT** | McGill 原始藏文 BERT 基座 |
| **TiBERT-classical-spm-500k** | 在 500k 古典藏文语料上继续 MLM，SentencePiece 32k vocab |
| **POS Classifier** | BERT-CRF，冻结前 9 层，微调最后 3 层 + 分类头，36 类 |

### 训练 POS 分类器

```bash
# 1. 准备数据集
.venv/bin/python scripts/prepare_pos_dataset.py

# 2. 训练
.venv/bin/python scripts/train_pos_classifier.py
```

---

## 实验结果（SegPOS，120,853 token，77 类）

| 指标 | 数值 |
|------|------|
| 加权 F1 | 0.825 |
| 宏平均 F1 | 0.659 |
| **加权-宏差距** | **68.5 pp** |
| 格助词（闭词类）F1 | 0.954 |
| 形容词（开放词类）F1 | 0.269 |

> **诊断结论**：闭词类（助词）已接近解决；开放词类（形容词、稀有名词子类）严重失败。聚合加权 F1 掩盖了对低频形态类别的系统性失效。

---

## API 端点

### 主服务（port 8000）

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/pos` | gemma few-shot POS 标注 |
| POST | `/analyze` | 完整分析（POS + LLM 解释） |
| POST | `/segment` | gemma 藏文分词 |
| POST | `/learn/particles` | 格助词学习数据 |
| POST | `/learn/verbs` | 动词学习数据 |
| POST | `/learn/drill` | gemma 生成练习题 |
| POST | `/learn/check` | gemma 评分 |
| GET | `/corpus/stats` | 语料库统计 |

### RAG 服务（port 8000）

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/rag` | RAG 问答（ChromaDB + gemma） |
| POST | `/rag/retrieval` | 仅向量检索 |
| POST | `/segment` | 藏文音节分词 |
| POST | `/lookup` | 词典查询 |
| GET | `/rag/stats` | 索引统计 |

---
