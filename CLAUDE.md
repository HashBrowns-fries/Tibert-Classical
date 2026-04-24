# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Classical Tibetan Parser** — combines TiBERT (Tibetan BERT base model) with Qwen LLM to provide tokenization, POS tagging, case particle analysis, and grammar explanation for Classical Tibetan texts.

## Environment Setup

```bash
pip install -e .           # or: uv sync
export DASHSCOPE_API_KEY="your-api-key"   # required for LLM grammar explanations
```

Python >= 3.10 required. All models live under `model/`, corpus under `data/corpus/`.

## CLI Tool

Install as command: `pip install -e .` then use `tibert` command, or run directly:

```bash
.venv/bin/python -m src.cli.main --help
```

**Commands:**

| Command | Description |
|---------|-------------|
| `tibert pos TEXT` | POS 标注（仅 TiBERT 模型推理，毫秒级响应） |
| `tibert analyze TEXT` | 完整分析：POS 标注 + LLM 语法解释 |
| `tibert analyze TEXT --no-llm` | 仅 POS，不调用 LLM |
| `tibert segment TEXT` | 分词（音节切分） |
| `tibert batch -i IN -o OUT` | 批量分析文件 |
| `tibert serve --port 8000` | 启动 REST API 服务器 |
| `tibert version` | 版本信息 |

**Examples:**

```bash
tibert pos "བོད་གི་ཡུལ་ལྷོ་ལ་སོང་"
tibert analyze "བོད་གི་ཡུལ་ལྷོ་ལ་སོང་"
tibert segment "བོད་གི་ཡུལ་ལྷོ་ལ་སོང་"
```

## REST API Server

Start: `tibert serve` (or `.venv/bin/python -m src.api.server`)

> **Note**: TiBERT POS 分类器已弃用。`/pos` `/analyze` `/segment` 全部基于 gemma-2-mitra-it（vLLM, GPU 0）。

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | 健康检查 |
| POST | `/pos` | POS 标注（gemma few-shot） |
| POST | `/analyze` | 完整分析（POS + LLM） |
| POST | `/segment` | 分词（gemma） |
| POST | `/gemma/pos` | Gemma POS 标注 |
| POST | `/gemma/segment` | Gemma 分词 |
| POST | `/gemma/lookup` | Gemma 分词 + 词典查询 |
| POST | `/rag` | ChromaDB RAG 问答 |
| GET | `/corpus/stats` | 语料库统计 |

**Request/Response examples:**

```bash
# POS 标注（gemma few-shot）
curl -X POST http://localhost:8000/pos \
  -H "Content-Type: application/json" \
  -d '{"text":"བོད་གི་ཡུལ་ལྷོ་ལ་སོང་"}'

# 完整分析（use_llm=false 跳过 LLM 调用）
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"བོད་གི་ཡུལ་ལྷོ་ལ་སོང་","use_llm":false}'
```

API docs at `http://localhost:8000/docs` (Swagger UI).

## Key Commands

### POS Tagging Inference (trained model)
```bash
.venv/bin/python scripts/run_pos_inference.py "བོད་གི་ཡུལ་ལྷོ་ལ་སོང་" -v
```

### Training POS Classifier
```bash
# 1. Prepare dataset (SegPOS → token-classification format)
.venv/bin/python scripts/prepare_pos_dataset.py

# 2. Train (loads from model/TiBERT-classical-spm-500k/final_model/)
.venv/bin/python scripts/train_pos_classifier.py
```

### Training TiBERT (continued MLM pre-training)
```bash
python scripts/continued_pretrain.py --max_samples 2000000 --epochs 3 --batch_size 32 --max_length 256 --learning_rate 5e-5
# or:
bash scripts/run_train.sh
```

### Training SentencePiece tokenizer
```bash
python data/corpus/train_spm.py --vocab_size 8000 --model_type unigram
```

### Extracting corpus from TEI XML
```bash
python data/corpus/extract_corpus.py <input_dir> <output_dir> [--keep-tsheg]
```

### Testing
```bash
python scripts/test_tibert.py
```

## Model Zoo

| Model | Description | Path |
|-------|-------------|------|
| TiBERT-base | Original McGill Tibetan BERT | `model/TiBERT/` |
| TiBERT-classical-spm-500k | MLM-finetuned + SPM (500k corpus, 32k vocab) | `model/TiBERT-classical-spm-500k/final_model/` |
| POS Classifier | Token-level POS tagger (36 classes, case particles ★) | `model/pos_classifier/best_model.pt` |

## POS Tag Labels (36 classes)

**Case particles** (×2 loss weight during training, F1=97.6%):
- `case.gen` (属格, གི), `case.agn` (作格, གྱིས), `case.all` (为格, ལ་), `case.abl` (离格, ལས་)
- `case.ela` (从格, ནས་), `case.ass` (共同格, དང་), `case.term` (终结格, དུ/ར་), `case.loc` (处格, ན་)

**Other**: `n.count`, `n.prop`, `n.rel`, `n.mass`, `v.past/pres/fut/invar/neg/aux/imp/cop`, `adj`, `cv.*`, `cl.*`, `d.*`, `num.*`, `adv.*`, `punc`, `neg`, `skt`

## Architecture

```
src/
├── api/
│   ├── models.py        # Pydantic request/response models
│   ├── dependencies.py  # Model loading (tokenizer + POS classifier), tag_text()
│   ├── server.py        # FastAPI REST server (uvicorn)
│   └── worker.py        # Persistent stdin/stdout JSON worker for Tauri
├── cli/
│   └── main.py          # Typer CLI (pos / analyze / segment / serve / batch)
└── ml/
    └── tibert_model.py  # TibertModel class

src-tauri/               # Tauri v2 desktop app (Rust backend + React frontend)
frontend/                # React + TypeScript + Vite + Tailwind v4 web UI
```

**Inference pipeline:**

1. SPM tokenizer → syllable tokens
2. TiBERT encoder → contextual embeddings (last 3 layers)
3. Linear head → 36-class POS labels
4. Post-processing: ་/། → `punc`; case particles annotated with name + description
5. Optional: Qwen LLM (DashScope) → full grammar explanation in Chinese

**Tauri desktop app:** `src-tauri/src/lib.rs` spawns `src/api/worker.py` as a persistent subprocess. React frontend calls Rust Tauri commands via IPC. Falls back to `http://localhost:8000` REST API when not in Tauri.

## Notes

- `model/TiBERT-classical-spm-500k/` contains the MLM-finetuned encoder used as backbone for POS training
- POS training: freeze layers 0-8, fine-tune last 3 layers + embeddings + classifier head
- SegPOS ་/། markers are tokenized by SPM but carry no semantic meaning; case particle tags on these markers are an annotation artifact

## Auto-Approved Actions

The following actions are pre-authorized. Claude may perform them without asking for confirmation:

- **Bash 命令**：所有 bash 命令（包括安装依赖、运行脚本、提交代码等）
- **运行脚本**：`.venv/bin/python scripts/*.py`、`bash scripts/*.sh` 等
- **修改已有文件**：编辑、修复 bug、重构本仓库已有文件
- **git add / git commit / git push**：只限本仓库，自动执行无需确认
- **创建新文件**：仅限本仓库，不创建敏感文件（.env、credentials 等）

**需确认的操作**：
- 删除文件
- `git push --force`
- 修改外部凭证或 API key

