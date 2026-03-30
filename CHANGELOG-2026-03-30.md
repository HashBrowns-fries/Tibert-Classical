# 开发日志 2026-03-30

## 主题：SPM tokenizer 修复 + 全量数据 POS 训练准备

---

## 1. 问题定位

### 1.1 `བདག → O` 的根本原因

推理时 `src/api/dependencies.py` 的 `spm_tokenize()` 用贪心最长匹配直接调用 SPM，没有先按藏文天然分词符号 ་/། 切分，导致音节被切碎。

但实际上 tokenizer 词表中**已经包含完整音节 `▁བདག`（→ ID 112）**，只是推理路径没有正确使用。

### 1.2 tokenizer 隐藏 bug

`_convert_token_to_id()` 对 U+2581（SPM 的空格前缀符）的二次 strip 导致空字符串被映射到 [PAD] 而非 [UNK]。

---

## 2. 修改内容

### 2.1 `src/api/dependencies.py` — 重写 `spm_tokenize()`

**修改前**：贪心最长匹配，直接走 SPM，音节被切碎

```python
# 修改前（错误）
def spm_tokenize(text: str, tokenizer) -> list[str]:
    tokens = []
    i = 0
    while i < len(text):
        for n in range(min(10, len(text) - i), 0, -1):
            sub = text[i:i+n]
            tid = tokenizer._convert_token_to_id(sub)
            unk = tokenizer._token2id.get("[UNK]")
            if tid != unk:
                best = sub
                break
        ...
```

**修改后**：先按 ་/། 天然切分，再对每段 encode

```python
# 修改后
def spm_tokenize(text: str, tokenizer) -> list[str]:
    # 1. 先按 ་/། 天然切分
    parts = []
    i = 0
    start = 0
    while i < len(text):
        ch = text[i]
        if ch in ("་", "།"):
            if start < i:
                parts.append(text[start:i])
            parts.append(ch)
            i += 1
            start = i
        else:
            i += 1
    if start < len(text):
        parts.append(text[start:])

    # 2. 对每个非符号部分做 tokenizer encode
    result = []
    for part in parts:
        if part in ("་", "།"):
            result.append(part)
            continue
        if not any(0x0F40 <= ord(c) <= 0xFBC for c in part):
            continue
        tokens = tokenizer._tokenize(part)
        result.extend(tokens)
    return result
```

**效果**：`'བདག' → ['བདག'] → ID 112`（完整音节，非碎片）

### 2.2 `scripts/continued_pretrain.py` — 修复 `_convert_token_to_id()`

**修改前**：二次 strip 导致空字符串

```python
# 修改前
def _convert_token_to_id(self, token: str) -> int:
    if token.startswith("\u2581"):
        token = token[1:]
    if not token:
        return self._token2id["[PAD]"]
    return self._token2id.get(token, self._token2id["[UNK]"])
```

**修改后**：保留 token 而非返回 [PAD]

```python
# 修改后
def _convert_token_to_id(self, token: str) -> int:
    if token.startswith("\u2581"):
        stripped = token[1:]
        token = stripped if stripped else token  # 保留完整 U+2581 而非空字符串
    if not token:
        return self._token2id["[PAD]"]
    return self._token2id.get(token, self._token2id["[UNK]"])
```

### 2.3 `scripts/train_pos_classifier.py` — 动态读取 num_labels

**修改前**：硬编码 `num_labels = 91`

```python
class Config:
    num_labels = 91  # 硬编码
```

**修改后**：从 label_map.json 动态读取（数据生成时会发现 92 个标签）

```python
class Config:
    _label_map_path = DATA_DIR / "label_map.json"
    if _label_map_path.exists():
        with open(_label_map_path, encoding="utf-8") as _f:
            _lm = json.load(_f)
        num_labels = _lm["num_labels"]  # 92 (O + 91 POS tags)
    else:
        num_labels = 91
```

---

## 3. 验证结果

```python
# 测试：所有音节均得到正确 ID，零 UNK
'བདག'  → tokens=['བདག']   ids=[112]    UNK=0  ✅
'་'     → tokens=['་']       ids=[226]    UNK=0  ✅
'།'     → tokens=['།']        ids=[5]      UNK=0  ✅
'གི'    → tokens=['གི']       ids=[51]     UNK=0  ✅
'བོད'   → tokens=['བོད']    ids=[931]    UNK=0  ✅
```

---

## 4. 当前状态：全量数据生成（后台运行中）

**脚本**：`scripts/prepare_pos_dataset.py`（全量，MAX_SENTS_PER_COLLECTION=None）

**进度（截至 2026-03-30 20:25）**：

| 语料库 | 句子数 | 状态 |
|--------|--------|------|
| DharmaDownload | 1,890,610 | ✅ 完成 |
| DrikungChetsang | 912,785 | ✅ 完成 |
| GuruLamaworks | 4,708,400 | ✅ 完成 |
| KarmaDelek | 1,855,616 | 🔄 处理中 |
| Shechen | — | ⏳ 待处理 |
| Various | — | ⏳ 待处理 |
| PalriParkhang | — | ⏳ 待处理 |
| VajraVidya | — | ⏳ 待处理 |
| TulkuSangag | — | ⏳ 待处理 |
| eKangyur | — | ⏳ 待处理 |
| eTengyur | — | ⏳ 待处理 |

**预期标签**：92 个（O + 91 个原始 SegPOS 标签）

**训练配置**（`train_pos_classifier.py`）：

| 参数 | 值 |
|------|----|
| num_labels | 92（动态读取） |
| encoder_freeze | False（全量数据，解冻 encoder） |
| unfreeze_layers | 6（最后 6 层 + embeddings） |
| batch_size | 64 |
| gradient_accum | 4（有效 batch = 256） |
| lr_encoder | 2e-5 |
| lr_head | 5e-4 |
| max_epochs | 20 |
| early_stop_patience | 8 |
| case_weight | 2.0×（18 种 case.* 标签） |

---

## 5. 标点遮掩数据增强（Punctuation-Drop Augmentation）

**目的**：让模型同时处理有标点（་/།）和无标点的藏文文本。SegPOS 语料本身有完整标点，但网上古籍、OCR 输出等文本往往缺少标点。

**实现方式**：训练时随机将 ་/། token 替换为 UNK（ID=1），对应标签置 -100（不参与 loss 计算）。模型在有标点和无标点输入上都能正确预测。

**参数**：`drop_punct_prob = 0.3`（30% 的 ་/། 被遮掩）

**实现位置**：`scripts/train_pos_classifier.py` — `PosDataset` 类

```python
def _drop_punct_augment(ids, labs, prob):
    mask = (ids == _TSEG_ID) | (ids == _SHAD_ID)  # ་=226, །=5
    drop = mask & (torch.rand(ids.size(0)) < prob)
    ids  = ids.masked_fill(drop, 1)   # → UNK
    labs = labs.masked_fill(drop, -100)  # → ignore loss
    return ids, labs
```

## 6. 全量数据生成结果

| 指标 | 数量 |
|------|------|
| 总句子数 | **18,587,316**（1860万） |
| Train | 14,869,852 |
| Dev | 1,858,731 |
| Test | 1,858,733 |
| 标签数 | 92（O + 91 原始 SegPOS 标签） |

**Top 标签频次**：

| 标签 | 频次 |
|------|------|
| `n.count` | 84,749,403 |
| `punc` | 26,162,408 |
| `case.gen` | 25,959,699 |
| `case.term` | 23,674,798 |
| `case.ass` | 9,374,516 |
| `case.agn` | 9,206,431 |

## 8. Qwen2.5-3B 微调方案（暂停）

**状态：暂停，待 GPU 空闲后继续**

已完成：
- `models/Qwen2.5-3B-Instruct/` 下载完毕（3.1B 参数，5.8 GB）
- `scripts/finetune_qwen_pos.py` 初版完成（有 bug 待修）

待完成：
- [ ] 修复数据生成脚本 bug（tokenizer 混用问题）
- [ ] 生成 5 万条训练数据
- [ ] QLoRA 微调训练

## 9. Qwen2.5-3B 微调方案

### 模型下载
- `models/Qwen2.5-3B-Instruct/` 已下载（3.1B 参数，5.8 GB）
- HuggingFace repo: `Qwen/Qwen2.5-3B-Instruct`

### 微调方案（QLoRA）

**训练数据**（自动从 SegPOS 生成）：
```json
{
  "messages": [
    {"role": "system", "content": "你是古典藏文语法学家..."},
    {"role": "user", "content": "分析：藏文原文 + POS标注 + 词典释义"},
    {"role": "assistant", "content": "## 词语解析\n- བདག[n.count·名词]：...\n## 句法结构\n..."}
  ]
}
```

**QLoRA 配置**（4-bit，一张 3090 可跑）：

| 参数 | 值 |
|------|----|
| 模型 | Qwen2.5-3B-Instruct |
| 量化 | 4-bit NF4 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q/k/v/o_proj, gate/up/down_proj |
| 有效 batch | 4×8=32 |
| 训练 epoch | 3 |

**启动训练**：
```bash
# 1. 生成训练数据（5万条）
.venv/bin/python scripts/finetune_qwen_pos.py --generate_data --num_samples 50000

# 2. 开始微调（需 GPU 空闲）
.venv/bin/python scripts/finetune_qwen_pos.py --train \
    --data_file data/corpus/grammar_analysis_train_50000.jsonl \
    --num_epochs 3 --batch_size 4 --gradient_accum 8
```

**训练脚本**：`scripts/finetune_qwen_pos.py`

### 训练数据生成流程

```
SegPOS 句子（1860万）
    ↓
parse_segpos_line() → [(word, tag), ...]
    ↓
POS tagger（TiBERT）→ 标注结果
    ↓
词典查询（lookup_word）→ 释义
    ↓
format_instruction_sample() → ChatML JSONL
    ↓
grammar_analysis_train_50k.jsonl
```

## 10. 仓库结构整理

### 问题

```
当前结构（混乱）:
  app/                    ← 空目录，旧 Streamlit 遗留
  frontend-design/        ← 旧设计实验
  dict/                   ← 原始词典压缩包（源码）
  extern/                 ← 原始词干词典（源码）
  model/                   ← TiBERT 系列模型
  models/                 ← Qwen2.5-3B（应合并到 model/）
  data/corpus/            ← 混合了数据和脚本
  scripts/                 ← 混合了测试、日志、训练脚本
  src/                    ← FastAPI 后端（api/ cli/ dict/ ml/）
  src-tauri/              ← Tauri 桌面应用（独立项目）
  frontend/               ← React + Tauri Web UI
  frontend/src-tauri/     ← Tauri Web UI 的 Rust 配置
```

### 整理计划

```bash
# 1. 移动 Qwen 模型
mv models/Qwen2.5-3B-Instruct model/

# 2. 移动 corpus 处理脚本
mv data/corpus/extract_corpus.py scripts/
mv data/corpus/train_spm.py scripts/
mv data/corpus/tokenizer.py scripts/

# 3. 删除空目录
rmdir app

# 4. 归档旧设计
mv frontend-design _archive_frontend-design

# 5. 清理 scripts 目录
# 删除：test_*.py（测试脚本）、*.log（日志）、eval_*.json
# 保留：train_*.py, prepare_*.py, continued_pretrain.py, finetune_*.py,
#       run_*.sh, run_*.py, gpu_train_watcher.sh, analyze_*.py,
#       import_dict_to_sqlite.py
```

### 目标结构

```
Tibert-Classical/
├── model/                       ← 所有模型（TiBERT + POS + Qwen）
│   ├── TiBERT-classical-spm-500k/
│   ├── pos_classifier/
│   └── Qwen2.5-3B-Instruct/     ← 从 models/ 移入
├── data/
│   ├── segpos_extracted/        ← 原始 SegPOS 语料
│   └── corpus/
│       ├── pos_dataset/         ← 训练数据
│       ├── extracted/           ← 提取后语料
│       └── spm/                ← SPM 模型
├── scripts/                     ← 训练和工具脚本
│   ├── train_pos_classifier.py
│   ├── prepare_pos_dataset.py
│   ├── continued_pretrain.py
│   ├── finetune_qwen_pos.py
│   ├── extract_corpus.py       ← 从 data/corpus/ 移入
│   └── gpu_train_watcher.sh
├── src/                        ← FastAPI 后端
│   ├── api/
│   ├── cli/
│   └── dict/
├── frontend/                   ← React + Tauri Web UI
│   └── src-tauri/
├── src-tauri/                  ← Tauri 桌面应用（独立）
├── dict/                       ← 原始词典压缩包（源码数据）
├── extern/                     ← 原始词干词典（源码数据）
├── CLAUDE.md
├── CHANGELOG-*.md
└── README.md
```

### 状态
- [x] Qwen 模型已在 `model/Qwen2.5-3B-Instruct/`
- [x] corpus 处理脚本移入 `scripts/`：extract_corpus.py, train_spm.py, tokenizer.py, train_tibetan_spm.py
- [x] 删除空 `app/` 目录
- [x] `frontend-design/` → `_archive_frontend-design/`
- [x] `scripts/` 清理，测试和日志归档至 `_archive_scripts/`
- [x] 更新 `.gitignore`（忽略 71GB pos_dataset、5.8GB Qwen 模型等大文件）
- [x] 清理所有 `__pycache__/`

