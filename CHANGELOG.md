# Changelog

## 2026-04-07 — npm run 前后端集成 + vLLM prompt dedup cache 修复

### 前后端集成（`npm run dev`）

项目根目录新建 `package.json`，使用 `concurrently` 同时启动前后端：

```bash
# 前后端一起启动（API server + Vite dev server）
npm run dev

# 仅后端（gemma-2-mitra-it API）
npm run dev:backend

# 仅前端
npm run dev:frontend

# 构建前端
npm run build
```

### `/analyze` 修复：前端显示 gemma POS 标签（不再"见词典释义"）

**问题**：`useAnalysis.ts` 的 `analyze()` 先调用 `/pos`（返回 `pos_zh="见词典释义"` 的占位符），再调用 `/analyze`（返回 gemma 真实 POS）。用户看到占位符闪烁。

**修复**：删除冗余 `/pos` 预加载，`/analyze` 直接返回 gemma 分词+POS 标签。

### vLLM prompt dedup cache 修复（UUID nonce）

**问题**：相同文本第二次请求触发 vLLM prompt deduplication cache，返回空输出（`len=0`）或重复输出。

**修复**：在 `_gemma_segment_and_pos()` 和 `call_gemma_llm()` 的 prompt 开头追加 UUID nonce：
```python
nonce = uuid.uuid4().hex[:12]
prompt = f"<start_of_turn>user\n[{nonce}]\n..."
```
每次请求 prompt hash 不同，绕过 cache。

### Tibetan Unicode NFD/NFC 修复

**问题**：输入使用 NFD 形式（如 `LA+HASANTA+VOWEL_O`），词典存储 NFC 形式，FMM 匹配失败。

**修复**：构建 Trie 时同时插入 NFC 和 NFD 两种形式。

### 修改文件

| 文件 | 说明 |
|------|------|
| `package.json` | 新增根级 package.json，concurrently 编排前后端 |
| `frontend/src/hooks/useAnalysis.ts` | 删除冗余 `/pos` 预加载，analyze() 直连 `/analyze` |
| `src/api/server.py` | UUID nonce 防 cache；NFC/NFD 双插入 Trie |

---

## 2026-04-06 — gemma-2-mitra-it 统一 pipeline（替换 TiBERT + Qwen）

### 当前状态（2026-04-06 22:00）

**统一服务器**: `src/api/server.py`（port 8000）
- `/pos` `/analyze` — FMM 分词（词典 Trie）+ SQLite 词典查询 + 一次 gemma 调用
- `/segment` `/gemma/segment` — gemma 分词
- `/gemma/pos` — gemma few-shot POS 标注
- `/rag` — ChromaDB RAG 问答（gemma 生成）
- **TiBERT POS 已弃用**（vocab_size 80007 vs tokenizer 32007，标注全错）
- **Qwen 已弃用**，全部迁移到 gemma-2-mitra-it
- ChromaDB 索引（~340k chunks）

### `/analyze` 完整 pipeline（一次 gemma 调用）

1. **FMM 分词**：基于 541k 词典 SQLite → 内存 Trie → 正向最大匹配
2. **SQLite 词典查询**：每个词 → 词典来源 + 释义（第一行，≤80字）
3. **gemma 生成**：词典上下文 → 中藏英三语语法分析

### FMM 分词（`src/api/server.py`）

- 启动时构建 Trie（1.6M 节点，~4s，一次加载复用）
- `re.split(r'(་|།)', text)` → 分段 → Trie 最长匹配
- 词典未收录词 → 单字保留

### POS 标注（`/gemma/pos`）

- gemma few-shot 生成式标注（`< 名词 >` `< 属格 >` 格式）
- 本地解析 + 藏文→POS 映射表 → TokenResponse

### 修改文件

| 文件 | 说明 |
|------|------|
| `src/api/server.py` | 新增 `_dict_trie` / `_build_dict_trie()` / `_fmm_segment()` / `_lookup_words()`；`/analyze` 改为 FMM+词典+gemma；`/pos` 改为 FMM+词典；Qwen → gemma |
| `src/api/models.py` | 新增 `GemmaPosRequest` / `GemmaPosResponse` |
| `CHANGELOG.md` / `CLAUDE.md` | 更新 |

### 启动命令

```bash
# vLLM gemma-2-mitra-it（需要 ~20GB GPU 显存，max_model_len=8192）
HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m src.api.server
```

### TODO

- [ ] 解决 vLLM `max_model_len=8192` 时 KV cache OOM（需 GPU 显存充裕）
- [ ] 前端 `/dict` LookupPage 接入 FMM 分词结果

---

## 2026-04-01 — POS 分类器训练与改进

### 训练结果（TiBERT 已弃用，仅作记录）

| Epoch | Train Loss | Dev Acc | Dev WF1 | Case WF1 |
|-------|-----------|---------|---------|----------|
| 30 | 7.95 | **83.22%** | **81.77%** | **96.24%** |

最优：`model/pos_classifier/best_model.pt`（Epoch 30）

### POS 标签（36 类）

**格助词**（×2 loss weight，F1=96.2%）:
`case.gen`(属格) `case.agn`(作格) `case.all`(为格) `case.abl`(离格)
`case.ela`(从格) `case.ass`(共同格) `case.term`(终结格) `case.loc`(处格)

**名词**: `n.count` `n.prop` `n.rel` `n.mass`
**动词**: `v.past` `v.pres` `v.fut` `v.invar` `v.neg` `v.aux` `v.imp` `v.cop`
**其他**: `adj` `adv` `cv.*` `cl.*` `d.*` `num.*` `punc` `neg` `skt`
