"""
Pure RAG API Server — gemma-2-mitra-it + ChromaDB
===================================================
仅加载：vLLM gemma-2-mitra-it (GPU 0) + ChromaDB
不加载 TiBERT POS 模型。

启动：
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m src.api.rag_server
    或
    .venv/bin/python -m src.api.rag_server

端点：
    GET  /health          — 健康检查
    POST /rag             — RAG 问答
    POST /rag/retrieval   — 仅检索（不生成）
    GET  /rag/stats       — 索引统计
"""

from __future__ import annotations

import os
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

import torch

log = logging.getLogger("rag-api")

# ── Config ────────────────────────────────────────────────────────────────────
VLLM_PORT = 8001
VLLM_MODEL = "buddhist-nlp/gemma-2-mitra-it"
INDEX_DIR = Path("/mnt/drive1/chenhao/Tibert-Classical/data/rag_index")

# ── Global state ─────────────────────────────────────────────────────────────
_vllm_engine = None
_embedder = None
_chroma_client = None
_index_built = False


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vllm_engine, _embedder, _chroma_client, _index_built
    log.info("启动 RAG 服务...")
    t0 = time.time()

    # 1. Embedder (CPU)
    from sentence_transformers import SentenceTransformer
    log.info("加载 embedding 模型...")
    _embedder = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
    )
    log.info(f"Embedding 模型加载完成（{time.time()-t0:.1f}s）")

    # 2. vLLM gemma-2-mitra-it (GPU 0)
    import os as _os
    _orig_cuda = _os.environ.get("CUDA_VISIBLE_DEVICES", None)
    _orig_hf = _os.environ.get("HF_ENDPOINT", None)
    _orig_hf_offline = _os.environ.get("HF_HUB_OFFLINE", None)
    _os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    _os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    _os.environ["HF_HUB_OFFLINE"] = "1"
    log.info("加载 gemma-2-mitra-it via vLLM...")
    t1 = time.time()
    from vllm import LLM
    _vllm_engine = LLM(
        model=VLLM_MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.80,
        max_model_len=1536,
        enforce_eager=True,
        trust_remote_code=True,
    )
    log.info(f"gemma-2-mitra-it 加载完成（{time.time()-t1:.1f}s）")
    if _orig_cuda is not None:
        _os.environ["CUDA_VISIBLE_DEVICES"] = _orig_cuda
    else:
        _os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    if _orig_hf is not None:
        _os.environ["HF_ENDPOINT"] = _orig_hf
    else:
        _os.environ.pop("HF_ENDPOINT", None)
    if _orig_hf_offline is not None:
        _os.environ["HF_HUB_OFFLINE"] = _orig_hf_offline
    else:
        _os.environ.pop("HF_HUB_OFFLINE", None)

    # 3. ChromaDB index
    import chromadb
    from chromadb.config import Settings
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _chroma_client = chromadb.PersistentClient(
        path=str(INDEX_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        col = _chroma_client.get_collection("tibetan_corpus")
        log.info(f"RAG 索引就绪（{col.count()} chunks）")
    except Exception:
        log.info("构建 RAG 索引...")
        from src.api.rag import build_index
        n = build_index()
        log.info(f"RAG 索引构建完成（{n} chunks）")

    log.info(f"RAG 服务启动完成（总计 {time.time()-t0:.1f}s）")
    yield
    log.info("RAG 服务关闭")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TiBERT RAG API",
    description="基于藏文佛典语料库的 RAG 问答服务",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ──────────────────────────────────────────────────────────

class RagRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    language: str = Field(default="藏文")
    top_k: int = Field(default=5, ge=1, le=20)


class RagChunk(BaseModel):
    text: str
    source: str
    distance: float


class RagResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: List[RagChunk]
    retrieve_time_s: float


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalResponse(BaseModel):
    chunks: List[RagChunk]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _retrieve(query: str, top_k: int) -> List[RagChunk]:
    """Vector retrieval from ChromaDB."""
    global _embedder, _chroma_client
    collection = _chroma_client.get_collection("tibetan_corpus")
    q_emb = _embedder.encode([query], convert_to_numpy=True)[0]
    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append(RagChunk(
            text=results["documents"][0][i],
            source=results["metadatas"][0][i].get("source", "unknown"),
            distance=results["distances"][0][i],
        ))
    return chunks


def _generate(prompt: str, max_tokens: int = 512) -> str:
    """Call vLLM engine."""
    global _vllm_engine
    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=0.3,
        max_tokens=max_tokens,
        stop=["<end_of_turn>", "#"],
    )
    outputs = _vllm_engine.generate([prompt], sp)
    return outputs[0].outputs[0].text.strip()


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["健康检查"])
async def health():
    return {"status": "ok", "version": "1.0.0", "model": VLLM_MODEL}


@app.post("/rag", response_model=RagResponse, tags=["RAG 问答"])
async def rag_qa(req: RagRequest):
    """
    RAG 问答：
    - ChromaDB 向量检索（paraphrase-multilingual-MiniLM-L12-v2）
    - gemma-2-mitra-it 生成（vLLM, GPU 0）
    """
    t0 = time.time()
    chunks = _retrieve(req.question, req.top_k)

    context = "\n\n".join(
        f"[{i+1}] {c.text} (来源: {c.source})"
        for i, c in enumerate(chunks)
    )

    prompt = (
        f"<start_of_turn>user\n"
        f"You are a Classical Tibetan Buddhist scholar. "
        f"Answer questions based ONLY on the provided context. "
        f"If the answer is not in the context, say so honestly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {req.question}\n"
        f"Answer in {req.language}:\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    answer = _generate(prompt)

    return RagResponse(
        question=req.question,
        answer=answer,
        retrieved_chunks=chunks,
        retrieve_time_s=round(time.time() - t0, 2),
    )


@app.post("/rag/retrieval", response_model=RetrievalResponse, tags=["检索"])
async def retrieval_only(req: RetrievalRequest):
    """仅检索相关文本块，不调用 LLM。"""
    chunks = _retrieve(req.query, req.top_k)
    return RetrievalResponse(chunks=chunks)


# ── Segmentation ───────────────────────────────────────────────────────────────

class SegmentRequest(BaseModel):
    text: str = Field(..., description="待分词的藏文文本")
    language: Optional[str] = Field("藏文", description="语言（藏文/中文/英文）")


class SegmentResponse(BaseModel):
    text: str
    syllables: List[str]
    method: str = "gemma-2-mitra-it"


def _segment_with_gemma(text: str) -> str:
    """
    Use gemma-2-mitra-it for Classical Tibetan syllable segmentation.
    Tibetan syllables are separated by ་, sentences by །.
    """
    prompt = (
        "<start_of_turn>user\n"
        "Segment this Tibetan text — add ་ between syllables and ། at sentence ends. "
        "Output ONLY the segmented Tibetan, no explanation.\n"
        f"Text: {text}\n"
        "Output:\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    result = _generate(prompt, max_tokens=256).strip()
    # Find first Tibetan character and take from there
    for i, ch in enumerate(result):
        if "\u0f00" <= ch <= "\u0fff":
            result = result[i:]
            break
    else:
        return text  # fallback
    # Strip anything after a second <end_of_turn> or obvious English leak
    if "<end_of_turn>" in result:
        result = result.split("<end_of_turn>")[0]
    return result.strip()


@app.post("/segment", response_model=SegmentResponse, tags=["分词"])
async def segment(req: SegmentRequest):
    """
    使用 gemma-2-mitra-it 对藏文进行音节分词。
    藏文分词符：་（音节分隔）、།（句子结束）。
    也支持中文、英文（按空格/词语边界分词）。
    """
    if not req.text.strip():
        return SegmentResponse(text=req.text, syllables=[], method="gemma-2-mitra-it")

    t0 = time.time()
    if req.language == "藏文":
        segmented = _segment_with_gemma(req.text)
        syllables = [s for s in segmented.split("་") if s.strip()]
    else:
        # Chinese/English: simple word split
        syllables = req.text.strip().split()
        segmented = req.text.strip()

    elapsed = time.time() - t0
    log.info("分词耗时 %.2fs，语言=%s，音节数=%d", elapsed, req.language, len(syllables))
    return SegmentResponse(text=req.text, syllables=syllables, method="gemma-2-mitra-it")


# ── Dictionary Lookup ───────────────────────────────────────────────────────────

DICT_DB = Path("/mnt/drive1/chenhao/Tibert-Classical/data/tibert_dict.db")

def _dict_lookup(word: str) -> List[dict]:
    """Look up a Tibetan syllable in SQLite dictionary."""
    import sqlite3
    conn = sqlite3.connect(f"file:{DICT_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT word, dict_name, definition FROM dict_entries WHERE word = ? LIMIT 5",
        (word,),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


class LookupRequest(BaseModel):
    text: str = Field(..., description="待查询的藏文文本")
    dict_names: Optional[List[str]] = Field(None, description="限定词典名称，如 [\"RangjungYeshe\"]")


class DictEntry(BaseModel):
    word: str
    dict_name: str
    definition: str


class LookupResponse(BaseModel):
    syllables: List[str]
    entries: List[DictEntry]
    total: int


@app.post("/lookup", response_model=LookupResponse, tags=["词典查询"])
async def dict_lookup(req: LookupRequest):
    """
    藏文词典查询：
    1. gemma 分词 → 音节列表
    2. 各音节查询 SQLite 词典（dict_entries，541,618 条）
    """
    import sqlite3

    # 1. Segment
    if not req.text.strip():
        return LookupResponse(syllables=[], entries=[], total=0)

    t0 = time.time()
    if all("\u0f00" <= c <= "\u0fff" or c in "་། " for c in req.text):
        segmented = _segment_with_gemma(req.text)
        syllables = [s for s in segmented.split("་") if s.strip()]
    else:
        # Non-Tibetan: just split on spaces
        syllables = req.text.strip().split()

    # 2. Query dictionary
    conn = sqlite3.connect(f"file:{DICT_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(syllables))
    query = f"SELECT word, dict_name, definition FROM dict_entries WHERE word IN ({placeholders}) LIMIT 200"
    if req.dict_names:
        placeholders2 = ",".join("?" * len(req.dict_names))
        query = f"SELECT word, dict_name, definition FROM dict_entries WHERE word IN ({placeholders}) AND dict_name IN ({placeholders2}) LIMIT 200"
        rows = conn.execute(query, syllables + req.dict_names).fetchall()
    else:
        rows = conn.execute(query, syllables).fetchall()
    conn.close()

    entries = [DictEntry(word=r["word"], dict_name=r["dict_name"], definition=r["definition"]) for r in rows]
    log.info("词典查询：%d 音节，%d 条目，耗时 %.2fs", len(syllables), len(entries), time.time() - t0)
    return LookupResponse(syllables=syllables, entries=entries, total=len(entries))


@app.get("/rag/stats", tags=["统计"])
async def rag_stats():
    """RAG 索引统计。"""
    global _chroma_client
    if _chroma_client is None:
        raise HTTPException(status_code=503, detail="RAG 服务尚未初始化")
    try:
        col = _chroma_client.get_collection("tibetan_corpus")
        count = col.count()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RAG 索引未构建: {e}")
    return {
        "collection": "tibetan_corpus",
        "total_chunks": count,
        "index_dir": str(INDEX_DIR),
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "llm_model": VLLM_MODEL,
    }


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "src.api.rag_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    main()
