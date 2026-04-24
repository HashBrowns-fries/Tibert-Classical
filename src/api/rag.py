"""
RAG Pipeline for Classical Tibetan using gemma-2-mitra-it
=========================================================
- Embedder: paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers, CPU)
- Vector store: ChromaDB (persistent, incremental writes)
- LLM: buddhist-nlp/gemma-2-mitra-it via vLLM (GPU 0)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer

log = logging.getLogger("tibert-rag")

# ── Config ──────────────────────────────────────────────────────────────────────

CORPUS_PATH = Path("/mnt/drive1/chenhao/Tibert-Classical/data/corpus/extracted/combined.json")
INDEX_DIR = Path("/mnt/drive1/chenhao/Tibert-Classical/data/rag_index")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VLLM_MODEL = "buddhist-nlp/gemma-2-mitra-it"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 5
EMBED_BATCH_SIZE = 128      # embed + write every N chunks
CHROMA_FLUSH_SIZE = 5000   # flush to ChromaDB every N chunks

_chroma_client = None
_embedder = None


# ── Embedder ────────────────────────────────────────────────────────────────────

def get_embedder():
    global _embedder
    if _embedder is None:
        log.info("加载 embedding 模型: %s", EMBEDDING_MODEL)
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        _embedder.eval()
        log.info("Embedding 模型加载完成，维度=%d", _embedder.get_sentence_embedding_dimension())
    return _embedder


# ── ChromaDB ─────────────────────────────────────────────────────────────────────

def get_chroma():
    global _chroma_client
    if _chroma_client is None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=str(INDEX_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


# ── Chunking ─────────────────────────────────────────────────────────────────────

def chunk_text(lines: list[str], doc_key: str) -> list[dict]:
    """Split document lines into overlapping chunks of ~CHUNK_SIZE chars."""
    full_text = "\n".join(lines)
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(full_text):
        end = start + CHUNK_SIZE
        if end < len(full_text):
            newline_pos = full_text.rfind("\n", start, end)
            if newline_pos > start:
                end = newline_pos + 1
        chunk_text = full_text[start:end].strip()
        if chunk_text:
            chunks.append({
                "id": f"{doc_key}_{chunk_id}",
                "text": chunk_text,
                "metadata": {"source": doc_key},
            })
            chunk_id += 1
        start = end - CHUNK_OVERLAP
        if start >= len(full_text):
            break
    return chunks


# ── Index builder (incremental writes — never accumulates full dataset in RAM) ──

def build_index():
    """
    Build ChromaDB index from combined.json.
    Writes to ChromaDB every CHROMA_FLUSH_SIZE chunks to keep RAM usage bounded.
    """
    embedder = get_embedder()
    client = get_chroma()

    try:
        client.delete_collection("tibetan_corpus")
    except Exception:
        pass

    collection = client.create_collection(
        name="tibetan_corpus",
        metadata={"description": "Classical Tibetan Buddhist corpus"},
    )

    file_size = os.path.getsize(CORPUS_PATH)
    log.info("语料库: %s (%.1f GB)", CORPUS_PATH, file_size / 1024**3)

    t0 = time.time()
    total_chunks = 0
    total_docs = 0
    batch_ids, batch_texts, batch_metas = [], [], []

    def flush_batch():
        nonlocal batch_ids, batch_texts, batch_metas, total_chunks
        if not batch_ids:
            return
        embs = embedder.encode(batch_texts, convert_to_numpy=True)
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metas,
            embeddings=[e.tolist() for e in embs],
        )
        total_chunks += len(batch_ids)
        log.info("  已写入 %d chunks (%.1fs)", total_chunks, time.time() - t0)
        batch_ids.clear()
        batch_texts.clear()
        batch_metas.clear()

    # Stream via ijson
    try:
        import ijson
        log.info("流式解析...")
        with open(CORPUS_PATH, "rb") as f:
            for source_name, docs in ijson.kvitems(f, ""):
                for doc_id, lines in docs.items():
                    total_docs += 1
                    if not lines or not lines[0].strip():
                        continue
                    chunks = chunk_text(lines, f"{source_name}/{doc_id}")
                    for c in chunks:
                        batch_ids.append(c["id"])
                        batch_texts.append(c["text"])
                        batch_metas.append(c["metadata"])
                        if len(batch_ids) >= EMBED_BATCH_SIZE:
                            flush_batch()
                    if len(batch_ids) >= CHROMA_FLUSH_SIZE:
                        flush_batch()
    except Exception as e:
        log.warning("ijson 流式失败 (%s)，回退到全量加载...", e)
        flush_batch()  # flush any pending
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        total_docs = sum(len(v) for v in corpus.values())
        log.info("全量加载: %d 集合，%d 文档", len(corpus), total_docs)
        for source_name, docs in corpus.items():
            for doc_id, lines in docs.items():
                total_docs += 1
                if not lines:
                    continue
                chunks = chunk_text(lines, f"{source_name}/{doc_id}")
                for c in chunks:
                    batch_ids.append(c["id"])
                    batch_texts.append(c["text"])
                    batch_metas.append(c["metadata"])
                    if len(batch_ids) >= EMBED_BATCH_SIZE:
                        flush_batch()
                if len(batch_ids) >= CHROMA_FLUSH_SIZE:
                    flush_batch()

    flush_batch()
    log.info("索引构建完成！共 %d chunks / %d 文档（%.1fs）",
              total_chunks, total_docs, time.time() - t0)
    return total_chunks


# ── Retrieval ───────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Vector retrieval from ChromaDB."""
    client = get_chroma()
    collection = client.get_collection("tibetan_corpus")
    embedder = get_embedder()
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]
    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "distance": results["distances"][0][i],
        })
    return chunks


def get_stats() -> dict:
    """Return ChromaDB index statistics."""
    client = get_chroma()
    try:
        col = client.get_collection("tibetan_corpus")
        return {
            "collection": "tibetan_corpus",
            "total_chunks": col.count(),
            "index_dir": str(INDEX_DIR),
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": VLLM_MODEL,
        }
    except Exception as e:
        return {"error": str(e)}


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true")
    args = parser.parse_args()
    if args.build_index:
        n = build_index()
        print(f"索引完成: {n} chunks")
