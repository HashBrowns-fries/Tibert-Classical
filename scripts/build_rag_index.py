#!/usr/bin/env python3
"""
Build ChromaDB RAG index from individual corpus JSON files.
File-by-file processing to avoid OOM. Skips very large files.
"""
import gc, json, logging, os, shutil, sys, time
from pathlib import Path

import ijson

# ── Setup ──────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build-rag")

ROOT = Path("/home/chenhao/Tibert-Classical")
INDEX_DIR = ROOT / "data/rag_index"
CORPUS_DIR = ROOT / "data/corpus/extracted"

# Files to index (skip UCB-OCR.json which is 1.3GB and caused OOM)
FILES = [
    ("trial.json",          1.2),
    ("Various.json",         32),
    ("PalriParkhang.json",   31),
    ("TulkuSangag.json",     30),
    ("VajraVidya.json",      26),
    ("DrikungChetsang.json",  81),
    ("KarmaDelek.json",      165),
    ("DharmaDownload.json",  179),
    ("Shechen.json",         115),
    ("GuruLamaWorks.json",   489),
    ("eKangyur.json",        211),
    # UCB-OCR.json (1300 MB) skipped — causes OOM
]

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
EMBED_BATCH = 64          # embed every N chunks
CHROMA_FLUSH = 2000       # flush to ChromaDB every N chunks

sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ── Nuke old index ──────────────────────────────────────────────────────────────
if INDEX_DIR.exists():
    log.info("删除旧索引...")
    shutil.rmtree(INDEX_DIR)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ── Embedder ────────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
log.info("加载 embedder (CPU)...")
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
log.info("Embedding 维度: %d", embedder.get_sentence_embedding_dimension())

# ── ChromaDB ────────────────────────────────────────────────────────────────────
import chromadb
client = chromadb.PersistentClient(path=str(INDEX_DIR))
collection = client.create_collection(
    name="tibetan_corpus",
    metadata={"description": "Classical Tibetan Buddhist corpus"},
)

# ── Chunking ─────────────────────────────────────────────────────────────────────
def chunk_text(lines, doc_id):
    """Split lines into overlapping chunks."""
    text = "\n".join(lines)
    if not text.strip():
        return []
    if len(text) <= CHUNK_SIZE:
        return [{"id": f"{doc_id}-0", "text": text, "metadata": {"source": doc_id}}]
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]
        # If not at end, try to break at newline or space
        if end < len(text):
            for sep in ["\n", "་", "།", " "]:
                last_sep = chunk_text.rfind(sep)
                if last_sep > CHUNK_SIZE // 2:
                    chunk_text = chunk_text[:last_sep + 1]
                    end = start + len(chunk_text)
                    break
        chunks.append({
            "id": f"{doc_id}-{idx}",
            "text": chunk_text.strip(),
            "metadata": {"source": doc_id},
        })
        start = end - CHUNK_OVERLAP
        idx += 1
    return chunks

# ── Flush helper ────────────────────────────────────────────────────────────────
batch_ids, batch_texts, batch_metas = [], [], []
total_chunks = 0

def flush():
    global batch_ids, batch_texts, batch_metas, total_chunks
    if not batch_ids:
        return
    t0 = time.time()
    embs = embedder.encode(batch_texts, convert_to_numpy=True)
    collection.add(
        ids=list(batch_ids),
        documents=list(batch_texts),
        metadatas=list(batch_metas),
        embeddings=[e.tolist() for e in embs],
    )
    total_chunks += len(batch_ids)
    log.info("  +%d chunks → total %d  (%.1fs)",
             len(batch_ids), total_chunks, time.time() - t0)
    batch_ids.clear(); batch_texts.clear(); batch_metas.clear()
    gc.collect()

# ── Process files ───────────────────────────────────────────────────────────────
def _make_processor():
    """Return a closure that accumulates per-file chunks."""
    fc = [0]  # mutable counter

    def process_lines(lines, doc_id):
        if not lines or not lines[0].strip():
            return
        for c in chunk_text(lines, doc_id):
            batch_ids.append(c["id"])
            batch_texts.append(c["text"])
            batch_metas.append(c["metadata"])
            fc[0] += 1
            if len(batch_ids) >= EMBED_BATCH:
                flush()
        if len(batch_ids) >= CHROMA_FLUSH:
            flush()

    return process_lines, fc

t0 = time.time()
for fname, size_mb in FILES:
    fpath = CORPUS_DIR / fname
    if not fpath.exists():
        log.warning("跳过不存在: %s", fname)
        continue
    log.info("处理 %s (%.0f MB)...", fname, size_mb)
    process_lines, fc = _make_processor()

    try:
        # Stream with ijson — auto-detect format:
        # Small files:  { collection: { doc_id: [lines] } }
        # Large files:  { doc_id: [lines] }  (flat, no collection wrapper)
        source_label = fname.replace(".json", "")
        with open(fpath, "rb") as f:
            for key, value in ijson.kvitems(f, ""):
                if isinstance(value, dict):
                    # nested format: { collection: { doc_id: [lines] } }
                    for doc_id, lines in value.items():
                        process_lines(lines, f"{source_label}/{key}/{doc_id}")
                else:
                    # flat format: { doc_id: [lines] }
                    process_lines(value, f"{source_label}/{key}")
        flush()  # flush after each file
        gc.collect()
        log.info("  → %s: %d chunks", fname, fc[0])
    except Exception as e:
        log.error("处理 %s 失败: %s", fname, e)
        flush()
        gc.collect()

log.info("索引构建完成！总计 %d chunks，耗时 %.1fs", total_chunks, time.time() - t0)
log.info("索引目录: %s", INDEX_DIR)

# Verify
log.info("验证: col.count() = %d", collection.count())
