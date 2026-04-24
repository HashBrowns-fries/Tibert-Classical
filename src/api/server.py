"""
Classical Tibetan REST API Server
=================================
FastAPI 服务，提供藏文分词、POS 标注、词典查询、语料库等。

启动：python -m src.api.server
依赖：lotsawa (CAI PoS tagger) for 分词+POS
"""

from __future__ import annotations
import json
import os
import re
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── MiniMax LLM helper ─────────────────────────────────────────────────────────

_MINIMAX_CLIENT = None

def _get_minimax_client():
    global _MINIMAX_CLIENT
    if _MINIMAX_CLIENT is None:
        import anthropic
        _MINIMAX_CLIENT = anthropic.Anthropic(
            api_key=os.environ.get("MINIMAX_API_KEY", ""),
            base_url=os.environ.get("ANTHROPIC_BASE_URL", ""),
        )
    return _MINIMAX_CLIENT

MINIMAX_MODEL = "claude-3-5-sonnet"

SYSTEM_PROMPT = """你是一位专业的古典藏文（Classical Tibetan）语言学家，精通分词、词性标注、格助词（case particle）解读和语法结构分析。

输入数据包含：
1. 原文（藏文）
2. 分词与词性标注结果
3. 词典释义（来自 RangjungYeshe、MonlamTibEng、DagYig 等词典）

请结合上述信息，用中文对原文进行详细的语法解释，包括：
- 每个实词的含义（结合词典释义）
- 格助词的语法功能（属格、作格、为格、离格等）
- 整句的句法结构

回复要求：
- 使用中文
- 结构清晰，分段说明
- 如有不确定之处，直接说明，不要编造"""

def _call_minimax(prompt: str) -> str:
    """
    Call MiniMax (Anthropic-compatible API) for grammar explanation.
    Returns the text response, or raises on error.
    """
    try:
        client = _get_minimax_client()
        msg = client.messages.create(
            model=MINIMAX_MODEL,
            max_tokens=1024,
            temperature=0.3,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract text from response blocks (ignore ThinkingBlock)
        parts = []
        for block in msg.content:
            if hasattr(block, "text") and block.text:
                parts.append(block.text)
        return "\n".join(parts) if parts else "(无内容)"
    except Exception as e:
        log.warning(f"MiniMax API 调用失败: {e}")
        raise

def _call_gemma(prompt: str, max_tokens: int = 512) -> str:
    """
    Call gemma-2-mitra-it via vLLM (GPU 0) for RAG answer generation.
    Lazy-loads the vLLM engine on first call.
    """
    global _gemma_engine
    if _gemma_engine is None:
        import os as _os
        from vllm import LLM as _LLM
        _orig_cuda = _os.environ.get("CUDA_VISIBLE_DEVICES", None)
        _orig_hf = _os.environ.get("HF_ENDPOINT", None)
        _orig_hf_offline = _os.environ.get("HF_HUB_OFFLINE", None)
        _os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        _os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        _os.environ["HF_HUB_OFFLINE"] = "1"
        log.info("加载 gemma-2-mitra-it via vLLM...")
        _gemma_engine = _LLM(
            model="buddhist-nlp/gemma-2-mitra-it",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.80,
            max_model_len=1536,
            enforce_eager=True,
            trust_remote_code=True,
        )
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
        log.info("gemma-2-mitra-it 引擎就绪")
    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=0.3,
        max_tokens=max_tokens,
        stop=["<end_of_turn>", "#"],
    )
    outputs = _gemma_engine.generate([prompt], sp)
    return outputs[0].outputs[0].text.strip()
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    PosRequest, AnalyzeRequest, SegmentRequest,
    PosResponse, AnalyzeResponse, SegmentResponse, TokenResponse,
    LookupRequest, LookupResponse, LookupEntry,
    GemmaPosRequest, GemmaPosResponse,
    GemmaSegmentRequest, GemmaSegmentResponse,
    GemmaLookupRequest, GemmaLookupResponse,
    RagRequest, RagResponse, RagChunk,
    CorpusStatsResponse, CorpusSentencesResponse,
    LearnerParticlesResponse, LearnerVerbsResponse,
    CaseParticleDrill, VerbDrill, LearnerDrillResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("tibert-api")

# ── Global state ────────────────────────────────────────────────────────────────
_corpus_cache = None
_corpus_collections = None
_dict_trie = None
_learner_data = None
_lotsawa_pos = None   # lazy-loaded PartOfSpeechTagger
_gemma_engine = None  # lazy-loaded vLLM gemma engine

# ── Lotsawa POS tagger ─────────────────────────────────────────────────────────

def _init_lotsawa_pos():
    global _lotsawa_pos
    sys_path = "/mnt/drive1/chenhao/Tibert-Classical/.venv/lib/python3.11/site-packages"
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from cai_manas.part_of_speech.pos_tagger import PartOfSpeechTagger
    log.info("Loading lotsawa PartOfSpeechTagger...")
    _lotsawa_pos = PartOfSpeechTagger("part-of-speech-intersyllabic-olive-cormorant")
    _lotsawa_pos.cuda()
    log.info("lotsawa POS tagger ready (GPU)")

def _get_lotsawa_pos():
    global _lotsawa_pos
    if _lotsawa_pos is None:
        _init_lotsawa_pos()
    return _lotsawa_pos

# ── Case particle extraction ──────────────────────────────────────────────────
CASE_PARTICLE_SYLLABLES: dict[str, str] = {
    "གི": "属格", "གྱི": "属格", "ཀྱི": "属格",
    "གྱིས": "作格", "གྱས": "作格",
    "ལ": "为格", "ལ་": "为格",
    "ལས": "离格",
    "ནས": "从格",
    "དང": "共同格",
    "དུ": "终结格", "ར": "终结格",
    "ན": "处格",
    "སྟེ": "终结格",
}

def _extract_case_name(token: str) -> Optional[str]:
    # Strip [SEP]/[CLS] then get 2nd-to-last syllable
    # (last real syllable; handles trailing ་ which splits to empty string)
    token = token.replace("[SEP]", "").replace("[CLS]", "").strip()
    parts = [p for p in token.replace("།", "་").split("་") if p.strip()]
    if len(parts) >= 2:
        return CASE_PARTICLE_SYLLABLES.get(parts[-1]) or CASE_PARTICLE_SYLLABLES.get(parts[-2])
    return CASE_PARTICLE_SYLLABLES.get(parts[-1]) if parts else None

# ── POS zh name map ───────────────────────────────────────────────────────────
POS_ZH: dict[str, str] = {
    "NOUN": "名词", "PROPN": "专有名词", "NUM": "数词", "DET": "限定词",
    "VERB": "动词", "ADJ": "形容词", "ADV": "副词",
    "PRON": "代词", "PART": "助词", "ADP": "格助词",
    "AUX": "助动词", "CCONJ": "连词", "PUNCT": "标点",
    "unk": "未识别",
    "NOUN+ADP": "名词+属格", "NOUN+PART": "名词+助词",
    "PROPN+ADP": "专有名词+属格", "PROPN+PART": "专有名词+助词",
    "VERB+ADP": "动词+助词", "VERB+PART": "动词+助词",
    "ADJ+ADP": "形容词+属格", "ADJ+PART": "形容词+助词",
    "ADV+ADP": "副词+属格", "ADV+PART": "副词+助词",
    "PRON+ADP": "代词+属格", "PRON+PART": "代词+助词",
    "PART+ADP": "助词+属格", "PART+PART": "助词+助词",
    "NUM+ADP": "数词+属格", "NUM+PART": "数词+助词",
    "DET+ADP": "限定词+属格", "DET+PART": "限定词+助词",
    "unk+ADP": "未识别+属格", "unk+unk": "未识别",
    "n.count": "普通名词", "n.prop": "专有名词",
    "v.pres": "动词(现在)", "v.past": "动词(过去)",
    "v.fut": "动词(将来)", "v.imp": "动词(命令)",
    "adj": "形容词", "adv": "副词",
    "case.gen": "属格助词", "case.agn": "作格助词",
    "case.all": "为格助词", "case.abl": "离格助词",
    "case.ela": "从格助词", "case.ass": "共同格助词",
    "case.term": "终结格助词", "case.loc": "处格助词",
    "punc": "标点",
}

# ── Dictionary lookup via SQLite ───────────────────────────────────────────────

_db_path = Path(__file__).parent.parent.parent / "data" / "tibert_dict.db"

def _lookup_word(word: str) -> list[dict]:
    """Look up a Tibetan word in the SQLite dictionary."""
    import sqlite3
    try:
        conn = sqlite3.connect(str(_db_path))
        cur = conn.cursor()
        cur.execute(
            "SELECT dict_name, definition, entry_type FROM dict_entries WHERE word = ?",
            (word,),
        )
        rows = cur.fetchall()
        conn.close()
        return [{"dict_name": r[0], "definition": r[1]} for r in rows]
    except Exception as e:
        log.warning(f"词典查询失败 {word}: {e}")
        return []

# ── Corpus ─────────────────────────────────────────────────────────────────────
def _load_corpus():
    global _corpus_cache, _corpus_collections
    if _corpus_cache is not None:
        return
    corpus_file = Path(__file__).parent.parent.parent / "data" / "corpus" / "extracted" / "combined.json"
    if not corpus_file.exists():
        _corpus_cache = {}
        _corpus_collections = []
        return
    t0 = time.time()
    with open(corpus_file, encoding="utf-8") as f:
        _corpus_cache = json.load(f)
    _corpus_collections = list(_corpus_cache.keys())
    total = sum(len(v) for v in _corpus_cache.values())
    log.info(f"语料库已加载（{len(_corpus_collections)} 收藏集，{total} 段落，{time.time()-t0:.1f}s）")

def _get_corpus_sentences(collection=None, page=1, page_size=20, search=None):
    _load_corpus()
    all_sentences = []
    collections = _corpus_collections if collection is None else [collection]
    for coll in collections:
        docs = _corpus_cache.get(coll, {})
        for doc_id, paras in docs.items():
            for para in paras:
                if isinstance(para, str) and para.strip():
                    all_sentences.append((f"{coll}#{doc_id}", coll, para.strip()))
    if search:
        all_sentences = [(s, c, t) for s, c, t in all_sentences if search in t]
    total = len(all_sentences)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    page_sentences = all_sentences[start:start + page_size]
    result = []
    for sid, coll, text in page_sentences:
        parts = re.split(r'(་|།)', text)
        syllables = [p for p in parts if p and p not in ("་", "།")]
        result.append({"id": sid, "collection": coll, "text": text, "syllables": syllables})
    return result, total, _corpus_collections

def _get_corpus_stats() -> dict:
    _load_corpus()
    coll_counts = {}
    for coll in _corpus_collections:
        docs = _corpus_cache.get(coll, {})
        coll_counts[coll] = sum(len(v) for v in docs.values())
    return {
        "total_sentences": sum(coll_counts.values()),
        "total_collections": len(coll_counts),
        "collections": [{"name": k, "count": v} for k, v in coll_counts.items()],
        "pos_dataset_stats": {},
    }

# ── Learner data ─────────────────────────────────────────────────────────────
def _load_learner_data():
    global _learner_data
    if _learner_data is not None:
        return _learner_data
    fpath = Path(__file__).parent.parent.parent / "data" / "learner_data.json"
    if fpath.exists():
        with open(fpath, encoding="utf-8") as f:
            _learner_data = json.load(f)
        log.info(f"学习数据已加载（{len(_learner_data.get('particles', []))} 组格助词）")
    else:
        _learner_data = {"particles": [], "verbs": []}
        log.warning(f"学习数据文件不存在: {fpath}")
    return _learner_data

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("启动中，加载 lotsawa POS tagger...")
    try:
        _init_lotsawa_pos()
        log.info("lotsawa POS tagger 就绪")
    except Exception as e:
        log.error(f"lotsawa 加载失败: {e}")
    yield
    log.info("服务器关闭")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Classical Tibetan API",
    description="古典藏文分词、POS 标注、词典查询、格助词学习",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ─────────────────────────────────────────────────────────────────────
def _spm_syllables_of(text: str) -> list[str]:
    """Split a lotsawa word into bare syllables (strip tsheg/tshang)."""
    return [p for p in text.replace("།", "་").split("་") if p.strip()]

def _align_tibert_to_lotsawa(
    text: str,
    lotsawa_words: list[str],
    tibert_syllables: list[str],
    tibert_tags: list[str],
    lm: dict,
) -> list[tuple[str, str]]:
    """
    Align TiBERT syllable-level tags to lotsawa word-level output.
    Uses longest-match: for each lotsawa word, find all consecutive
    TiBERT syllables that belong to it, then pick the most specific tag
    (case particle takes precedence).
    """
    # Strip tsheg markers from TiBERT syllables to match _spm_syllables_of output
    bare_sylls, bare_tags = [], []
    for s, t in zip(tibert_syllables, tibert_tags):
        if s not in ("་", "།", "༔"):
            bare_sylls.append(s)
            bare_tags.append(t)

    result = []
    t_idx = 0  # index into bare_sylls / bare_tags

    for word in lotsawa_words:
        word2 = (word.replace("[SEP]", "").replace("[CLS]", "").strip())
        if not word2:
            continue
        sylls = _spm_syllables_of(word2)
        n = len(sylls)

        # Find where these n syllables appear in TiBERT output (bare syllables)
        match_start = None
        search_from = t_idx
        while search_from <= len(bare_sylls) - n:
            if bare_sylls[search_from:search_from + n] == sylls:
                match_start = search_from
                break
            search_from += 1

        if match_start is not None:
            t_idx = match_start + n  # advance past matched syllables
            seg_tags = [bare_tags[match_start + i] for i in range(n)]
        else:
            # No match found — TiBERT syllable count may differ from lotsawa's word count.
            # Take N syllables from current TiBERT position (both process text in order).
            seg_tags = bare_tags[t_idx : t_idx + n]
            t_idx += n

        # Case particle: TiBERT wins if any tag is a case particle
        case_tags = [t for t in seg_tags if t.startswith("case.")]
        if case_tags:
            result.append((word2, case_tags[0], seg_tags))
        else:
            result.append((word2, seg_tags[0] if seg_tags else "O", seg_tags))

    return result

def _run_tibert(text: str) -> tuple[list[str], list[str], dict]:
    import sys, torch
    sys.path.insert(0, "/mnt/drive1/chenhao/Tibert-Classical")
    from src.api.dependencies import get_tokenizer, get_pos_model, spm_tokenize, get_label_map
    tok = get_tokenizer()
    model, device = get_pos_model()
    lm = get_label_map()
    sylls = spm_tokenize(text, tok)
    ids = [tok.bos_token_id] + [tok._convert_token_to_id(s) for s in sylls] + [tok.eos_token_id]
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(ids_t, attention_mask=(ids_t != 0).long())
        preds = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
    tags = [lm.get(lid, "O") for lid in preds[1:-1]]
    case_count = sum(1 for t in tags if t.startswith("case."))
    log.info(f"TiBERT: sylls={sylls} case_particles={case_count} tags={tags}")
    return sylls, tags, lm

def _build_tokens(text: str, lotsawa_words: list[str], lotsawa_tags: list[str]):
    """
    Hybrid pipeline:
      - Segmentation: lotsawa (precise word boundaries)
      - Case particle type: TiBERT (F1=97.6% on 17 case particle types)
      - Other POS: TiBERT syllable-level sequential tags
    """
    from src.api.dependencies import get_tokenizer, spm_tokenize
    tok = get_tokenizer()

    # Filter [CLS]/[SEP] from lotsawa output
    clean_words, clean_lotsawa_tags = [], []
    for w, t in zip(lotsawa_words, lotsawa_tags):
        w2 = w.replace("[SEP]", "").replace("[CLS]", "").strip()
        if w2:
            clean_words.append(w2)
            clean_lotsawa_tags.append(t)

    # Run TiBERT on full text — get sequential syllables and tags
    tibert_sylls, tibert_tags, lm = _run_tibert(text)

    # Build per-word case detection from alignment
    aligned = _align_tibert_to_lotsawa(text, clean_words, tibert_sylls, tibert_tags, lm)

    tokens_out, noun_count, verb_count, case_count = [], 0, 0, 0

    # Walk TiBERT syllables in order
    t_idx = 0
    while t_idx < len(tibert_sylls):
        syl = tibert_sylls[t_idx]
        tag = tibert_tags[t_idx]

        if syl in ("[CLS]", "[SEP]"):
            t_idx += 1
            continue

        # Tsheg/tshang → punctuation
        if syl in ("་", "།", "༔"):
            tokens_out.append(TokenResponse(
                token=syl, pos="punc", pos_zh="标点",
                is_case_particle=False, case_name=None, case_desc=None, dict_entries=[],
            ))
            t_idx += 1
            continue

        # Find which lotsawa word this syllable belongs to
        word_idx = 0
        for wi, word in enumerate(clean_words):
            syllables = spm_tokenize(word, tok)
            for si, syl2 in enumerate(syllables):
                if syl2 in ("་", "།", "༔"):
                    continue
                if syl2 == syl:
                    word_idx = wi
                    break
            else:
                continue
            break

        base = tag.split("+")[0] if "+" in tag else tag
        is_syl_case = base.startswith("case.")

        # Count noun/verb
        if base not in ("punc", "punc.seg", "punc.eos"):
            if base in ("NOUN", "PROPN", "PRON", "DET", "NUM", "p.pers", "n.count", "n.prop", "n.rel", "n.mass"):
                noun_count += 1
            elif base in ("VERB", "v.past", "v.pres", "v.fut", "v.imp", "v.invar", "v.aux", "v.cop", "v.neg"):
                verb_count += 1

        # case_name: first case-tagged syllable of each case-particle word
        if is_syl_case and (not tokens_out or not tokens_out[-1].is_case_particle):
            case_count += 1
            case_name = POS_ZH.get(tag, "格助词")
        elif is_syl_case:
            case_name = tokens_out[-1].case_name
        else:
            case_name = None

        syl_entries = _lookup_word(syl)
        syl_dict = [LookupEntry(dict_name=e["dict_name"], definition=e["definition"]) for e in syl_entries]
        tokens_out.append(TokenResponse(
            token=syl, pos=tag, pos_zh=POS_ZH.get(tag, tag),
            is_case_particle=is_syl_case,
            case_name=case_name,
            case_desc=None, dict_entries=syl_dict,
        ))
        t_idx += 1

    syllable_count = sum(len(spm_tokenize(w, tok)) for w in clean_words)
    syllables_str = " · ".join(clean_words)
    stats = {"nouns": noun_count, "verbs": verb_count,
             "case_particles": case_count, "syllable_count": syllable_count}
    return tokens_out, stats, syllables_str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}

@app.post("/pos", response_model=PosResponse)
async def pos_tag(req: PosRequest):
    """lotsawa 分词 + POS 标注。"""
    try:
        tagger = _get_lotsawa_pos()
        result = tagger.tag(req.text)
        tokens_out, stats, syllables_str = _build_tokens(req.text, result["words"], result["tags"])
        return PosResponse(original=req.text, syllables=syllables_str, tokens=tokens_out, stats=stats)
    except Exception as e:
        log.exception("POS 标注失败")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """lotsawa 分词+POS + 词典查词 + MiniMax 语法解释。"""
    try:
        tagger = _get_lotsawa_pos()
        result = tagger.tag(req.text)
        words, pos_tags = result["words"], result["tags"]
        tokens_out, stats, syllables_str = _build_tokens(req.text, words, pos_tags)

        # Build dictionary lookup context
        dict_lines = []
        seen = set()
        for tok in tokens_out:
            if tok.token in ("་", "།", "༔"):
                continue
            entries = _lookup_word(tok.token)
            key = (tok.token, tuple(sorted(set(e["dict_name"] for e in entries))))
            if key not in seen:
                seen.add(key)
                if entries:
                    defs = " | ".join(f"[{e['dict_name']}] {e['definition'][:80]}" for e in entries)
                    dict_lines.append(f"  {tok.token}: {tok.pos_zh} → {defs}")

        # Build MiniMax prompt
        pos_lines = "\n".join(
            f"  {t.token}: {t.pos}（{t.pos_zh}）{('★ ' + t.case_name if t.case_name else '')}"
            for t in tokens_out if t.token not in ("་", "།", "༔")
        )
        prompt = f"""请分析以下古典藏文：

【原文】
{req.text}

【分词与词性】
{pos_lines}

【词典释义】
{dict_lines if dict_lines else "  （词典中未找到对应条目）"}
"""
        # Call MiniMax (only when use_llm=True)
        llm_explanation = None
        if req.use_llm:
            try:
                llm_explanation = _call_minimax(prompt)
            except Exception as e:
                log.warning(f"MiniMax 调用失败: {e}")
                llm_explanation = None

        return AnalyzeResponse(
            original=req.text,
            syllables=syllables_str,
            tokens=tokens_out,
            stats=stats,
            llm_explanation=llm_explanation,
            structure=syllables_str,
        )
    except Exception as e:
        log.exception("分析失败")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/segment", response_model=SegmentResponse)
async def segment(req: SegmentRequest):
    """lotsawa 分词。"""
    try:
        tagger = _get_lotsawa_pos()
        result = tagger.tag(req.text)
        return SegmentResponse(original=req.text, syllables=result["words"])
    except Exception as e:
        log.exception("分词失败")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/lookup", response_model=LookupResponse)
async def lookup(req: LookupRequest):
    """词典查词。"""
    word = req.word.strip()
    if not word:
        return LookupResponse(word=word, entries=[], verb_entries=[])
    entries = _lookup_word(word)
    return LookupResponse(
        word=word,
        entries=[LookupEntry(dict_name=e["dict_name"], definition=e["definition"]) for e in entries],
        verb_entries=[],
    )

@app.post("/gemma/pos", response_model=GemmaPosResponse)
async def gemma_pos(req: GemmaPosRequest):
    """lotsawa POS（兼容格式）。"""
    try:
        tagger = _get_lotsawa_pos()
        result = tagger.tag(req.text)
        tokens_out, stats, syllables_str = _build_tokens(req.text, result["words"], result["tags"])
        return GemmaPosResponse(original=req.text, syllables=syllables_str, tokens=tokens_out, stats=stats)
    except Exception as e:
        log.exception("gemma/pos 失败")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gemma/segment", response_model=GemmaSegmentResponse)
async def gemma_segment(req: GemmaSegmentRequest):
    """lotsawa 分词（兼容格式）。"""
    try:
        tagger = _get_lotsawa_pos()
        result = tagger.tag(req.text)
        return GemmaSegmentResponse(original=req.text, syllables=result["words"], method="lotsawa-olive-cormorant")
    except Exception as e:
        log.exception("gemma/segment 失败")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gemma/lookup", response_model=GemmaLookupResponse)
async def gemma_lookup(req: GemmaLookupRequest):
    """词典查词（兼容格式）：将文本按音节分割后查词典。"""
    text = req.text.strip()
    if not text:
        return GemmaLookupResponse(syllables=[], entries=[], total=0)
    # Split into syllables (bare syllables only)
    syllables = [p.strip() for p in text.replace("།", "་").split("་") if p.strip()]
    all_entries = []
    for syll in syllables:
        entries = _lookup_word(syll)
        for e in entries:
            all_entries.append({"word": syll, "dict_name": e["dict_name"], "definition": e["definition"]})
    return GemmaLookupResponse(syllables=syllables, entries=all_entries, total=len(all_entries))

@app.post("/rag", response_model=RagResponse)
async def rag(req: RagRequest):
    """ChromaDB RAG 检索 + gemma-2-mitra-it 生成答案。"""
    try:
        from src.api.rag import retrieve
        t0 = time.time()
        chunks = retrieve(req.question, top_k=req.top_k)
        total_time = time.time() - t0

        # gemma-2-mitra-it 生成答案
        answer = None
        if chunks:
            context = "\n\n".join(
                f"[{i+1}] {c['text']} (来源: {c['source']})"
                for i, c in enumerate(chunks)
            )
            prompt = (
                "<start_of_turn>user\n"
                "You are a Classical Tibetan Buddhist scholar. "
                "Answer questions based ONLY on the provided context. "
                "If the answer is not in the context, say so honestly.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {req.question}\n"
                f"Answer in {req.language}:\n"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
            )
            try:
                answer = _call_gemma(prompt)
            except Exception as gen_err:
                log.warning(f"gemma 生成失败: {gen_err}")
                answer = f"（检索到 {len(chunks)} 条相关段落，见下方）"

        return RagResponse(
            question=req.question,
            answer=answer or f"（检索到 {len(chunks)} 条相关段落，见下方）",
            retrieved_chunks=[RagChunk(**c) for c in chunks],
            retrieve_time_s=round(total_time, 2),
        )
    except Exception as e:
        log.exception("RAG 查询失败")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/stats")
async def rag_stats():
    try:
        from src.api.rag import get_stats
        return get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/corpus/stats", response_model=CorpusStatsResponse)
async def corpus_stats():
    try:
        return CorpusStatsResponse(**_get_corpus_stats())
    except Exception as e:
        log.exception("语料库统计失败")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/corpus/sentences", response_model=CorpusSentencesResponse)
async def corpus_sentences(
    collection: str | None = None,
    page: int = 1,
    page_size: int = 20,
    search: str | None = None,
):
    try:
        sentences, total, collections = _get_corpus_sentences(collection, page, page_size, search)
        return CorpusSentencesResponse(
            sentences=sentences, total=total, page=page,
            page_size=page_size, collections=collections,
        )
    except Exception as e:
        log.exception("语料库查询失败")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learn/particles", response_model=LearnerParticlesResponse)
async def learn_particles():
    data = _load_learner_data()
    return LearnerParticlesResponse(
        particles=[CaseParticleDrill(**p) for p in data.get("particles", [])],
        total_sentences=sum(p.get("count", 0) for p in data.get("particles", [])),
        total_words=sum(p.get("word_count", 0) for p in data.get("particles", [])),
    )

@app.get("/learn/verbs", response_model=LearnerVerbsResponse)
async def learn_verbs():
    data = _load_learner_data()
    verbs = [VerbDrill(**v) for v in data.get("verbs", [])]
    total = sum(len(v.examples) for v in verbs)
    return LearnerVerbsResponse(verbs=verbs, total_verb_examples=total)

@app.post("/learn/drill", response_model=LearnerDrillResponse)
async def learn_drill(req: LearnerDrillRequest = None):
    import random
    data = _load_learner_data()
    particles = data.get("particles", [])
    if not particles:
        raise HTTPException(status_code=503, detail="学习数据未加载")

    # Filter by particle_tag if provided
    candidates = particles
    if req and req.particle_tag:
        candidates = [p for p in particles if p.get("tag") == req.particle_tag]
        if not candidates:
            candidates = particles

    p = random.choice(candidates)
    examples = p.get("examples", [p.get("example", "")])
    example = random.choice(examples) if examples else ""
    tag = p.get("tag", "ADP")

    # Try to use gemma for explanation, fall back to template
    try:
        explanation_prompt = (
            f"Explain the Tibetan case particle '{p.get('syllable', '')}' "
            f"in the sentence: {example}\n"
            f"Describe its grammatical function briefly in Chinese."
        )
        explanation = _call_gemma(explanation_prompt, max_tokens=128)
    except Exception:
        explanation = f"格助词 {p.get('syllable', '')} 表示{CASE_PARTICLE_SYLLABLES.get(p.get('syllable', ''), '格助词')}。"

    return LearnerDrillResponse(
        drill_type="particle_identify",
        question_type="fill_blank",
        sentence=example,
        answer=p.get("syllable", ""),
        target=tag,
        explanation=explanation,
    )

class LearnerCheckRequest(BaseModel):
    drill_type: str
    answer: str
    user_answer: str
    target: str
    sentence: str

@app.post("/learn/check")
async def learn_check(req: LearnerCheckRequest):
    correct = req.user_answer.strip() == req.answer.strip()
    score = 1.0 if correct else 0.0
    if correct:
        feedback = f"正确！{req.target} {req.answer}"
    else:
        feedback = f"错误。正确答案是：{req.answer}"
    return {"correct": correct, "feedback": feedback, "score": score}
