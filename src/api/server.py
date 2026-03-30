"""
TiBERT Classical REST API Server
=================================
FastAPI 服务，提供藏文 POS 标注、完整语法分析、分词、词典查询等端点。

启动：
    python -m src.api.server
    或
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

API 端点：
    GET  /health          — 健康检查
    POST /pos             — POS 标注
    POST /analyze         — 完整分析（POS + LLM 解释）
    POST /segment         — 分词
    POST /lookup          — 词典查询（StarDict + 动词词干）
    GET  /corpus/stats    — 语料库统计
"""

from __future__ import annotations
import os
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.models import (
    PosRequest, AnalyzeRequest, SegmentRequest,
    PosResponse, AnalyzeResponse, SegmentResponse,
    CorpusStatsResponse, TokenResponse,
    LookupRequest, LookupResponse, LookupEntry,
)
from src.api import dependencies as dep

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("tibert-api")

# ── Lifespan: preload models on startup ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("启动中，正在加载 TiBERT 模型...")
    t0 = time.time()
    try:
        # 预加载 tokenizer 和 label map
        dep.get_tokenizer()
        dep.get_label_map()
        # 预加载 POS model
        model, device = dep.get_pos_model()
        log.info(f"模型加载完成（{time.time()-t0:.1f}s），设备={device}")
    except Exception as e:
        log.error(f"模型加载失败: {e}")
        raise
    yield
    log.info("服务器关闭")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TiBERT Classical API",
    description="古典藏文 NLP：POS 标注、格助词分析、语法解释",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS：允许本地 Streamlit 访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_llm_prompt(
    text: str,
    tagged: list[dict],
    dict_results: dict[str, dict],
) -> str:
    """
    Build concise structured prompt for Qwen LLM.
    Includes POS tagging + dictionary definitions for richer context.
    """
    # ── Per-token info: POS + dict definitions ────────────────────────────────
    token_lines = []
    for t in tagged:
        if t["token"] in ("་", "།"):
            continue
        pos_zh = t["pos_zh"]
        note = ""
        if t["is_case_particle"]:
            note = f"【{t['case_name']}·{t['case_desc']}】"

        # Dictionary definitions
        dict_info = ""
        dr = dict_results.get(t["token"])
        if dr:
            defns = []
            for e in dr.get("entries", []):
                defn = e.get("definition", "").split("\n")[0][:120]  # first line, 120 chars
                if defn:
                    defns.append(f"[{e['dict_name']}] {defn}")
            if defns:
                dict_info = "｜".join(defns[:3])  # max 3 dicts to keep prompt short

        line = f"· {t['token']}  [{pos_zh}]{note}"
        if dict_info:
            line += f"\n    → {dict_info}"
        token_lines.append(line)

    tokens_block = "\n".join(token_lines) if token_lines else "（无有效音节）"

    # ── Structure line ─────────────────────────────────────────────────────────
    structure = []
    for t in tagged:
        if t["token"] in ("་", "།"):
            continue
        if t["is_case_particle"]:
            structure.append(f"[{t['case_name']}]")
        elif t["pos"].startswith("n."):
            structure.append(f"[名]{t['token']}")
        elif t["pos"].startswith("v."):
            structure.append(f"[动]{t['token']}")
        elif t["pos"] == "adj":
            structure.append(f"[形]{t['token']}")
        elif t["pos"] == "neg":
            structure.append(f"[否]{t['token']}")
        else:
            structure.append(t["token"])
    structure_line = " ".join(structure)

    return f"""分析以下古典藏文，输出简洁结构化的中英双语解释。

【原文】
{text}

【标注与词典释义】
{tokens_block}

【句法结构】
{structure_line}

请按以下格式输出，每项简洁，不写废话：

## 整句翻译
<中文>
<English>

## 词语解析
- <词>：<含义>（<格助词功能>）

## 句法结构
<简要>

## 格助词（如有）
- <词><格名>：<功能>
"""


def call_qwen_llm(prompt: str) -> str:
    """Call Qwen LLM via DashScope."""
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        return "⚠️ 未设置 DASHSCOPE_API_KEY，无法生成语法解释"

    import dashscope
    from dashscope import Generation
    from dashscope.api_entities.dashscope_response import Message

    dashscope.api_key = api_key

    SYSTEM = (
        "You are a Classical Tibetan linguist. Analyse the given Classical Tibetan sentence "
        "and respond in concise, structured Chinese and English. Follow the output format exactly."
    )

    try:
        resp = Generation.call(
            model="qwen-plus-2025-07-28",
            messages=[
                Message(role="system", content=SYSTEM),
                Message(role="user", content=prompt),
            ],
            temperature=0.2,
            top_p=0.9,
            api_key=api_key,
        )
        if resp.status_code != 200:
            return f"⚠️ LLM 调用失败: {resp.code} — {getattr(resp, 'message', '')}"
        return resp.output.text
    except Exception as e:
        return f"⚠️ LLM 调用异常: {e}"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["健康检查"])
async def health():
    """健康检查端点。"""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/pos", response_model=PosResponse, tags=["POS 标注"])
async def pos_tag(req: PosRequest):
    """
    POS 标注：对输入藏文进行词性标注。

    - 基于 TiBERT + POS 分类器
    - 毫秒级响应，无需 LLM
    - 识别 10 种格助词（属格/作格/为格/离格/从格/共同格/终结格/处格/比格/连格）
    """
    try:
        tokens, stats, syllables_str = dep.tag_text(req.text)
        return PosResponse(
            original=req.text,
            syllables=syllables_str,
            tokens=[TokenResponse(**t) for t in tokens],
            stats=stats,
        )
    except Exception as e:
        log.exception("POS 标注失败")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResponse, tags=["完整分析"])
async def analyze(req: AnalyzeRequest):
    """
    完整语法分析：POS 标注 + LLM 自然语言解释。

    - 首先进行 POS 标注
    - 然后调用 Qwen LLM 生成详细语法解释
    - 设置 use_llm=False 可跳过 LLM（仅返回 POS）
    """
    try:
        tokens, stats, syllables_str = dep.tag_text(req.text)
        llm_explanation = None
        structure = None

        if req.use_llm:
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            if api_key:
                # 并发查词典：每个非分隔符 token 独立查询
                from src.dict import lookup_word
                import asyncio

                unique_words = list({
                    t["token"]: True
                    for t in tokens
                    if t["token"] not in ("་", "།")
                }.keys())

                async def fetch_dict(w: str) -> tuple[str, dict]:
                    loop = asyncio.get_running_loop()
                    return w, await loop.run_in_executor(None, lookup_word, w, None, True)

                dict_results = {}
                if unique_words:
                    results = await asyncio.gather(*[fetch_dict(w) for w in unique_words])
                    dict_results = dict(results)

                prompt = build_llm_prompt(req.text, tokens, dict_results)
                log.info("dict_results sample: %s",
                         {k: v for k, v in list(dict_results.items())[:2]})
                llm_explanation = call_qwen_llm(prompt)

                # Build structure string
                parts = []
                for t in tokens:
                    if t["token"] in ("་", "།"):
                        continue
                    if t["is_case_particle"]:
                        parts.append(f"[{t['case_name']}:{t['token']}]")
                    elif t["pos"].startswith("n."):
                        parts.append(f"[名:{t['token']}]")
                    elif t["pos"].startswith("v."):
                        parts.append(f"[动:{t['token']}]")
                    elif t["pos"] == "adj":
                        parts.append(f"[形:{t['token']}]")
                    else:
                        parts.append(f"[{t['token']}]")
                structure = " ".join(parts)
            else:
                llm_explanation = "⚠️ 未设置 DASHSCOPE_API_KEY，无法生成语法解释"

        return AnalyzeResponse(
            original=req.text,
            syllables=syllables_str,
            tokens=[TokenResponse(**t) for t in tokens],
            stats=stats,
            llm_explanation=llm_explanation,
            structure=structure,
        )
    except Exception as e:
        log.exception("分析失败")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lookup", response_model=LookupResponse, tags=["词典查询"])
async def lookup(req: LookupRequest):
    """
    词典查询：在多个 StarDict 格式藏英词典和动词词干词典中查询词条释义。

    - 支持指定词典名称（dict_name）或查询所有
    - 动词词干词典接受 Wylie 罗马字输入（如 'kum', 'ker'）
    - 藏文词语请直接使用 Unicode 藏文字符
    """
    try:
        from src.dict import lookup_word

        result = lookup_word(
            word=req.word,
            dict_names=[req.dict_name] if req.dict_name else None,
            include_verbs=req.include_verbs,
        )
        return LookupResponse(
            word=result["word"],
            entries=[LookupEntry(**e) for e in result["entries"]],
            verb_entries=result.get("verb_entries"),
        )
    except Exception as e:
        log.exception("词典查询失败")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment", response_model=SegmentResponse, tags=["分词"])
async def segment(req: SegmentRequest):
    """分词：将藏文切分为音节列表。"""
    try:
        tokenizer = dep.get_tokenizer()
        syllables = dep.spm_tokenize(req.text, tokenizer)
        return SegmentResponse(
            original=req.text,
            syllables=dep.spm_tokenize(req.text, tokenizer),
        )
    except Exception as e:
        log.exception("分词失败")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/corpus/stats", response_model=CorpusStatsResponse, tags=["语料库"])
async def corpus_stats():
    """语料库统计信息。"""
    try:
        stats = dep.get_corpus_stats()
        return CorpusStatsResponse(**stats)
    except Exception as e:
        log.exception("语料库统计失败")
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
