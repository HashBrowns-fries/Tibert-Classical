"""
Pydantic models for TiBERT Classical REST API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Request models ──────────────────────────────────────────────────────────────

class PosRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="古典藏文文本")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="古典藏文文本")
    use_llm: bool = Field(default=True, description="是否调用 LLM 生成语法解释")


class SegmentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="古典藏文文本")


# ── Response models ────────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    """单个音节 token 的标注结果。"""
    token: str = Field(..., description="藏语音节")
    pos: str = Field(..., description="POS 标签")
    pos_zh: str = Field(..., description="中文标签名")
    is_case_particle: bool = Field(default=False, description="是否为格助词")
    case_name: Optional[str] = Field(default=None, description="格助词名称（属格/作格等）")
    case_desc: Optional[str] = Field(default=None, description="格助词功能描述")


class PosResponse(BaseModel):
    """POST /pos 响应：POS 标注结果。"""
    original: str = Field(..., description="原始输入文本")
    syllables: str = Field(..., description="音节分隔的原文")
    tokens: List[TokenResponse] = Field(..., description="每个音节的标注")
    stats: dict = Field(..., description="词类统计")


class AnalyzeResponse(BaseModel):
    """POST /analyze 响应：完整分析结果。"""
    original: str
    syllables: str
    tokens: List[TokenResponse]
    stats: dict
    llm_explanation: Optional[str] = Field(default=None, description="LLM 自然语言语法解释")
    structure: Optional[str] = Field(default=None, description="句法结构简写")
    error: Optional[str] = Field(default=None, description="LLM 调用失败时的错误信息")


class SegmentResponse(BaseModel):
    """POST /segment 响应：分词结果。"""
    original: str
    syllables: List[str] = Field(..., description="音节列表")


class CorpusStatsResponse(BaseModel):
    """GET /corpus/stats 响应：语料库统计。"""
    total_sentences: int
    total_collections: int
    collections: List[dict]
    pos_dataset_stats: dict


# ── Dictionary lookup models ────────────────────────────────────────────────────

class LookupRequest(BaseModel):
    """POST /lookup 请求：词典查询。"""
    word: str = Field(..., min_length=1, max_length=200, description="要查询的藏文词语或动词词干（Wylie）")
    dict_name: Optional[str] = Field(default=None, description="词典名称；None = 所有 StarDict 词典")
    include_verbs: bool = Field(default=True, description="是否查询动词词干词典")


class LookupEntry(BaseModel):
    """单个词典的查询结果。"""
    dict_name: str = Field(..., description="词典名称")
    definition: str = Field(..., description="释义文本")


class LookupResponse(BaseModel):
    """POST /lookup 响应：词典查询结果。"""
    word: str
    entries: List[LookupEntry] = Field(default_factory=list, description="StarDict 词典结果")
    verb_entries: Optional[List[dict]] = Field(default=None, description="动词词干词典结果")
