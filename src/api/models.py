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
    dict_entries: List["LookupEntry"] = Field(default_factory=list, description="词典释义列表")


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


class CorpusSentence(BaseModel):
    """单条语料句子。"""
    id: str
    collection: str
    text: str
    syllables: List[str] = Field(default_factory=list)


class CorpusSentencesResponse(BaseModel):
    """GET /corpus/sentences 响应：分页语料列表。"""
    sentences: List[CorpusSentence]
    total: int
    page: int
    page_size: int
    collections: List[str] = Field(default_factory=list)


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


# ── Gemma Segmentation & Dictionary models ──────────────────────────────────────

class GemmaSegmentRequest(BaseModel):
    """POST /gemma/segment 请求：gemma-2-mitra-it 分词。"""
    text: str = Field(..., description="待分词的藏文文本")
    language: str = Field(default="藏文", description="语言（藏文/中文/英文）")


class GemmaSegmentResponse(BaseModel):
    """POST /gemma/segment 响应：分词结果。"""
    text: str
    syllables: List[str]
    method: str = "gemma-2-mitra-it"


class GemmaLookupRequest(BaseModel):
    """POST /gemma/lookup 请求：gemma 分词 + SQLite 词典查询。"""
    text: str = Field(..., description="待查询的藏文文本")
    dict_names: Optional[List[str]] = Field(default=None, description="限定词典名称")


class GemmaLookupEntry(BaseModel):
    word: str
    dict_name: str
    definition: str


class GemmaLookupResponse(BaseModel):
    """POST /gemma/lookup 响应：分词 + 词典结果。"""
    syllables: List[str]
    entries: List[GemmaLookupEntry]
    total: int


# ── Gemma POS tagging models ──────────────────────────────────────────────────────

class GemmaPosRequest(BaseModel):
    """POST /gemma/pos 请求：gemma-2-mitra-it POS 标注。"""
    text: str = Field(..., min_length=1, max_length=5000, description="古典藏文文本")


class GemmaPosResponse(BaseModel):
    """POST /gemma/pos 响应：gemma POS 标注结果。"""
    original: str
    syllables: str = Field(..., description="音节分隔的原文")
    tokens: List[TokenResponse] = Field(..., description="每个音节的标注")
    stats: dict = Field(..., description="词类统计")


# ── RAG models ──────────────────────────────────────────────────────────────────

class RagRequest(BaseModel):
    """POST /rag 请求：基于语料库的 RAG 问答。"""
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题（藏文/中文/英文）")
    language: str = Field(default="藏文", description="回答语言偏好（藏文/中文/英文）")
    top_k: int = Field(default=5, ge=1, le=20, description="检索 chunk 数量")
    build_index: bool = Field(default=False, description="是否在请求时重建索引（如语料已更新）")


class RagChunk(BaseModel):
    """单个检索到的文本块。"""
    text: str = Field(..., description="文本内容")
    source: str = Field(..., description="来源文档")
    distance: float = Field(..., description="向量距离（越小越相关）")


class RagResponse(BaseModel):
    """POST /rag 响应：RAG 问答结果。"""
    question: str
    answer: str
    retrieved_chunks: List[RagChunk] = Field(..., description="检索到的相关文本块")
    retrieve_time_s: float = Field(..., description="检索耗时（秒）")


# ── Learner tool models ─────────────────────────────────────────────────────────

class CaseParticleExample(BaseModel):
    """格助词例句。"""
    particle: str = Field(..., description="格助词藏文形式")
    particle_tag: str = Field(..., description="POS 标签")
    noun: Optional[str] = Field(default=None, description="所附着的名词")
    sentence: str = Field(..., description="完整句子（原始 SegPOS 格式）")
    context: str = Field(..., description="上下文（前后各4个词）")
    collection: str = Field(..., description="语料来源")


class CaseParticleDrill(BaseModel):
    """单个格助词的练习数据。"""
    tag: str = Field(..., description="POS 标签")
    tibetan: str = Field(..., description="藏文形式")
    name: str = Field(..., description="格助名称")
    english: str = Field(..., description="英文名称")
    chinese: str = Field(..., description="中文含义")
    function: str = Field(..., description="语法功能")
    count: int = Field(..., description="语料库中出现次数")
    examples: List[CaseParticleExample] = Field(..., description="例句（最多50条）")


class LearnerParticlesResponse(BaseModel):
    """GET /learn/particles 响应：格助词练习数据。"""
    particles: List[CaseParticleDrill]
    total_sentences: int
    total_words: int


class VerbFormExample(BaseModel):
    """动词变形例句。"""
    form: str = Field(..., description="动词形式")
    sentence: str = Field(..., description="所在完整句子")
    lexicon_meaning: str = Field(default="", description="词典释义")


class VerbDrill(BaseModel):
    """单个动词标签的练习数据。"""
    tag: str = Field(..., description="POS 标签")
    count: int = Field(..., description="出现次数")
    examples: List[VerbFormExample] = Field(..., description="例句")


class LearnerVerbsResponse(BaseModel):
    """GET /learn/verbs 响应：动词练习数据。"""
    verbs: List[VerbDrill]
    total_verb_examples: int


class LearnerDrillRequest(BaseModel):
    """POST /learn/drill 请求：生成一道练习题。"""
    type: str = Field(..., description="题目类型：particle_identify | verb_conjugate | sentence_parse")
    particle_tag: Optional[str] = Field(default=None, description="限定格助词类型")
    text: Optional[str] = Field(default=None, description="用户提供的句子（如不提供则从语料随机选取）")


class LearnerDrillResponse(BaseModel):
    """POST /learn/drill 响应：生成的练习题。"""
    drill_type: str = Field(default="particle_identify", description="题目类型")
    question_type: str
    sentence: str = Field(..., description="练习句子")
    target: str = Field(..., description="题目目标（如格助词名称）")
    hint: Optional[str] = Field(default=None, description="提示")
    answer: str = Field(..., description="正确答案")
    explanation: str = Field(..., description="详细解释（含 gemma 生成内容）")
