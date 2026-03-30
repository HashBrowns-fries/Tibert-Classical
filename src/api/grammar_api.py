"""
Grammar Analysis API — Qwen LLM 驱动的藏文语法分析

提供分词、词性标注、格助词解读、语法分析功能。
"""

from typing import List, Dict, Optional
from dataclasses import dataclass

from config import QWEN_API_KEY, QWEN_MODEL


@dataclass
class TokenResult:
    token: str
    pos: str
    meaning: str
    grammar_note: str = ""


@dataclass
class AnalysisResult:
    original: str
    tokens: List[TokenResult]
    grammar: str
    raw_response: str = ""


class GrammarAnalyzer:
    """基于 Qwen LLM 的语法分析器"""

    SYSTEM_PROMPT = (
        "你是一位专业的古典藏文语言学家，擅长分词、词性标注、"
        "格助词解读和语法结构分析。请对输入的古典藏文进行详细分析，"
        "严格按照以下 JSON 格式输出，不要添加任何额外文字：\n\n"
        "{"
        '"tokens": ['
        '{"token": "词1", "pos": "词性", "meaning": "含义", "grammar_note": "语法注释"},'
        '{"token": "词2", "pos": "词性", "meaning": "含义", "grammar_note": "语法注释"}'
        "], "
        '"grammar": "整句语法结构说明"'
        "}"
    )

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        初始化分析器。

        Args:
            api_key: DashScope API Key，默认从环境变量读取
            model: Qwen 模型名称，默认从 config 读取
        """
        self.api_key = api_key or QWEN_API_KEY
        self.model = model or QWEN_MODEL

    def _call_llm(self, text: str) -> str:
        """调用 Qwen LLM 获取分析结果"""
        import dashscope
        from dashscope import Generation
        from dashscope.api_entities.dashscope_response import Message

        dashscope.api_key = self.api_key

        response = Generation.call(
            model=self.model,
            messages=[
                Message(role="system", content=self.SYSTEM_PROMPT),
                Message(role="user", content=text),
            ],
            temperature=0.1,
            api_key=self.api_key,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Qwen API error: {response.code} — {getattr(response, 'message', '')}"
            )

        return response.output.text

    def _parse_response(self, text: str, original: str) -> AnalysisResult:
        """解析 LLM 返回的 JSON 响应"""
        import json, re

        # 提取 JSON 块
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return AnalysisResult(
                original=original,
                tokens=[],
                grammar=text,
                raw_response=text,
            )

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return AnalysisResult(
                original=original,
                tokens=[],
                grammar=text,
                raw_response=text,
            )

        tokens = [
            TokenResult(
                token=t["token"],
                pos=t.get("pos", "未知"),
                meaning=t.get("meaning", ""),
                grammar_note=t.get("grammar_note", ""),
            )
            for t in data.get("tokens", [])
        ]

        return AnalysisResult(
            original=original,
            tokens=tokens,
            grammar=data.get("grammar", ""),
            raw_response=text,
        )

    def analyze(self, text: str) -> AnalysisResult:
        """
        分析一段古典藏文。

        Args:
            text: 输入藏文文本

        Returns:
            AnalysisResult，包含分词、词性、语法分析结果
        """
        if not self.api_key:
            raise RuntimeError(
                "Qwen API key 未配置，请设置 DASHSCOPE_API_KEY 环境变量"
            )

        prompt = f"请分析以下古典藏文：{text}"
        response = self._call_llm(prompt)
        return self._parse_response(response, text)

    def segment(self, text: str) -> List[str]:
        """
        仅做分词。

        Args:
            text: 输入藏文文本

        Returns:
            分词结果列表
        """
        result = self.analyze(text)
        return [t.token for t in result.tokens]
