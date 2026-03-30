"""
词典注册表
===========
管理所有可用词典的注册、lazy load 和查询接口。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .stardict import StarDict
from .verb_lexicon import VerbLexicon


# ── Path resolution ─────────────────────────────────────────────────────────────

_DICT_ROOT = Path(__file__).parent.parent.parent / "dict"
_STARDICT_DIR = _DICT_ROOT / "All_Tibetan_Digtal_Dictionary-master"
_VERB_XML = (
    _DICT_ROOT / "lexicon-of-tibetan-verb-stems-master" / "dictionary.xml"
)


# ── StarDict registry ─────────────────────────────────────────────────────────

# name → (base_path, description)
_STARDICT_MANIFEST: dict[str, tuple[str, str]] = {
    "RangjungYeshe": (
        str(_STARDICT_DIR / "RangjungYeshe"),
        "Rangjung Yeshe Tibetan-English Dictionary (276k entries)",
    ),
    "DagYig": (
        str(_STARDICT_DIR / "DagYigSarDrig"),
        "Dag Yig Sar Drig — Comprehensive Tibetan Glossary (7k entries)",
    ),
    "Dungkar": (
        str(_STARDICT_DIR / "Dungkar"),
        "Dungkar Tibetan Dictionary (large)",
    ),
    "MonlamTibetan": (
        str(_STARDICT_DIR / "MonlamTibetan"),
        "Monlam Tibetan-Tibetan Dictionary",
    ),
    "MonlamTibEng": (
        str(_STARDICT_DIR / "MonlamTibEng"),
        "Monlam Tibetan-English Dictionary",
    ),
    "MonlamEngTib": (
        str(_STARDICT_DIR / "MonlamEngTib"),
        "Monlam English-Tibetan Dictionary",
    ),
    "MonlamTbTb": (
        str(_STARDICT_DIR / "MonlamTbTb"),
        "Monlam Tibetan-Tibetan (mdx)",
    ),
    "MonlamTbEn": (
        str(_STARDICT_DIR / "MonlamTbEn"),
        "Monlam Tibetan-English (mdx)",
    ),
    "tibChinmo": (
        str(_STARDICT_DIR / "tibChinmo"),
        "Tibetan-Chinese Mo? Dictionary",
    ),
    "HanTb": (
        str(_STARDICT_DIR / "HanTb"),
        "Han-Tibetan (Chinese-Tibetan)",
    ),
    "dz-en": (
        str(_STARDICT_DIR / "dz-en"),
        "Dzongkha-English",
    ),
    "en-dz": (
        str(_STARDICT_DIR / "en-dz"),
        "English-Dzongkha",
    ),
    "dzongkha": (
        str(_STARDICT_DIR / "dzongkha"),
        "General Dzongkha Dictionary",
    ),
    "T-C-E-ITDICT": (
        str(_STARDICT_DIR / "T-C-E-ITDICT"),
        "Tibetan-Chinese-English IT Dictionary",
    ),
}


@lru_cache(maxsize=1)
def get_dict(name: str) -> Optional[StarDict]:
    """
    获取指定名称的 StarDict 词典（lazy load，缓存复用）。

    Parameters
    ----------
    name : str
        词典名称（如 "RangjungYeshe"）

    Returns
    -------
    StarDict | None
        词典实例；文件不存在时返回 None
    """
    if name not in _STARDICT_MANIFEST:
        return None
    base_path, _ = _STARDICT_MANIFEST[name]
    if not Path(base_path + ".dict").exists():
        return None
    return StarDict(base_path)


@lru_cache(maxsize=1)
def get_verb_lexicon() -> VerbLexicon:
    """获取动词词干词典（lazy load，缓存复用）。"""
    return VerbLexicon(str(_VERB_XML))


def list_dicts() -> dict[str, str]:
    """
    列出所有可用的 StarDict 词典名称和描述。
    只返回实际文件存在的词典。
    """
    result = {}
    for name, (_, desc) in _STARDICT_MANIFEST.items():
        base, _ = _STARDICT_MANIFEST[name]
        if Path(base + ".dict").exists():
            result[name] = desc
    return result


def list_all_dict_names() -> list[str]:
    """返回所有已注册的 StarDict 词典名称列表。"""
    return list(_STARDICT_MANIFEST.keys())


# ── Unified lookup API ─────────────────────────────────────────────────────────

def lookup_word(
    word: str,
    dict_names: Optional[list[str]] = None,
    include_verbs: bool = True,
) -> dict:
    """
    统一查询接口：在多个词典中查询词条。

    Parameters
    ----------
    word : str
        要查询的词语
    dict_names : list[str] | None
        要查询的词典名称列表；None = 查询所有
    include_verbs : bool
        是否查询动词词干词典

    Returns
    -------
    dict
        {
            "word": str,
            "entries": [{"dict_name": str, "definition": str}, ...],
            "verb_entries": [verb_entry.to_dict(), ...] or None,
        }
    """
    results: list[dict] = []

    # 确定要查哪些词典
    if dict_names:
        names_to_query = [n for n in dict_names if n in _STARDICT_MANIFEST]
    else:
        names_to_query = list_all_dict_names()

    for name in names_to_query:
        sd = get_dict(name)
        if sd is None:
            continue
        defn = sd.lookup(word)
        if defn:
            results.append({"dict_name": name, "definition": defn})

    verb_entries_out: Optional[list[dict]] = None
    if include_verbs:
        vl = get_verb_lexicon()
        entries = vl.lookup(word)
        if entries:
            verb_entries_out = [e.to_dict() for e in entries]

    return {
        "word": word,
        "entries": results,
        "verb_entries": verb_entries_out,
    }
