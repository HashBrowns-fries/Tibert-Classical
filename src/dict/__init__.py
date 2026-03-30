"""
src.dict — 藏文词典查询模块
============================
提供 StarDict 格式词典和动词词干词典的统一查询接口。

Usage:
    from src.dict import lookup_word, get_dict, get_verb_lexicon, list_dicts
"""

from .stardict import StarDict
from .verb_lexicon import VerbLexicon, VerbEntry
from .registry import (
    get_dict,
    get_verb_lexicon,
    list_dicts,
    list_all_dict_names,
)

# 优先使用 SQLite 查询（毫秒级），降级到内存 StarDict
try:
    from .sqlite_lookup import sqlite_lookup_word as lookup_word
except FileNotFoundError:
    from .registry import lookup_word


__all__ = [
    "StarDict",
    "VerbLexicon",
    "VerbEntry",
    "get_dict",
    "get_verb_lexicon",
    "list_dicts",
    "list_all_dict_names",
    "lookup_word",
]
