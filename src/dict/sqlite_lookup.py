"""
SQLite-backed dictionary lookup.
比 StarDict 文件查询快 ~100 倍（索引查找 vs. 每次 open+seek），
无需加载任何 .idx/.dict 文件到内存。
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Optional

_DB_PATH = Path(__file__).parent.parent.parent / "data" / "tibert_dict.db"


@lru_cache(maxsize=1)
def _get_conn() -> sqlite3.Connection:
    """单例连接，进程生命周期内复用。"""
    if not _DB_PATH.exists():
        raise FileNotFoundError(
            f"SQLite DB not found at {_DB_PATH}.\n"
            "Run: python scripts/import_dict_to_sqlite.py"
        )
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA cache_size = 20000")   # 20k page cache ≈ 80 MB
    conn.execute("PRAGMA query_only = ON")      # read-only, safe for concurrency
    return conn


def sqlite_lookup_word(
    word: str,
    dict_names: Optional[list[str]] = None,
    include_verbs: bool = True,
) -> dict:
    """
    查询 SQLite — 返回格式与 registry.lookup_word() 完全一致。

    Parameters
    ----------
    word       : 要查询的词语（藏文）
    dict_names : 仅查指定词典，None = 查全部
    include_verbs : 是否含动词词干词典

    Returns
    -------
    dict  {"word": str, "entries": [...], "verb_entries": [...] or None}
    """
    conn = _get_conn()
    cur = conn.cursor()

    # ── StarDict 条目 ──────────────────────────────────────────────────────────
    if dict_names:
        placeholders = ", ".join("?" * len(dict_names))
        cur.execute(
            f"SELECT dict_name, definition FROM dict_entries "
            f"WHERE word = ? AND dict_name IN ({placeholders}) "
            f"ORDER BY dict_name",
            (word, *dict_names),
        )
    else:
        if include_verbs:
            cur.execute(
                "SELECT dict_name, definition FROM dict_entries "
                "WHERE word = ? AND entry_type = 'stardict' "
                "ORDER BY dict_name",
                (word,),
            )
        else:
            cur.execute(
                "SELECT dict_name, definition FROM dict_entries "
                "WHERE word = ? AND entry_type = 'stardict' "
                "ORDER BY dict_name",
                (word,),
            )

    rows = cur.fetchall()
    entries = [{"dict_name": r[0], "definition": r[1]} for r in rows]

    # ── 动词词干条目 ────────────────────────────────────────────────────────────
    verb_entries_out: Optional[list[dict]] = None
    if include_verbs:
        cur.execute(
            "SELECT definition FROM dict_entries "
            "WHERE word = ? AND entry_type = 'verb'",
            (word,),
        )
        verb_rows = cur.fetchall()
        if verb_rows:
            verb_entries_out = [{"definition": r[0]} for r in verb_rows]

    cur.close()
    return {"word": word, "entries": entries, "verb_entries": verb_entries_out}
