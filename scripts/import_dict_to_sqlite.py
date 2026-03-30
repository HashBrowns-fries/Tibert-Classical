"""
一次性将所有 StarDict 词典 + 动词词干词典 bulk insert 到 SQLite。
运行一次即可，后续查询直接走 SQLite，不再碰 .idx/.dict 文件。

Usage:
    python scripts/import_dict_to_sqlite.py
"""
import sys, time
from pathlib import Path

# 确保 src 在路径里
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dict.registry import list_all_dict_names, get_dict, get_verb_lexicon

DB_PATH = Path(__file__).parent.parent / "data" / "tibert_dict.db"


def main():
    import sqlite3

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # ── 创建表和索引 ────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE dict_entries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            word        TEXT    NOT NULL,
            dict_name   TEXT    NOT NULL,
            definition  TEXT    NOT NULL,
            entry_type  TEXT    NOT NULL DEFAULT 'stardict'
        )
    """)
    # 精确匹配索引
    cur.execute("CREATE INDEX idx_word     ON dict_entries(word);")
    # 前缀搜索索引
    cur.execute("CREATE INDEX idx_word_ops  ON dict_entries(word);")
    # 词典名索引
    cur.execute("CREATE INDEX idx_dict_name ON dict_entries(dict_name);")

    total = 0

    # ── 导入 StarDict 词典 ──────────────────────────────────────────────────────
    for name in list_all_dict_names():
        sd = get_dict(name)
        if sd is None:
            print(f"  [SKIP] {name}: files not found")
            continue
        sd.load_index()   # lru_cache(maxsize=1) 保证后续字典全部命中缓存
        count = 0
        batch = []
        for word, (offset, size) in sd._index.items():
            defn = sd.lookup(word) or ""
            batch.append((word, name, defn, "stardict"))
            if len(batch) >= 5000:
                cur.executemany(
                    "INSERT INTO dict_entries(word, dict_name, definition, entry_type) VALUES (?,?,?,?)",
                    batch,
                )
                count += len(batch)
                batch = []
        if batch:
            cur.executemany(
                "INSERT INTO dict_entries(word, dict_name, definition, entry_type) VALUES (?,?,?,?)",
                batch,
            )
            count += len(batch)

        conn.commit()
        total += count
        print(f"  ✓ {name}: {count} entries")

    # ── 导入动词词干词典 ────────────────────────────────────────────────────────
    vl = get_verb_lexicon()
    vl.load()
    entries = vl.all_entries()
    verb_rows = []
    for e in entries:
        meaning = e.meaning or ""
        defn_parts = []
        for label, val in [("现在时", e.present), ("过去时", e.past),
                            ("将来时", e.future), ("命令式", e.imperative), ("释义", meaning)]:
            if val:
                defn_parts.append(f"{label}: {val}")
        defn = " | ".join(defn_parts) if defn_parts else meaning
        verb_rows.append((e.headword, "verb_lexicon", defn, "verb"))

    cur.executemany(
        "INSERT INTO dict_entries(word, dict_name, definition, entry_type) VALUES (?,?,?,?)",
        verb_rows,
    )
    conn.commit()
    total += len(verb_rows)
    print(f"  ✓ verb_lexicon: {len(verb_rows)} entries")

    print(f"\nTotal imported: {total} entries  →  {DB_PATH}")
    cur.close()
    conn.close()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Done in {time.time()-t0:.1f}s")
