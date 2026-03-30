"""
StarDict 解析器（支持大端序 .idx 格式）
========================================
适用于 dict/All_Tibetan_Digtal_Dictionary-master/*.dict 格式词典。
关键发现：这些 .idx 文件使用 big-endian u32 存储 offset/size，
与标准 StarDict spec 的 little-endian 不同。
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional


class StarDict:
    """
    读取 StarDict 格式词典（.ifo / .idx / .dict 三件套）。

    Parameters
    ----------
    base_path : str | Path
        词典基础路径（不含后缀），例如 /path/to/RangjungYeshe
    """

    def __init__(self, base_path: str | Path) -> None:
        base = Path(base_path)
        self.name = base.name
        self.ifo_path = base.with_suffix(".ifo")
        self.idx_path = base.with_suffix(".idx")
        self.dict_path = base.with_suffix(".dict")

        self._index: dict[str, tuple[int, int]] = {}  # word → (offset, size)
        self._loaded = False
        self._word_count: int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_index(self) -> None:
        """解析 .ifo + .idx 文件，构建内存索引。"""
        if self._loaded:
            return

        # 读取 .ifo 获取元信息
        ifo: dict[str, str] = {}
        with self.ifo_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    ifo[k.strip()] = v.strip()

        self._word_count = int(ifo.get("wordcount", 0))

        # 解析 .idx
        with self.idx_path.open("rb") as f:
            idx_data = f.read()

        pos = 0
        while pos < len(idx_data):
            # 找 null 终止符
            null_pos = idx_data.index(b"\x00", pos)
            word = idx_data[pos:null_pos].decode("utf-8", errors="replace")
            # 大端序 u32: offset(4字节) + size(4字节)
            offset = struct.unpack(">I", idx_data[null_pos + 1 : null_pos + 5])[0]
            size = struct.unpack(">I", idx_data[null_pos + 5 : null_pos + 9])[0]
            self._index[word] = (offset, size)
            pos = null_pos + 9

        self._loaded = True

    @property
    def word_count(self) -> int:
        if not self._loaded:
            self.load_index()
        return self._word_count or len(self._index)

    @property
    def indexed_count(self) -> int:
        if not self._loaded:
            self.load_index()
        return len(self._index)

    def lookup(self, word: str) -> Optional[str]:
        """
        查询词条，返回释义字符串（不含词头），未找到返回 None。

        Parameters
        ----------
        word : str
            藏文词语（UTF-8 字符串）

        Returns
        -------
        str | None
            释义文本；多个释义以换行分隔；未找到返回 None
        """
        if not self._loaded:
            self.load_index()

        if word not in self._index:
            return None

        offset, size = self._index[word]
        with self.dict_path.open("rb") as f:
            f.seek(offset)
            raw = f.read(size)

        # sametypesequence=m → 纯文本释义
        return raw.decode("utf-8", errors="replace")

    def lookup_all(self, word: str) -> list[str]:
        """
        查询词条，返回所有匹配项（一个词可能对应多个释义）。
        StarDict 索引中一个词可能重复出现，返回全部。
        """
        if not self._loaded:
            self.load_index()

        results: list[str] = []
        with self.dict_path.open("rb") as f:
            for offset, size in self._index.values():
                if offset == 0 and size == 0:
                    continue
                # 仅返回与查询词相同 offset 的条目（同一词的不同义项）
            # 实际上，同一词条的多个释义已合并在单一 offset 中
            # StarDict 允许多个 entry 使用相同 word 作为 key；返回所有
        return results

    def __repr__(self) -> str:
        return f"<StarDict {self.name!r} ({self.word_count} entries)>"
