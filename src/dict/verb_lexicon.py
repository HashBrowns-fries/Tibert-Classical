"""
动词词干词典解析器
==================
解析 lexicon-of-tibetan-verb-stems-master/dictionary.xml（2,489 词条），
返回动词的 Present / Past / Future / Imperative 形态及英文释义。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


class VerbEntry:
    """单个动词词条的解析结果。"""

    def __init__(
        self,
        head: str,          # 字母分组 (ka, kha, ...)
        headword: str,      # 动词词干（罗马字 Wylie）
        present: Optional[str] = None,
        past: Optional[str] = None,
        future: Optional[str] = None,
        imperative: Optional[str] = None,
        meaning: Optional[str] = None,
    ) -> None:
        self.head = head
        self.headword = headword
        self.present = present
        self.past = past
        self.future = future
        self.imperative = imperative
        self.meaning = meaning

    def to_dict(self) -> dict:
        return {
            "head": self.head,
            "headword": self.headword,
            "present": self.present,
            "past": self.past,
            "future": self.future,
            "imperative": self.imperative,
            "meaning": self.meaning,
        }

    def __repr__(self) -> str:
        return (
            f"VerbEntry({self.headword!r}, "
            f"pres={self.present!r}, past={self.past!r}, "
            f"fut={self.future!r}, imp={self.imperative!r})"
        )


def _extract_text(element: Optional[ET.Element]) -> Optional[str]:
    """从 XML 元素中提取所有文本节点，拼接为字符串。"""
    if element is None:
        return None
    parts: list[str] = []
    for child in element:
        if child.text:
            parts.append(child.text.strip())
        # 递归收集嵌套 orth 标签内容
        for sub in child:
            if sub.text:
                parts.append(sub.text.strip())
    result = " ".join(parts).strip()
    return result if result else None


class VerbLexicon:
    """
    动词词干词典加载器。

    Parameters
    ----------
    xml_path : str | Path
        dictionary.xml 文件路径
    """

    def __init__(self, xml_path: str | Path) -> None:
        self.path = Path(xml_path)
        self._entries: list[VerbEntry] = []
        self._loaded = False
        self._index: dict[str, list[int]] = {}  # headword → list of entry indices

    def load(self) -> None:
        """解析 XML 文件，构建内存索引。"""
        if self._loaded:
            return

        tree = ET.parse(str(self.path))
        root = tree.getroot()

        for div in root:
            head_el = div.find("head")
            head = ""
            if head_el is not None:
                orth_el = head_el.find("orth")
                if orth_el is not None and orth_el.text:
                    head = orth_el.text.strip()

            for entry in div:
                hw_el = entry.find("headword/orth")
                if hw_el is None or not hw_el.text:
                    continue
                headword = hw_el.text.strip()

                verb_entry = VerbEntry(
                    head=head,
                    headword=headword,
                    present=_extract_text(entry.find("wrI")),
                    past=_extract_text(entry.find("wrII")),
                    future=_extract_text(entry.find("wrIII")),
                    imperative=_extract_text(entry.find("wrIV")),
                    meaning=_extract_text(entry.find("trans")),
                )
                idx = len(self._entries)
                self._entries.append(verb_entry)
                self._index.setdefault(headword, []).append(idx)

        self._loaded = True

    @property
    def entry_count(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._entries)

    def lookup(self, headword: str) -> list[VerbEntry]:
        """
        查询动词词干，返回所有匹配的 VerbEntry 列表。

        Parameters
        ----------
        headword : str
            动词词干（Wylie 罗马字，如 'kum', 'ker', 'klog'）

        Returns
        -------
        list[VerbEntry]
            匹配条目（同一词干可能有多条，如 'klub' 有两个不同释义）
        """
        if not self._loaded:
            self.load()

        indices = self._index.get(headword, [])
        return [self._entries[i] for i in indices]

    def all_entries(self) -> list[VerbEntry]:
        """返回所有动词条目。"""
        if not self._loaded:
            self.load()
        return list(self._entries)

    def __repr__(self) -> str:
        return f"<VerbLexicon {self.path.name!r} ({self.entry_count} entries)>"
