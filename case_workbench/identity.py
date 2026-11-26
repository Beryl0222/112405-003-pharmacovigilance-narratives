"""匿名病例身份归并与来源独立性工具。"""

from __future__ import annotations

import re
from difflib import SequenceMatcher


_PUNCT_RE = re.compile(r"[\s，。；;：:、“”\"'‘’（）()\[\]【】!?！？·\-_—…]+")


def normalize_text(text: str) -> str:
    """去除标点和空白，用于跨渠道重复文本比较。"""

    return _PUNCT_RE.sub("", text or "").lower()


def text_similarity(left: str, right: str) -> float:
    a, b = normalize_text(left), normalize_text(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def char_bigrams(text: str) -> set[str]:
    normalized = normalize_text(text)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[i : i + 2] for i in range(len(normalized) - 1)}


def bigram_jaccard(left: str, right: str) -> float:
    a, b = char_bigrams(left), char_bigrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def duplicate_confidence(left: str, right: str) -> float:
    """综合字符序列和二元组相似度，避免短文本偶然重合。"""

    seq = text_similarity(left, right)
    jac = bigram_jaccard(left, right)
    return round(0.65 * seq + 0.35 * jac, 3)


def union_find_merge(groups: dict[str, set[str]], a: str, b: str) -> None:
    """把 a/b 所在集合合并到 groups（就地更新所有键）。"""

    group_a = next((value for value in groups.values() if a in value), {a})
    group_b = next((value for value in groups.values() if b in value), {b})
    merged = set(group_a) | set(group_b)
    for key in list(groups):
        if groups[key] is group_a or groups[key] is group_b or groups[key] & merged:
            groups[key] = merged
    groups.setdefault(a, merged)
    groups.setdefault(b, merged)
