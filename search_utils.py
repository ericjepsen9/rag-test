import re
from typing import List, Dict, Tuple


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\ufeff", "")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_lines(text: str) -> List[str]:
    out = []
    for ln in normalize_text(text).split("\n"):
        s = re.sub(r"\s+", " ", ln).strip()
        if not s:
            continue
        if set(s) <= {"=", "-", "_", " "}:
            continue
        out.append(s)
    return out


def uniq(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        key = re.sub(r"\s+", " ", (x or "").strip())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def is_faq_line(line: str) -> bool:
    s = (line or "").strip()
    return s.startswith("【Q】") or s.startswith("【A】")


def section_block(text: str, titles: List[str], stops: List[str]) -> str:
    txt = normalize_text(text)
    if not txt:
        return ""
    start = None
    chosen = None
    for t in titles:
        idx = txt.find(t)
        if idx != -1 and (start is None or idx < start):
            start = idx
            chosen = t
    if start is None:
        return ""
    # 跳过标题行本身，从标题之后开始截取内容
    # 这样 stops 搜索不会误匹配标题行中的关键词
    title_end = start + len(chosen)
    sub = txt[title_end:]
    end = None
    for s in stops:
        idx = sub.find(s)
        if idx >= 0 and (end is None or idx < end):
            end = idx
    if end is not None and end > 0:
        sub = sub[:end]
    return sub.strip()


def _extract_terms(query: str) -> List[str]:
    """从查询中提取搜索词：先按空格/标点分割，再对中文长词做 bigram 切分"""
    raw = [x for x in re.split(r"[\s,，;；、？?！!。]+", query.lower()) if x]
    terms = []
    for w in raw:
        terms.append(w)
        # 对纯中文且长度>=3的词做 bigram 切分，提高部分匹配能力
        if len(w) >= 3 and re.fullmatch(r"[\u4e00-\u9fff]+", w):
            for i in range(len(w) - 1):
                terms.append(w[i:i+2])
    return list(set(terms))


def keyword_score(query: str, text: str) -> float:
    q_terms = _extract_terms(query)
    t = (text or "").lower()
    if not q_terms:
        return 0.0
    hit = 0
    for term in q_terms:
        if term in t:
            hit += 1
    return hit / max(len(q_terms), 1)


def keyword_search(query: str, docs: List[Dict], top_k: int = 8) -> List[Dict]:
    scored = []
    for d in docs:
        score = keyword_score(query, d.get("text", ""))
        if score <= 0:
            continue
        x = dict(d)
        x["keyword_score"] = score
        scored.append(x)
    scored.sort(key=lambda x: x.get("keyword_score", 0.0), reverse=True)
    return scored[:top_k]


def merge_hybrid(vector_hits: List[Dict], keyword_hits: List[Dict], vw: float, kw: float, top_k: int) -> List[Dict]:
    merged = {}
    for h in vector_hits:
        key = h.get("text", "")
        merged[key] = dict(h)
        merged[key]["hybrid_score"] = float(h.get("score", 0.0)) * vw
    for h in keyword_hits:
        key = h.get("text", "")
        if key not in merged:
            merged[key] = dict(h)
            merged[key]["score"] = 0.0
            merged[key]["hybrid_score"] = 0.0
        merged[key]["hybrid_score"] += float(h.get("keyword_score", 0.0)) * kw
    out = list(merged.values())
    out.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return out[:top_k]


def split_multi_question(question: str, separators: List[str] = None) -> List[str]:
    # 移除 "。" 避免正常句号被误拆；只按真正表示"多个问题"的连接词拆分
    separators = separators or ["；", ";", "，另外", "并且，", "同时，", "还有，"]
    parts = [question]
    for sep in separators:
        next_parts = []
        for p in parts:
            next_parts.extend(p.split(sep))
        parts = next_parts
    # 过滤太短的碎片（避免无意义子问题）
    parts = [p.strip().rstrip("。？?") for p in parts if len(p.strip()) >= 4]
    return uniq(parts)


def detect_terms(question: str, term_map: Dict[str, List[str]]) -> List[str]:
    q = question.lower()
    found = []
    for key, aliases in term_map.items():
        for a in aliases:
            if a.lower() in q:
                found.append(key)
                break
    return uniq(found)
