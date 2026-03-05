import math
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
    seen = set()
    for w in raw:
        if w not in seen:
            terms.append(w)
            seen.add(w)
        # 对纯中文且长度>=3的词做 bigram 切分，提高部分匹配能力
        if len(w) >= 3 and re.fullmatch(r"[\u4e00-\u9fff]+", w):
            for i in range(len(w) - 1):
                bg = w[i:i+2]
                if bg not in seen:
                    terms.append(bg)
                    seen.add(bg)
    return terms


def _count_term(term: str, text: str) -> int:
    """统计 term 在 text 中出现的次数"""
    count = 0
    start = 0
    while True:
        idx = text.find(term, start)
        if idx == -1:
            break
        count += 1
        start = idx + len(term)
    return count


def bm25_score(query: str, text: str, avg_dl: float, n_docs: int,
               doc_freqs: Dict[str, int], k1: float = 1.5, b: float = 0.75) -> float:
    """BM25 评分：考虑词频、文档长度、逆文档频率"""
    q_terms = _extract_terms(query)
    t = (text or "").lower()
    dl = len(t)
    if not q_terms or dl == 0:
        return 0.0

    score = 0.0
    for term in q_terms:
        tf = _count_term(term, t)
        if tf == 0:
            continue
        df = doc_freqs.get(term, 0)
        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        # TF normalization
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
        score += idf * tf_norm
    return score


_bm25_cache: Dict[int, Tuple[List[str], int, float]] = {}  # id(docs) -> (texts, n_docs, avg_dl)


def _get_bm25_corpus(docs: List[Dict]) -> Tuple[List[str], int, float]:
    """缓存 docs 的小写文本和平均长度，避免每次请求重复计算"""
    key = id(docs)
    cached = _bm25_cache.get(key)
    if cached and cached[1] == len(docs):
        return cached
    texts = [(d.get("text") or "").lower() for d in docs]
    n_docs = len(texts)
    avg_dl = sum(len(t) for t in texts) / max(n_docs, 1)
    result = (texts, n_docs, avg_dl)
    _bm25_cache[key] = result
    return result


def keyword_search(query: str, docs: List[Dict], top_k: int = 8) -> List[Dict]:
    if not docs:
        return []

    q_terms = _extract_terms(query)
    if not q_terms:
        return []

    texts, n_docs, avg_dl = _get_bm25_corpus(docs)

    doc_freqs: Dict[str, int] = {}
    for term in q_terms:
        df = sum(1 for t in texts if term in t)
        doc_freqs[term] = df

    scored = []
    for i, d in enumerate(docs):
        s = bm25_score(query, texts[i], avg_dl, n_docs, doc_freqs)
        if s <= 0:
            continue
        x = dict(d)
        x["keyword_score"] = s
        scored.append(x)

    scored.sort(key=lambda x: x.get("keyword_score", 0.0), reverse=True)

    # 归一化到 [0, 1] 区间，使其与向量分数可比
    if scored:
        max_score = scored[0]["keyword_score"]
        if max_score > 0:
            for x in scored:
                x["keyword_score"] = x["keyword_score"] / max_score

    return scored[:top_k]


def _hit_key(h: Dict) -> str:
    """生成检索结果的唯一键（优先用 source_file+chunk_id，兜底用 text）"""
    meta = h.get("meta", {})
    src = meta.get("source_file", "")
    cid = meta.get("chunk_id", "")
    if src and cid:
        return f"{src}#{cid}"
    return h.get("text", "")


def merge_hybrid(vector_hits: List[Dict], keyword_hits: List[Dict], vw: float, kw: float, top_k: int) -> List[Dict]:
    merged = {}
    for h in vector_hits:
        key = _hit_key(h)
        merged[key] = dict(h)
        merged[key]["hybrid_score"] = float(h.get("score", 0.0)) * vw
    for h in keyword_hits:
        key = _hit_key(h)
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
