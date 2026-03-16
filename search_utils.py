import hashlib
import math
import re
import unicodedata
from functools import lru_cache
from typing import Any, List, Dict, Tuple

from rag_runtime_config import BM25_K1, BM25_B, SIGMOID_SCALE, CACHE_MAX_PRODUCTS, ROUTE_BOOST, JIEBA_ENABLED

# jieba 分词延迟加载
_jieba = None
_jieba_initialized = False


def _get_jieba():
    """延迟加载 jieba 并添加医美领域自定义词典"""
    global _jieba, _jieba_initialized
    if _jieba_initialized:
        return _jieba
    if not JIEBA_ENABLED:
        _jieba_initialized = True
        return None
    try:
        import jieba as _jieba_mod
        # 添加医美领域术语，防止被错误切分
        _CUSTOM_WORDS = [
            "菲罗奥", "赛洛菲", "聚己内酯", "透明质酸", "玻尿酸", "谷胱甘肽",
            "胶原蛋白", "法令纹", "苹果肌", "下颌线", "鱼尾纹", "泪沟",
            "光子嫩肤", "热玛吉", "热拉提", "超声刀", "皮秒", "德玛莎",
            "水光针", "微针", "中胚层", "淤青", "瘀青",
            "术后护理", "不良反应", "禁忌人群", "防伪鉴别",
            "HiddenTag", "PCL", "MTS", "IPL",
        ]
        for w in _CUSTOM_WORDS:
            _jieba_mod.add_word(w)
        _jieba = _jieba_mod
    except ImportError:
        print("[WARN] jieba 未安装，回退到 bigram 分词")
        _jieba = None
    _jieba_initialized = True
    return _jieba

# 预编译常用正则（避免每次函数调用时隐式编译）
_RE_WHITESPACE = re.compile(r"\s+")
_RE_TERM_SPLIT = re.compile(r"[\s,，;；、？?！!。]+")
_RE_CJK_WORD = re.compile(r"^[\u4e00-\u9fff]+$")
_RE_TIME_PATTERN = re.compile(
    r"术后[第]?\d+[-~到]?\d*[天日周月]|术后\d+小时|术后\d+个月|术后当天"
)
# 分隔线字符集：用于快速判断一行是否为纯分隔线（避免每次 set(ln) 构造临时集合）
_SEPARATOR_CHARS = frozenset("=-_ ")


def _sigmoid_norm(raw_score: float) -> float:
    """BM25 分数归一化：sigmoid(score/scale)，钳位防 exp 溢出。
    将 raw_score 量化到 2 位小数以提高 LRU 缓存命中率。"""
    z = round(raw_score / SIGMOID_SCALE, 2)
    return _sigmoid_cached(max(-20.0, min(20.0, z)))


@lru_cache(maxsize=512)
def _sigmoid_cached(z: float) -> float:
    """LRU 缓存的 sigmoid：典型 BM25 分数分布在有限区间，命中率高。"""
    return 1.0 / (1.0 + math.exp(-z))

# 中文医美术语同义词映射表：将口语/变体统一为规范术语，提升 BM25 召回率
# key: 变体形式, value: 规范形式
_SYNONYM_MAP = {
    # === 注射动作 ===
    "打": "注射", "注": "注射", "施术": "注射", "扎": "注射",
    # === 产品动作 ===
    "做": "操作", "做了": "操作",
    # === 疼痛 ===
    "痛": "疼痛", "疼": "疼痛", "刺痛": "疼痛", "酸痛": "疼痛", "胀痛": "疼痛",
    # === 肿胀 ===
    "肿": "肿胀", "肿了": "肿胀", "浮肿": "肿胀", "水肿": "肿胀",
    # === 淤青 ===
    "青": "淤青", "瘀青": "淤青", "发青": "淤青", "乌青": "淤青", "发紫": "淤青",
    # === 恢复 ===
    "好了": "恢复", "消了": "消退", "退了": "消退",
    # === 医生 ===
    "大夫": "医生", "主治": "医生", "主任": "医生",
    # === 玻尿酸 / 透明质酸 ===
    "玻尿酸": "透明质酸", "HA": "透明质酸",
    # === 禁忌 ===
    "不适合": "禁忌", "不能做": "禁忌", "不能打": "禁忌",
    # === 效果 ===
    "见效": "效果", "有用": "效果", "管用": "效果", "有效": "效果",
    # === 反应 ===
    "术后反应": "不良反应", "副作用": "不良反应",
    # === 护理 ===
    "保养": "护理", "养护": "护理",
    # === 脸部 ===
    "脸": "面部", "面": "面部",
    # === 皱纹 ===
    "纹": "皱纹", "细纹": "皱纹", "法令纹": "皱纹",
    # === 紧致 ===
    "拉紧": "紧致", "提拉": "紧致", "提升": "紧致",
    # === 胶原蛋白 ===
    "胶原蛋白": "胶原",

    # === 医美项目俗称 → 规范名 ===
    "瘦脸针": "肉毒素注射", "瘦咬肌": "肉毒素注射", "botox": "肉毒素注射",
    "美白针": "谷胱甘肽注射", "美白点滴": "谷胱甘肽注射",
    "超皮秒": "皮秒激光", "蜂巢皮秒": "皮秒激光",
    "热玛吉": "射频紧肤", "热拉提": "射频紧肤", "RF": "射频",
    "超声刀": "聚焦超声", "超声提升": "聚焦超声",
    "光子": "IPL光子嫩肤", "强脉冲光": "IPL光子嫩肤",
    "点阵": "点阵激光", "飞梭": "点阵激光",
    "溶脂针": "注射溶脂", "消脂针": "注射溶脂",
    "线雕": "埋线提升", "蛋白线": "埋线提升", "提线": "埋线提升",
    "自体脂肪": "脂肪填充", "脂肪移植": "脂肪填充",

    # === 皮肤问题口语 → 规范术语 ===
    "斑": "色斑", "晒斑": "色斑", "黄褐斑": "色斑", "雀斑": "色斑",
    "痘印": "痘坑", "痘疤": "痘坑", "痤疮疤痕": "痘坑",
    "毛孔大": "毛孔粗大", "毛孔很大": "毛孔粗大",
    "皮肤松": "皮肤松弛", "脸松": "皮肤松弛", "松了": "皮肤松弛", "下垂": "松弛",
    "暗沉": "肤色不均", "暗黄": "肤色不均", "发黄": "肤色不均",
    "干": "干燥", "缺水": "干燥", "起皮": "干燥", "脱皮": "干燥",
    "黑眼圈": "眼周色素", "眼袋": "眼周",
    "红血丝": "毛细血管扩张", "泛红": "毛细血管扩张",
    "鸡皮": "毛周角化", "鸡皮肤": "毛周角化",

    # === 部位口语 → 规范名 ===
    "嘴角纹": "木偶纹", "苦笑纹": "木偶纹",
    "抬头纹": "额纹", "鱼尾纹": "眼周皱纹", "川字纹": "眉间纹",
    "双下巴": "下颌脂肪", "大饼脸": "面部轮廓",
    "太阳穴": "颞部", "颧骨": "颧部",

    # === 术后/护理口语 ===
    "忌口": "饮食禁忌", "能吃辣吗": "辛辣", "吃海鲜": "海鲜",
    "能喝酒吗": "禁酒", "涂防晒": "防晒", "能化妆吗": "化妆",
    "能洗脸吗": "洗脸", "能运动吗": "运动",
    "多久恢复": "恢复期", "恢复要多久": "恢复期",
}

# 反向映射：同义词扩展（检索时同时搜索同义词）
_SYNONYM_EXPAND = {}
for _k, _v in _SYNONYM_MAP.items():
    _SYNONYM_EXPAND.setdefault(_v, set()).add(_k)
    _SYNONYM_EXPAND.setdefault(_k, set()).add(_v)


def expand_synonyms(query: str) -> str:
    """在查询中追加同义词，提升 BM25 召回率。
    例如 "打菲罗奥疼吗" → "打菲罗奥疼吗 注射 疼痛"
    支持时间模式扩展：术后第N天 → 追加恢复/消退关键词
    """
    extra = set()
    q_lower = query.lower()
    for term, synonyms in _SYNONYM_EXPAND.items():
        if term in q_lower:
            for syn in synonyms:
                if syn not in q_lower:
                    extra.add(syn)
    # 时间模式扩展：术后第X天/周/月/小时/范围 → 补充恢复相关词
    if _RE_TIME_PATTERN.search(q_lower):
        for w in ("恢复", "消退", "正常"):
            if w not in q_lower:
                extra.add(w)
    if extra:
        expanded = query + " " + " ".join(extra)
        return expanded[:2000]  # 防止同义词膨胀过长
    return query


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # NFC 标准化：统一 CJK 字符和全角/半角变体，避免同义词匹配遗漏
    text = unicodedata.normalize("NFC", text)
    return text


def normalize_lines(text: str) -> List[str]:
    out = []
    for ln in normalize_text(text).split("\n"):
        s = " ".join(ln.split())
        if not s:
            continue
        if not (set(s) - _SEPARATOR_CHARS):
            continue
        out.append(s)
    return out


def uniq(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        key = " ".join((x or "").split())
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


def _extract_terms_bigram(query: str) -> List[str]:
    """原始 bigram 分词：按空格/标点分割，再对中文长词做 bigram 切分（jieba 不可用时的回退方案）"""
    raw = [x for x in _RE_TERM_SPLIT.split(query.lower()) if x]
    terms = []
    seen = set()
    for w in raw:
        if w not in seen:
            terms.append(w)
            seen.add(w)
        # 对纯中文且长度>=3的词做 bigram 切分，提高部分匹配能力
        if len(w) >= 3 and _RE_CJK_WORD.match(w):
            for i in range(len(w) - 1):
                bg = w[i:i+2]
                if bg not in seen:
                    terms.append(bg)
                    seen.add(bg)
    return terms


def _extract_terms_jieba(query: str) -> List[str]:
    """jieba 分词 + bigram 补充：先用 jieba 精确分词，再对长中文词补充 bigram 提高部分匹配能力"""
    jieba = _get_jieba()
    q_lower = query.lower()
    # jieba 切词（搜索引擎模式：更细粒度，召回更高）
    raw_words = list(jieba.cut_for_search(q_lower))
    terms = []
    seen = set()
    for w in raw_words:
        w = w.strip()
        if not w or w in seen:
            continue
        # 过滤纯标点和空白
        if _RE_TERM_SPLIT.match(w):
            continue
        terms.append(w)
        seen.add(w)
        # 对 jieba 切出的长中文词仍做 bigram 补充，增加部分匹配能力
        if len(w) >= 4 and _RE_CJK_WORD.match(w):
            for i in range(len(w) - 1):
                bg = w[i:i+2]
                if bg not in seen:
                    terms.append(bg)
                    seen.add(bg)
    return terms


def _extract_terms(query: str) -> List[str]:
    """从查询中提取搜索词：优先 jieba 分词，不可用时回退 bigram"""
    if _get_jieba() is not None:
        return _extract_terms_jieba(query)
    return _extract_terms_bigram(query)


def _count_term(term: str, text: str) -> int:
    """统计 term 在 text 中出现的次数（非重叠匹配）"""
    if not term:
        return 0
    return text.count(term)


def bm25_score(query_or_terms, text: str, avg_dl: float, n_docs: int,
               doc_freqs: Dict[str, int], k1: float = BM25_K1, b: float = BM25_B) -> float:
    """BM25 评分：考虑词频、文档长度、逆文档频率。
    query_or_terms: 预提取的查询词列表，或查询字符串（自动提取）。"""
    if isinstance(query_or_terms, str):
        query_or_terms = _extract_terms(query_or_terms)
    dl = len(text)
    if not query_or_terms or dl == 0:
        return 0.0

    score = 0.0
    effective_n = max(n_docs, 100)
    for term in query_or_terms:
        tf = text.count(term)
        if tf == 0:
            continue
        df = doc_freqs.get(term, 0)
        # IDF: 对小语料库平滑，防止稀有词分数膨胀
        idf = math.log((effective_n - df + 0.5) / (df + 0.5) + 1.0)
        # TF normalization
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
        score += idf * tf_norm
    return score


_CACHE_MAX_SIZE = CACHE_MAX_PRODUCTS
_bm25_cache: Dict[Any, Tuple[List[str], int, float]] = {}


def _cache_put(cache: dict, key: Any, value: Any, max_size: int = 0) -> None:
    """写入缓存，超过上限时清除最早的条目。已有 key 时直接覆盖不淘汰。"""
    limit = max_size if max_size > 0 else _CACHE_MAX_SIZE
    if key in cache:
        cache[key] = value
        return
    while len(cache) >= limit:
        try:
            oldest = next(iter(cache))
            cache.pop(oldest, None)
        except (StopIteration, RuntimeError):
            break
    cache[key] = value


def _corpus_cache_key(docs: List[Dict]) -> Tuple:
    """生成稳定的缓存键：使用确定性哈希（跨进程重启不变）"""
    n = len(docs)
    if n == 0:
        return (0,)
    first = (docs[0].get("text") or "")[:64]
    last = (docs[-1].get("text") or "")[:64]
    mid_idx = n // 2
    mid = (docs[mid_idx].get("text") or "")[:32] if n > 2 else ""
    digest = hashlib.md5(f"{first}|{mid}|{last}".encode()).hexdigest()[:12]
    return (n, digest)


def _get_bm25_corpus(docs: List[Dict]) -> Tuple[List[str], int, float, Tuple]:
    """缓存 docs 的小写文本和平均长度，避免每次请求重复计算。
    返回 (texts, n_docs, avg_dl, corpus_key)，调用方可复用 corpus_key。"""
    key = _corpus_cache_key(docs)
    cached = _bm25_cache.get(key)
    if cached:
        return cached[0], cached[1], cached[2], key
    texts = [(d.get("text") or "").lower() for d in docs]
    n_docs = len(texts)
    avg_dl = sum(len(t) for t in texts) / max(n_docs, 1)
    _cache_put(_bm25_cache, key, (texts, n_docs, avg_dl))
    return texts, n_docs, avg_dl, key


_df_cache: Dict[Any, Dict[str, int]] = {}  # corpus_key -> {term: df}


def _get_doc_freq(term: str, texts: List[str], corpus_key: Any) -> int:
    """获取 term 的文档频率，带缓存"""
    cached = _df_cache.get(corpus_key)
    if cached is None:
        cached = {}
        _cache_put(_df_cache, corpus_key, cached)
    if term not in cached:
        cached[term] = sum(1 for t in texts if term in t)
    return cached[term]


def _batch_doc_freqs(terms: List[str], texts: List[str], corpus_key: Any) -> Dict[str, int]:
    """批量计算多个 term 的文档频率，减少对 texts 的重复遍历。
    对缓存未命中的 term 只做一次 O(n) 扫描（n=文档数），而非每个 term 扫一次。"""
    cached = _df_cache.get(corpus_key)
    if cached is None:
        cached = {}
        _cache_put(_df_cache, corpus_key, cached)

    # 找出未缓存的 terms（用 set 加速后续内循环 membership 测试）
    uncached = [t for t in terms if t not in cached]
    if uncached:
        # 一次遍历文档列表，同时统计所有未缓存 term 的 df
        counts = {t: 0 for t in uncached}
        for doc_text in texts:
            for t in uncached:
                if t in doc_text:
                    counts[t] += 1
        cached.update(counts)
    # 直接从缓存取值，避免构建中间 dict（大多数 terms 已缓存）
    return {t: cached.get(t, 0) for t in terms}


# ============================================================
# 倒排索引：加速 BM25 检索，跳过不包含任何查询词的文档
# ============================================================

_inverted_index_cache: Dict[Any, Dict[str, List[int]]] = {}  # corpus_key -> {term: [doc_ids]}
_doc_len_cache: Dict[Any, List[int]] = {}  # corpus_key -> [doc_len_per_doc]


def _get_inverted_index(texts: List[str], corpus_key: Any) -> Dict[str, List[int]]:
    """构建/获取倒排索引：{term: [doc_indices]}，按 bigram 粒度索引。
    惰性构建，缓存复用。索引粒度为 2-gram 以匹配 _extract_terms 的输出。"""
    cached = _inverted_index_cache.get(corpus_key)
    if cached is not None:
        return cached
    inv: Dict[str, List[int]] = {}
    for i, text in enumerate(texts):
        # 对每个文档建立字符级 bigram 索引
        for j in range(len(text) - 1):
            bg = text[j:j+2]
            if bg not in inv:
                inv[bg] = [i]
            elif inv[bg][-1] != i:  # 去重：同一文档只记录一次
                inv[bg].append(i)
    _cache_put(_inverted_index_cache, corpus_key, inv)
    return inv


def _get_candidate_docs(terms: List[str], inv_index: Dict[str, List[int]],
                         n_docs: int) -> List[int]:
    """用倒排索引找出包含至少一个查询词的文档集合。
    返回排序后的文档索引列表。"""
    candidate_set: set = set()
    for term in terms:
        # 对每个查询词，查找包含它的文档
        # 如果 term 长度 >= 2，用其 bigram 做交集过滤
        if len(term) >= 2:
            # 取 term 的第一个 bigram 作为候选过滤
            bg = term[:2]
            doc_ids = inv_index.get(bg)
            if doc_ids is not None:
                candidate_set.update(doc_ids)
        elif len(term) == 1:
            # 单字符 term：遍历所有包含该字符的 bigram
            for bg_key, doc_ids in inv_index.items():
                if term in bg_key:
                    candidate_set.update(doc_ids)
    return sorted(candidate_set)


def keyword_search(query: str, docs: List[Dict], top_k: int = 8,
                    skip_synonym_expand: bool = False) -> List[Dict]:
    if not docs:
        return []

    # 同义词扩展：增加召回（当调用方已做扩展时跳过，避免双重扩展噪音）
    if not skip_synonym_expand:
        query = expand_synonyms(query)
    q_terms = _extract_terms(query)
    if not q_terms:
        return []

    texts, n_docs, avg_dl, corpus_key = _get_bm25_corpus(docs)

    doc_freqs = _batch_doc_freqs(q_terms, texts, corpus_key)

    # 倒排索引加速：只对包含查询词的文档计算 BM25，跳过无关文档
    inv_index = _get_inverted_index(texts, corpus_key)
    candidates = _get_candidate_docs(q_terms, inv_index, n_docs)

    scored = []
    for i in candidates:
        s = bm25_score(q_terms, texts[i], avg_dl, n_docs, doc_freqs)
        if s <= 0:
            continue
        scored.append({**docs[i], "keyword_score": s})

    scored.sort(key=lambda x: x.get("keyword_score", 0.0), reverse=True)

    # 归一化到 [0, 1] 区间：sigmoid 函数使不同 query 间的分数可比
    # sigmoid(x/scale) 中 scale=5 使典型 BM25 分数（0~15）映射到 (0.5, 0.95) 区间
    for x in scored:
        x["keyword_score"] = _sigmoid_norm(x["keyword_score"])

    return scored[:top_k]


def _hit_key(h: Dict) -> str:
    """生成检索结果的唯一键（优先用 source_file+chunk_id，兜底用 text）"""
    meta = h.get("meta", {})
    src = meta.get("source_file", "")
    cid = meta.get("chunk_id", "")
    if src and cid:
        return f"{src}#{cid}"
    return h.get("text", "")


def merge_hybrid(vector_hits: List[Dict], keyword_hits: List[Dict], vw: float, kw: float, top_k: int,
                  route: str = "") -> List[Dict]:
    merged = {}
    for h in vector_hits:
        key = _hit_key(h)
        # 安全钳位：FAISS 余弦相似度理论范围 [0,1]，防止超界
        vs = min(1.0, max(0.0, float(h.get("score", 0.0))))
        new_score = vs * vw
        if key in merged:
            # 同一文档重复出现时取最高分，而非累加
            if new_score > merged[key].get("hybrid_score", 0.0):
                merged[key] = {**h, "hybrid_score": new_score}
        else:
            merged[key] = {**h, "hybrid_score": new_score}
    for h in keyword_hits:
        key = _hit_key(h)
        if key not in merged:
            merged[key] = {**h, "score": 0.0, "hybrid_score": 0.0}
        kw_contribution = float(h.get("keyword_score", 0.0)) * kw
        # 同一文档多次出现在 keyword_hits 中（如来自不同 store）时取最高分，
        # 而非累加，防止分数膨胀
        current_kw = merged[key].get("_kw_contribution", 0.0)
        if kw_contribution > current_kw:
            merged[key]["hybrid_score"] += kw_contribution - current_kw
            merged[key]["_kw_contribution"] = kw_contribution

    # 路由感知加分：匹配目标章节标题的 chunk 得到小幅提升
    if route:
        _apply_route_boost(merged, route)

    out = list(merged.values())
    # 清理内部追踪字段
    for h in out:
        h.pop("_kw_contribution", None)
    out.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return out[:top_k]


# 路由→章节标题关键词：chunk 文本中包含这些关键词即视为匹配该路由的章节
_ROUTE_SECTION_MARKERS = {
    "basic":             ["产品基础信息", "产品名称", "备案信息", "规格"],
    "ingredient":        ["核心成分与作用", "聚己内酯", "PCL", "透明质酸"],
    "operation":         ["操作方法与注射指南", "注射参数", "推荐方式"],
    "aftercare":         ["术后护理与注意事项", "术后护理", "冰敷", "运动与出汗",
                          "化妆与护肤", "防晒与日晒", "洗浴与高温", "游泳"],
    "anti_fake":         ["防伪鉴别方法", "HiddenTag", "防伪步骤"],
    "risk":              ["风险与不良反应", "常见术后反应", "处理建议"],
    "contraindication":  ["禁忌人群", "免疫系统疾病", "妊娠期"],
    "combo":             ["联合方案与项目搭配", "可联合项目", "联合注意事项"],
    "effect":            ["效果与维持时间", "起效时间", "维持时间"],
    "pre_care":          ["术前准备", "术前检查", "术前注意事项"],
    "design":            ["方案设计与面部评估", "面部评估要点", "方案设计原则"],
    "repair":            ["修复与补救方案", "需要修复的情况", "修复原则"],
    # 跨实体路由
    "complication":      ["并发症", "决策树", "分级处理", "术后正常恢复", "警惕信号"],
    "course":            ["疗程规划", "疗程方案", "时间轴", "间隔", "总周期"],
    "anatomy_q":         ["面部分区", "额部", "苹果肌", "法令纹", "下颌线", "颈部"],
    "indication_q":      ["皮肤松弛", "干燥缺水", "毛孔粗大", "色斑", "痘坑", "皱纹", "年龄段"],
    "procedure_q":       ["项目概述", "操作流程", "适用人群", "适应症"],
    "equipment_q":       ["设备概述", "设备参数", "适配产品", "针头规格"],
    "script":            ["客户常见顾虑", "话术", "怎么解释", "合规要点", "沟通技巧"],
}
_ROUTE_BOOST = ROUTE_BOOST


def _apply_route_boost(merged: Dict[str, Dict], route: str) -> None:
    markers = _ROUTE_SECTION_MARKERS.get(route, [])
    if not markers:
        return
    for h in merged.values():
        text = (h.get("text") or "")[:800]  # 查看 chunk 前800字符，覆盖更多章节标记
        if any(m in text for m in markers):
            h["hybrid_score"] += _ROUTE_BOOST


# split_multi_question 预编译正则（避免每次调用隐式编译）
_RE_LIST_SPLIT = re.compile(r"^(.+?)(分别|各自)(是什么|有哪些|怎么样|怎么办)$")
_RE_PAIR_SPLIT = re.compile(r"^(.+?)和(.+?)(分别|各自)?(是什么|有哪些|怎么样)$")
_RE_ENUM_TAIL = re.compile(r"^(.+?)(怎么选|怎么样|是什么|有什么区别|哪个好)$")
_RE_ENUM_ITEMS = re.compile(r"[、和]")
_RE_COMMA_SPLIT = re.compile(r"[，,]")


def split_multi_question(question: str, separators: List[str] = None) -> List[str]:
    # 选择式问题（"A还是B"）：如果没有问号分隔的多个子问题，不拆分
    # 但如果有多个子问题（如"水光还是微针？术后怎么护理？"），仍需按问号拆分
    if "还是" in question and "？" not in question and "?" not in question:
        return [question.strip().rstrip("。？?")]

    separators = separators or [
        "？", "?",        # 问号分隔多个问题
        "；", ";",        # 分号
        "，另外", "，还有", "，同时",  # 连接词
        "并且，", "同时，", "还有，",
    ]
    parts = [question]
    for sep in separators:
        next_parts = []
        for p in parts:
            next_parts.extend(p.split(sep))
        parts = next_parts

    # "A和B分别是什么" → ["A是什么", "B是什么"]
    # "A、B和C分别是什么" → ["A是什么", "B是什么", "C是什么"]
    expanded = []
    for p in parts:
        ps = p.strip()
        # 先尝试带顿号的多项列举："A、B、C分别是什么"
        m_list = _RE_LIST_SPLIT.match(ps)
        if m_list:
            items_str = m_list.group(1)
            suffix = m_list.group(3)
            # 用顿号和"和"分割列举项
            items = _RE_ENUM_ITEMS.split(items_str)
            if len(items) >= 2:
                for item in items:
                    item = item.strip()
                    if item:
                        expanded.append(item + suffix)
                continue
        # 简单的 "A和B是什么" 模式
        m = _RE_PAIR_SPLIT.match(ps)
        if m:
            suffix = m.group(4)
            expanded.append(m.group(1).strip() + suffix)
            expanded.append(m.group(2).strip() + suffix)
        else:
            expanded.append(p)

    # 顿号分隔列举：当包含顿号且有共同的疑问尾时拆分
    # "水光、微针、光电怎么选" → ["水光怎么选", "微针怎么选", "光电怎么选"]
    enum_expanded = []
    for p in expanded:
        if "、" in p:
            m_enum = _RE_ENUM_TAIL.match(p.strip())
            if m_enum:
                items_str = m_enum.group(1)
                suffix = m_enum.group(2)
                items = [x.strip() for x in items_str.split("、") if x.strip()]
                if len(items) >= 2:
                    enum_expanded.extend(item + suffix for item in items)
                    continue
        enum_expanded.append(p)

    # 逗号分隔：仅当两侧都 ≥6 字符时才拆分（避免 "术后1天，可以洗脸" 被误拆）
    final = []
    for p in enum_expanded:
        comma_parts = _RE_COMMA_SPLIT.split(p)
        if len(comma_parts) >= 2 and all(len(cp.strip()) >= 6 for cp in comma_parts):
            final.extend(comma_parts)
        else:
            final.append(p)

    result = [s.rstrip("。？?") for p in final if len(s := p.strip()) >= 2]
    return uniq(result)


_detect_terms_cache: Dict[int, Dict[str, List[str]]] = {}  # id(term_map) -> lowered map


def detect_terms(question: str, term_map: Dict[str, List[str]]) -> List[str]:
    """检测问题中提到的实体（产品/项目等），返回匹配的 key 列表。
    每个 key 最多匹配一次（内部 break），结果天然唯一，无需 uniq。"""
    # 缓存小写别名映射（term_map 通常是模块级常量，id 稳定）
    map_id = id(term_map)
    lowered = _detect_terms_cache.get(map_id)
    if lowered is None:
        lowered = {k: [a.lower() for a in aliases] for k, aliases in term_map.items()}
        _detect_terms_cache[map_id] = lowered
    q = question.lower()
    found = []
    for key, aliases in lowered.items():
        for a in aliases:
            if a in q:
                found.append(key)
                break
    return found


# ============================================================
# Reranker：使用 BGE-M3 compute_score 对候选文档重排序
# ============================================================

# Rerank 结果缓存：避免相同 query + 相同候选集重复调用 compute_score
_rerank_cache: Dict[str, List[Tuple[str, float]]] = {}  # cache_key -> [(hit_key, score), ...]
_RERANK_CACHE_MAX = 512


def _rerank_cache_key(query: str, hits: List[Dict]) -> str:
    """生成 rerank 缓存键：query + 候选文档标识的组合哈希"""
    hit_keys = "|".join(_hit_key(h) for h in hits)
    return hashlib.md5(f"{query}||{hit_keys}".encode()).hexdigest()


def rerank_hits(query: str, hits: List[Dict], model, top_k: int) -> List[Dict]:
    """使用 BGE-M3 的 compute_score 对混合检索结果重排序。

    model: 已加载的 BGEM3FlagModel 实例（复用嵌入模型，无额外模型开销）。
    返回按 rerank_score 降序排列的 top_k 结果。
    带缓存：相同 query + 相同候选集直接返回缓存分数，跳过 compute_score。
    """
    if not hits or not model:
        return hits[:top_k]

    # 缓存查找
    cache_key = _rerank_cache_key(query, hits)
    cached = _rerank_cache.get(cache_key)
    if cached is not None:
        # 用缓存的分数重建排序
        score_map = dict(cached)
        for h in hits:
            hk = _hit_key(h)
            if hk in score_map:
                h["rerank_score"] = score_map[hk]
                h["hybrid_score_before_rerank"] = h.get("hybrid_score", 0.0)
                h["hybrid_score"] = score_map[hk]
        hits.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        return hits[:top_k]

    sentence_pairs = [[query, (h.get("text") or "")[:512]] for h in hits]
    try:
        scores = model.compute_score(sentence_pairs)
    except Exception as e:
        print(f"[WARN] rerank 失败，回退原排序: {e}")
        return hits[:top_k]

    # compute_score 返回 {"colbert": [...], "sparse": [...], "dense": [...], "score": [...]}
    # 或 {"colbert+sparse+dense": [...]} 取决于 FlagEmbedding 版本
    if isinstance(scores, dict):
        # 优先取融合分数
        final_scores = (
            scores.get("colbert+sparse+dense")
            or scores.get("score")
            or scores.get("dense")
        )
        if final_scores is None:
            # 手动加权
            col = scores.get("colbert", [0.0] * len(hits))
            spa = scores.get("sparse", [0.0] * len(hits))
            den = scores.get("dense", [0.0] * len(hits))
            final_scores = [0.4 * c + 0.2 * s + 0.4 * d
                           for c, s, d in zip(col, spa, den)]
    elif isinstance(scores, (list, tuple)):
        final_scores = scores
    else:
        return hits[:top_k]

    # 归一化 rerank 分数到 [0, 1]
    if final_scores:
        max_s = max(final_scores) if final_scores else 1.0
        min_s = min(final_scores) if final_scores else 0.0
        rng = max_s - min_s if max_s > min_s else 1.0
        cache_entries = []
        for i, h in enumerate(hits):
            raw = final_scores[i] if i < len(final_scores) else 0.0
            normalized = (raw - min_s) / rng
            h["rerank_score"] = normalized
            h["hybrid_score_before_rerank"] = h.get("hybrid_score", 0.0)
            h["hybrid_score"] = normalized
            cache_entries.append((_hit_key(h), normalized))
        # 写入缓存
        _cache_put(_rerank_cache, cache_key, cache_entries, max_size=_RERANK_CACHE_MAX)

    hits.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return hits[:top_k]


# ============================================================
# 动态阈值：根据检索分数分布自适应调整过滤阈值
# ============================================================

def compute_dynamic_threshold(hits: List[Dict], route_threshold: float,
                               ratio: float = 0.40, floor_ratio: float = 0.70) -> float:
    """根据 top-1 分数动态计算阈值：
    - 如果 top-1 分数很高，阈值适当升高，过滤低质量结果
    - 如果 top-1 分数偏低，阈值适当降低，保留更多候选
    返回 max(route_threshold * floor_ratio, top1_score * ratio)
    """
    if not hits:
        return route_threshold
    top1 = max(h.get("hybrid_score", 0.0) for h in hits)
    dynamic = top1 * ratio
    floor = route_threshold * floor_ratio
    return max(floor, min(dynamic, route_threshold))
