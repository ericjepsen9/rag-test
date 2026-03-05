import re
from typing import Dict, Any, List, Optional
from rag_runtime_config import (
    PRODUCT_ALIASES, PROJECT_ALIASES, TIME_TERMS, SYMPTOM_TERMS,
    QUESTION_ROUTES,
)
from search_utils import detect_terms, uniq, split_multi_question

# 路由专属检索扩展词：当检测到某路由时，补充高区分度关键词帮助 BM25 命中正确 chunk
_ROUTE_EXPANSION = {
    "ingredient":       ["PCL", "聚己内酯", "透明质酸", "生长因子"],
    "anti_fake":        ["HiddenTag", "防伪", "正品认证"],
    "combo":            ["联合", "间隔", "搭配"],
    "risk":             ["红肿", "不良反应", "就医"],
    "contraindication": ["禁忌", "妊娠", "哺乳", "禁忌人群"],
}

# 指代词模式：命中时用产品名**替换**指代词
_PRONOUN_PATTERNS = re.compile(
    r"^(它|这个|那个|该产品|这款|那款)"
    r"|^(那|那么).{0,2}(呢|吗|怎么样)"
)

# 追问/延续模式：命中时在问题前**补充**产品名
_FOLLOWUP_PATTERNS = re.compile(
    r"^还有(别的|其他|吗)"
    r"|(呢|吗|怎么样|有哪些|是什么|是多少|怎么办)$"
)

# 明确的非领域词：命中这些词的问题大概率与产品无关，不应补全
_OFFTOPIC_PATTERNS = re.compile(
    r"(天气|新闻|股票|电影|音乐|美食|旅游|游戏|足球|篮球|奥运)"
)

# 隐含关联模式：问题本身是关于某个主题但缺少产品主语
# 例如 "安全吗" "效果怎样" "多少钱" "要恢复多久"
_IMPLICIT_TOPIC_PATTERNS = re.compile(
    r"(安全|效果|价格|多少钱|持续|维持|恢复|疗程|几次|多久|保质期|保存"
    r"|区别|对比|优势|好处|原理|机制|作用|功效"
    r"|痛|疼|会不会|能不能|可不可以|需要|注意)"
)

# 所有路由关键词汇集，用于判断问题是否包含领域词
_ALL_ROUTE_KEYWORDS = set()
for _kws in QUESTION_ROUTES.values():
    _ALL_ROUTE_KEYWORDS.update(kw.lower() for kw in _kws)


def _extract_history_context(history: List[Dict]) -> Dict[str, Any]:
    """从对话历史中提取结构化上下文信息。

    返回:
        product: 最近提到的产品标准中文名
        product_id: 产品 ID
        route: 最近的路由主题
        last_user_q: 最近的用户问题原文
    """
    ctx: Dict[str, Any] = {
        "product": "", "product_id": "", "route": "", "last_user_q": ""
    }
    for item in reversed(history):
        content = item.get("content", "")
        if item.get("role") != "user":
            continue

        if not ctx["last_user_q"]:
            ctx["last_user_q"] = content

        if not ctx["product"]:
            products = detect_terms(content, PRODUCT_ALIASES)
            if products:
                pid = products[0]
                aliases = PRODUCT_ALIASES.get(pid, [])
                if aliases:
                    ctx["product"] = aliases[0]
                    ctx["product_id"] = pid

        if not ctx["route"]:
            content_lower = content.lower()
            for route, keywords in QUESTION_ROUTES.items():
                if any(kw.lower() in content_lower for kw in keywords):
                    ctx["route"] = route
                    break

        if ctx["product"] and ctx["route"]:
            break
    return ctx


def _resolve_context(question: str, history_ctx: Dict[str, Any]) -> str:
    """基于已提取的历史上下文解析指代和省略，补全当前问题。

    参数 history_ctx 来自 _extract_history_context()，避免重复解析。

    策略（从高优先级到低）：
    1. 当前问题已包含产品名 → 不需要补全
    2. 指代词（它/这个/那个）→ 从历史替换为具体产品名
    3. 追问句式（"…呢/吗/怎么样"）→ 补充产品名
    4. 隐含关联（含领域词但无主语，如"安全吗""效果怎样"）→ 补充产品名
    5. 纯领域词问题（含路由关键词但无产品名）→ 补充产品名
    """
    q = question.strip()

    history_product = history_ctx.get("product", "")
    if not history_product:
        return q

    # 当前问题已有明确产品名，不需要补全
    if detect_terms(q, PRODUCT_ALIASES):
        return q

    # 排除明显的非领域问题（天气、新闻等）
    if _OFFTOPIC_PATTERNS.search(q):
        return q

    # 模式1: 指代词替换 — "它的成分呢" → "菲罗奥的成分呢"
    if _PRONOUN_PATTERNS.search(q):
        resolved = _PRONOUN_PATTERNS.sub(history_product, q, count=1)
        return resolved

    # 模式2: 追问/延续补全 — "还有别的吗" "成分呢" "禁忌人群有哪些" → 补充产品名
    if _FOLLOWUP_PATTERNS.search(q):
        return f"{history_product} {q}"

    # 模式3: 隐含关联 — "安全吗" "效果怎样" "需要几次" → 补充产品名
    if _IMPLICIT_TOPIC_PATTERNS.search(q) and len(q) <= 30:
        return f"{history_product} {q}"

    # 模式4: 含路由关键词但无产品名 — "术后能洗脸吗" "注射深度多少"
    q_lower = q.lower()
    has_route_keyword = any(kw in q_lower for kw in _ALL_ROUTE_KEYWORDS)
    if has_route_keyword and len(q) <= 40:
        return f"{history_product} {q}"

    return q


def _detect_route_for_expansion(q: str) -> List[str]:
    """检测问题命中的路由，返回匹配到的路由列表"""
    q_lower = q.lower()
    matched = []
    for route, keywords in QUESTION_ROUTES.items():
        if any(kw.lower() in q_lower for kw in keywords):
            matched.append(route)
    return matched


def _build_history_summary(history: List[Dict], max_turns: int = 3) -> str:
    """从对话历史中构建多轮摘要，供 LLM 理解完整对话脉络。

    提取最近 max_turns 轮用户问题，让 LLM 看到话题演变过程。
    例如: "菲罗奥成分是什么 → 安全吗 → 术后注意什么"
    """
    user_qs = []
    for item in history:
        if item.get("role") == "user":
            content = item.get("content", "").strip()
            if content:
                user_qs.append(content)
    # 取最近 max_turns 轮
    recent = user_qs[-max_turns:] if len(user_qs) > max_turns else user_qs
    return " → ".join(recent)


def rewrite_query(question: str, history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    raw = (question or "").strip()

    # 提取历史上下文（只解析一次，传给后续所有需要的函数）
    history_ctx: Dict[str, Any] = {}
    if history:
        history_ctx = _extract_history_context(history)

    # 上下文补全：解析指代词和省略
    q = _resolve_context(raw, history_ctx) if history_ctx else raw
    context_resolved = (q != raw)

    products = detect_terms(q, PRODUCT_ALIASES)
    projects = detect_terms(q, PROJECT_ALIASES)

    times = [x for x in TIME_TERMS if x in q]
    symptoms = [x for x in SYMPTOM_TERMS if x in q]

    expanded_terms = []
    for pid in products:
        expanded_terms.extend(PRODUCT_ALIASES.get(pid, [])[:4])
    for pj in projects:
        expanded_terms.extend(PROJECT_ALIASES.get(pj, [])[:3])
    expanded_terms.extend(times)
    expanded_terms.extend(symptoms)

    # 路由感知扩展：补充路由专属的高区分度词
    for route in _detect_route_for_expansion(q):
        expanded_terms.extend(_ROUTE_EXPANSION.get(route, []))

    sub_questions = split_multi_question(q)
    expanded_query = " ".join(uniq([q] + expanded_terms))

    # 构建多轮历史摘要供 LLM 使用
    history_summary = ""
    last_user_q = ""
    if history and context_resolved:
        history_summary = _build_history_summary(history)
        last_user_q = history_ctx.get("last_user_q", "")

    return {
        "original": q,
        "raw_input": raw,
        "context_resolved": context_resolved,
        "expanded": expanded_query,
        "products": products,
        "projects": projects,
        "times": uniq(times),
        "symptoms": uniq(symptoms),
        "sub_questions": sub_questions,
        "history_summary": history_summary,   # 多轮摘要，供 LLM prompt
        "last_user_q": last_user_q,           # 上一轮用户问题，供路由继承
    }
