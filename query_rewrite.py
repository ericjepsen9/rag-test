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

# 指代词/省略句模式：命中时需要从历史中补全主语
_PRONOUN_PATTERNS = re.compile(
    r"^(它|这个|那个|该产品|这款|那款)"
    r"|^(那|那么).{0,2}(呢|吗|怎么样)"
    r"|^还有(别的|其他|吗)"
)

# 追问模式：保留当前问题但从历史补充产品/项目上下文
_FOLLOWUP_PATTERNS = re.compile(
    r"(呢|吗|怎么样|有哪些|是什么|是多少|怎么办)$"
)


def _resolve_context(question: str, history: Optional[List[Dict]]) -> str:
    """基于对话历史解析指代和省略，补全当前问题的上下文。

    策略：
    1. 如果当前问题已包含产品名 → 不需要补全
    2. 如果当前问题含指代词（它/这个/那个）→ 从历史中提取产品名替换
    3. 如果当前问题是追问（"那成分呢"）→ 从历史中继承产品名
    """
    if not history:
        return question

    q = question.strip()

    # 当前问题已有明确产品名，不需要补全
    if detect_terms(q, PRODUCT_ALIASES):
        return q

    # 从历史中提取最近的产品名和路由上下文
    history_product = ""
    history_topic = ""
    for item in reversed(history):
        content = item.get("content", "")
        if item.get("role") == "user":
            products = detect_terms(content, PRODUCT_ALIASES)
            if products and not history_product:
                # 找到产品的中文名
                for pid in products:
                    aliases = PRODUCT_ALIASES.get(pid, [])
                    if aliases:
                        history_product = aliases[0]  # 取第一个别名（标准中文名）
                        break
            if not history_topic:
                history_topic = content

    if not history_product:
        return q

    # 模式1: 指代词替换 — "它的成分呢" → "菲罗奥的成分呢"
    if _PRONOUN_PATTERNS.search(q):
        resolved = _PRONOUN_PATTERNS.sub(history_product, q, count=1)
        return resolved

    # 模式2: 追问补全 — "成分呢" / "禁忌人群有哪些" → 补充产品名
    if _FOLLOWUP_PATTERNS.search(q) and len(q) <= 15:
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


def rewrite_query(question: str, history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    q = (question or "").strip()

    # 上下文补全：解析指代词和省略
    q = _resolve_context(q, history)

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

    return {
        "original": q,
        "expanded": expanded_query,
        "products": products,
        "projects": projects,
        "times": uniq(times),
        "symptoms": uniq(symptoms),
        "sub_questions": sub_questions,
    }
