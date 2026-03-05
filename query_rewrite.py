from typing import Dict, Any, List
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


def _detect_route_for_expansion(q: str) -> List[str]:
    """检测问题命中的路由，返回匹配到的路由列表"""
    q_lower = q.lower()
    matched = []
    for route, keywords in QUESTION_ROUTES.items():
        if any(kw.lower() in q_lower for kw in keywords):
            matched.append(route)
    return matched


def rewrite_query(question: str) -> Dict[str, Any]:
    q = (question or "").strip()
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
