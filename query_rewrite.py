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
    "aftercare":        ["术后护理", "冰敷", "面膜", "洗脸", "禁酒", "辛辣",
                         "运动", "化妆", "防晒", "洗澡", "游泳", "桑拿", "上班"],
    "operation":        ["注射", "深度", "进针", "剂量", "操作", "间距"],
    "effect":           ["效果", "维持", "胶原再生", "疗程"],
    "pre_care":         ["术前", "准备", "检查", "停药"],
    "design":           ["方案设计", "面部评估", "松弛", "用量"],
    "repair":           ["修复", "补救", "不理想", "硬块处理"],
    # 跨实体路由
    "complication":     ["并发症", "分级处理", "警惕信号", "就医", "观察", "复诊", "术后恢复"],
    "course":           ["疗程规划", "治疗周期", "间隔时间", "次数", "时间表"],
    "anatomy_q":        ["面部分区", "部位", "治疗方案", "区域特征"],
    "indication_q":     ["皮肤状态", "症状改善", "推荐方案", "适应症"],
    "procedure_q":      ["项目概述", "操作流程", "适用人群", "效果对比"],
    "equipment_q":      ["仪器参数", "设备特性", "适配产品", "针头规格"],
    "script":           ["客户沟通", "话术", "顾虑解答", "合规"],
}

# 指代词模式：命中时用产品名**替换**指代词
# 支持句首和句中匹配（如 "怎么样，这个产品呢？"）
_PRONOUN_PATTERNS = re.compile(
    r"(它|这个|那个|该产品|这款|那款)"
    r"|(那|那么).{0,2}(呢|吗|怎么样)"
)

# 追问/延续模式：命中时在问题前**补充**产品名
_FOLLOWUP_PATTERNS = re.compile(
    r"^还有(别的|其他|什么|吗)"
    r"|(呢|吗|怎么样|有哪些|是什么|是多少|怎么办)$"
)

# 明确的非领域词：命中这些词的问题大概率与产品无关，不应补全
_OFFTOPIC_PATTERNS = re.compile(
    r"(天气|新闻|股票|电影|音乐|美食|旅游|游戏|足球|篮球|奥运)"
)

# 产品切换意图：用户想问另一个/不同的产品，不应继承历史产品
_SWITCH_PATTERNS = re.compile(
    r"(换一个|另一个|另一款|其他产品|别的产品|不同的产品|有没有其他|还有什么产品)"
)

# 隐含关联模式：问题本身是关于某个主题但缺少产品主语
# 例如 "安全吗" "效果怎样" "多少钱" "要恢复多久"
# 注意：单独出现容易误匹配的词（如"效果""作用"）须搭配疑问尾才触发
_IMPLICIT_TOPIC_PATTERNS = re.compile(
    r"(安全|价格|多少钱|持续多久|维持多久|恢复多久|多久恢复|恢复期|疗程|几次|保质期|保存"
    r"|区别|对比|优势|好处|原理|机制"
    r"|痛不痛|疼不疼|会不会|能不能|可不可以|需要几|需要多|注意什么|注意事项"
    r"|要不要停药|需不需要|要做什么准备|怎么设计|打几支|怎么修复"
    r"|会不会烧伤|会不会感染|正常吗|怎么办|多久见效|多久消|几天消"
    r"|有没有用|有效果吗|靠谱吗|值得做吗|推荐吗|划算吗)"
    r"|(效果|功效|作用).{0,4}(吗|呢|怎么样|如何|好不好|怎样)"
)

# 非提问：问候、致谢、确认等不需要检索的输入
_CHITCHAT_PATTERNS = re.compile(
    r"^(你好|嗨|hi|hello|hey|您好|在吗|在不在)[啊呀哇~！!？?。.]*$"
    r"|^(谢谢|感谢|多谢|辛苦了|好的|明白了|了解了|知道了|收到|OK|ok|嗯|嗯嗯|哦|噢)[啊呀~！!。.]*$"
    r"|^(再见|拜拜|bye)[~！!。.]*$"
    r"|^[？?！!。\.…·\s~]+$",
    re.IGNORECASE,
)

# 纠正前缀：用户纠正上一个回答时的引导语，检索时应去除
_CORRECTION_PREFIX = re.compile(
    r"^(不对[，,]?|不是[，,]?(这个[，,]?)?|我(想|要)?问的是|我说的是)"
)

# 所有路由关键词汇集，用于判断问题是否包含领域词
_ALL_ROUTE_KEYWORDS = set()
for _kws in QUESTION_ROUTES.values():
    _ALL_ROUTE_KEYWORDS.update(kw.lower() for kw in _kws)

# 预计算小写路由关键词，避免 _detect_route_for_expansion / _extract_history_context 每次调用 .lower()
_QUESTION_ROUTES_LOWER = {
    route: [kw.lower() for kw in keywords]
    for route, keywords in QUESTION_ROUTES.items()
}


_MAX_HISTORY_SCAN = 12  # 最多回溯的用户消息数，支持深度对话链


def _extract_history_context(history: List[Dict]) -> Dict[str, Any]:
    """从对话历史中提取结构化上下文信息。

    最多回溯 _MAX_HISTORY_SCAN 条用户消息。当 product + route + projects
    全部找到时提前退出；否则扫完限额后返回已找到的部分。

    返回:
        product: 最近提到的产品标准中文名
        product_id: 产品 ID
        projects: 最近提到的项目名列表（水光/微针等）
        route: 最近的路由主题
        last_user_q: 最近的用户问题原文
    """
    ctx: Dict[str, Any] = {
        "product": "", "product_id": "", "projects": [],
        "route": "", "last_user_q": "",
        "last_routed_q": "",  # 最近一条含路由关键词的用户问题（用于路由继承）
    }
    user_count = 0
    # 从末尾向前扫描，用切片避免 reversed() 创建完整拷贝
    scan_slice = history[-(2 * _MAX_HISTORY_SCAN):] if len(history) > 2 * _MAX_HISTORY_SCAN else history
    for item in reversed(scan_slice):
        raw_content = item.get("content")
        content = (str(raw_content) if raw_content is not None else "")[:500]
        if item.get("role") != "user":
            continue
        user_count += 1

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

        if not ctx["projects"]:
            projects = detect_terms(content, PROJECT_ALIASES)
            if projects:
                proj_names = []
                for pj in projects:
                    aliases = PROJECT_ALIASES.get(pj, [])
                    if aliases:
                        proj_names.append(aliases[0])
                ctx["projects"] = proj_names

        if not ctx["route"]:
            content_lower = content.lower()
            for route, keywords in _QUESTION_ROUTES_LOWER.items():
                if any(kw in content_lower for kw in keywords):
                    ctx["route"] = route
                    # 记录含路由关键词的原始问题，而非可能是 "还有吗" 的 last_user_q
                    if not ctx["last_routed_q"]:
                        ctx["last_routed_q"] = content
                    break

        # 全部找到 → 提前退出
        if ctx["product"] and ctx["route"] and ctx["projects"]:
            break
        # 已找到产品+路由（最重要的两项），且已扫描足够消息 → 提前退出
        if ctx["product"] and ctx["route"] and user_count >= 3:
            break
        if user_count >= _MAX_HISTORY_SCAN:
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

    # 排除产品切换意图（"换一个""另一款"等）——不应继承历史产品
    if _SWITCH_PATTERNS.search(q):
        return q

    # 构建补全前缀：产品名 + 历史中的项目名（如果当前问题中没有）
    prefix_parts = [history_product]
    if not detect_terms(q, PROJECT_ALIASES):
        for pj in history_ctx.get("projects", []):
            prefix_parts.append(pj)
    prefix = " ".join(prefix_parts)

    # 模式1: 指代词替换 — "它的成分呢" → "菲罗奥的成分呢"
    # 仅替换句首或标点后的指代词，排除介词/连词后的宾语位（"和它""跟它""比它"等）
    m = _PRONOUN_PATTERNS.search(q)
    if m:
        pos = m.start()
        # 句首直接替换
        if pos == 0:
            resolved = prefix + q[m.end():]
            return resolved
        # 句中：仅标点/空格后替换，排除 "和它""跟它""比它""与它" 宾语位
        prev = q[pos - 1]
        if prev in "，。？！；、,;!? " and (pos < 2 or q[pos - 2] not in "和跟比与"):
            resolved = q[:pos] + prefix + q[m.end():]
            return resolved

    # 模式2: 追问/延续补全 — "还有别的吗" "成分呢" "禁忌人群有哪些" → 补充上下文
    if _FOLLOWUP_PATTERNS.search(q):
        return f"{prefix} {q}"

    # 模式3: 隐含关联 — "安全吗" "效果怎样" "需要几次" → 补充上下文
    if _IMPLICIT_TOPIC_PATTERNS.search(q) and len(q) <= 30:
        return f"{prefix} {q}"

    # 模式4: 含路由关键词但无产品名 — "术后能洗脸吗" "注射深度多少"
    q_lower = q.lower()
    has_route_keyword = any(kw in q_lower for kw in _ALL_ROUTE_KEYWORDS)
    if has_route_keyword and len(q) <= 40:
        return f"{prefix} {q}"

    return q


def _detect_route_for_expansion(q: str) -> List[str]:
    """检测问题命中的路由，返回匹配到的路由列表"""
    q_lower = q.lower()
    matched = []
    for route, keywords in _QUESTION_ROUTES_LOWER.items():
        if any(kw in q_lower for kw in keywords):
            matched.append(route)
    return matched


def _build_history_summary_and_pairs(
    history: List[Dict], max_turns: int = 3, max_pairs: int = 3
) -> tuple:
    """单次反向扫描同时构建多轮摘要和 Q&A 对，避免两次独立遍历。

    返回 (summary: str, pairs: List[Dict])。
    summary: "菲罗奥成分是什么 → 安全吗 → 术后注意什么" 格式。
    pairs: [{"user": "...", "assistant": "..."}, ...] 格式。
    """
    recent_questions: List[str] = []
    pairs: List[Dict] = []
    current_assistant = ""
    summary_done = False
    pairs_done = False

    for item in reversed(history):
        if summary_done and pairs_done:
            break
        role = item.get("role", "")
        raw = item.get("content")
        content = (str(raw) if raw is not None else "").strip()

        if role == "assistant":
            if not pairs_done and not current_assistant:
                current_assistant = content[:200]
        elif role == "user":
            if not summary_done and content:
                recent_questions.append(content)
                if len(recent_questions) >= max_turns:
                    summary_done = True
            if not pairs_done and current_assistant:
                pairs.append({"user": content, "assistant": current_assistant})
                current_assistant = ""
                if len(pairs) >= max_pairs:
                    pairs_done = True

    recent_questions.reverse()
    pairs.reverse()
    return " → ".join(recent_questions), pairs


def _build_history_summary(history: List[Dict], max_turns: int = 3) -> str:
    """向后兼容包装：委托给合并后的单次扫描函数。"""
    summary, _ = _build_history_summary_and_pairs(history, max_turns=max_turns, max_pairs=0)
    return summary


def _build_history_pairs(history: List[Dict], max_pairs: int = 3) -> List[Dict]:
    """向后兼容包装：委托给合并后的单次扫描函数。"""
    _, pairs = _build_history_summary_and_pairs(history, max_turns=0, max_pairs=max_pairs)
    return pairs


def rewrite_query(question: str, history: Optional[List[Dict]] = None,
                   _cached_ctx: Optional[Dict] = None) -> Dict[str, Any]:
    raw = (question or "").strip()

    # 快速判断：非提问（问候/致谢/确认）直接返回，跳过后续处理
    is_chitchat = bool(_CHITCHAT_PATTERNS.match(raw))

    # 清理纠正前缀："不对，我问的是禁忌人群" → "禁忌人群"（在上下文补全之前清理）
    cleaned = _CORRECTION_PREFIX.sub("", raw).strip() if _CORRECTION_PREFIX.search(raw) else raw
    has_correction = (cleaned != raw)

    # 提取历史上下文（支持缓存复用，避免多子问题重复解析）
    history_ctx: Dict[str, Any] = {}
    if _cached_ctx is not None:
        history_ctx = _cached_ctx
    elif history:
        history_ctx = _extract_history_context(history)

    # 上下文补全：解析指代词和省略（用清理后的文本做补全，更干净）
    q = _resolve_context(cleaned, history_ctx) if (history_ctx and not is_chitchat) else cleaned
    context_resolved = (q != raw)

    # search_query：用于检索的查询（已去除纠正前缀）
    # original：保留完整补全结果（含产品名），供路由检测和 LLM 使用
    search_q = q

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
    detected_routes = _detect_route_for_expansion(q)
    for rt in detected_routes:
        expanded_terms.extend(_ROUTE_EXPANSION.get(rt, []))

    # 当没有检测到路由但有历史路由时，也补充对应的扩展词（支持追问继承场景）
    if not detected_routes and history_ctx.get("route"):
        inherited_rt = history_ctx["route"]
        expanded_terms.extend(_ROUTE_EXPANSION.get(inherited_rt, []))

    sub_questions = split_multi_question(q)
    expanded_query = " ".join(uniq([search_q] + expanded_terms))

    # 构建多轮历史摘要供 LLM 使用（单次扫描同时提取 summary + pairs）
    history_summary = ""
    history_pairs: List[Dict] = []
    last_user_q = ""
    if history:
        history_summary, history_pairs = _build_history_summary_and_pairs(history)
        last_user_q = history_ctx.get("last_user_q", "")

    return {
        "original": q,
        "raw_input": raw,
        "search_query": search_q,             # 清理后的检索用查询（去除纠正前缀等噪音）
        "is_chitchat": is_chitchat,           # 非提问标记，调用方可直接返回礼貌回复
        "context_resolved": context_resolved,
        "expanded": expanded_query,
        "products": products,
        "projects": projects,
        "times": uniq(times),
        "symptoms": uniq(symptoms),
        "sub_questions": sub_questions,
        "detected_routes": detected_routes,   # rewrite 阶段检测到的路由（供 answer_one 参考）
        "history_summary": history_summary,   # 多轮摘要，供 LLM prompt
        "history_pairs": history_pairs,       # 完整 Q&A 对，供 LLM 深度理解
        "last_user_q": last_user_q,           # 上一轮用户问题
        "last_routed_q": history_ctx.get("last_routed_q", ""),  # 最近含路由的问题，供路由继承
    }
