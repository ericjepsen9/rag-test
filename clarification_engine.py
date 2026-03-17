"""消歧引导引擎：当用户查询模糊且缺乏上下文时，生成候选选项帮助用户精确提问。

设计原则：
- 不是每次都弹选项，仅在查询短且歧义大时触发
- 同时返回一个"最可能的答案"方向 + 候选选项，不阻断流程
- 选项数量控制在 2-4 个，外加一个"直接搜索"兜底
- 如果上下文已明确（历史对话中有产品/项目），跳过消歧
"""

import re
from typing import Dict, List, Any, Optional

from rag_runtime_config import (
    PRODUCT_ALIASES, PROJECT_ALIASES, QUESTION_ROUTES,
    CLARIFICATION_ENABLED, CLARIFICATION_MIN_QUERY_LEN, CLARIFICATION_MAX_QUERY_LEN,
)

# ============================================================
# 消歧规则表：模糊关键词 → 候选场景
# key: 触发词（用户输入中包含即匹配）
# value: 候选选项列表，每个选项包含 label（展示）、query（替代查询）、route（路由提示）
# ============================================================

_CLARIFICATION_RULES: Dict[str, List[Dict[str, str]]] = {
    # === 症状类模糊查询 ===
    "红肿": [
        {"label": "注射后红肿（正常反应及处理）", "query": "注射后红肿是正常反应吗 怎么处理", "route": "aftercare"},
        {"label": "红肿不消退（可能的并发症）", "query": "红肿一直不消退怎么办 并发症", "route": "complication"},
        {"label": "如何预防红肿", "query": "术前术后如何预防红肿", "route": "pre_care"},
    ],
    "肿": [
        {"label": "术后正常肿胀及消退时间", "query": "术后肿胀多久消退 正常反应", "route": "aftercare"},
        {"label": "肿胀不消退（异常情况）", "query": "肿胀一直不消退 并发症处理", "route": "complication"},
        {"label": "如何减轻肿胀", "query": "术后如何减轻肿胀 冷敷", "route": "aftercare"},
    ],
    "疼": [
        {"label": "注射过程中的疼痛", "query": "注射时疼痛程度 麻醉方式", "route": "operation"},
        {"label": "术后疼痛及缓解方法", "query": "术后疼痛正常吗 如何缓解", "route": "aftercare"},
        {"label": "异常疼痛（需要就医）", "query": "术后剧烈疼痛 异常反应 就医", "route": "complication"},
    ],
    "痛": [
        {"label": "注射过程中的疼痛", "query": "注射时疼痛程度 麻醉方式", "route": "operation"},
        {"label": "术后疼痛及缓解方法", "query": "术后疼痛正常吗 如何缓解", "route": "aftercare"},
        {"label": "异常疼痛（需要就医）", "query": "术后剧烈疼痛 异常反应 就医", "route": "complication"},
    ],
    "硬": [
        {"label": "注射后硬结/硬块", "query": "注射后出现硬结硬块 正常吗 怎么处理", "route": "complication"},
        {"label": "填充物变硬", "query": "填充物变硬是什么原因 处理方法", "route": "risk"},
        {"label": "术后皮肤触感变硬", "query": "术后皮肤触感变硬 多久恢复", "route": "aftercare"},
    ],
    "硬块": [
        {"label": "硬块的原因及处理", "query": "注射后硬块原因及处理方法", "route": "complication"},
        {"label": "硬块多久能消退", "query": "注射后硬块多久消退 恢复期", "route": "aftercare"},
        {"label": "硬块是否需要就医", "query": "注射后硬块需要就医吗 严重吗", "route": "risk"},
    ],
    "结节": [
        {"label": "结节的原因及处理", "query": "注射后出现结节原因 处理方法", "route": "complication"},
        {"label": "结节多久消退", "query": "注射后结节多久消退 恢复", "route": "aftercare"},
        {"label": "结节是否需要就医", "query": "注射后结节需要就医吗", "route": "risk"},
    ],
    "淤青": [
        {"label": "术后淤青正常吗", "query": "术后淤青是正常反应吗 多久消退", "route": "aftercare"},
        {"label": "如何加速淤青消退", "query": "术后淤青如何加速消退 处理方法", "route": "aftercare"},
        {"label": "如何预防淤青", "query": "术前如何预防淤青", "route": "pre_care"},
    ],
    "过敏": [
        {"label": "注射后过敏反应处理", "query": "注射后过敏反应 红疹瘙痒处理", "route": "complication"},
        {"label": "过敏体质能做吗", "query": "过敏体质能注射吗 禁忌人群", "route": "contraindication"},
        {"label": "术前过敏测试", "query": "术前需要做过敏测试吗", "route": "pre_care"},
    ],
    "感染": [
        {"label": "感染的症状及处理", "query": "注射后感染症状 处理方法 就医", "route": "complication"},
        {"label": "如何预防感染", "query": "注射前后如何预防感染", "route": "aftercare"},
        {"label": "感染的严重程度判断", "query": "注射后感染严重程度判断 就医信号", "route": "risk"},
    ],

    # === 效果类模糊查询 ===
    "效果": [
        {"label": "效果怎么样/能维持多久", "query": "注射效果怎么样 维持多久", "route": "effect"},
        {"label": "多久见效", "query": "注射后多久见效 起效时间", "route": "effect"},
        {"label": "效果不理想怎么办", "query": "注射效果不理想 修复补救", "route": "repair"},
    ],
    "多久": [
        {"label": "恢复期多久", "query": "术后恢复期多久 多久消肿", "route": "aftercare"},
        {"label": "效果维持多久", "query": "注射效果能维持多久", "route": "effect"},
        {"label": "多久见效", "query": "注射后多久能看到效果", "route": "effect"},
        {"label": "多久能做下一次", "query": "疗程间隔多久 多久做一次", "route": "course"},
    ],
    "恢复": [
        {"label": "恢复期及注意事项", "query": "术后恢复期多久 注意事项", "route": "aftercare"},
        {"label": "恢复过程中的正常反应", "query": "术后恢复过程中正常反应有哪些", "route": "complication"},
        {"label": "如何加速恢复", "query": "术后如何加速恢复", "route": "aftercare"},
    ],

    # === 操作类模糊查询 ===
    "怎么打": [
        {"label": "注射操作方法/参数", "query": "注射操作方法 深度 剂量 参数", "route": "operation"},
        {"label": "注射部位方案设计", "query": "注射部位选择 方案设计 面部评估", "route": "design"},
        {"label": "需要打几次/疗程", "query": "需要打几次 疗程规划", "route": "course"},
    ],
    "注射": [
        {"label": "注射操作方法", "query": "注射操作方法 深度 参数 注意事项", "route": "operation"},
        {"label": "注射后护理", "query": "注射后护理注意事项", "route": "aftercare"},
        {"label": "注射风险与不良反应", "query": "注射后风险 不良反应", "route": "risk"},
    ],

    # === 禁忌类模糊查询 ===
    "能做吗": [
        {"label": "禁忌人群（哪些人不能做）", "query": "哪些人不能做 禁忌人群", "route": "contraindication"},
        {"label": "特殊时期能做吗（孕期/哺乳等）", "query": "孕期哺乳期能做吗 禁忌", "route": "contraindication"},
        {"label": "正在用药能做吗", "query": "吃药期间能做吗 用药禁忌", "route": "contraindication"},
    ],
    "能不能": [
        {"label": "禁忌人群", "query": "哪些人不能做 禁忌人群", "route": "contraindication"},
        {"label": "术后能不能（护理相关）", "query": "术后注意事项 禁忌行为", "route": "aftercare"},
    ],

    # === 成分类模糊查询 ===
    "成分": [
        {"label": "核心成分及作用", "query": "核心成分是什么 作用机制", "route": "ingredient"},
        {"label": "成分安全性", "query": "成分安全吗 有没有副作用", "route": "risk"},
    ],

    # === 产品类模糊查询 ===
    "是什么": [
        {"label": "产品基本介绍", "query": "产品基础信息 是什么 有什么用", "route": "basic"},
        {"label": "核心成分与功效", "query": "核心成分 功效 作用", "route": "ingredient"},
        {"label": "适合什么人/什么场景", "query": "适合什么人群 适应症", "route": "indication_q"},
    ],

    # === 护理类模糊查询 ===
    "护理": [
        {"label": "术后护理注意事项", "query": "术后护理注意事项 全面指南", "route": "aftercare"},
        {"label": "术前准备事项", "query": "术前准备 注意事项", "route": "pre_care"},
    ],
    "注意": [
        {"label": "术后注意事项", "query": "术后护理注意事项", "route": "aftercare"},
        {"label": "术前注意事项", "query": "术前注意事项 准备", "route": "pre_care"},
        {"label": "禁忌事项", "query": "禁忌人群 禁忌事项", "route": "contraindication"},
    ],

    # === 安全类模糊查询 ===
    "安全": [
        {"label": "产品安全性及认证", "query": "产品安全吗 认证 备案", "route": "basic"},
        {"label": "风险与不良反应", "query": "有什么风险 不良反应", "route": "risk"},
        {"label": "禁忌人群", "query": "哪些人不能做 禁忌", "route": "contraindication"},
    ],

    # === 搭配类模糊查询 ===
    "搭配": [
        {"label": "可搭配的项目", "query": "可以和什么项目搭配 联合方案", "route": "combo"},
        {"label": "搭配注意事项及间隔", "query": "联合方案注意事项 间隔时间", "route": "combo"},
    ],
    "联合": [
        {"label": "可联合的项目有哪些", "query": "可以联合哪些项目 搭配方案", "route": "combo"},
        {"label": "联合注意事项及间隔", "query": "联合方案注意事项 间隔时间", "route": "combo"},
    ],
}

# 预编译：触发词长度排序（长词优先匹配，避免短词过早命中）
_SORTED_TRIGGER_KEYS = sorted(_CLARIFICATION_RULES.keys(), key=len, reverse=True)

# 不触发消歧的上下文关键词：当用户输入中已包含这些词时，说明意图已明确
_CLEAR_INTENT_PATTERNS = re.compile(
    r"(术后|术前|注射后|打完|做完|之前|之后|恢复期|并发症|禁忌|操作|怎么办|怎么处理"
    r"|多久消|正常吗|需要就医|严重吗|怎么缓解|如何预防|能不能做|多久见效"
    r"|维持多久|效果怎么样|安全吗|成分是|第\d+天|当天|一周|一个月)"
)

# 从配置读取阈值
_MIN_QUERY_LEN_FOR_CLARIFY = CLARIFICATION_MIN_QUERY_LEN
_MAX_QUERY_LEN_FOR_CLARIFY = CLARIFICATION_MAX_QUERY_LEN


def should_clarify(
    question: str,
    products: list,
    projects: list,
    detected_routes: list,
    history_product: str = "",
    history_route: str = "",
    is_chitchat: bool = False,
    is_offtopic: bool = False,
) -> bool:
    """判断是否需要触发消歧引导。

    触发条件（全部满足）：
    1. 非闲聊、非离题
    2. 查询长度在 [2, _MAX_QUERY_LEN_FOR_CLARIFY] 字符之间
    3. 查询中不包含明确意图词（_CLEAR_INTENT_PATTERNS）
    4. 以下至少一个为真：
       a. 无产品上下文（当前 + 历史都没有）
       b. 检测到的路由模糊（多于1个候选或仅 basic）
       c. 查询过短（< _MIN_QUERY_LEN_FOR_CLARIFY 字符）且路由不明确
    """
    if not CLARIFICATION_ENABLED:
        return False
    if is_chitchat or is_offtopic:
        return False

    q = question.strip()
    q_len = len(q)

    # 太短（1字）或太长（>15字）不触发
    if q_len < 2 or q_len > _MAX_QUERY_LEN_FOR_CLARIFY:
        return False

    # 已有明确意图表达 → 不需要消歧
    if _CLEAR_INTENT_PATTERNS.search(q):
        return False

    # 有产品上下文 + 有明确路由 → 不需要消歧
    has_product = bool(products) or bool(history_product)
    has_clear_route = (
        bool(detected_routes)
        and len(detected_routes) == 1
        and detected_routes[0] != "basic"
    )
    if has_product and has_clear_route:
        return False

    # 查询较短且缺少上下文 → 需要消歧
    if q_len < _MIN_QUERY_LEN_FOR_CLARIFY:
        return True

    # 路由模糊（多候选或仅 basic 或无）→ 需要消歧
    if not detected_routes or (len(detected_routes) == 1 and detected_routes[0] == "basic"):
        return True

    # 多路由候选 → 需要消歧
    if len(detected_routes) > 1:
        return True

    return False


def generate_clarification(
    question: str,
    products: list = None,
    projects: list = None,
    history_product: str = "",
) -> Optional[Dict[str, Any]]:
    """生成消歧选项。

    返回:
        None: 未命中任何消歧规则
        Dict: {
            "message": 提示语,
            "options": [
                {"label": "注射后红肿", "query": "...", "route": "aftercare"},
                ...
            ],
            "fallback_option": {"label": "直接搜索", "query": 原始查询}
        }
    """
    q = question.strip()
    q_lower = q.lower()

    # 查找匹配的消歧规则（长词优先）
    matched_options = None
    matched_key = None
    for key in _SORTED_TRIGGER_KEYS:
        if key in q_lower:
            matched_options = _CLARIFICATION_RULES[key]
            matched_key = key
            break

    if not matched_options:
        return None

    # 如果有产品上下文，在选项的 query 中补充产品名
    product_prefix = ""
    if products:
        pid = products[0]
        aliases = PRODUCT_ALIASES.get(pid, [])
        if aliases:
            product_prefix = aliases[0]
    elif history_product:
        product_prefix = history_product

    # 构建选项（注入产品上下文）
    options = []
    for opt in matched_options:
        enriched_query = opt["query"]
        if product_prefix and product_prefix not in enriched_query:
            enriched_query = f"{product_prefix} {enriched_query}"
        options.append({
            "label": opt["label"],
            "query": enriched_query,
            "route": opt.get("route", ""),
        })

    # 兜底选项
    fallback_query = q
    if product_prefix and product_prefix not in q:
        fallback_query = f"{product_prefix} {q}"

    return {
        "message": f"请问您想了解关于「{matched_key}」的哪方面问题？",
        "options": options,
        "fallback_option": {
            "label": f"直接搜索「{q}」相关内容",
            "query": fallback_query,
        },
    }


def format_clarification_text(clarification: Dict[str, Any]) -> str:
    """将消歧选项格式化为文本（用于非结构化响应场景）。

    输出格式：
    请问您想了解关于「红肿」的哪方面问题？
    1. 注射后红肿（正常反应及处理）
    2. 红肿不消退（可能的并发症）
    3. 如何预防红肿
    4. 直接搜索「红肿」相关内容

    您可以直接回复数字选择，或继续描述您的问题。
    """
    lines = [clarification["message"]]
    for i, opt in enumerate(clarification["options"], 1):
        lines.append(f"{i}. {opt['label']}")
    fb = clarification.get("fallback_option")
    if fb:
        lines.append(f"{len(clarification['options']) + 1}. {fb['label']}")
    lines.append("")
    lines.append("您可以直接回复数字选择，或继续描述您的问题。")
    return "\n".join(lines)


def resolve_numeric_choice(
    user_input: str,
    previous_clarification: Dict[str, Any],
) -> Optional[str]:
    """解析用户的数字选择，返回对应的查询字符串。

    如果用户输入不是数字或超出范围，返回 None。
    """
    s = user_input.strip()
    if not s.isdigit():
        return None
    idx = int(s)
    options = previous_clarification.get("options", [])
    fb = previous_clarification.get("fallback_option")

    if 1 <= idx <= len(options):
        return options[idx - 1]["query"]
    if fb and idx == len(options) + 1:
        return fb["query"]
    return None
