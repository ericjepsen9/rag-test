import re
import time
import threading
from collections import OrderedDict
from typing import Dict, Any, List, Optional
from rag_runtime_config import (
    PRODUCT_ALIASES, PROJECT_ALIASES, TIME_TERMS, SYMPTOM_TERMS,
    QUESTION_ROUTES, USE_OPENAI, OPENAI_MODEL, OPENAI_API_BASE,
    PROCEDURE_ALIASES, EQUIPMENT_ALIASES, INDICATION_KEYWORDS,
    LLM_REWRITE_ENABLED,
)
from search_utils import detect_terms, uniq, split_multi_question, lookup_learned_synonym
from clarification_engine import should_clarify, generate_clarification

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

# ===== LLM 查询改写（方案3）=====
# 当静态同义词/别名无法识别用户术语时，用 LLM 将其映射到知识库已有概念
# 触发条件：查询未匹配任何已知产品/项目/路由关键词（即静态手段完全失效）
# 有 LRU 缓存，避免相同查询重复调用

_LLM_REWRITE_ENABLED = USE_OPENAI and LLM_REWRITE_ENABLED
_LLM_REWRITE_CACHE_SIZE = 256

# 构建知识库已知术语列表（告知 LLM 可以映射到哪些词）
_KNOWN_VOCAB_PARTS = []
for _aliases in PRODUCT_ALIASES.values():
    _KNOWN_VOCAB_PARTS.extend(_aliases[:2])
for _aliases in PROJECT_ALIASES.values():
    _KNOWN_VOCAB_PARTS.extend(_aliases[:2])
for _aliases in PROCEDURE_ALIASES.values():
    _KNOWN_VOCAB_PARTS.extend(_aliases[:3])
for _aliases in EQUIPMENT_ALIASES.values():
    _KNOWN_VOCAB_PARTS.extend(_aliases[:2])
for _kws in INDICATION_KEYWORDS.values():
    _KNOWN_VOCAB_PARTS.extend(_kws[:2])
_KNOWN_VOCAB = "、".join(sorted(set(_KNOWN_VOCAB_PARTS)))


class _LRUCache:
    """线程安全 LRU 缓存，支持 TTL 过期"""
    def __init__(self, maxsize: int = 256, ttl: float = 3600.0):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl  # 秒，默认 1 小时
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                ts, value = entry
                # TTL 过期检查
                if time.monotonic() - ts > self._ttl:
                    self._cache.pop(key, None)
                    self._misses += 1
                    return None
                self._cache.move_to_end(key)
                self._hits += 1
                return value
            self._misses += 1
        return None

    def put(self, key: str, value: str) -> None:
        now = time.monotonic()
        with self._lock:
            if key in self._cache:
                self._cache[key] = (now, value)
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = (now, value)

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"size": len(self._cache), "hits": self._hits,
                    "misses": self._misses, "maxsize": self._maxsize}


_llm_rewrite_cache = _LRUCache(_LLM_REWRITE_CACHE_SIZE)

# 预编译：用于判断 LLM 改写结果是否有效
_RE_NO_REWRITE = re.compile(r"(无法|不能|不确定|抱歉|sorry|NO_REWRITE)", re.IGNORECASE)


def _should_trigger_llm_rewrite(question: str, products: list, projects: list,
                                 detected_routes: list, is_chitchat: bool,
                                 is_offtopic: bool) -> bool:
    """判断是否需要触发 LLM 查询改写。

    改进策略：不仅在静态手段完全失效时触发，还在以下「边界情况」触发：
    1. 未识别到任何产品/项目且路由为 basic（最弱命中）
    2. 问题含有模糊/口语化表达，但只命中了一个通用路由关键词
    这样可以让更多「用词不精确」的查询被 LLM 纠正到规范术语。
    """
    if not _LLM_REWRITE_ENABLED:
        return False
    if is_chitchat or is_offtopic:
        return False
    # 问题太短（≤2字）或太长（>50字）不触发
    q = question.strip()
    if len(q) <= 2 or len(q) > 50:
        return False
    # 已识别到产品 + 明确路由 → 静态手段工作良好，不需要 LLM
    if products and detected_routes and not (len(detected_routes) == 1 and detected_routes[0] == "basic"):
        return False
    # 已识别到项目 + 明确路由 → 不需要 LLM
    if projects and detected_routes and not (len(detected_routes) == 1 and detected_routes[0] == "basic"):
        return False
    # 如果只命中了 basic 路由或没命中任何路由 → 触发 LLM 改写
    # 即使识别到了产品/项目，路由不明确也值得让 LLM 帮助理解用户意图
    if not detected_routes or (len(detected_routes) == 1 and detected_routes[0] == "basic"):
        return True
    # 检查是否包含足够的已知路由关键词（命中数 >= 2 才认为静态手段可靠）
    q_lower = q.lower()
    route_kw_hits = sum(1 for kw in _ALL_ROUTE_KEYWORDS if kw in q_lower)
    if route_kw_hits >= 2:
        return False
    # 只命中一个路由关键词且没有产品/项目 → 边界情况，触发 LLM
    if not products and not projects:
        return True
    return False


def _llm_rewrite_query(question: str) -> str:
    """调用 LLM 将用户查询中的未知术语映射到知识库已有概念。

    返回改写后的查询字符串。如果 LLM 认为无需改写或改写失败，返回空字符串。

    设计原则：
    - Prompt 精简，控制 token 消耗（~200 input tokens）
    - 只做术语映射，不改变用户意图
    - 带缓存，相同查询不重复调用
    """
    # 缓存命中
    cached = _llm_rewrite_cache.get(question)
    if cached is not None:
        return cached

    # 优先通过 llm_client 获取对话用 client，回退到 rag_answer 单例
    _chat_model = OPENAI_MODEL
    try:
        from llm_client import get_client as _get_multi_client, get_model as _get_multi_model, is_enabled as _is_enabled
        if _is_enabled("chat"):
            client = _get_multi_client("chat")
            if client:
                _chat_model = _get_multi_model("chat") or OPENAI_MODEL
        else:
            client = None
    except ImportError:
        client = None
    if client is None:
        try:
            from rag_answer import _get_openai_client
            client = _get_openai_client()
        except Exception:
            client = None
    if client is None:
        # 不缓存：client 可能稍后变为可用（如用户配置了 API key）
        return ""

    system_prompt = (
        "你是医美知识库的查询改写助手。用户的问题可能包含俗称、缩写或口语化表达，"
        "请将其改写为知识库能理解的规范术语。\n"
        "规则：\n"
        "1. 只改写术语，保留用户的原始意图和问题结构\n"
        "2. 如果用户用语已经规范或你不确定对应关系，原样返回用户问题\n"
        "3. 只输出改写后的问题，不要解释\n"
        f"\n知识库已有的术语包括：{_KNOWN_VOCAB}\n"
    )

    try:
        resp = client.chat.completions.create(
            model=_chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_tokens=150,
        )
        if not resp.choices:
            _llm_rewrite_cache.put(question, "")
            return ""
        result = (resp.choices[0].message.content or "").strip()
        # 验证结果有效性
        if not result or _RE_NO_REWRITE.search(result) or result == question:
            _llm_rewrite_cache.put(question, "")
            return ""
        _llm_rewrite_cache.put(question, result)
        return result
    except Exception as e:
        try:
            from rag_logger import log_error
            log_error("llm_rewrite_query", f"LLM 查询改写失败: {e}",
                      meta={"question": question[:100]})
        except Exception:
            pass
        # 不缓存 API 异常结果，允许后续重试（仅缓存 LLM 明确返回"无需改写"的情况）
        return ""


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

# 完全脱离医美领域的问题检测：命中时直接拒绝回答，避免返回无关知识库内容
# 注意：只拦截明显与医美无关的问题，医美行业相关（产品、手术、仪器、皮肤、
# 解剖、并发症、成分、护理等）的问题都应放行
_OFFTOPIC_FULL_PATTERNS = re.compile(
    r"(做菜|菜谱|食谱|烹饪|炒菜|炖|煮|蒸|烤|红烧|清蒸|糖醋|锅包|麻辣|酸辣|宫保|鱼香"
    r"|天气预报|气温|下雨|下雪|台风|暴雨"
    r"|股票|基金|理财|炒股|A股|港股|美股|比特币|加密货币|区块链"
    r"|电影|电视剧|综艺|动漫|追剧|票房|演员|导演|明星八卦"
    r"|歌曲|歌手|演唱会|专辑|歌词"
    r"|旅游攻略|景点|机票|酒店预订|签证|出境"
    r"|足球|篮球|排球|乒乓球|羽毛球|网球|世界杯|NBA|奥运|体育"
    r"|编程|代码|Python|Java|JavaScript|数据库|算法|软件开发"
    r"|数学题|物理|化学方程|历史事件|地理|语文|英语翻译|考试|高考|考研"
    r"|小说|诗词|散文|作文|写作技巧"
    r"|汽车|发动机|轮胎|驾照|交通违章|加油"
    r"|手机|电脑|笔记本|平板|耳机|显卡|CPU|内存"
    r"|外卖|快递|物流|淘宝|京东|拼多多|网购"
    r"|装修|家具|水电|房贷|租房|房价|买房"
    r"|宠物|养狗|养猫|猫粮|狗粮"
    r"|星座|算命|风水|周公解梦|占卜"
    r"|减肥食谱|健身计划|跑步|瑜伽课程"
    r"|政治|选举|法律咨询|打官司|离婚|遗产"
    r"|种菜|种花|园艺|农业|养殖)",
    re.IGNORECASE,
)

# 医美领域保护词：即使命中了非领域词，如果同时包含这些词则仍认为是医美问题
_MEDAES_GUARD_PATTERNS = re.compile(
    r"(注射|填充|玻尿酸|肉毒|胶原蛋白|光子嫩肤|激光|超声刀|热玛吉|水光|微针"
    r"|术后|术前|恢复期|禁忌|并发症|红肿|肿胀|硬块|瘢痕|疤痕"
    r"|皮肤|面部|脸部|额头|法令纹|苹果肌|下颌|鼻部|眼周|颈部"
    r"|医美|美容|整形|抗衰|紧致|提升|嫩肤|祛斑|祛痘|脱毛"
    r"|菲罗奥|赛罗菲|FILLOUP|PCL|聚己内酯|透明质酸|生长因子"
    r"|仪器|设备|水光仪|微针仪|射频|超声|冷冻溶脂"
    r"|操作|治疗|方案|疗程|护理|敷麻|麻醉)"
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

    # 非医美领域检测：命中非领域关键词且不含医美保护词 → 标记为 offtopic
    is_offtopic = bool(
        _OFFTOPIC_FULL_PATTERNS.search(raw)
        and not _MEDAES_GUARD_PATTERNS.search(raw)
    )

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
    # 区分"纠正前缀去除"和"上下文补全"：只有实际做了指代/省略补全才标记 context_resolved
    context_resolved = (q != cleaned)

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

    # ---- 学习词库查找 + LLM 查询改写（方案3）----
    # 先查已学习词库（零成本），再决定是否调用 LLM（消耗 token）
    llm_rewritten = ""
    if _should_trigger_llm_rewrite(q, products, projects, detected_routes,
                                    is_chitchat, is_offtopic):
        # 优先查已学习词库：如果之前 LLM 改写过相同/相似的查询，直接复用
        learned = lookup_learned_synonym(q)
        if learned:
            llm_rewritten = learned
        else:
            llm_rewritten = _llm_rewrite_query(q)
            if llm_rewritten:
                # 自动沉淀：将成功的 LLM 改写持久化到学习词库
                try:
                    from synonym_store import save_learned
                    save_learned(q, llm_rewritten)
                    # 增量更新运行时同义词表（避免全量 reload 的 I/O 开销）
                    from search_utils import add_learned_synonym
                    add_learned_synonym(q, llm_rewritten)
                except Exception:
                    pass  # 沉淀失败不影响主流程
        if llm_rewritten:
            # 改写成功：用改写结果替换检索查询，同时保留原始查询做混合检索
            search_q = llm_rewritten
            # 重新检测改写后的产品/项目/路由（可能映射到了已知实体）
            products = detect_terms(llm_rewritten, PRODUCT_ALIASES) or products
            projects = detect_terms(llm_rewritten, PROJECT_ALIASES) or projects
            detected_routes = _detect_route_for_expansion(llm_rewritten) or detected_routes
            # 补充改写后的扩展词
            for rt in detected_routes:
                expanded_terms.extend(_ROUTE_EXPANSION.get(rt, []))
            # 重建 expanded_query：原始查询 + LLM 改写 + 扩展词
            expanded_query = " ".join(uniq([q, llm_rewritten] + expanded_terms))
            sub_questions = split_multi_question(llm_rewritten)

    # 构建多轮历史摘要供 LLM 使用（单次扫描同时提取 summary + pairs）
    history_summary = ""
    history_pairs: List[Dict] = []
    last_user_q = ""
    if history:
        history_summary, history_pairs = _build_history_summary_and_pairs(history)
        last_user_q = history_ctx.get("last_user_q", "")

    # ---- 消歧引导：查询模糊且缺乏上下文时生成候选选项 ----
    clarification = None
    needs_clarification = False
    if not is_chitchat and not is_offtopic:
        history_product = history_ctx.get("product", "")
        history_route = history_ctx.get("route", "")
        if should_clarify(
            raw, products, projects, detected_routes,
            history_product=history_product,
            history_route=history_route,
            is_chitchat=is_chitchat,
            is_offtopic=is_offtopic,
        ):
            clarification = generate_clarification(
                raw,
                products=products,
                projects=projects,
                history_product=history_product,
            )
            if clarification:
                needs_clarification = True

    return {
        "original": q,
        "raw_input": raw,
        "search_query": search_q,             # 清理后的检索用查询（去除纠正前缀等噪音）
        "is_chitchat": is_chitchat,           # 非提问标记，调用方可直接返回礼貌回复
        "is_offtopic": is_offtopic,           # 非医美领域标记，调用方可直接返回拒绝回复
        "context_resolved": context_resolved,
        "expanded": expanded_query,
        "products": products,
        "projects": projects,
        "times": uniq(times),
        "symptoms": uniq(symptoms),
        "sub_questions": sub_questions,
        "detected_routes": detected_routes,   # rewrite 阶段检测到的路由（供 answer_one 参考）
        "llm_rewritten": llm_rewritten,       # LLM 改写结果（空=未触发或无改写）
        "history_summary": history_summary,   # 多轮摘要，供 LLM prompt
        "history_pairs": history_pairs,       # 完整 Q&A 对，供 LLM 深度理解
        "last_user_q": last_user_q,           # 上一轮用户问题
        "last_routed_q": history_ctx.get("last_routed_q", ""),  # 最近含路由的问题，供路由继承
        "needs_clarification": needs_clarification,  # 是否需要消歧引导
        "clarification": clarification,              # 消歧选项（None 表示不需要）
    }
