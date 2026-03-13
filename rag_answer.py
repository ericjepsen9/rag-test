import os
import sys
import json
import re
import threading
from pathlib import Path
from typing import List, Dict

os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

from rag_runtime_config import (
    KNOWLEDGE_DIR, STORE_ROOT, OUT_PATH, DEFAULT_MODE, DEFAULT_TOP_K,
    USE_OPENAI, OPENAI_MODEL, QUESTION_ROUTES, SECTION_RULES,
    PRODUCT_ALIASES, VECTOR_TOP_K, KEYWORD_TOP_K,
    HYBRID_VECTOR_WEIGHT, HYBRID_KEYWORD_WEIGHT, QUESTION_TYPE_CONFIG,
    MAX_SUB_QUESTIONS, MAX_EVIDENCE_CHUNKS,
    EMBED_MODEL_NAME, EMBED_USE_FP16, EMBED_BATCH_SIZE_QUERY, EMBED_MAX_LENGTH_QUERY,
    LLM_TEMPERATURE, LLM_MAX_TOKENS_BRIEF, LLM_MAX_TOKENS_FULL,
    RELATIONS_FILE,
    PRICE_REPLY, COMPARISON_REPLY, LOCATION_REPLY,
)
from search_utils import (
    normalize_lines, uniq, is_faq_line, section_block,
    keyword_search, merge_hybrid, detect_terms
)
from query_rewrite import rewrite_query
from answer_formatter import format_structured_answer
from relation_engine import enrich_answer as relation_enrich
from rag_logger import log_qa

_model = None
_faiss = None
_BGEM3 = None
_store_cache = {}  # {product: (index, docs, mtime)} — 进程内缓存，避免每次请求重读文件
_store_lock = threading.Lock()   # 保护 _store_cache 读写
_search_lock = threading.Lock()  # 保护 FAISS index.search（非线程安全）

# 跨实体路由：这些路由需要检索共享知识库（_shared store）
_SHARED_ROUTES = {"complication", "course", "anatomy_q", "indication_q",
                  "procedure_q", "equipment_q", "script"}
# 混合路由：同时检索产品库和共享库
_HYBRID_ENTITY_ROUTES = {"complication", "course", "anatomy_q", "indication_q"}


def invalidate_store_cache(product: str) -> None:
    """线程安全地清除指定产品的索引缓存，供 rebuild 后调用"""
    with _store_lock:
        _store_cache.pop(product, None)


def get_faiss():
    global _faiss
    if _faiss is None:
        import faiss as _faiss_mod
        _faiss = _faiss_mod
    return _faiss


def get_bg_cls():
    global _BGEM3
    if _BGEM3 is None:
        from FlagEmbedding import BGEM3FlagModel as _cls
        _BGEM3 = _cls
    return _BGEM3


def _get_out_path() -> Path:
    """支持通过环境变量 RAG_ANSWER_FILE 指定输出路径，避免并发覆盖"""
    env_path = os.environ.get("RAG_ANSWER_FILE", "").strip()
    if env_path:
        return Path(env_path)
    return OUT_PATH


def save_answer(text: str):
    _get_out_path().write_text((text or "").strip() + "\n", encoding="utf-8-sig")


def get_model():
    global _model
    if _model is None:
        _model = get_bg_cls()(EMBED_MODEL_NAME, use_fp16=EMBED_USE_FP16)
    return _model


def embed_query(text: str) -> np.ndarray:
    model = get_model()
    out = model.encode([text], batch_size=EMBED_BATCH_SIZE_QUERY, max_length=EMBED_MAX_LENGTH_QUERY)
    if isinstance(out, dict):
        if "dense_vecs" in out:
            vec = out["dense_vecs"]
        elif "dense" in out:
            vec = out["dense"]
        elif "embeddings" in out:
            vec = out["embeddings"]
        else:
            raise ValueError("未找到查询向量字段")
    else:
        vec = out
    vec = np.asarray(vec, dtype="float32")
    get_faiss().normalize_L2(vec)
    return vec


def _auto_rebuild_index(product: str, docs: List[Dict], store_dir: Path):
    """当 docs.jsonl 与 index.faiss 条目数不一致时，自动重建索引。
    在运行时使用 rag_answer 的 embed 模型（已加载），避免再次导入 build_faiss。
    返回新的 FAISS index，失败时返回 None。"""
    try:
        texts = [(d.get("text") or "").strip() for d in docs]
        texts = [t if t else " " for t in texts]  # 空文本用占位符
        model = get_model()
        out = model.encode(texts, batch_size=EMBED_BATCH_SIZE_QUERY, max_length=EMBED_MAX_LENGTH_QUERY)
        if isinstance(out, dict):
            vecs = out.get("dense_vecs") or out.get("dense") or out.get("embeddings")
        else:
            vecs = out
        if vecs is None:
            print(f"[WARN] {product}: 自动重建失败 — 未获取到向量")
            return None
        vecs = np.asarray(vecs, dtype="float32")
        faiss = get_faiss()
        faiss.normalize_L2(vecs)
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        # 写回磁盘
        index_path = store_dir / "index.faiss"
        faiss.write_index(index, str(index_path))
        print(f"[INFO] {product}: 索引自动重建成功 ({len(docs)} vectors, dim={dim})")
        return index
    except Exception as e:
        print(f"[WARN] {product}: 自动重建索引失败 — {e}")
        return None


def load_store(product: str):
    store_dir = STORE_ROOT / product
    index_path = store_dir / "index.faiss"
    docs_path = store_dir / "docs.jsonl"
    if not docs_path.exists():
        return None, []

    # 缓存键：docs.jsonl 的 mtime（即使没有 index.faiss 也能缓存 docs）
    mtime = docs_path.stat().st_mtime
    with _store_lock:
        cached = _store_cache.get(product)
        if cached and cached[2] == mtime:
            return cached[0], cached[1]

    # 加载文档（在锁外进行，IO 耗时）
    docs = []
    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 跳过损坏行，不中断整体加载

    # 加载向量索引（可选：无 index.faiss 时仅使用关键词检索）
    index = None
    if index_path.exists():
        try:
            index = get_faiss().read_index(str(index_path))
            # 检测索引与文档数不一致（索引过期未重建）→ 自动重建
            if index.ntotal != len(docs):
                print(f"[INFO] {product}: index.faiss has {index.ntotal} vectors but docs.jsonl has {len(docs)} records. 自动重建索引...")
                index = _auto_rebuild_index(product, docs, store_dir)
        except Exception:
            index = None

    with _store_lock:
        _store_cache[product] = (index, docs, mtime)
    return index, docs


def vector_search(product: str, query: str, top_k: int) -> List[Dict]:
    index, docs = load_store(product)
    if index is None or not docs:
        return []
    qv = embed_query(query)
    with _search_lock:
        scores, ids = index.search(qv, min(top_k, index.ntotal))
    hits = []
    for i, idx in enumerate(ids[0]):
        if idx < 0 or idx >= len(docs):
            continue
        d = dict(docs[idx])
        d["score"] = float(scores[0][i])
        hits.append(d)
    return hits


def read_knowledge_file(product: str, fname: str) -> str:
    p = KNOWLEDGE_DIR / product / fname
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def _is_product_dir(name: str) -> bool:
    """判断 knowledge/ 下的目录是否为产品目录（排除共享知识目录）"""
    from rag_runtime_config import SHARED_ENTITY_DIRS
    shared_names = set(SHARED_ENTITY_DIRS.values())
    p = KNOWLEDGE_DIR / name
    return p.is_dir() and (p / "main.txt").exists() and name not in shared_names


def detect_product(question: str) -> str:
    found = detect_terms(question, PRODUCT_ALIASES)
    if found:
        return found[0]
    if (KNOWLEDGE_DIR / "feiluoao").exists():
        return "feiluoao"
    if KNOWLEDGE_DIR.exists():
        dirs = [x.name for x in KNOWLEDGE_DIR.iterdir() if _is_product_dir(x.name)]
    else:
        dirs = []
    return dirs[0] if dirs else "feiluoao"


def _detect_special_intent(q: str) -> str:
    """检测无知识兜底的特殊意图：价格、对比、地点。返回意图名或空字符串。"""
    price_kws = ["多少钱", "价格", "费用", "贵不贵", "便宜", "一支多少", "疗程多少钱",
                 "收费", "报价", "花多少", "怎么收费", "优惠", "打折", "一次多少",
                 "一针多少", "总花费", "预算多少"]
    if any(k in q for k in price_kws):
        return "price"

    # 对比意图：需排除产品自身成分对比（如"PCL和透明质酸的作用"属于 ingredient）
    compare_kws = ["区别", "对比", "vs", "哪个好", "哪个更", "差别",
                   "不同点", "优劣", "比较", "哪里比.*好", "和.*区别",
                   "和.*哪个", "好还是"]
    # 排除成分层面的内部对比
    _internal_compare = ["PCL", "聚己内酯", "透明质酸", "成分"]
    if not any(ic in q for ic in _internal_compare):
        for k in compare_kws:
            if re.search(k, q):
                return "comparison"

    location_kws = ["哪里可以做", "哪家医院", "附近", "哪里有", "哪能做", "去哪",
                    "哪个城市", "北京能做", "上海能做", "哪里能打", "哪里做",
                    "哪个机构", "哪个诊所", "推荐医院"]
    if any(k in q for k in location_kws):
        return "location"
    return ""


def detect_route(question: str) -> str:
    q = (question or "").lower()
    # 跨实体路由优先于产品路由（更具体的先匹配）
    order = [
        # 跨实体路由
        "complication", "script", "procedure_q", "equipment_q",
        "anatomy_q", "indication_q", "course",
        # 产品路由
        "risk", "repair", "combo", "aftercare", "operation", "anti_fake",
        "contraindication", "design", "effect", "pre_care", "ingredient", "basic",
    ]

    # 收集每个 route 的匹配关键词
    matched = {}
    for route in order:
        hits = [kw for kw in QUESTION_ROUTES.get(route, []) if kw.lower() in q]
        if hits:
            matched[route] = hits

    if not matched:
        return "basic"

    # 消歧：禁忌适用性信号 → 优先 contraindication
    contra_signals = ["体质", "人群", "可以用", "可以打", "适合", "能用", "能打",
                      "能做", "可以做", "能不能", "能打吗", "可以吗"]
    has_contra_signal = any(s in q for s in contra_signals)

    if "contraindication" in matched and has_contra_signal:
        return "contraindication"
    if "risk" in matched and "contraindication" in matched and has_contra_signal:
        return "contraindication"

    # 消歧：联合方案信号 → 优先 combo（"一起做""搭配" 优先于 operation/aftercare）
    combo_signals = ["一起做", "联合", "搭配", "同做", "配合", "间隔多久"]
    if "combo" in matched and any(s in q for s in combo_signals):
        return "combo"

    # 消歧：修复意图信号 → 优先 repair（在症状消歧之前，因为"硬块需要修复吗"应归repair）
    repair_signals = ["修复", "补救", "返修", "重新做", "做坏", "做失败", "效果差"]
    if "repair" in matched and any(s in q for s in repair_signals):
        return "repair"

    # 消歧：complication vs risk — 术后时间线信号（"术后第N天""术后N天"）→ complication
    # 长期异常（"术后3个月""半年后"）→ risk
    temporal_short = re.search(r"术后(第?\d+天|当天|1-3天|一周|1周)", q)
    temporal_long = re.search(r"(术后\d+个月|半年|一年|长期)", q)
    if "complication" in matched and "risk" in matched:
        if temporal_long:
            return "risk"
        if temporal_short:
            return "complication"

    # 消歧：疼痛体感询问 — "疼不疼/痛不痛/疼吗" 是术前体感问题 → operation
    pain_inquiry = re.search(r"(疼不疼|痛不痛|疼吗|痛吗|会不会疼|会不会痛)", q)
    if pain_inquiry and "operation" in matched:
        return "operation"

    # 消歧：术后症状 → risk 优先于 aftercare/operation
    symptom_kws = ["红肿", "肿胀", "肿", "硬块", "结节", "疼痛", "感染", "淤青", "瘀青",
                   "发紫", "发黑", "红疹", "疹子", "痒", "化脓", "溃烂", "坏死", "不消",
                   "越来越"]
    if "risk" in matched and ("aftercare" in matched or "operation" in matched):
        if any(s in q for s in symptom_kws):
            return "risk"

    # 消歧：疼痛体感 — "打的时候疼"是 operation（术中），"打完疼"是 risk（术后异常）
    if "risk" in matched and "operation" in matched:
        pre_pain = re.search(r"(打的时候|注射时|术中|操作中).{0,4}(疼|痛)", q)
        if pre_pain:
            return "operation"

    # 消歧：生活限制问题 → aftercare 优先（"能运动吗""能化妆吗"）
    lifestyle_kws = ["运动", "健身", "游泳", "桑拿", "化妆", "上妆", "防晒", "晒太阳",
                     "洗澡", "喝酒", "饮酒", "上班", "出汗", "泡澡", "汗蒸"]
    if "aftercare" in matched and any(s in q for s in lifestyle_kws):
        return "aftercare"

    # 消歧：疗程规划 vs 操作参数（"疗程" 同时出现在 operation 和 course 中）
    course_signals = ["安排", "规划", "几次", "间隔多久", "总共", "多长时间",
                      "时间表", "多少钱", "费用", "预算"]
    if "course" in matched and "operation" in matched:
        if any(s in q for s in course_signals):
            return "course"
        return "operation"

    # 消歧：部位/适应症 vs 方案设计（"怎么打""打哪里""几支" 是设计信号）
    design_signals = ["怎么打", "打哪里", "几支", "怎么设计", "方案", "用量"]
    if ("anatomy_q" in matched or "indication_q" in matched) and "design" in matched:
        if any(s in q for s in design_signals):
            return "design"

    # 按优先级返回第一个命中的 route
    for route in order:
        if route in matched:
            return route
    return "basic"


def _truncate_to_sentence(text: str, max_chars: int = 450) -> str:
    """截断文本到最近的句子边界，避免截断关键信息"""
    if len(text) <= max_chars:
        return text
    # 在 max_chars 范围内找最后一个句子结束符
    truncated = text[:max_chars]
    # 中文和英文句子结束符
    for sep in ["。", "；", "！", "？", ". ", "! ", "? ", "\n"]:
        pos = truncated.rfind(sep)
        if pos > max_chars // 3:  # 至少保留 1/3 内容
            return truncated[:pos + len(sep)].strip()
    return truncated.strip()


def build_evidence(hits: List[Dict]) -> List[Dict]:
    ev = []
    for h in hits[:MAX_EVIDENCE_CHUNKS]:
        ev.append({
            "meta": h.get("meta", {}),
            "text": _truncate_to_sentence((h.get("text") or "").strip()),
        })
    return ev


def parse_anti_fake(main_text: str, faq_text: str, mode: str) -> List[str]:
    rule = SECTION_RULES["anti_fake"]
    block = section_block(main_text, rule["titles"], rule["stops"])
    if not block:
        block = section_block(faq_text, ["防伪", "HiddenTag"], [])
    if not block:
        return []

    lines = [ln for ln in normalize_lines(block) if not is_faq_line(ln)]
    subject, official, notes = [], [], []
    steps = {i: [] for i in range(1, 6)}
    current = None
    in_notes = False

    for ln in lines:
        if "防伪验证主体" in ln:
            current = None
            in_notes = False
            continue
        if "官方验证方式" in ln:
            current = None
            in_notes = False
            continue
        if "【防伪步骤】" in ln:
            current = None
            in_notes = False
            continue
        if "【防伪注意事项】" in ln:
            current = None
            in_notes = True
            continue

        m = re.match(r"STEP\s*(\d+)", ln, re.I)
        if m:
            current = int(m.group(1))
            in_notes = False
            continue

        clean = ln.lstrip("-").strip()
        if not clean:
            continue

        if in_notes:
            notes.append(clean)
            continue

        if current in steps:
            steps[current].append(clean)
            continue

        if "G-international" in clean and ("公司" in clean or len(clean) <= 40):
            subject.append(clean)
            continue

        # 官方验证方式只收核心句，避免把 step1 吃掉
        if ("HiddenTag APP 扫描" in clean) or ("扫码方式无效" in clean) or ("其他扫码方式无效" in clean):
            official.append(clean)
            continue

    # FAQ 兜底补缺
    faq_lines = normalize_lines(faq_text)
    if not subject:
        for ln in faq_lines:
            if "G-international" in ln and ("官方认证" in ln or "公司" in ln):
                subject.append("韩国(株)G-international 公司")
                break

    if not official:
        for x in ["使用 HiddenTag APP 扫描验证", "其他扫码方式无效（资料描述）"]:
            official.append(x)

    # 步骤硬兜底：避免 step 消失
    defaults = {
        1: ["在手机应用商店下载 HiddenTag APP"],
        2: ["打开 APP，点击“正品认证”"],
        3: ["肉眼确认产品标签是否为正品标签"],
        4: ["扫描产品上的 HiddenTag 标签", "建议在不反光环境下扫描，提高识别成功率"],
        5: ["验证成功后，APP 显示韩国(株)G-international 官方认证图片"],
    }
    for i in range(1, 6):
        if not steps[i]:
            steps[i] = defaults[i][:]

    if not notes:
        notes = ["仅 HiddenTag APP 可用于验证", "标签保持平整、避免反光", "以官方认证结果为准"]

    out = ["防伪验证主体："]
    for x in uniq(subject or ["韩国(株)G-international 公司"]):
        out.append(x)
    out.append("官方验证方式：")
    for x in uniq(official):
        out.append(x)
    out.append("【防伪步骤】")
    for i in range(1, 6):
        out.append(f"STEP {i}：")
        lim = 1 if mode == "brief" and i in (1, 2, 3, 5) else 2
        for x in uniq(steps[i])[:lim]:
            out.append(x)
    out.append("【防伪注意事项】")
    for x in uniq(notes)[:(2 if mode == "brief" else 6)]:
        out.append(x)
    return out


def _accept_line(clean: str, route: str) -> bool:
    """判断一行是否应保留——按路由类型使用不同策略"""
    # 小节标题始终保留
    if re.match(r"^\d+[）\)]", clean):
        return True
    # 分隔线、纯标记跳过
    if set(clean) <= {"=", "-", "_", " ", "—"}:
        return False

    # 各路由的关键词白名单
    route_keywords = {
        "aftercare": ["术后", "洗脸", "辛辣", "禁酒", "面膜", "保湿", "熬夜", "按摩",
                       "洁面仪", "多喝水", "水果", "蔬菜", "冰敷", "清洁", "饮食",
                       "睡眠", "面霜", "生活", "运动", "健身", "出汗", "化妆", "上妆",
                       "卸妆", "防晒", "SPF", "紫外线", "日晒", "洗澡", "水温",
                       "桑拿", "汗蒸", "泡澡", "游泳", "泳池", "上班", "工作",
                       "海鲜", "喝酒", "饮酒", "避免", "恢复"],
        "operation": ["针头", "深度", "注射", "0.8", "1.0", "0.3ml", "2cm", "MTS",
                      "水光", "涂抹", "微针", "仪器", "全脸", "进针", "间距",
                      "用量", "方式", "中胚层", "安全声明", "专业人员", "不建议",
                      "麻醉", "敷麻", "疼", "痛", "耐受", "分钟", "操作"],
        "contraindication": ["免疫", "妊娠", "哺乳", "过敏", "18", "风湿", "皮肤疾病",
                             "感染", "炎症", "禁忌", "敏感", "年龄", "男性", "性别",
                             "自行", "自己", "在家", "抗凝", "阿司匹林", "停药",
                             "未成年", "评估", "医生"],
        "risk": ["红肿", "疼痛", "结节", "硬块", "感染", "过敏", "淤青", "肿胀",
                 "冰敷", "就医", "反应", "处理", "缓解", "发热", "化脓",
                 "异常", "严重", "正规医疗", "专业医师", "注意事项",
                 "消退", "正常", "天", "瘀青", "发紫", "疹子", "红疹", "痒",
                 "坏死", "溃烂", "不消", "越来越", "色沉"],
        "combo": ["联合", "搭配", "间隔", "水光", "微针", "光电", "填充", "同日",
                  "恢复", "建议", "不建议", "周"],
        "ingredient": ["PCL", "聚己内酯", "透明质酸", "玻尿酸", "谷胱甘肽", "肽",
                       "生长因子", "矿物质", "聚乙二醇", "胶原", "抗氧化", "修复",
                       "再生", "保湿", "提升", "弹性"],
        "effect": ["效果", "维持", "见效", "胶原", "再生", "疗程", "持续",
                   "显现", "最佳", "差异", "防晒", "保湿", "护理", "因人而异",
                   "肤质", "年龄", "生活习惯", "注意事项"],
        "pre_care": ["术前", "检查", "病史", "过敏史", "用药", "A酸", "果酸",
                     "饮酒", "抗凝", "素颜", "沟通", "知情", "同意书", "费用",
                     "方案", "准备", "评估", "皮肤状态", "感染", "炎症",
                     "妊娠", "哺乳", "禁忌", "风险", "注意事项"],
        "design": ["设计", "方案", "评估", "松弛", "皱纹", "轮廓", "法令纹",
                   "下颌", "苹果肌", "毛孔", "弹性", "含水量", "用量", "支",
                   "导入", "注射", "全脸", "区域", "疗程", "间隔", "保守",
                   "安全声明", "仅供专业人员", "治疗史"],
        "repair": ["修复", "补救", "不理想", "不均匀", "不对称", "结节", "硬块",
                   "吸收", "层次", "自行", "按压", "揉捏", "间隔", "医生",
                   "评估", "安全", "循序渐进", "记录", "效果差",
                   "既往", "注射项目", "效果不佳", "注意事项"],
        # 跨实体路由
        "complication": ["并发症", "红肿", "结节", "硬块", "感染", "过敏", "淤青",
                         "坏死", "栓塞", "正常", "异常", "就医", "复诊", "观察",
                         "警惕", "急诊", "分级", "处理", "恢复", "冰敷", "消退",
                         "术后", "天", "周", "当天"],
        "course": ["疗程", "次", "间隔", "周期", "总共", "持续", "规划", "时间表",
                   "周", "月", "方案", "维护", "效果", "预算", "费用"],
        "anatomy_q": ["部位", "区域", "额头", "额部", "眼周", "苹果肌", "法令纹",
                      "下颌线", "颈部", "鼻部", "手部", "手背", "面部", "分区",
                      "推荐项目", "常见问题", "注意事项"],
        "indication_q": ["松弛", "干燥", "毛孔", "色斑", "痘坑", "皱纹", "缺水",
                         "粗糙", "暗沉", "细纹", "改善", "推荐", "适合", "年龄",
                         "敏感肌", "油性", "屏障", "备孕", "孕期"],
        "procedure_q": ["项目", "操作", "流程", "对比", "区别", "优势", "原理",
                        "适用", "适应症", "效果", "疗程", "搭配", "水光针",
                        "微针", "光电", "填充", "射频", "激光"],
        "equipment_q": ["仪器", "设备", "机器", "参数", "针头", "深度", "功能",
                        "兼容", "适配", "品牌", "维护", "水光仪", "微针仪"],
        "script": ["话术", "解释", "回答", "沟通", "顾虑", "说法", "合规",
                   "介绍", "客户", "担心", "推销", "预期"],
    }

    keywords = route_keywords.get(route)
    if keywords:
        return any(k in clean for k in keywords)

    # basic 路由：保留产品属性相关的行
    basic_keywords = ["产品", "品牌", "备案", "规格", "形态", "保质期", "储存",
                      "适用", "肤质", "松弛", "下垂", "缺水", "细纹", "紧致",
                      "FILLOUP", "菲罗奥", "赛罗菲", "CELLOFILL",
                      "ml", "瓶", "液体", "常温"]
    if any(k in clean for k in basic_keywords):
        return True
    return len(clean) <= 60


def parse_bullets_from_section(main_text: str, faq_text: str, route: str, mode: str) -> List[str]:
    rule = SECTION_RULES.get(route)
    if not rule:
        return []
    block = section_block(main_text, rule["titles"], rule["stops"])
    if not block:
        block = section_block(faq_text, rule["titles"], [])
    if not block:
        return []

    lines = [ln for ln in normalize_lines(block) if not is_faq_line(ln)]
    items = []
    for ln in lines:
        clean = ln.lstrip("-").strip()
        if not clean:
            continue
        if _accept_line(clean, route):
            items.append(clean)

    items = uniq(items)

    # 路由特定后处理
    if route == "contraindication":
        items = [x for x in items if not ("术后一周内不要" in x or "洁面仪" in x)]
        if "具体是否适用需由专业医生评估。" not in items:
            items.append("具体是否适用需由专业医生评估。")

    if route == "risk":
        if not any("医" in x for x in items):
            items.append("术后如有任何异常，请及时联系操作医生。")

    # 各路由的条目数限制 (brief, full)
    limits = {
        "aftercare":       (10, 28),
        "operation":       (10, 24),
        "contraindication": (8,  18),
        "risk":            (8,  20),
        "combo":           (8,  16),
        "ingredient":      (10, 24),
        "basic":           (10, 22),
        "effect":          (8,  18),
        "pre_care":        (8,  18),
        "design":          (10, 24),
        "repair":          (10, 22),
    }
    brief_lim, full_lim = limits.get(route, (8, 20))
    limit = brief_lim if mode == "brief" else full_lim

    return items[:limit]


# 共享路由 → 共享知识目录名 的映射
_SHARED_ROUTE_DIR = {
    "complication": "complications",
    "course":       "courses",
    "anatomy_q":    "anatomy",
    "indication_q": "indications",
    "procedure_q":  "procedures",
    "equipment_q":  "equipment",
    "script":       "scripts",
}


def _read_shared_knowledge(dir_name: str) -> str:
    """读取共享知识目录内容：单文件直接读，多实例拼接所有 main.txt"""
    shared_dir = KNOWLEDGE_DIR / dir_name
    if not shared_dir.exists():
        return ""
    main_file = shared_dir / "main.txt"
    if main_file.exists():
        return main_file.read_text(encoding="utf-8")
    # 多实例目录：拼接所有子目录的 main.txt
    parts = []
    for inst in sorted(shared_dir.iterdir()):
        if inst.is_dir() and (inst / "main.txt").exists():
            parts.append(inst.joinpath("main.txt").read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def parse_answer(route: str, product: str, mode: str) -> List[str]:
    # 共享路由：从共享知识目录读取
    shared_dir_name = _SHARED_ROUTE_DIR.get(route)
    if shared_dir_name:
        main_text = _read_shared_knowledge(shared_dir_name)
        faq_text = ""
    else:
        main_text = read_knowledge_file(product, "main.txt")
        faq_text = read_knowledge_file(product, "faq.txt")

    if route == "anti_fake":
        return parse_anti_fake(main_text, faq_text, mode)
    return parse_bullets_from_section(main_text, faq_text, route, mode)


def _extract_faq_from_hits(hits: List[Dict], question: str) -> List[str]:
    """从检索结果中提取与问题高度相关的 FAQ 回答。
    当 FAQ 条目的 Q 与用户问题有 bigram 重叠时，直接提取 A 的内容。"""
    q_lower = question.lower().replace(" ", "")
    if len(q_lower) < 2:
        return []
    # 用 bigram（2字组合）做模糊匹配，解决中文不分词问题
    q_bigrams = set(q_lower[i:i+2] for i in range(len(q_lower) - 1))

    faq_candidates = []
    for h in hits:
        meta = h.get("meta", {})
        if meta.get("source_type") != "faq":
            continue
        text = (h.get("text") or "").strip()
        if "【Q】" not in text or "【A】" not in text:
            continue
        q_part = text.split("【A】")[0].replace("【Q】", "").lower().replace(" ", "")
        # 计算 FAQ 问题与用户问题的 bigram 重叠率
        faq_bigrams = set(q_part[i:i+2] for i in range(len(q_part) - 1))
        if not faq_bigrams:
            continue
        overlap = len(q_bigrams & faq_bigrams)
        # 用比率而非绝对数量：重叠 bigram 占用户问题 bigram 的比例 ≥30%
        # 同时要求至少 3 个重叠（避免极短问题误匹配）
        ratio = overlap / max(len(q_bigrams), 1)
        if overlap >= 3 and ratio >= 0.3:
            a_part = text.split("【A】")[1].strip()
            if a_part:
                faq_candidates.append((ratio, a_part))
    # 按重叠率排序，取最相关的
    faq_candidates.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in faq_candidates[:2]]


def _try_faq_fast_path(hits: List[Dict], question: str, route: str,
                       rewrite: dict, log_meta: dict) -> str:
    """FAQ 精确匹配快速路径：当检索结果中有高置信度 FAQ 条目时直接返回。

    触发条件（全部满足才走快速路径）：
    1. 检索结果 top-1 是 FAQ 类型 chunk
    2. FAQ 的 Q 与用户问题 bigram 重叠率 ≥50%（高置信度）
    3. 检索分数 ≥0.40（排除低分噪音命中）

    返回格式化答案或空字符串（不满足条件时）。
    """
    if not hits:
        return ""

    top_hit = hits[0]
    # 条件1：top-1 必须是 FAQ
    meta = top_hit.get("meta", {})
    if meta.get("source_type") != "faq":
        return ""
    # 条件3：检索分数门槛
    score = top_hit.get("hybrid_score", top_hit.get("score", 0.0))
    if score < 0.40:
        return ""

    text = (top_hit.get("text") or "").strip()
    if "【Q】" not in text or "【A】" not in text:
        return ""

    q_part = text.split("【A】")[0].replace("【Q】", "").lower().replace(" ", "")
    a_part = text.split("【A】")[1].strip()
    if not a_part:
        return ""

    # 条件2：bigram 高重叠率
    q_lower = question.lower().replace(" ", "")
    q_bigrams = set(q_lower[i:i+2] for i in range(len(q_lower) - 1))
    faq_bigrams = set(q_part[i:i+2] for i in range(len(q_part) - 1))
    if not q_bigrams or not faq_bigrams:
        return ""
    overlap = len(q_bigrams & faq_bigrams)
    ratio = overlap / max(len(q_bigrams), 1)
    if overlap < 3 or ratio < 0.50:
        return ""

    # 构建答案
    body_lines = [a_part]
    evidence = build_evidence(hits[:1])
    add_risk = route in ("risk", "complication", "contraindication")
    answer = format_structured_answer(route, body_lines, evidence, add_risk_note=add_risk)

    log_qa(question, answer, rewritten_query=rewrite.get("expanded", ""),
           matched_sources=evidence, hit=True,
           meta={**log_meta, "method": "faq_fast_path", "faq_score": score,
                 "faq_overlap_ratio": round(ratio, 3)})
    return answer


def _build_context(hits: List[Dict], max_chars: int = 3000) -> str:
    """将检索结果拼接为 LLM context 字符串，按完整 chunk 粒度截断"""
    parts = []
    total = 0
    for i, h in enumerate(hits, 1):
        text = (h.get("text") or "").strip()
        if not text:
            continue
        source = h.get("meta", {}).get("source_file", "unknown")
        chunk_id = h.get("meta", {}).get("chunk_id", "?")
        score = h.get("hybrid_score", h.get("score", 0.0))
        header = f"[片段{i} | {source}#{chunk_id} | 相关度:{score:.2f}]"
        part = f"{header}\n{text}"
        # 至少保留 1 个片段；之后按完整 chunk 粒度截断
        if parts and total + len(part) > max_chars:
            break
        parts.append(part)
        total += len(part)
    return "\n\n".join(parts)


def _get_openai_client():
    """获取 OpenAI client，失败返回 None"""
    if not USE_OPENAI:
        return None
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=key)
    except Exception:
        return None


def llm_generate_answer(question: str, context: str, route: str, mode: str,
                        history_summary: str = "",
                        history_pairs: list = None) -> str:
    """基于检索 context 用 LLM 生成答案（真正的 RAG）。

    当 history_pairs 非空时，将完整 Q&A 对纳入 prompt，帮助 LLM 理解
    用户在多轮对话中的真实意图。history_summary 作为简洁的话题脉络辅助。
    """
    client = _get_openai_client()
    if client is None:
        return ""

    length_hint = "简洁扼要，控制在300字以内" if mode == "brief" else "详细全面，可适当展开"
    route_hints = {
        "risk": "重点说明可能的不良反应、处理建议，并提醒需医生评估。",
        "aftercare": "按时间线整理术后护理要点。",
        "operation": "重点说明操作参数（深度、剂量、间距等）。",
        "anti_fake": "按步骤说明防伪验证方法。",
        "contraindication": "列出禁忌人群和情况，提醒需医生评估。",
        "combo": "说明联合方案和间隔时间。",
        "basic": "介绍产品基本信息。",
        "effect": "说明效果起效时间、维持时间和影响因素。",
        "pre_care": "列出术前需要做的准备和注意事项。",
        "design": "说明方案设计的要点和评估维度，提醒需医生面诊制定。",
        "repair": "说明修复和补救的思路，强调需医生评估，不要自行处理。",
    }

    history_block = ""
    if history_pairs:
        # 完整 Q&A 对让 LLM 理解对话全貌
        pairs_text = "\n".join(
            f"   用户：{p['user']}\n   助手：{p['assistant']}"
            for p in history_pairs
        )
        history_block = (
            "7. 以下是之前的对话记录，请结合上下文理解用户当前问题的真实意图，\n"
            "   不要重复回答用户已经问过的内容，聚焦当前问题：\n"
            f"{pairs_text}\n"
        )
    elif history_summary:
        # 降级：只有问题摘要时用箭头形式
        history_block = (
            "7. 用户之前的对话脉络如下，请结合对话上下文理解用户当前问题的真实意图，\n"
            "   不要重复回答用户已经问过的内容，聚焦当前问题：\n"
            f"   对话脉络：「{history_summary}」\n"
        )

    system_prompt = (
        "你是一位医美产品知识库问答助手。请严格基于以下检索到的知识库片段回答用户问题。\n"
        "规则：\n"
        "1. 只使用知识库中的信息，不要编造或补充任何事实\n"
        '2. 如果知识库中没有相关信息，明确说明"当前知识库未覆盖该问题"\n'
        "3. 回答使用结构化格式（分点列出）\n"
        '4. 末尾加上"以上信息仅供参考，具体请咨询专业医师。"\n'
        f"5. 回答要求：{length_hint}\n"
        f"6. {route_hints.get(route, '')}\n"
        f"{history_block}"
    )

    user_prompt = f"知识库检索结果：\n{context}\n\n用户问题：{question}"

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS_BRIEF if mode == "brief" else LLM_MAX_TOKENS_FULL,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def openai_rewrite_answer(text: str, route: str) -> str:
    """Fallback: 当 LLM RAG 未启用时，用 LLM 润色规则提取的答案"""
    client = _get_openai_client()
    if client is None:
        return text
    try:
        prompt = (
            "请在不改变事实的前提下，将以下基于知识库的回答整理得更专业、更自然。"
            "不要新增事实。保留结构化格式。\n\n" + text
        )
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS_BRIEF,
        )
        return (resp.choices[0].message.content or "").strip() or text
    except Exception:
        return text


def _fallback_from_hits(hits: List[Dict], max_lines: int = 8,
                        query: str = "") -> List[str]:
    """当规则提取失败时，从检索结果中提取文本作为 fallback 答案。
    优先选择包含查询关键词的行，其次取各 chunk 的前几行。"""
    query_terms = [t for t in re.split(r"[\s,，;；、？?！!。]+", query.lower()) if len(t) >= 2]
    priority_lines = []
    other_lines = []
    for h in hits:
        text = (h.get("text") or "").strip()
        if not text:
            continue
        for ln in text.split("\n"):
            ln = ln.strip()
            if not ln or len(ln) <= 4:
                continue
            ln_lower = ln.lower()
            if query_terms and any(t in ln_lower for t in query_terms):
                priority_lines.append(ln)
            else:
                other_lines.append(ln)
    # 关键词匹配行优先，再补充其他行
    result = uniq(priority_lines + other_lines)
    return result[:max_lines]


def answer_one(question: str, mode: str, rewrite: dict = None,
               route_override: str = "") -> str:
    product = detect_product(question)
    route = route_override or detect_route(question)
    if rewrite is None:
        rewrite = rewrite_query(question)

    # 日志基础 meta：包含上下文补全信息便于生产调试
    raw_input = rewrite.get("raw_input", "")
    _log_meta = {"product": product, "route": route, "mode": mode}
    if raw_input and raw_input != rewrite.get("original", ""):
        _log_meta["raw_input"] = raw_input
        _log_meta["resolved_question"] = rewrite["original"]

    # 根据问题类型使用不同的检索参数
    route_cfg = QUESTION_TYPE_CONFIG.get(route, {})
    route_top_k = route_cfg.get("k", DEFAULT_TOP_K)
    route_threshold = route_cfg.get("threshold", 0.30)

    # 向量检索用 search_query（去除纠正前缀等噪音，语义更聚焦）
    # 关键词检索用扩展查询（别名/同义词有助于 term 匹配）
    search_q = rewrite.get("search_query", rewrite["original"])

    # 决定搜索哪些 store
    search_product = route not in _SHARED_ROUTES or route in _HYBRID_ENTITY_ROUTES
    search_shared = route in _SHARED_ROUTES

    vector_hits, keyword_hits = [], []
    if search_product:
        vector_hits = vector_search(product, search_q, VECTOR_TOP_K)
        _, docs = load_store(product)
        keyword_hits = keyword_search(rewrite["expanded"], docs, KEYWORD_TOP_K) if docs else []
    if search_shared:
        shared_v = vector_search("_shared", search_q, VECTOR_TOP_K)
        _, shared_docs = load_store("_shared")
        shared_kw = keyword_search(rewrite["expanded"], shared_docs, KEYWORD_TOP_K) if shared_docs else []
        vector_hits = vector_hits + shared_v
        keyword_hits = keyword_hits + shared_kw

    # 路由感知权重：精确参数类问题提高关键词权重
    vw = route_cfg.get("vw", HYBRID_VECTOR_WEIGHT)
    kw = route_cfg.get("kw", HYBRID_KEYWORD_WEIGHT)
    hits = merge_hybrid(vector_hits, keyword_hits, vw, kw, route_top_k, route=route) if (vector_hits or keyword_hits) else []
    # 过滤低于 threshold 的结果
    hits = [h for h in hits if h.get("hybrid_score", h.get("score", 0.0)) >= route_threshold]

    # ---- 策略0: FAQ 精确匹配快速路径 ----
    # 当检索结果中有高置信度 FAQ 条目与问题高度吻合时，直接返回 FAQ 回答，
    # 跳过 LLM/规则提取，提升常见问题的响应质量和速度。
    if hits:
        faq_answer = _try_faq_fast_path(hits, question, route, rewrite, _log_meta)
        if faq_answer:
            return faq_answer

    # ---- 策略1: LLM RAG（优先）——检索结果作为 context 让 LLM 生成答案 ----
    if hits and USE_OPENAI:
        context = _build_context(hits)
        if context:
            history_summary = rewrite.get("history_summary", "") if rewrite else ""
            history_pairs = rewrite.get("history_pairs", []) if rewrite else []
            llm_answer = llm_generate_answer(question, context, route, mode,
                                             history_summary=history_summary,
                                             history_pairs=history_pairs)
            if llm_answer:
                log_qa(question, llm_answer, rewritten_query=rewrite["expanded"],
                       matched_sources=build_evidence(hits), hit=True,
                       meta={**_log_meta, "method": "llm_rag"})
                return llm_answer

    # ---- 策略2: 规则提取（Fallback）——从知识库文档中按章节规则提取条目 ----
    body_lines = parse_answer(route, product, mode)

    # 补充 FAQ 命中：如果检索结果中有 FAQ 条目与查询高度相关，追加到答案中
    if body_lines and hits:
        faq_supplement = _extract_faq_from_hits(hits, question)
        if faq_supplement:
            body_lines = faq_supplement + [""] + body_lines

    # 关联数据补充：从 relations.json 中提取跨实体信息
    relation_lines = relation_enrich(route, product, question)
    if relation_lines:
        body_lines = body_lines + ["", "【关联信息】"] + relation_lines[:6]

    if not body_lines:
        # 规则也提取失败，从检索结果中摘要
        fallback_lines = _fallback_from_hits(hits, query=question)
        if fallback_lines:
            body_lines = fallback_lines
        else:
            fallback = [
                "当前知识库未覆盖该问题的直接结论。",
                "可确认方向：请核对产品主文档、FAQ 或补充对应知识库章节。",
                "该问题可能涉及医生判断范围，建议由专业医师评估。",
            ]
            text = format_structured_answer(route, fallback, build_evidence(hits), add_risk_note=(route == "risk"))
            log_qa(question, text, rewritten_query=rewrite["expanded"],
                   matched_sources=build_evidence(hits), hit=False,
                   meta={**_log_meta, "method": "no_hit"})
            return text

    text = format_structured_answer(route, body_lines, build_evidence(hits), add_risk_note=(route == "risk"))
    if USE_OPENAI:
        text = openai_rewrite_answer(text, route)
    log_qa(question, text, rewritten_query=rewrite["expanded"],
           matched_sources=build_evidence(hits), hit=True,
           meta={**_log_meta, "method": "rule_extract"})
    return text


_NO_MATCH_REPLY = "抱歉，暂时无法回答该问题。请尝试询问产品成分、术后护理、禁忌人群等相关问题。"

# 非提问的礼貌回复映射
_CHITCHAT_REPLIES = {
    "greeting": "您好！我是医美产品知识库助手，请问有什么可以帮您的？",
    "thanks":   "不客气！如有其他问题，随时可以继续问我。",
    "bye":      "再见！祝您一切顺利。",
    "ack":      "好的，如有其他问题请继续提问。",
}

def _chitchat_reply(raw: str) -> str:
    """根据非提问输入类型返回礼貌回复"""
    if re.match(r"^(你好|嗨|hi|hello|hey)$", raw, re.IGNORECASE):
        return _CHITCHAT_REPLIES["greeting"]
    if re.match(r"^(谢谢|感谢|多谢|辛苦了)$", raw):
        return _CHITCHAT_REPLIES["thanks"]
    if re.match(r"^(再见|拜拜|bye)$", raw, re.IGNORECASE):
        return _CHITCHAT_REPLIES["bye"]
    return _CHITCHAT_REPLIES["ack"]


def _detect_route_with_history(question: str, rewrite: dict) -> str:
    """路由检测：优先用当前问题检测，若结果为 basic 且有历史上下文，
    尝试从历史中继承更精确的路由。

    典型场景：用户先问"菲罗奥术后注意什么"(aftercare)，再问"还有别的吗"，
    当前问题被补全为"菲罗奥 还有别的吗"，路由检测可能落入 basic，
    但用户真实意图是继续问 aftercare。

    使用 last_routed_q（最近一条含路由关键词的历史问题）做路由继承，
    而非 last_user_q（可能是"还有吗"这类无路由词的追问），
    解决连续追问链（Q1:aftercare → Q2:"还有吗" → Q3:"还有别的吗"）
    中路由丢失的问题。
    """
    route = detect_route(question)
    if route != "basic":
        return route

    # 当前路由为 basic 且发生了上下文补全 → 用最近含路由关键词的问题的路由
    if rewrite.get("context_resolved"):
        # 优先用 last_routed_q（含路由关键词的问题），回退到 last_user_q
        routed_q = rewrite.get("last_routed_q") or rewrite.get("last_user_q", "")
        if routed_q:
            history_route = detect_route(routed_q)
            if history_route != "basic":
                return history_route

    return route


def answer_question(question: str, mode: str, history: list = None,
                    rewrite: dict = None) -> str:
    q = (question or "").strip()
    if not q:
        return _NO_MATCH_REPLY
    if rewrite is None:
        rewrite = rewrite_query(q, history=history)

    # 非提问快速路径：问候/致谢/确认等直接返回礼貌回复，跳过检索
    if rewrite.get("is_chitchat"):
        reply = _chitchat_reply(rewrite.get("raw_input", q))
        log_qa(q, reply, rewritten_query="", matched_sources=[], hit=False,
               meta={"method": "chitchat"})
        return reply

    # 特殊意图快速路径：价格/对比/地点等无知识覆盖的问题
    special = _detect_special_intent(q)
    if special:
        _SPECIAL_REPLIES = {"price": PRICE_REPLY, "comparison": COMPARISON_REPLY,
                            "location": LOCATION_REPLY}
        reply = _SPECIAL_REPLIES.get(special, "")
        if reply:
            log_qa(q, reply, rewritten_query="", matched_sources=[], hit=False,
                   meta={"method": "special_intent", "intent": special})
            return reply

    outputs = []
    seen_routes = set()
    for subq in rewrite["sub_questions"][:MAX_SUB_QUESTIONS]:
        # 如果子问题与原问题相同，复用已有的 rewrite 结果；
        # 否则用 history 重新 rewrite，使子问题也能继承上下文
        # （如 "成分是什么？禁忌人群呢？" 拆分后 "禁忌人群呢" 需要产品名）
        sub_rewrite = rewrite if subq == rewrite["original"] else rewrite_query(subq, history=history)
        route = _detect_route_with_history(subq, sub_rewrite)
        # 同路由的子问题只回答一次（避免重复检索相同 chunk）
        if route in seen_routes:
            continue
        seen_routes.add(route)
        ans = answer_one(subq, mode, rewrite=sub_rewrite, route_override=route)
        key = ans.strip()
        if key:
            outputs.append(ans)
    return "\n\n".join(outputs) if outputs else _NO_MATCH_REPLY


def main():
    if len(sys.argv) < 2:
        print('Usage: python rag_answer.py "你的问题" [k] [brief|full]')
        return
    question = sys.argv[1].strip()
    mode = DEFAULT_MODE
    if len(sys.argv) >= 4 and sys.argv[3].strip() in ("brief", "full"):
        mode = sys.argv[3].strip()
    elif len(sys.argv) >= 3 and sys.argv[2].strip() in ("brief", "full"):
        mode = sys.argv[2].strip()

    ans = answer_question(question, mode)
    save_answer(ans)
    print("\n===== Answer saved =====")
    print(f"Saved to: {_get_out_path()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        save_answer("ERROR: " + repr(e))
        print("\n===== Answer saved =====")
        print(f"Saved to: {_get_out_path()}")
