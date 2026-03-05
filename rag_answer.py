import os
import sys
import json
import re
import threading
from pathlib import Path
from typing import List, Dict, Tuple

os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

from rag_runtime_config import (
    KNOWLEDGE_DIR, STORE_ROOT, OUT_PATH, DEFAULT_MODE, DEFAULT_TOP_K,
    USE_OPENAI, OPENAI_MODEL, DEBUG, QUESTION_ROUTES, SECTION_RULES,
    PRODUCT_ALIASES, PROJECT_ALIASES, VECTOR_TOP_K, KEYWORD_TOP_K,
    HYBRID_VECTOR_WEIGHT, HYBRID_KEYWORD_WEIGHT, QUESTION_TYPE_CONFIG
)
from search_utils import (
    normalize_text, normalize_lines, uniq, is_faq_line, section_block,
    keyword_search, merge_hybrid, detect_terms
)
from query_rewrite import rewrite_query
from answer_formatter import format_structured_answer
from rag_logger import log_qa

_model = None
_faiss = None
_BGEM3 = None
_store_cache = {}  # {product: (index, docs, mtime)} — 进程内缓存，避免每次请求重读文件
_search_lock = threading.Lock()  # 保护 FAISS index.search（非线程安全）
MAX_SUB_QUESTIONS = 4  # 单次问答最多拆分的子问题数
MAX_EVIDENCE_CHUNKS = 6  # build_evidence 保留的最大片段数


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
        _model = get_bg_cls()("BAAI/bge-m3", use_fp16=False)
    return _model


def embed_query(text: str) -> np.ndarray:
    model = get_model()
    out = model.encode([text], batch_size=1, max_length=1024)
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


def load_store(product: str):
    store_dir = STORE_ROOT / product
    index_path = store_dir / "index.faiss"
    docs_path = store_dir / "docs.jsonl"
    if not index_path.exists() or not docs_path.exists():
        return None, []

    # 按 index.faiss 的修改时间判断是否需要重新加载
    mtime = index_path.stat().st_mtime
    cached = _store_cache.get(product)
    if cached and cached[2] == mtime:
        return cached[0], cached[1]

    index = get_faiss().read_index(str(index_path))
    docs = []
    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    _store_cache[product] = (index, docs, mtime)
    return index, docs


def vector_search(product: str, query: str, top_k: int) -> List[Dict]:
    index, docs = load_store(product)
    if index is None or not docs:
        return []
    qv = embed_query(query)
    with _search_lock:
        scores, ids = index.search(qv, min(top_k, len(docs)))
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


def detect_product(question: str) -> str:
    found = detect_terms(question, PRODUCT_ALIASES)
    if found:
        return found[0]
    if (KNOWLEDGE_DIR / "feiluoao").exists():
        return "feiluoao"
    dirs = [x.name for x in KNOWLEDGE_DIR.iterdir() if x.is_dir()] if KNOWLEDGE_DIR.exists() else []
    return dirs[0] if dirs else "feiluoao"


def detect_route(question: str) -> str:
    q = (question or "").lower()
    order = ["risk", "combo", "aftercare", "operation", "anti_fake", "contraindication", "ingredient", "basic"]

    # 收集每个 route 的匹配关键词
    matched = {}
    for route in order:
        hits = [kw for kw in QUESTION_ROUTES.get(route, []) if kw.lower() in q]
        if hits:
            matched[route] = hits

    if not matched:
        return "basic"

    # 消歧：当 risk 和 contraindication 同时命中时，
    # 含 "体质/人群/可以用/可以打/适合" 等倾向禁忌
    if "risk" in matched and "contraindication" in matched:
        contra_signals = ["体质", "人群", "可以用", "可以打", "适合", "能用", "能打"]
        if any(s in q for s in contra_signals):
            return "contraindication"

    # 按优先级返回第一个命中的 route
    for route in order:
        if route in matched:
            return route
    return "basic"


def build_evidence(hits: List[Dict]) -> List[Dict]:
    ev = []
    for h in hits[:MAX_EVIDENCE_CHUNKS]:
        ev.append({
            "meta": h.get("meta", {}),
            "text": (h.get("text") or "")[:200],
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
                       "睡眠", "面霜", "生活"],
        "operation": ["针头", "深度", "注射", "0.8", "1.0", "0.3ml", "2cm", "MTS",
                      "水光", "涂抹", "微针", "仪器", "全脸", "进针", "间距",
                      "用量", "方式", "中胚层"],
        "contraindication": ["免疫", "妊娠", "哺乳", "过敏", "18", "风湿", "皮肤疾病",
                             "感染", "炎症", "禁忌"],
        "risk": ["红肿", "疼痛", "结节", "硬块", "感染", "过敏", "淤青", "肿胀",
                 "冰敷", "就医", "反应", "处理", "缓解", "发热", "化脓",
                 "异常", "严重"],
        "combo": ["联合", "搭配", "间隔", "水光", "微针", "光电", "填充", "同日",
                  "恢复", "建议", "不建议", "周"],
        "ingredient": ["PCL", "聚己内酯", "透明质酸", "玻尿酸", "谷胱甘肽", "肽",
                       "生长因子", "矿物质", "聚乙二醇", "胶原", "抗氧化", "修复",
                       "再生", "保湿", "提升", "弹性"],
    }

    keywords = route_keywords.get(route)
    if keywords:
        return any(k in clean for k in keywords)

    # basic 和未知路由：保留较短的行
    return len(clean) <= 80


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

    # 各路由的条目数限制
    limits = {
        "aftercare":       (20, 28),
        "operation":       (16, 24),
        "contraindication": (12, 18),
        "risk":            (12, 20),
        "combo":           (10, 16),
        "ingredient":      (16, 24),
        "basic":           (14, 22),
    }
    brief_lim, full_lim = limits.get(route, (12, 20))
    limit = brief_lim if mode == "brief" else full_lim

    return items[:limit]


def parse_answer(route: str, product: str, mode: str) -> List[str]:
    main_text = read_knowledge_file(product, "main.txt")
    faq_text = read_knowledge_file(product, "faq.txt")
    if route == "anti_fake":
        return parse_anti_fake(main_text, faq_text, mode)
    return parse_bullets_from_section(main_text, faq_text, route, mode)


def _build_context(hits: List[Dict], max_chars: int = 3000) -> str:
    """将检索结果拼接为 LLM context 字符串"""
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
        if total + len(part) > max_chars:
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


def llm_generate_answer(question: str, context: str, route: str, mode: str) -> str:
    """基于检索 context 用 LLM 生成答案（真正的 RAG）"""
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
    }

    system_prompt = (
        "你是一位医美产品知识库问答助手。请严格基于以下检索到的知识库片段回答用户问题。\n"
        "规则：\n"
        "1. 只使用知识库中的信息，不要编造或补充任何事实\n"
        '2. 如果知识库中没有相关信息，明确说明"当前知识库未覆盖该问题"\n'
        "3. 回答使用结构化格式（分点列出）\n"
        '4. 末尾加上"以上信息仅供参考，具体请咨询专业医师。"\n'
        f"5. 回答要求：{length_hint}\n"
        f"6. {route_hints.get(route, '')}\n"
    )

    user_prompt = f"知识库检索结果：\n{context}\n\n用户问题：{question}"

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024 if mode == "brief" else 2048,
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
            temperature=0.3,
            max_tokens=1024,
        )
        return (resp.choices[0].message.content or "").strip() or text
    except Exception:
        return text


def _fallback_from_hits(hits: List[Dict], max_lines: int = 8) -> List[str]:
    """当规则提取失败时，从检索结果中提取文本作为 fallback 答案"""
    lines = []
    for h in hits:
        text = (h.get("text") or "").strip()
        if not text:
            continue
        # 取每个 chunk 的前几行作为摘要
        for ln in text.split("\n"):
            ln = ln.strip()
            if ln and len(ln) > 4:
                lines.append(ln)
            if len(lines) >= max_lines:
                break
        if len(lines) >= max_lines:
            break
    return uniq(lines)


def answer_one(question: str, mode: str, rewrite: dict = None) -> str:
    product = detect_product(question)
    route = detect_route(question)
    if rewrite is None:
        rewrite = rewrite_query(question)

    # 根据问题类型使用不同的检索参数
    route_cfg = QUESTION_TYPE_CONFIG.get(route, {})
    route_top_k = route_cfg.get("k", DEFAULT_TOP_K)
    route_threshold = route_cfg.get("threshold", 0.30)

    # 向量检索用原始查询（避免别名扩展稀释语义方向）
    # 关键词检索用扩展查询（别名/同义词有助于 term 匹配）
    vector_hits = vector_search(product, rewrite["original"], VECTOR_TOP_K)
    _, docs = load_store(product)
    keyword_hits = keyword_search(rewrite["expanded"], docs, KEYWORD_TOP_K) if docs else []
    hits = merge_hybrid(vector_hits, keyword_hits, HYBRID_VECTOR_WEIGHT, HYBRID_KEYWORD_WEIGHT, route_top_k) if (vector_hits or keyword_hits) else []
    # 过滤低于 threshold 的结果
    hits = [h for h in hits if h.get("hybrid_score", h.get("score", 0.0)) >= route_threshold]

    # ---- 策略1: LLM RAG（优先）——检索结果作为 context 让 LLM 生成答案 ----
    if hits and USE_OPENAI:
        context = _build_context(hits)
        if context:
            llm_answer = llm_generate_answer(question, context, route, mode)
            if llm_answer:
                log_qa(question, llm_answer, rewritten_query=rewrite["expanded"],
                       matched_sources=build_evidence(hits), hit=True,
                       meta={"product": product, "route": route, "mode": mode,
                              "method": "llm_rag"})
                return llm_answer

    # ---- 策略2: 规则提取（Fallback）——从知识库文档中按章节规则提取条目 ----
    body_lines = parse_answer(route, product, mode)

    if not body_lines:
        # 规则也提取失败，从检索结果中摘要
        fallback_lines = _fallback_from_hits(hits)
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
                   meta={"product": product, "route": route, "mode": mode,
                          "method": "no_hit"})
            return text

    text = format_structured_answer(route, body_lines, build_evidence(hits), add_risk_note=(route == "risk"))
    if USE_OPENAI:
        text = openai_rewrite_answer(text, route)
    log_qa(question, text, rewritten_query=rewrite["expanded"],
           matched_sources=build_evidence(hits), hit=True,
           meta={"product": product, "route": route, "mode": mode,
                  "method": "rule_extract"})
    return text


def answer_question(question: str, mode: str) -> str:
    rewrite = rewrite_query(question)
    outputs = []
    seen = set()
    for subq in rewrite["sub_questions"][:MAX_SUB_QUESTIONS]:
        # 如果子问题与原问题相同，复用已有的 rewrite 结果
        sub_rewrite = rewrite if subq == rewrite["original"] else None
        ans = answer_one(subq, mode, rewrite=sub_rewrite)
        key = ans.strip()
        if key and key not in seen:
            seen.add(key)
            outputs.append(ans)
    return "\n\n".join(outputs)


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
