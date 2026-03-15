from typing import List, Dict
from rag_runtime_config import REFERENCE_NOTE, RISK_NOTE, MAX_EVIDENCE_CHUNKS

_SAFETY_ROUTES = frozenset(("contraindication", "complication", "repair", "operation"))

_TITLE_MAP = {
    "basic": "基础资料",
    "operation": "操作说明",
    "aftercare": "术后护理",
    "risk": "风险/异常反应",
    "combo": "联合方案",
    "anti_fake": "防伪鉴别",
    "contraindication": "禁忌人群",
    "ingredient": "核心成分与作用",
    "effect": "效果与维持时间",
    "pre_care": "术前准备",
    "design": "方案设计与面部评估",
    "repair": "修复与补救方案",
    "complication": "术后并发症处理",
    "course": "疗程规划",
    "anatomy_q": "部位治疗方案",
    "indication_q": "适应症推荐",
    "procedure_q": "项目介绍",
    "equipment_q": "设备信息",
    "script": "客户沟通话术",
}


def format_structured_answer(
    route: str,
    body_lines: List[str],
    evidence: List[Dict] = None,
    add_risk_note: bool = False,
) -> str:
    title = _TITLE_MAP.get(route, "回答")
    out = [f"【{title}】", ""]
    for ln in body_lines:
        # 保留已有的格式化（如 - 开头的行），否则加上 bullet
        stripped = ln.strip()
        if not stripped:
            out.append("")
        elif stripped.startswith(("-", "•", "·", "【")):
            out.append(stripped)
        else:
            out.append(f"- {stripped}")

    # 安全提醒（自然语气）
    out.append("")
    if add_risk_note or route in _SAFETY_ROUTES:
        out.append(f"⚠️ {RISK_NOTE}如有不适请及时就医。")
    out.append(f"💡 {REFERENCE_NOTE}")
    return "\n".join(out).strip()
