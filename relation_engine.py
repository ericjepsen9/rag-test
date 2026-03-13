"""实体关联引擎：加载 relations.json 并提供跨实体关联查询。

用途：
1. combo 路由 → 补充产品-项目兼容性和联合禁忌
2. contraindication 路由 → 补充药物交互警告
3. indication_q / anatomy_q 路由 → 补充推荐方案
4. equipment_q 路由 → 补充设备-项目对应关系
"""
import json
import threading
from pathlib import Path
from typing import List, Dict, Optional

from rag_runtime_config import RELATIONS_FILE, PRODUCT_ALIASES, PROCEDURE_ALIASES, EQUIPMENT_ALIASES

_relations: Optional[Dict] = None
_lock = threading.Lock()


def _load() -> Dict:
    global _relations
    if _relations is not None:
        return _relations
    with _lock:
        if _relations is not None:
            return _relations
        if not RELATIONS_FILE.exists():
            _relations = {}
            return _relations
        try:
            _relations = json.loads(RELATIONS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _relations = {}
    return _relations


def _product_label(pid: str) -> str:
    """产品 ID → 中文名"""
    aliases = PRODUCT_ALIASES.get(pid, [])
    return aliases[0] if aliases else pid


def _procedure_label(pid: str) -> str:
    """项目 ID → 中文名"""
    aliases = PROCEDURE_ALIASES.get(pid, [])
    return aliases[0] if aliases else pid


def _equipment_label(eid: str) -> str:
    """设备 ID → 中文名"""
    aliases = EQUIPMENT_ALIASES.get(eid, [])
    return aliases[0] if aliases else eid


# ===== 公开查询接口 =====

def get_combo_info(product_id: str) -> List[str]:
    """获取产品的联合方案信息 + 联合禁忌，用于 combo 路由补充。"""
    data = _load()
    lines = []

    # 产品-项目兼容性
    for rel in data.get("product_procedure", []):
        if rel.get("product") == product_id:
            proc = _procedure_label(rel.get("procedure", ""))
            spacing = rel.get("spacing", "")
            note = rel.get("note", "")
            equip = _equipment_label(rel.get("equipment", "")) if rel.get("equipment") else ""
            line = f"与{proc}联合"
            if equip:
                line += f"（使用{equip}）"
            if spacing:
                line += f"：{spacing}"
            if note:
                line += f"。{note}"
            lines.append(line)

    # 联合禁忌
    for rule in data.get("combo_contraindications", []):
        severity = rule.get("severity", "")
        text = rule.get("rule", "")
        if severity == "禁止":
            lines.append(f"【禁止】{text}")
        elif text:
            lines.append(f"【注意】{text}")

    return lines


def get_drug_interactions(route: str = "") -> List[str]:
    """获取药物交互信息，用于 contraindication / pre_care 路由补充。"""
    data = _load()
    lines = []
    for item in data.get("drug_interactions", []):
        drug = item.get("drug", "")
        impact = item.get("impact", "")
        action = item.get("action", "")
        if drug and (impact or action):
            line = f"{drug}：{impact}"
            if action:
                line += f"，{action}"
            lines.append(line)
    return lines


def get_indication_recommendations(query: str) -> List[str]:
    """根据查询中的适应症关键词，返回推荐方案。"""
    data = _load()
    lines = []
    q_lower = query.lower()
    for item in data.get("indication_product", []):
        indication = item.get("indication", "")
        if indication and indication in q_lower:
            products = [_product_label(p) for p in item.get("products", [])]
            procedures = [_procedure_label(p) for p in item.get("procedures", [])]
            note = item.get("note", "")
            parts = []
            if products:
                parts.append("推荐产品：" + "、".join(products))
            if procedures:
                parts.append("推荐项目：" + "、".join(procedures))
            if note:
                parts.append(note)
            if parts:
                lines.append(f"【{indication}】" + "；".join(parts))
    return lines


def get_anatomy_recommendations(query: str) -> List[str]:
    """根据查询中的部位关键词，返回推荐方案。"""
    data = _load()
    lines = []
    q_lower = query.lower()
    for item in data.get("anatomy_product", []):
        area = item.get("area", "")
        if area and area in q_lower:
            products = [_product_label(p) for p in item.get("products", [])]
            procedures = [_procedure_label(p) for p in item.get("procedures", [])]
            note = item.get("note", "")
            parts = []
            if products:
                parts.append("推荐产品：" + "、".join(products))
            if procedures:
                parts.append("推荐项目：" + "、".join(procedures))
            if note:
                parts.append(note)
            if parts:
                lines.append(f"【{area}】" + "；".join(parts))
    return lines


def get_procedure_equipment(query: str) -> List[str]:
    """根据查询中的项目关键词，返回对应设备信息。"""
    data = _load()
    lines = []
    q_lower = query.lower()
    # 检查查询中提到的项目
    for proc_id, aliases in PROCEDURE_ALIASES.items():
        if any(a.lower() in q_lower for a in aliases):
            for rel in data.get("procedure_equipment", []):
                if rel.get("procedure") == proc_id:
                    equip = _equipment_label(rel.get("equipment", ""))
                    note = rel.get("note", "")
                    required = "必需" if rel.get("required") else "可选"
                    line = f"{_procedure_label(proc_id)} → {equip}（{required}）"
                    if note:
                        line += f"：{note}"
                    lines.append(line)
    return lines


def enrich_answer(route: str, product_id: str, question: str) -> List[str]:
    """根据路由类型，从 relations.json 中提取补充信息。

    返回补充行列表，调用方拼接到答案末尾。
    """
    data = _load()
    if not data:
        return []

    lines = []
    if route == "combo":
        lines = get_combo_info(product_id)
    elif route in ("contraindication", "pre_care"):
        lines = get_drug_interactions(route)
    elif route == "indication_q":
        lines = get_indication_recommendations(question)
    elif route == "anatomy_q":
        lines = get_anatomy_recommendations(question)
    elif route == "equipment_q":
        lines = get_procedure_equipment(question)

    return lines
