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

# 预建倒排索引：加载时一次性构建，避免每次查询 O(n) 遍历
_idx_indication: Optional[Dict[str, List[Dict]]] = None   # indication -> [items]
_idx_anatomy: Optional[Dict[str, List[Dict]]] = None      # area -> [items]
_idx_product_proc: Optional[Dict[str, List[Dict]]] = None  # product_id -> [rels]
_idx_proc_equip: Optional[Dict[str, List[Dict]]] = None    # procedure_id -> [rels]


def invalidate_relations_cache() -> None:
    """清除关联数据缓存（relations.json 更新后调用）"""
    global _relations, _idx_indication, _idx_anatomy, _idx_product_proc, _idx_proc_equip
    with _lock:
        _relations = None
        _idx_indication = None
        _idx_anatomy = None
        _idx_product_proc = None
        _idx_proc_equip = None


def _build_indices(data: Dict) -> None:
    """从 relations 数据构建倒排索引，O(n) 一次性遍历"""
    global _idx_indication, _idx_anatomy, _idx_product_proc, _idx_proc_equip

    # indication -> items
    idx_ind: Dict[str, List[Dict]] = {}
    for item in data.get("indication_product", []):
        ind = item.get("indication", "")
        if ind:
            idx_ind.setdefault(ind, []).append(item)
    _idx_indication = idx_ind

    # anatomy area -> items
    idx_anat: Dict[str, List[Dict]] = {}
    for item in data.get("anatomy_product", []):
        area = item.get("area", "")
        if area:
            idx_anat.setdefault(area, []).append(item)
    _idx_anatomy = idx_anat

    # product_id -> product_procedure rels
    idx_pp: Dict[str, List[Dict]] = {}
    for rel in data.get("product_procedure", []):
        pid = rel.get("product", "")
        if pid:
            idx_pp.setdefault(pid, []).append(rel)
    _idx_product_proc = idx_pp

    # procedure_id -> procedure_equipment rels
    idx_pe: Dict[str, List[Dict]] = {}
    for rel in data.get("procedure_equipment", []):
        pid = rel.get("procedure", "")
        if pid:
            idx_pe.setdefault(pid, []).append(rel)
    _idx_proc_equip = idx_pe


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
            data = json.loads(RELATIONS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            from rag_logger import log_error
            log_error("relation_engine", f"relations.json 加载失败: {exc}")
            _relations = {}
            return _relations
        _build_indices(data)
        _relations = data
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

    # 产品-项目兼容性（使用倒排索引）
    for rel in (_idx_product_proc or {}).get(product_id, []):
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
    _load()
    lines = []
    q_lower = query.lower()
    for indication, items in (_idx_indication or {}).items():
        if indication in q_lower:
            for item in items:
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
    _load()
    lines = []
    q_lower = query.lower()
    for area, items in (_idx_anatomy or {}).items():
        if area in q_lower:
            for item in items:
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
    _load()
    lines = []
    q_lower = query.lower()
    # 检查查询中提到的项目（使用倒排索引）
    for proc_id, aliases in PROCEDURE_ALIASES.items():
        if any(a.lower() in q_lower for a in aliases):
            for rel in (_idx_proc_equip or {}).get(proc_id, []):
                equip = _equipment_label(rel.get("equipment", ""))
                note = rel.get("note", "")
                required = "必需" if rel.get("required") else "可选"
                line = f"{_procedure_label(proc_id)} → {equip}（{required}）"
                if note:
                    line += f"：{note}"
                lines.append(line)
    return lines


def get_temporal_constraints(product_id: str) -> List[str]:
    """获取产品相关的时序约束：各项目之间的间隔要求。"""
    data = _load()
    lines = []
    seen = set()

    # 产品-项目间隔（使用倒排索引）
    for rel in (_idx_product_proc or {}).get(product_id, []):
        proc = _procedure_label(rel.get("procedure", ""))
        spacing = rel.get("spacing", "")
        if spacing and spacing not in seen:
            lines.append(f"与{proc}：{spacing}")
            seen.add(spacing)

    # 产品-产品间隔
    for rel in data.get("product_product", []):
        if rel.get("product_a") == product_id or rel.get("product_b") == product_id:
            other_id = rel["product_b"] if rel["product_a"] == product_id else rel["product_a"]
            other = _product_label(other_id)
            days = rel.get("min_interval_days", 0)
            note = rel.get("note", "")
            if days:
                line = f"与{other}间隔至少{days}天"
                if note:
                    line += f"（{note}）"
                if line not in seen:
                    lines.append(line)
                    seen.add(line)

    return lines


def validate_combo_safety(product_id: str, question: str) -> List[str]:
    """检查问题中提到的联合方案是否存在安全风险。

    返回警告信息列表（空列表表示无风险或无法判断）。
    """
    data = _load()
    warnings = []
    q_lower = question.lower()

    # 检测问题中提到的项目
    mentioned_procs = []
    for proc_id, aliases in PROCEDURE_ALIASES.items():
        if any(a.lower() in q_lower for a in aliases):
            mentioned_procs.append(proc_id)

    if len(mentioned_procs) < 2:
        return []

    # 检查联合禁忌
    for rule in data.get("combo_contraindications", []):
        severity = rule.get("severity", "")
        text = rule.get("rule", "")
        if severity == "禁止":
            # 同一部位同日多种注射 → 检查是否有多个注射类项目
            injection_procs = {"water_light", "microneedling", "filling"}
            mentioned_injections = [p for p in mentioned_procs if p in injection_procs]
            if len(mentioned_injections) >= 2 and "同一部位" in text:
                warnings.append(f"⚠ {text}")

    # 检查时序冲突：如果两个项目都有 spacing 要求（使用倒排索引）
    for rel in (_idx_product_proc or {}).get(product_id, []):
        proc = rel.get("procedure", "")
        if proc in mentioned_procs:
            spacing = rel.get("spacing", "")
            if spacing and "同日" not in spacing:
                proc_label = _procedure_label(proc)
                warnings.append(f"时序提示：{proc_label} — {spacing}")

    return warnings


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
        # 追加安全风险检查
        safety = validate_combo_safety(product_id, question)
        if safety:
            lines.extend(safety)
    elif route in ("contraindication", "pre_care"):
        lines = get_drug_interactions(route)
    elif route == "indication_q":
        lines = get_indication_recommendations(question)
    elif route == "anatomy_q":
        lines = get_anatomy_recommendations(question)
    elif route == "equipment_q":
        lines = get_procedure_equipment(question)
    elif route == "course":
        # 疗程路由追加时序约束
        temporal = get_temporal_constraints(product_id)
        if temporal:
            lines.append("【间隔要求】")
            lines.extend(temporal)

    return lines
