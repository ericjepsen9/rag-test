import json
from pathlib import Path
from typing import List, Dict, Optional
from rag_runtime_config import KNOWLEDGE_DIR


def _load_product_media(product_id: str) -> List[Dict]:
    """加载某个产品的 media.json"""
    media_file = KNOWLEDGE_DIR / product_id / "media.json"
    if not media_file.exists():
        return []
    try:
        items = json.loads(media_file.read_text(encoding="utf-8"))
        return [it for it in items if isinstance(it, dict) and "title" in it]
    except Exception:
        return []


def find_media(question: str,
               product_id: Optional[str] = None,
               route: Optional[str] = None) -> List[Dict]:
    """根据产品+路由精准匹配媒体，关键词兜底。

    优先级：
    1. product_id + route 都命中 → 返回该产品该路由的媒体
    2. product_id 命中但 route 无匹配 → 用 keywords 在该产品媒体中兜底
    3. product_id 未知 → 空列表（不跨产品猜测）
    """
    if not product_id:
        return []

    items = _load_product_media(product_id)
    if not items:
        return []

    # 1) 路由精准匹配
    if route:
        route_hits = [it for it in items
                      if route in (it.get("routes") or [])]
        if route_hits:
            return route_hits[:6]

    # 2) 关键词兜底
    q = (question or "").lower()
    if not q:
        return []
    kw_hits = []
    for it in items:
        keys = it.get("keywords") or []
        if any(k.lower() in q for k in keys):
            kw_hits.append(it)
    return kw_hits[:6]
