# app/search.py
# 检索后端：给 rag_answer.py 调用
# 提供：
# - load_docs()
# - load_index()
# - embed_texts(model, texts)
# - 可选 search_text(...) 方便直接调试

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss

from config.settings import (
    EMBED_MODEL_NAME,
    USE_FP16,
    get_product_id,
    get_store_dir,
    get_index_path,
    get_docs_path,
)

# Windows UTF-8（避免控制台乱码）
os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _ensure_exists(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} 不存在: {path}")


def load_docs(product_id: Optional[str] = None) -> List[Dict[str, Any]]:
    docs_path = get_docs_path(product_id)
    _ensure_exists(docs_path, "docs.jsonl")
    docs: List[Dict[str, Any]] = []
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except Exception:
                # 跳过坏行，避免整库崩掉
                continue
    return docs


def load_index(product_id: Optional[str] = None):
    index_path = get_index_path(product_id)
    _ensure_exists(index_path, "index.faiss")
    return faiss.read_index(str(index_path))


def load_model():
    # 延迟加载，避免导入时就下载模型
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(EMBED_MODEL_NAME, use_fp16=USE_FP16)
    return model


def embed_texts(model, texts: List[str]) -> np.ndarray:
    """
    返回 L2 归一化后的 np.float32 dense 向量，shape = [N, dim]。
    归一化是必须的：build_faiss.py 使用 IndexFlatIP（内积索引），
    存入的文档向量已经归一化，查询向量也必须归一化才能正确计算余弦相似度。
    """
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)

    out = model.encode(
        texts,
        batch_size=min(16, max(1, len(texts))),
        # ── 修复 3：max_length 统一为 1024，与 build_faiss.py 建索引时一致 ──
        max_length=1024,
    )

    # 常见返回：
    # 1) dict: {"dense_vecs": np.ndarray, ...}
    # 2) np.ndarray
    # 3) list
    if isinstance(out, dict):
        vecs = out.get("dense_vecs")
        if vecs is None:
            raise RuntimeError("模型返回中未找到 dense_vecs")
    else:
        vecs = out

    vecs = np.asarray(vecs, dtype=np.float32)

    # 强制二维
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)

    # ── 修复 4：在此处统一做 L2 归一化 ──────────────────────────────
    # build_faiss.py 存储向量时已归一化，查询向量也必须归一化，
    # 否则 IndexFlatIP 的分数不等于余弦相似度，检索结果会出错。
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms

    return vecs


def search_text(question: str, k: int = 8, product_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    直接检索调试用（rag_answer.py 也可以不调用这个）
    """
    pid = get_product_id(product_id)
    docs = load_docs(pid)
    index = load_index(pid)
    model = load_model()

    qv = embed_texts(model, [question])
    # embed_texts 内部已归一化，无需再次 normalize_L2

    scores, ids = index.search(qv, int(k))
    out = []

    for i, idx in enumerate(ids[0]):
        if idx < 0 or idx >= len(docs):
            continue
        d = docs[idx]
        out.append({
            "score": float(scores[0][i]),
            "text": d.get("text", ""),
            "meta": d.get("meta", {}),
        })
    return out


if __name__ == "__main__":
    # 命令行快速测试
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "非罗奥 怎么验真伪"
    pid = sys.argv[2] if len(sys.argv) > 2 else None
    res = search_text(q, k=5, product_id=pid)
    for i, r in enumerate(res, 1):
        print(f"[{i}] score={r['score']:.4f} meta={r.get('meta')}")
        print((r.get("text") or "")[:200].replace("\n", " "))
        print("-" * 80)
