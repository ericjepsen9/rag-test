import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import List

os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel

from rag_runtime_config import KNOWLEDGE_DIR, STORE_ROOT

MODEL_NAME = "BAAI/bge-m3"
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "420"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "100"))
_model = None


def get_model():
    global _model
    if _model is None:
        print(f"[INFO] 加载模型：{MODEL_NAME}")
        _model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)
    return _model


# ====== 多编码读取 ======

def read_text_auto(p: Path) -> str:
    if not p.exists():
        return ""
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return p.read_text(encoding=enc)
        except Exception:
            continue
    return p.read_text(errors="replace")


# ====== 段落感知切块 ======

def normalize_text(text: str) -> str:
    t = (text or "").replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def is_title_like(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    patterns = [
        r"^[一二三四五六七八九十]+、",
        r"^\d+[）\.\)]",
        r"^第[一二三四五六七八九十0-9]+",
        r"^STEP\s*\d+",
        r"^【.+】$",
        r"^#+\s*",
        r"^=+$",
        r"^-{3,}$",
    ]
    return any(re.match(p, s, flags=re.IGNORECASE) for p in patterns)


def split_into_paragraphs(text: str) -> List[str]:
    lines = [ln.rstrip() for ln in (text or "").split("\n")]
    paras: List[str] = []
    buf: List[str] = []

    def flush():
        nonlocal buf
        if buf:
            s = "\n".join(buf).strip()
            if s:
                paras.append(s)
            buf = []

    for line in lines:
        s = line.strip()
        if not s:
            flush()
            continue
        if is_title_like(s):
            flush()
            paras.append(s)
            continue
        buf.append(s)

    flush()
    return paras


def merge_paragraphs_to_chunks(paragraphs: List[str], chunk_size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    cur = ""

    def add_chunk(x: str):
        x = x.strip()
        if x:
            chunks.append(x)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 超长段落切分
        if len(para) > int(chunk_size * 1.2):
            if cur:
                add_chunk(cur)
                cur = ""
            start = 0
            step = max(1, chunk_size - overlap)
            while start < len(para):
                add_chunk(para[start:start + chunk_size])
                start += step
            continue

        candidate = (cur + "\n" + para).strip() if cur else para
        if len(candidate) <= chunk_size:
            cur = candidate
        else:
            if cur:
                add_chunk(cur)
            if chunks and overlap > 0:
                tail = chunks[-1][-overlap:]
                cur = (tail + "\n" + para).strip()
                if len(cur) > int(chunk_size * 1.3):
                    cur = para
            else:
                cur = para

    if cur:
        add_chunk(cur)

    # 去重
    dedup: List[str] = []
    seen = set()
    for c in chunks:
        k = re.sub(r"\s+", " ", c.strip())
        if k and k not in seen:
            dedup.append(c.strip())
            seen.add(k)
    return dedup


def chunk_text(text: str) -> List[str]:
    t = normalize_text(text)
    if not t:
        return []
    paras = split_into_paragraphs(t)
    return merge_paragraphs_to_chunks(paras, CHUNK_SIZE, CHUNK_OVERLAP)


# ====== 向量编码 ======

def embed_texts(texts):
    model = get_model()
    out = model.encode(texts, batch_size=8, max_length=8192)
    vecs = None
    if isinstance(out, dict):
        if out.get("dense_vecs") is not None:
            vecs = out["dense_vecs"]
        elif out.get("dense") is not None:
            vecs = out["dense"]
        elif out.get("embeddings") is not None:
            vecs = out["embeddings"]
    elif isinstance(out, (list, tuple, np.ndarray)):
        vecs = out
    if vecs is None:
        raise ValueError("encode 输出中未找到向量字段")
    vecs = np.asarray(vecs, dtype="float32")
    if vecs.ndim != 2:
        raise ValueError(f"向量维度异常: {vecs.shape}")
    faiss.normalize_L2(vecs)
    return vecs


# ====== 构建索引 ======

def collect_product_records(product: str):
    pdir = KNOWLEDGE_DIR / product
    if not pdir.exists():
        raise FileNotFoundError(f"未找到产品目录：{pdir}")
    files = [("main.txt", "main"), ("faq.txt", "faq"), ("alias.txt", "alias")]
    records = []
    for fname, stype in files:
        f = pdir / fname
        if not f.exists():
            continue
        text = read_text_auto(f)
        # alias 不切块，整体入库
        chunks = [text] if stype == "alias" else chunk_text(text)
        print(f"[OK] {product}/{fname}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks, 1):
            records.append({
                "text": chunk,
                "meta": {
                    "product_id": product,
                    "source_file": f.name,
                    "source_type": stype,
                    "chunk_id": i,
                }
            })
    if not records:
        raise ValueError(f"{product} 没有可用文本")
    return records


def build_for_product(product: str):
    records = collect_product_records(product)
    texts = [r["text"] for r in records]
    print(f"[INFO] Total chunks: {len(texts)}")
    print(f"[INFO] Embedding {len(texts)} chunks ...")
    vecs = embed_texts(texts)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    out_dir = STORE_ROOT / product
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_path = out_dir / "docs.jsonl"
    index_path = out_dir / "index.faiss"

    with docs_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    faiss.write_index(index, str(index_path))

    print(f"[DONE] Built store")
    print(f"       product: {product}")
    print(f"       chunks : {len(records)}")
    print(f"       dim    : {dim}")


def list_products():
    if not KNOWLEDGE_DIR.exists():
        print(f"[ERROR] knowledge 目录不存在：{KNOWLEDGE_DIR}")
        return
    for p in sorted([x.name for x in KNOWLEDGE_DIR.iterdir() if x.is_dir()]):
        print(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", type=str, help="产品目录名")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_products()
        return
    if not args.product:
        ap.error("请使用 --product <name> 或 --list")
    build_for_product(args.product.strip())


if __name__ == "__main__":
    main()
