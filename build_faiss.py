import os
import sys
import re
import json
import argparse
import tempfile
from pathlib import Path
from typing import List

os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from rag_runtime_config import (
    KNOWLEDGE_DIR, STORE_ROOT,
    EMBED_MODEL_NAME, EMBED_USE_FP16, EMBED_BATCH_SIZE_BUILD, EMBED_MAX_LENGTH_BUILD,
    CHUNK_SIZE as _DEFAULT_CHUNK_SIZE, CHUNK_OVERLAP as _DEFAULT_CHUNK_OVERLAP,
)

MODEL_NAME = EMBED_MODEL_NAME
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", str(_DEFAULT_CHUNK_SIZE)))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", str(_DEFAULT_CHUNK_OVERLAP)))
_model = None
_np = None
_faiss = None


def _get_np():
    global _np
    if _np is None:
        import numpy as _np_mod
        _np = _np_mod
    return _np


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss as _faiss_mod
        _faiss = _faiss_mod
    return _faiss


def get_model():
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel
        print(f"[INFO] 加载模型：{MODEL_NAME}")
        _model = BGEM3FlagModel(MODEL_NAME, use_fp16=EMBED_USE_FP16)
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


def _is_major_section(line: str) -> bool:
    """判断是否为主章节标题（一、二、...）或编号子节标题（1）2）...），
    这些标题处应强制分 chunk，避免跨章节合并"""
    s = line.strip()
    return bool(re.match(r"^[一二三四五六七八九十]+、", s) or
                re.match(r"^\d+[）\)]\s*.{2,}", s))


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

        # 主章节标题处强制分 chunk，避免跨章节合并（如"禁忌人群"+"风险"混在一个 chunk）
        if _is_major_section(para) and cur:
            add_chunk(cur)
            cur = para
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
                # 防止 overlap 跨越主章节边界造成语义污染
                if _is_major_section(tail.split("\n")[0]):
                    cur = para
                else:
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


def _is_separator(para: str) -> bool:
    """判断是否为纯分隔线（====, ----, 空白符号行）"""
    return bool(re.fullmatch(r"[=\-_\s]+", para.strip()))


MIN_CHUNK_CHARS = 30  # 过短的 chunk 对检索无信息量，过滤噪音


def chunk_text(text: str) -> List[str]:
    t = normalize_text(text)
    if not t:
        return []
    paras = split_into_paragraphs(t)
    paras = [p for p in paras if not _is_separator(p)]
    chunks = merge_paragraphs_to_chunks(paras, CHUNK_SIZE, CHUNK_OVERLAP)
    # 过滤过短 chunk（纯标题行、残余片段等），减少索引噪音
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


# ====== 向量编码 ======

def embed_texts(texts):
    model = get_model()
    out = model.encode(texts, batch_size=EMBED_BATCH_SIZE_BUILD, max_length=EMBED_MAX_LENGTH_BUILD)
    vecs = None
    if isinstance(out, dict):
        if out.get("dense_vecs") is not None:
            vecs = out["dense_vecs"]
        elif out.get("dense") is not None:
            vecs = out["dense"]
        elif out.get("embeddings") is not None:
            vecs = out["embeddings"]
    else:
        vecs = out
    if vecs is None:
        raise ValueError("encode 输出中未找到向量字段")
    np = _get_np()
    vecs = np.asarray(vecs, dtype="float32")
    if vecs.ndim != 2:
        raise ValueError(f"向量维度异常: {vecs.shape}")
    if vecs.shape[0] != len(texts):
        raise ValueError(f"向量行数({vecs.shape[0]})与文本数({len(texts)})不一致")
    _get_faiss().normalize_L2(vecs)
    return vecs


# ====== 构建索引 ======

def _dedup_records(records: List[dict]) -> List[dict]:
    """跨来源去重：相同文本内容（忽略空白差异）只保留第一个来源的记录"""
    seen_hashes = set()
    deduped = []
    for r in records:
        # 用去空白后的文本做去重键
        key = re.sub(r"\s+", " ", r["text"].strip())
        if key in seen_hashes:
            continue
        seen_hashes.add(key)
        deduped.append(r)
    return deduped


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
        # 清除纯分隔线
        text = re.sub(r"^[=\-_]{3,}\s*$", "", text, flags=re.MULTILINE).strip()
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
    # 跨来源去重：main.txt + faq.txt 可能有重复段落
    before = len(records)
    records = _dedup_records(records)
    if len(records) < before:
        print(f"[INFO] {product}: 跨来源去重 {before} → {len(records)} records")
    return records


def build_for_product(product: str):
    records = collect_product_records(product)
    texts = [r["text"] for r in records]
    print(f"[INFO] Total chunks: {len(texts)}")
    print(f"[INFO] Embedding {len(texts)} chunks ...")
    vecs = embed_texts(texts)
    faiss = _get_faiss()
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    out_dir = STORE_ROOT / product
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_path = out_dir / "docs.jsonl"
    index_path = out_dir / "index.faiss"

    # 原子写入：先写临时文件，再 rename，防止进程中断导致文件损坏
    tmp_docs = out_dir / "docs.jsonl.tmp"
    tmp_index = out_dir / "index.faiss.tmp"
    with tmp_docs.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _get_faiss().write_index(index, str(tmp_index))
    os.replace(str(tmp_docs), str(docs_path))
    os.replace(str(tmp_index), str(index_path))

    print(f"[DONE] Built store")
    print(f"       product: {product}")
    print(f"       chunks : {len(records)}")
    print(f"       dim    : {dim}")


def collect_shared_records():
    """收集所有共享知识实体（procedures、equipment、anatomy 等）的文本记录。"""
    from rag_runtime_config import SHARED_ENTITY_DIRS
    records = []
    for entity_type, subdir in SHARED_ENTITY_DIRS.items():
        edir = KNOWLEDGE_DIR / subdir
        if not edir.exists():
            continue
        # 两种结构：1) subdir/main.txt (单文件实体) 2) subdir/{name}/main.txt (多实例)
        main_file = edir / "main.txt"
        if main_file.exists():
            # 单文件实体（anatomy、indications、complications、courses、scripts）
            text = read_text_auto(main_file)
            text = re.sub(r"^[=\-_]{3,}\s*$", "", text, flags=re.MULTILINE).strip()
            chunks = chunk_text(text)
            print(f"[OK] {subdir}/main.txt: {len(chunks)} chunks")
            for i, chunk in enumerate(chunks, 1):
                records.append({
                    "text": chunk,
                    "meta": {
                        "product_id": "_shared",
                        "entity_type": entity_type,
                        "source_file": f"{subdir}/main.txt",
                        "source_type": entity_type,
                        "chunk_id": i,
                    }
                })
        # 多实例子目录
        for inst in sorted(edir.iterdir()):
            if not inst.is_dir():
                continue
            for fname, stype in [("main.txt", "main"), ("faq.txt", "faq"), ("alias.txt", "alias")]:
                f = inst / fname
                if not f.exists():
                    continue
                text = read_text_auto(f)
                text = re.sub(r"^[=\-_]{3,}\s*$", "", text, flags=re.MULTILINE).strip()
                chunks = [text] if stype == "alias" else chunk_text(text)
                label = f"{subdir}/{inst.name}/{fname}"
                print(f"[OK] {label}: {len(chunks)} chunks")
                for i, chunk in enumerate(chunks, 1):
                    records.append({
                        "text": chunk,
                        "meta": {
                            "product_id": "_shared",
                            "entity_type": entity_type,
                            "entity_id": inst.name,
                            "source_file": label,
                            "source_type": stype,
                            "chunk_id": i,
                        }
                    })
    # 跨来源去重
    before = len(records)
    records = _dedup_records(records)
    if len(records) < before:
        print(f"[INFO] shared: 跨来源去重 {before} → {len(records)} records")
    return records


def build_shared():
    """构建共享知识索引（存储在 stores/_shared/）"""
    records = collect_shared_records()
    if not records:
        print("[WARN] 无共享知识可索引")
        return
    texts = [r["text"] for r in records]
    print(f"[INFO] Shared total chunks: {len(texts)}")
    print(f"[INFO] Embedding {len(texts)} chunks ...")
    vecs = embed_texts(texts)
    faiss = _get_faiss()
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    out_dir = STORE_ROOT / "_shared"
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_path = out_dir / "docs.jsonl"
    index_path = out_dir / "index.faiss"

    # 原子写入：先写临时文件，再 rename
    tmp_docs = out_dir / "docs.jsonl.tmp"
    tmp_index = out_dir / "index.faiss.tmp"
    with tmp_docs.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _get_faiss().write_index(index, str(tmp_index))
    os.replace(str(tmp_docs), str(docs_path))
    os.replace(str(tmp_index), str(index_path))

    print(f"[DONE] Built shared store")
    print(f"       chunks : {len(records)}")
    print(f"       dim    : {dim}")


def _is_product_dir(p: Path) -> bool:
    """顶层目录且有 main.txt，且不是共享知识目录"""
    from rag_runtime_config import SHARED_ENTITY_DIRS
    shared_names = set(SHARED_ENTITY_DIRS.values())
    return p.is_dir() and (p / "main.txt").exists() and p.name not in shared_names


def list_products():
    if not KNOWLEDGE_DIR.exists():
        print(f"[ERROR] knowledge 目录不存在：{KNOWLEDGE_DIR}")
        return
    # 列出产品目录（顶层有 main.txt 且不是共享知识目录）
    for p in sorted(KNOWLEDGE_DIR.iterdir()):
        if _is_product_dir(p):
            print(f"[product] {p.name}")
    # 列出共享知识目录
    from rag_runtime_config import SHARED_ENTITY_DIRS
    for entity_type, subdir in SHARED_ENTITY_DIRS.items():
        edir = KNOWLEDGE_DIR / subdir
        if not edir.exists():
            continue
        if (edir / "main.txt").exists():
            print(f"[{entity_type}] {subdir}/")
        for inst in sorted(edir.iterdir()):
            if inst.is_dir() and (inst / "main.txt").exists():
                print(f"[{entity_type}] {subdir}/{inst.name}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", type=str, help="产品目录名")
    ap.add_argument("--shared", action="store_true", help="构建共享知识索引")
    ap.add_argument("--all", action="store_true", help="构建所有产品+共享知识")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_products()
        return
    if args.all:
        # 构建所有产品（排除共享知识目录）
        for p in sorted(KNOWLEDGE_DIR.iterdir()):
            if _is_product_dir(p):
                print(f"\n{'='*40}\n构建产品: {p.name}\n{'='*40}")
                build_for_product(p.name)
        # 构建共享知识
        print(f"\n{'='*40}\n构建共享知识\n{'='*40}")
        build_shared()
        return
    if args.shared:
        build_shared()
        return
    if not args.product:
        ap.error("请使用 --product <name> / --shared / --all / --list")
    build_for_product(args.product.strip())


if __name__ == "__main__":
    main()
