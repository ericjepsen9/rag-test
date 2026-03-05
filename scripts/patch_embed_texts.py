import re
from pathlib import Path

p = Path("faiss_store.py")
s = p.read_text(encoding="utf-8", errors="replace")

# 用正则找到 def embed_texts(...) 到下一个 def / if __name__ 之间的块
m = re.search(r"(?ms)^def\s+embed_texts\s*\(.*?\):\s*\n(.*?)(?=^\s*def\s+|\Z)", s)
if not m:
    print("ERROR: 没找到 embed_texts 函数。请把 faiss_store.py 里 def embed_texts 附近贴出来。")
    raise SystemExit(1)

new_func = r'''def embed_texts(model, texts):
    """
    兼容两种返回：
    1) ndarray/list: SentenceTransformer.encode 常见返回
    2) dict: 某些 BGE-M3/自定义 wrapper 返回 {"dense_vecs": ...}
    """
    vecs = model.encode(texts, batch_size=1)
    # 若返回 dict，则取 dense_vecs
    if isinstance(vecs, dict):
        if "dense_vecs" in vecs:
            vecs = vecs["dense_vecs"]
        elif "sentence_embedding" in vecs:
            vecs = vecs["sentence_embedding"]
        else:
            raise TypeError(f"encode() returned dict but no dense vector key found: {list(vecs.keys())}")

    import numpy as np
    return np.asarray(vecs, dtype="float32")
'''

# 替换整个函数体（包括 def 行到函数末尾）
start = m.start()
end = m.end()
# m.start() 定位到 def 行；我们用另一种方法：找到 def 行开头再替换到块末
def_line = re.search(r"(?m)^def\s+embed_texts\s*\(.*?\):\s*$", s)
if not def_line:
    print("ERROR: 找不到 embed_texts 的 def 行。")
    raise SystemExit(1)

# 找到 embed_texts 块的结束位置：从 def 行后开始，遇到下一個顶格 def 或文件结束
after = s[def_line.end():]
next_def = re.search(r"(?m)^\s*def\s+\w+\s*\(.*?\):\s*$", after)
if next_def:
    block_end = def_line.end() + next_def.start()
else:
    block_end = len(s)

patched = s[:def_line.start()] + new_func + "\n" + s[block_end:]
p.write_text(patched, encoding="utf-8", newline="\n")
print("Patched embed_texts() in faiss_store.py OK")
