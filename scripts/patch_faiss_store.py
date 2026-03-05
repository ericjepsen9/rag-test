import re
from pathlib import Path

p = Path("faiss_store.py")
s = p.read_text(encoding="utf-8", errors="replace")

orig = s

# 1) 去掉 max_length=xxx（BGE-M3 的 SentenceTransformer.encode 不接受这个参数）
s = re.sub(r",\s*max_length\s*=\s*\d+\s*", ", ", s)

# 2) 去掉 ["dense_vecs"]（SentenceTransformer.encode 返回的是向量，不是 dict）
s = re.sub(r'\)\s*\[\s*["\']dense_vecs["\']\s*\]', ")", s)

# 3) 顺手清理可能出现的多余逗号/空格
s = re.sub(r",\s*\)", ")", s)
s = re.sub(r"\(\s*", "(", s)
s = re.sub(r"\s*\)", ")", s)
s = re.sub(r",\s*,", ",", s)

if s == orig:
    print("No changes made. (Pattern not found) -> 请把 embed_texts 函数附近那几行贴出来，我再给你精准补丁。")
else:
    p.write_text(s, encoding="utf-8", newline="\n")
    print("Patched faiss_store.py OK")
