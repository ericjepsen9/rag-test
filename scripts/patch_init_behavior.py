import re
from pathlib import Path

p = Path("faiss_store.py")
s = p.read_text(encoding="utf-8", errors="replace")

# 1) 把 init 里内置的 docs=[{...},{...}] 示例数据块替换成 docs = load_docs()
s2 = re.sub(
    r"(?ms)^\s*docs\s*=\s*\[\s*\{.*?\}\s*\]\s*\n",
    "    docs = load_docs()\n",
    s
)

# 2) 删除 init 中写回 DOCS_PATH 的代码块（避免覆盖你导入的 docs.jsonl）
s3 = re.sub(
    r"(?ms)^\s*with\s+open\(DOCS_PATH\s*,\s*[\"']w[\"'].*?\)\s+as\s+f:\s*\n(?:\s+.*\n)+",
    "",
    s2
)

if s3 == s:
    print("Patch failed (no changes). 需要更精确的补丁。")
else:
    p.write_text(s3, encoding="utf-8", newline="\n")
    print("Patched OK: init will no longer overwrite docs.jsonl.")
