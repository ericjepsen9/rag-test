import json
from pathlib import Path

src = Path("faiss_store/docs.jsonl")
dst = Path("faiss_store/docs_fixed.jsonl")

def build_cp1252_char_to_byte():
    """
    建立 cp1252 中“字节->字符”里那些 ord>255 的字符映射回原始字节。
    例如 0x9B -> '›' (U+203A)，我们需要 '›' -> 0x9B
    """
    m = {}
    for b in range(256):
        ch = bytes([b]).decode("cp1252", errors="ignore")
        if ch and ord(ch) > 255:
            m[ch] = b
    return m

CP1252_EXTRA = build_cp1252_char_to_byte()

def recover_bytes(s: str) -> bytes:
    """
    把“乱码字符串”还原成原始字节序列：
    - ord<=255 的字符直接当作单字节
    - ord>255 的字符（如 › • € …）用 cp1252 映射回原始字节
    """
    out = bytearray()
    for ch in s:
        o = ord(ch)
        if o <= 255:
            out.append(o)
        elif ch in CP1252_EXTRA:
            out.append(CP1252_EXTRA[ch])
        else:
            # 遇到无法映射的字符，直接保留为 '?'
            out.append(ord("?"))
    return bytes(out)

def fix_text(s: str) -> str:
    """
    关键修复：把字符串按“被错误解码后的字符”还原为原始字节，再按 UTF-8 解码。
    成功则得到中文；失败就原样返回。
    """
    try:
        b = recover_bytes(s)
        return b.decode("utf-8")
    except Exception:
        return s

fixed_lines = 0
total = 0
out_lines = []

with src.open("r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total += 1
        obj = json.loads(line)

        if isinstance(obj.get("text"), str):
            before = obj["text"]
            after = fix_text(before)
            if after != before:
                fixed_lines += 1
            obj["text"] = after

        out_lines.append(json.dumps(obj, ensure_ascii=False))

dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
print(f"Done. total={total}, fixed_lines={fixed_lines}")
print(f"Wrote: {dst}")
