# scripts/import_docs.py
# 用法示例：
#   python scripts/import_docs.py --from-dir . --product feiluoao
#   python scripts/import_docs.py --from-dir data/docs --product feiluoao
#   python scripts/import_docs.py --from-dir . --auto
#
# 作用：
# - 将旧文件（如 feiluoao_main.txt / feiluoao_faq.txt / feiluoao_alias.txt）
#   导入到新结构：
#   knowledge/<product_id>/main.txt
#   knowledge/<product_id>/faq.txt
#   knowledge/<product_id>/alias.txt

import os
import re
import sys
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional


# ===== Windows UTF-8 输出 =====
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_KNOWLEDGE_DIR = BASE_DIR / "knowledge"


def print_safe(*args):
    s = " ".join(str(x) for x in args)
    try:
        print(s)
    except Exception:
        print(s.encode("utf-8", "backslashreplace").decode("ascii", "ignore"))


def read_text_auto(path: Path) -> str:
    if not path.exists():
        return ""
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return ""


def write_text_utf8(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def clean_product_id(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def detect_source_type(file_name: str) -> Optional[str]:
    n = file_name.lower()
    if "main" in n:
        return "main"
    if "faq" in n:
        return "faq"
    if "alias" in n:
        return "alias"
    return None


def group_legacy_files(files: List[Path]) -> Dict[str, Dict[str, Path]]:
    """
    将旧命名文件分组：
    - feiluoao_main.txt -> product=feiluoao, source_type=main
    - abc_faq.txt      -> product=abc, source_type=faq
    """
    result: Dict[str, Dict[str, Path]] = {}
    for f in files:
        if not f.is_file():
            continue
        if f.suffix.lower() != ".txt":
            continue

        name = f.stem  # without .txt
        source_type = detect_source_type(f.name)
        if not source_type:
            continue

        # 从文件名里去掉 _main / _faq / _alias
        product_part = re.sub(r"(?i)[_\-]?(main|faq|alias)$", "", name).strip("_- ")
        product_id = clean_product_id(product_part)
        if not product_id:
            continue

        result.setdefault(product_id, {})
        result[product_id][source_type] = f

    return result


def import_one_product(from_dir: Path, product_id: str, knowledge_dir: Path):
    product_id = clean_product_id(product_id)

    candidates = {
        "main": [
            from_dir / f"{product_id}_main.txt",
            from_dir / "main.txt",  # 兼容直接放 main.txt 的情况
        ],
        "faq": [
            from_dir / f"{product_id}_faq.txt",
            from_dir / "faq.txt",
        ],
        "alias": [
            from_dir / f"{product_id}_alias.txt",
            from_dir / "alias.txt",
        ],
    }

    out_dir = knowledge_dir / product_id
    out_dir.mkdir(parents=True, exist_ok=True)

    imported = 0
    for stype, paths in candidates.items():
        src = next((p for p in paths if p.exists()), None)
        if not src:
            if stype == "main":
                print_safe(f"[WARN] 未找到 main 文件（{product_id}）")
            continue

        txt = read_text_auto(src)
        if not txt.strip():
            print_safe(f"[WARN] 空文件跳过: {src}")
            continue

        dst = out_dir / f"{stype}.txt"
        write_text_utf8(dst, txt)
        imported += 1
        print_safe(f"[OK] {src}  ->  {dst}")

    if imported == 0:
        print_safe(f"[SKIP] {product_id}: 未导入任何文件")
    else:
        print_safe(f"[DONE] {product_id}: 导入 {imported} 个文件")


def auto_import(from_dir: Path, knowledge_dir: Path):
    txt_files = [p for p in from_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    grouped = group_legacy_files(txt_files)

    if not grouped:
        print_safe("[INFO] 未发现可识别的旧命名文件（如 xxx_main.txt / xxx_faq.txt / xxx_alias.txt）")
        return

    print_safe("[INFO] 发现产品：", ", ".join(sorted(grouped.keys())))

    for product_id, mp in sorted(grouped.items()):
        out_dir = knowledge_dir / product_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for stype in ("main", "faq", "alias"):
            src = mp.get(stype)
            if not src:
                continue
            txt = read_text_auto(src)
            if not txt.strip():
                print_safe(f"[WARN] 空文件跳过: {src}")
                continue
            dst = out_dir / f"{stype}.txt"
            write_text_utf8(dst, txt)
            print_safe(f"[OK] {src.name} -> knowledge/{product_id}/{stype}.txt")

        print_safe(f"[DONE] {product_id}")


def main():
    parser = argparse.ArgumentParser(description="导入旧 txt 文件到 knowledge 新结构")
    parser.add_argument("--from-dir", type=str, default=str(BASE_DIR), help="旧文件所在目录（默认项目根目录）")
    parser.add_argument("--product", type=str, default="", help="指定产品目录名（如 feiluoao）")
    parser.add_argument("--auto", action="store_true", help="自动扫描 xxx_main.txt / xxx_faq.txt / xxx_alias.txt")
    parser.add_argument("--knowledge-dir", type=str, default=str(DEFAULT_KNOWLEDGE_DIR), help="knowledge 目录路径")
    args = parser.parse_args()

    from_dir = Path(args.from_dir).resolve()
    knowledge_dir = Path(args.knowledge_dir).resolve()

    if not from_dir.exists():
        print_safe(f"[ERROR] from-dir 不存在: {from_dir}")
        return

    knowledge_dir.mkdir(parents=True, exist_ok=True)

    if args.auto:
        auto_import(from_dir, knowledge_dir)
        return

    if args.product:
        import_one_product(from_dir, args.product, knowledge_dir)
        return

    print_safe("请使用以下任一方式：")
    print_safe("  python scripts/import_docs.py --from-dir . --product feiluoao")
    print_safe("  python scripts/import_docs.py --from-dir . --auto")


if __name__ == "__main__":
    main()