#!/usr/bin/env python3
"""知识库导入工具：提供原始文档，LLM 自动整理生成结构化知识库文件。

用法：
    # 导入产品知识
    python import_knowledge.py --type product --id my_product --input raw_doc.txt

    # 导入共享知识（项目/设备/材料）
    python import_knowledge.py --type procedure --id my_procedure --input raw_doc.txt

    # 导入单文件共享知识（并发症/疗程等，会追加到现有 main.txt）
    python import_knowledge.py --type complication --input raw_doc.txt

    # 从多个文件导入
    python import_knowledge.py --type product --id my_product --input doc1.txt doc2.txt

    # 导入后自动构建索引
    python import_knowledge.py --type product --id my_product --input raw.txt --build

    # 预览 LLM 整理结果（不写文件）
    python import_knowledge.py --type product --id my_product --input raw.txt --dry-run

支持的文档格式：纯文本(.txt)、Markdown(.md)、PDF(.pdf)
"""

import os
import sys
import argparse
import json
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from rag_runtime_config import (
    KNOWLEDGE_DIR, OPENAI_MODEL, OPENAI_API_BASE,
    SHARED_ENTITY_DIRS,
)

# 实体类型 → (目录类型, 是否单文件)
_ENTITY_TYPES = {
    # 产品
    "product":      ("product",     False),
    # 多实例共享知识
    "procedure":    ("procedures",  False),
    "equipment":    ("equipment",   False),
    "material":     ("materials",   False),
    # 单文件共享知识
    "anatomy":      ("anatomy",      True),
    "indication":   ("indications",  True),
    "complication": ("complications", True),
    "course":       ("courses",      True),
    "script":       ("scripts",      True),
}


def _get_openai_client():
    """获取知识库整理用 LLM client（优先 llm_client 多提供商，回退旧版）"""
    # 优先通过 llm_client 获取知识库专用 client
    try:
        from llm_client import get_client as _get_multi_client, is_enabled as _is_enabled
        if _is_enabled("knowledge"):
            client = _get_multi_client("knowledge")
            if client is not None:
                return client
    except ImportError:
        pass
    # 回退到旧版逻辑
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("未设置 OPENAI_API_KEY 环境变量，且未配置知识库整理用 LLM")
    try:
        from openai import OpenAI
        kwargs = {"api_key": key}
        if OPENAI_API_BASE:
            kwargs["base_url"] = OPENAI_API_BASE
        return OpenAI(**kwargs)
    except ImportError:
        raise RuntimeError("未安装 openai 库，请运行: pip install openai")


def _get_knowledge_model() -> str:
    """获取知识库整理用的模型名称"""
    try:
        from llm_client import get_model as _get_multi_model, is_enabled as _is_enabled
        if _is_enabled("knowledge"):
            m = _get_multi_model("knowledge")
            if m:
                return m
    except ImportError:
        pass
    return OPENAI_MODEL


def _read_input_file(path: str) -> str:
    """读取输入文件，支持 txt/md/pdf"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    suffix = p.suffix.lower()
    if suffix == ".pdf":
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(p) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_parts.append(t)
            return "\n\n".join(text_parts)
        except ImportError:
            raise RuntimeError("读取 PDF 需要安装 pdfplumber: pip install pdfplumber")
    else:
        # txt / md / 其他文本文件
        for enc in ("utf-8-sig", "utf-8", "gbk", "gb2312"):
            try:
                return p.read_text(encoding=enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return p.read_text(errors="replace")


def _llm_call(client, system_prompt: str, user_prompt: str,
              max_tokens: int = 4000) -> str:
    """调用 LLM API"""
    resp = client.chat.completions.create(
        model=_get_knowledge_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    if not resp.choices:
        return ""
    return (resp.choices[0].message.content or "").strip()


# ============================================================
# LLM Prompt 模板
# ============================================================

_SYSTEM_PRODUCT = """你是医美行业知识库整理专家。用户会提供一份关于某个医美产品的原始文档。
请将其整理为结构化的知识库文档。

输出要求（JSON 格式）：
{
  "main_txt": "整理后的主文档内容",
  "faq_txt": "FAQ 问答对",
  "alias_txt": "别名和关键词",
  "product_name": "产品中文名",
  "product_aliases": ["别名1", "别名2", ...]
}

main_txt 整理规则：
1. 按以下章节标题组织（用"一、二、三..."中文编号），内容不存在的章节可以跳过：
   一、产品基础信息（产品名、规格、备案、保质期、适用范围、适用人群）
   二、核心成分与作用（每个成分用"1）2）"编号，列出作用要点）
   三、产品特点与优势
   四、操作方法与注射指南（注射方式、参数、深度、剂量等）
   五、防伪鉴别方法（如有）
   六、术后护理与注意事项（冰敷、清洁、饮食、运动、化妆、防晒等）
   七、禁忌人群
   八、风险与不良反应
   九、联合方案与项目搭配（如有）
   十、效果与维持时间
   十一、术前准备
   十二、方案设计与面部评估（如有）
   十三、修复与补救方案（如有）
2. 内容要具体，保留所有数字参数（深度、剂量、时间等）
3. 不要编造原文中没有的信息，可以在条目后加"（资料描述）"标注
4. 用"- "做要点列表

faq_txt 整理规则：
1. 从原文中提取可能的常见问题，生成 10-20 个 FAQ
2. 格式：【Q】问题\n【A】回答\n\n（每对之间空一行）
3. 覆盖主要话题：产品介绍、操作方式、术后护理、禁忌、效果、防伪等
4. 回答简洁但完整，50-150 字

alias_txt 整理规则：
1. 每行一个别名或关键词
2. 包含：产品正式名、常见别称、英文名、缩写、常见错别字、核心关键词
3. 不超过 25 行"""

_SYSTEM_SHARED = """你是医美行业知识库整理专家。用户会提供一份关于{entity_label}的原始文档。
请将其整理为结构化的知识库文档。

输出要求（JSON 格式）：
{
  "main_txt": "整理后的主文档内容",
  "alias_txt": "别名和关键词（如适用，否则留空）",
  "entity_name": "实体中文名",
  "entity_aliases": ["别名1", "别名2", ...]
}

main_txt 整理规则：
1. 用"一、二、三..."中文编号组织章节
2. 内容结构根据{entity_label}类型自行组织，通常包含：
   - 概述/定义
   - 适用人群/适应症
   - 操作方法/流程
   - 注意事项/风险
   - 效果/预期
3. 内容要具体，保留所有数字参数
4. 不要编造原文中没有的信息
5. 用"- "做要点列表

alias_txt 整理规则：
1. 每行一个别名或关键词
2. 包含：正式名、常见别称、英文名、缩写
3. 不超过 15 行"""

_SYSTEM_SINGLE_FILE = """你是医美行业知识库整理专家。用户会提供一份关于{entity_label}的原始文档。
请将其整理为结构化的知识库内容，会追加到已有的知识库文件中。

输出要求（JSON 格式）：
{
  "main_txt": "整理后的内容"
}

main_txt 整理规则：
1. 用"一、二、三..."中文编号或适当的子标题组织
2. 内容要具体，保留所有数字参数
3. 不要编造原文中没有的信息
4. 用"- "做要点列表"""


_SYSTEM_REFINE = """你是医美行业知识库整理专家。用户对上一次的整理结果不满意，
请根据用户的修改意见进行修订。

规则：
1. 仅修改用户指出的问题部分，保留其他已满意的内容
2. 不要丢失原有的数字参数、具体数据
3. 输出完整的修订后 JSON（不是只输出修改部分）
4. 遵循与首次整理相同的格式规范
5. JSON 格式与首次整理一致（包含 main_txt, faq_txt, alias_txt 等字段）"""


def refine_knowledge(client, current: dict, feedback: str,
                     raw_text: str = "", entity_type: str = "product") -> dict:
    """根据用户反馈修订知识库内容。

    Args:
        client: OpenAI 兼容客户端
        current: 当前整理结果，包含 main_txt / faq_txt / alias_txt
        feedback: 用户修改意见
        raw_text: 可选，原始文档供 LLM 参考
        entity_type: 实体类型

    Returns:
        修订后的完整结果 dict
    """
    user_parts = []
    user_parts.append("【当前整理结果】")
    if current.get("main_txt"):
        user_parts.append(f"main_txt:\n{current['main_txt']}")
    if current.get("faq_txt"):
        user_parts.append(f"faq_txt:\n{current['faq_txt']}")
    if current.get("alias_txt"):
        user_parts.append(f"alias_txt:\n{current['alias_txt']}")

    user_parts.append(f"\n【用户修改意见】\n{feedback}")

    if raw_text:
        # 限制原始文档长度，避免超 token
        max_raw = 6000
        if len(raw_text) > max_raw:
            raw_text = raw_text[:max_raw] + f"\n...(原始文档省略 {len(raw_text) - max_raw} 字)"
        user_parts.append(f"\n【原始文档参考】\n{raw_text}")

    user_prompt = "\n\n".join(user_parts)

    # 根据实体类型决定输出 JSON 字段提示
    if entity_type == "product":
        field_hint = '输出 JSON 需包含字段: main_txt, faq_txt, alias_txt'
    else:
        _, is_single = _ENTITY_TYPES.get(entity_type, ("", False))
        if is_single:
            field_hint = '输出 JSON 需包含字段: main_txt'
        else:
            field_hint = '输出 JSON 需包含字段: main_txt, alias_txt'

    system_prompt = _SYSTEM_REFINE + f"\n\n{field_hint}"

    result_text = _llm_call(client, system_prompt, user_prompt, max_tokens=4000)

    # 解析 JSON（复用首次整理的解析逻辑）
    cleaned = result_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{[\s\S]*\}', cleaned)
        if m:
            try:
                result = json.loads(m.group())
            except json.JSONDecodeError:
                raise ValueError(f"LLM 返回内容无法解析为 JSON: {cleaned[:300]}")
        else:
            raise ValueError(f"LLM 返回内容中未找到 JSON: {cleaned[:300]}")

    return result


_ENTITY_LABELS = {
    "product":      "医美产品",
    "procedure":    "医美项目/手术",
    "equipment":    "医美设备/仪器",
    "material":     "医美材料/成分",
    "anatomy":      "面部解剖/分区治疗",
    "indication":   "皮肤适应症/问题",
    "complication": "术后并发症/处理",
    "course":       "疗程规划",
    "script":       "客户沟通话术",
}


def _generate_knowledge(client, raw_text: str, entity_type: str,
                         entity_id: str = "") -> dict:
    """调用 LLM 将原始文档整理为结构化知识库内容"""
    label = _ENTITY_LABELS.get(entity_type, entity_type)
    _, is_single = _ENTITY_TYPES[entity_type]

    if entity_type == "product":
        system = _SYSTEM_PRODUCT
    elif is_single:
        system = _SYSTEM_SINGLE_FILE.format(entity_label=label)
    else:
        system = _SYSTEM_SHARED.format(entity_label=label)

    user_prompt = f"以下是关于「{entity_id or label}」的原始文档，请整理为结构化知识库内容：\n\n{raw_text}"

    # 原始文档可能很长，分批处理
    # 限制发送给 LLM 的文本长度（约 12000 字 ≈ 6000 tokens）
    max_chars = 12000
    if len(raw_text) > max_chars:
        print(f"[INFO] 原始文档较长（{len(raw_text)} 字），将分段处理")
        # 先发送前半部分生成主结构
        part1 = raw_text[:max_chars]
        part2 = raw_text[max_chars:]

        user_prompt_1 = (
            f"以下是关于「{entity_id or label}」的原始文档（第1部分，共2部分）。"
            f"请先整理这部分内容：\n\n{part1}"
        )
        result_text = _llm_call(client, system, user_prompt_1, max_tokens=4000)

        # 再发送后半部分补充
        if len(part2) > max_chars:
            print(f"[WARN] 文档第2部分仍超长（{len(part2)} 字），截断至 {max_chars} 字，"
                  f"丢弃 {len(part2) - max_chars} 字内容")
        user_prompt_2 = (
            f"以下是文档的第2部分，请整理并补充到之前的结果中。"
            f"输出完整的最终 JSON（合并两部分内容）：\n\n"
            f"第1部分整理结果：\n{result_text}\n\n"
            f"第2部分原文：\n{part2[:max_chars]}"
        )
        result_text = _llm_call(client, system, user_prompt_2, max_tokens=4000)
    else:
        result_text = _llm_call(client, system, user_prompt, max_tokens=4000)

    # 解析 JSON 结果
    # LLM 可能返回 ```json ... ``` 包裹的内容
    cleaned = result_text.strip()
    if cleaned.startswith("```"):
        # 去除 markdown 代码块标记
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # 尝试提取 JSON 部分
        import re
        m = re.search(r'\{[\s\S]*\}', cleaned)
        if m:
            try:
                result = json.loads(m.group())
            except json.JSONDecodeError:
                raise ValueError(f"LLM 返回内容无法解析为 JSON：{cleaned[:500]}")
        else:
            raise ValueError(f"LLM 返回内容中未找到 JSON：{cleaned[:500]}")

    return result


def _write_knowledge_files(result: dict, entity_type: str, entity_id: str,
                            dry_run: bool = False) -> Path:
    """将 LLM 整理结果写入知识库文件"""
    _, is_single = _ENTITY_TYPES[entity_type]

    if entity_type == "product":
        out_dir = KNOWLEDGE_DIR / entity_id
    elif is_single:
        dir_name = _ENTITY_TYPES[entity_type][0]
        out_dir = KNOWLEDGE_DIR / dir_name
    else:
        dir_name = _ENTITY_TYPES[entity_type][0]
        out_dir = KNOWLEDGE_DIR / dir_name / entity_id

    if dry_run:
        print(f"\n{'='*60}")
        print(f"[DRY RUN] 目标目录: {out_dir}")
        print(f"{'='*60}")
        if result.get("main_txt"):
            print(f"\n--- main.txt ({len(result['main_txt'])} 字) ---")
            print(result["main_txt"][:2000])
            if len(result["main_txt"]) > 2000:
                print(f"\n... (省略 {len(result['main_txt']) - 2000} 字)")
        if result.get("faq_txt"):
            print(f"\n--- faq.txt ({len(result['faq_txt'])} 字) ---")
            print(result["faq_txt"][:1000])
        if result.get("alias_txt"):
            print(f"\n--- alias.txt ---")
            print(result["alias_txt"])
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    # main.txt
    main_txt = result.get("main_txt", "")
    if main_txt:
        main_path = out_dir / "main.txt"
        if is_single and main_path.exists():
            # 单文件模式：追加内容
            existing = main_path.read_text(encoding="utf-8")
            main_txt = existing.rstrip() + "\n\n" + main_txt
            print(f"[INFO] 追加内容到已有文件: {main_path}")
        main_path.write_text(main_txt, encoding="utf-8")
        print(f"[OK] 写入 {main_path} ({len(main_txt)} 字)")

    # faq.txt（仅产品类型）
    faq_txt = result.get("faq_txt", "")
    if faq_txt and entity_type == "product":
        faq_path = out_dir / "faq.txt"
        faq_path.write_text(faq_txt, encoding="utf-8")
        print(f"[OK] 写入 {faq_path} ({len(faq_txt)} 字)")

    # alias.txt
    alias_txt = result.get("alias_txt", "")
    if alias_txt:
        alias_path = out_dir / "alias.txt"
        alias_path.write_text(alias_txt, encoding="utf-8")
        print(f"[OK] 写入 {alias_path}")

    return out_dir


def _print_registration_hint(result: dict, entity_type: str, entity_id: str):
    """打印别名注册提示"""
    if entity_type == "product":
        aliases = result.get("product_aliases", [])
        name = result.get("product_name", entity_id)
        if aliases:
            alias_str = json.dumps(aliases, ensure_ascii=False)
            print(f"\n[提示] 请在 rag_runtime_config.py 的 PRODUCT_ALIASES 中添加：")
            print(f'    "{entity_id}": {alias_str},')
    elif entity_type == "procedure":
        aliases = result.get("entity_aliases", [])
        if aliases:
            alias_str = json.dumps(aliases, ensure_ascii=False)
            print(f"\n[提示] 请在 rag_runtime_config.py 的 PROCEDURE_ALIASES 中添加：")
            print(f'    "{entity_id}": {alias_str},')
    elif entity_type == "equipment":
        aliases = result.get("entity_aliases", [])
        if aliases:
            alias_str = json.dumps(aliases, ensure_ascii=False)
            print(f"\n[提示] 请在 rag_runtime_config.py 的 EQUIPMENT_ALIASES 中添加：")
            print(f'    "{entity_id}": {alias_str},')
    elif entity_type == "material":
        aliases = result.get("entity_aliases", [])
        if aliases:
            alias_str = json.dumps(aliases, ensure_ascii=False)
            print(f"\n[提示] 请在 rag_runtime_config.py 的 MATERIAL_ALIASES 中添加：")
            print(f'    "{entity_id}": {alias_str},')


def _build_index(entity_type: str, entity_id: str):
    """构建 FAISS 索引"""
    from build_faiss import build_for_product, build_shared

    _, is_single = _ENTITY_TYPES[entity_type]

    if entity_type == "product":
        print(f"\n[INFO] 构建产品索引: {entity_id}")
        build_for_product(entity_id)
    else:
        print(f"\n[INFO] 构建共享知识索引")
        build_shared()

    print("[DONE] 索引构建完成")


def main():
    ap = argparse.ArgumentParser(
        description="知识库导入工具：原始文档 → LLM 整理 → 结构化知识库文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 导入产品
  python import_knowledge.py --type product --id botox --input botox资料.txt --build

  # 导入项目
  python import_knowledge.py --type procedure --id thermage --input 热玛吉.txt --build

  # 预览（不写文件）
  python import_knowledge.py --type product --id test --input doc.txt --dry-run
        """,
    )
    ap.add_argument("--type", required=True, choices=list(_ENTITY_TYPES.keys()),
                    help="知识类型: product/procedure/equipment/material/anatomy/indication/complication/course/script")
    ap.add_argument("--id", type=str, default="",
                    help="实体ID（目录名），product/procedure/equipment/material 必填")
    ap.add_argument("--input", nargs="+", required=True,
                    help="输入文件路径（支持 .txt .md .pdf，可指定多个）")
    ap.add_argument("--build", action="store_true",
                    help="导入后自动构建 FAISS 索引")
    ap.add_argument("--dry-run", action="store_true",
                    help="仅预览 LLM 整理结果，不写入文件")
    args = ap.parse_args()

    entity_type = args.type
    entity_id = args.id.strip()
    _, is_single = _ENTITY_TYPES[entity_type]

    # 校验：非单文件类型必须提供 --id
    if not is_single and not entity_id:
        ap.error(f"--type {entity_type} 需要提供 --id 参数（作为目录名）")

    # 读取所有输入文件
    raw_parts = []
    for input_path in args.input:
        print(f"[INFO] 读取文件: {input_path}")
        text = _read_input_file(input_path)
        if text.strip():
            raw_parts.append(text.strip())
            print(f"  → {len(text)} 字")
        else:
            print(f"  → 文件为空，跳过")

    if not raw_parts:
        print("[ERROR] 所有输入文件均为空")
        sys.exit(1)

    raw_text = "\n\n---\n\n".join(raw_parts)
    print(f"[INFO] 总计 {len(raw_text)} 字原始内容")

    # 调用 LLM 整理
    print(f"[INFO] 调用 LLM 整理内容（模型: {_get_knowledge_model()}）...")
    client = _get_openai_client()
    result = _generate_knowledge(client, raw_text, entity_type, entity_id)
    print(f"[OK] LLM 整理完成")

    # 写入文件
    out_dir = _write_knowledge_files(result, entity_type, entity_id,
                                      dry_run=args.dry_run)

    # 注册提示
    if not args.dry_run:
        _print_registration_hint(result, entity_type, entity_id)

    # 构建索引
    if args.build and not args.dry_run:
        _build_index(entity_type, entity_id)

    print(f"\n{'='*60}")
    if args.dry_run:
        print("[完成] 预览模式，未写入任何文件")
    else:
        print(f"[完成] 知识已导入到: {out_dir}")
        if not args.build:
            if entity_type == "product":
                print(f"[提示] 运行以下命令构建索引：")
                print(f"  python build_faiss.py --product {entity_id}")
            else:
                print(f"[提示] 运行以下命令构建共享索引：")
                print(f"  python build_faiss.py --shared")


if __name__ == "__main__":
    main()
