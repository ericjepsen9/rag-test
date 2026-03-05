# config/settings.py
# 统一配置入口（给 app/search.py、build_faiss.py 等使用）

from pathlib import Path
import os

# 项目根目录（.../bge-m3-test）
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------
# 模型配置
# ---------------------------
EMBED_MODEL_NAME = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = int(os.environ.get("RAG_EMBED_DIM", "1024"))

# 是否优先 fp16（GPU 环境建议 True；CPU 环境可设 False）
USE_FP16 = os.environ.get("RAG_USE_FP16", "1") in ("1", "true", "True")

# ---------------------------
# 数据目录（新结构）
# ---------------------------
# 你的知识库源文件目录：knowledge/<product_id>/{main,faq,alias}.txt
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

# 向量库存储目录：stores/<product_id>/{index.faiss,docs.jsonl}
STORES_DIR = BASE_DIR / "stores"

# 兼容旧结构（可保留，不建议继续使用）
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"

# 默认产品（当没传 product_id 时使用）
DEFAULT_PRODUCT_ID = os.environ.get("RAG_PRODUCT_ID", "feiluoao")

# ---------------------------
# 当前产品对应的 store 路径（运行时可被环境变量覆盖）
# ---------------------------
def get_product_id(explicit_product_id: str | None = None) -> str:
    pid = (explicit_product_id or os.environ.get("RAG_PRODUCT_ID") or DEFAULT_PRODUCT_ID).strip()
    return pid or DEFAULT_PRODUCT_ID

def get_store_dir(explicit_product_id: str | None = None) -> Path:
    # 优先环境变量直接指定目录（调试用）
    env_store = os.environ.get("RAG_STORE_DIR", "").strip()
    if env_store:
        return Path(env_store)
    return STORES_DIR / get_product_id(explicit_product_id)

def get_index_path(explicit_product_id: str | None = None) -> Path:
    return get_store_dir(explicit_product_id) / "index.faiss"

def get_docs_path(explicit_product_id: str | None = None) -> Path:
    return get_store_dir(explicit_product_id) / "docs.jsonl"

# ── 修复 5：移除模块级静态常量 STORE_DIR / INDEX_PATH / DOCS_PATH ───────
# 原来的写法在模块导入时就固化了路径，之后修改 RAG_PRODUCT_ID 等环境变量
# 不会生效。所有调用方应改为直接调用上面的函数获取最新路径：
#   get_store_dir()   get_index_path()   get_docs_path()

# ---------------------------
# 构建参数（分块）
# ---------------------------
# 你后续可以按文档类型微调
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "100"))

# ---------------------------
# 检索参数（默认值）
# ---------------------------
TOP_K_DEFAULT = int(os.environ.get("RAG_TOP_K", "8"))

# ---------------------------
# 调试打印
# ---------------------------
def print_runtime_config():
    pid = get_product_id()
    print("[RAG settings]")
    print(f"  BASE_DIR       = {BASE_DIR}")
    print(f"  KNOWLEDGE_DIR  = {KNOWLEDGE_DIR}")
    print(f"  STORES_DIR     = {STORES_DIR}")
    print(f"  PRODUCT_ID     = {pid}")
    print(f"  STORE_DIR      = {get_store_dir(pid)}")
    print(f"  INDEX_PATH     = {get_index_path(pid)}")
    print(f"  DOCS_PATH      = {get_docs_path(pid)}")
    print(f"  EMBED_MODEL    = {EMBED_MODEL_NAME}")
    print(f"  EMBED_DIM      = {EMBED_DIM}")
    print(f"  USE_FP16       = {USE_FP16}")
