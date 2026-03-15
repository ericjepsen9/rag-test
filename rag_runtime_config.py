import os as _os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ===== 基础路径 =====
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
STORE_ROOT = BASE_DIR / "stores"
OUT_PATH = BASE_DIR / "answer.txt"

# ===== OpenAI / 兼容 API 开关 =====
# 支持 Cherry Studio 等 OpenAI 兼容 API 服务：
#   export RAG_USE_OPENAI=1
#   export OPENAI_API_KEY=cs-sk-...
#   export OPENAI_API_BASE=http://127.0.0.1:23333/v1
#   export RAG_OPENAI_MODEL=your-model-name
USE_OPENAI = _os.environ.get("RAG_USE_OPENAI", "").strip().lower() in ("1", "true", "yes")
OPENAI_MODEL = _os.environ.get("RAG_OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_BASE = _os.environ.get("OPENAI_API_BASE", "").strip() or None

# ===== 输出与调试 =====
DEFAULT_MODE = "brief"   # brief / full
DEFAULT_TOP_K = 8

# ===== 回答模板 =====
REFERENCE_NOTE = "结果仅供参考，需询问专业医师。"
RISK_NOTE = "仅供参考，需医生评估。"

# ===== 问题路由 =====
QUESTION_ROUTES = {
    "risk": ["红肿", "结节", "疼痛", "感染", "硬块", "异常", "副作用", "不良反应", "发热", "肿胀",
             "肿", "淤青", "瘀青", "发紫", "发黑", "变黑", "色沉", "红疹", "疹子", "痒", "瘙痒",
             "不消", "越来越", "化脓", "溃烂", "坏死"],
    "aftercare": ["术后", "护理", "恢复", "注意事项", "洗脸", "面膜", "保湿", "禁酒", "禁忌行为",
                  "运动", "健身", "出汗", "洗澡", "游泳", "桑拿", "汗蒸", "泡澡",
                  "化妆", "上妆", "卸妆", "防晒", "晒太阳", "紫外线", "日晒",
                  "喝酒", "饮酒", "辛辣", "吃辣", "海鲜", "咖啡", "喝茶",
                  "上班", "工作", "几天能", "多久能", "做完能", "做完可以"],
    "operation": ["注射", "深度", "参数", "微针", "水光", "操作", "针头", "0.8mm", "0.3ml", "2cm",
                  "疗程", "麻醉", "敷麻", "疼不疼", "痛不痛", "疼吗", "痛吗"],
    "anti_fake": ["防伪", "真伪", "HiddenTag", "正品", "鉴别", "验真"],
    "contraindication": ["禁忌", "禁忌人群", "不适合", "过敏体质", "妊娠", "哺乳",
                         "孕妇", "怀孕", "备孕", "敏感肌", "皮肤敏感",
                         "阿司匹林", "抗凝", "免疫抑制",
                         "多大年纪", "年龄", "岁能", "18岁", "未成年",
                         "男性", "男生", "男人", "自己打", "自己注射", "在家打"],
    "combo": ["联合", "同做", "一起做", "间隔", "配合", "搭配"],
    "ingredient": ["成分", "PCL", "聚己内酯", "透明质酸", "玻尿酸", "谷胱甘肽", "肽", "生长因子",
                    "HA", "胶原蛋白", "交联", "分子量"],
    "basic": ["是什么", "作用", "适用", "产品", "功效", "备案", "规格"],
    "effect": ["效果", "维持", "见效", "多久", "持续", "变白", "变亮", "能维持", "几个月"],
    "pre_care": ["术前", "准备", "检查", "停药", "素颜", "到院"],
    "design": ["设计", "方案", "评估", "规划", "怎么打", "打哪里", "用量",
               "几支", "松弛程度", "面部分析"],
    "repair": ["修复", "补救", "不理想", "不均匀", "不对称", "矫正", "返修",
               "做坏了", "做失败", "效果差", "重新做", "之前做的"],
    # --- 跨实体路由（检索共享知识库） ---
    "complication": ["术后第", "术后当天", "天了还", "正常吗", "并发症", "急诊", "就医",
                     "肿了", "化脓", "坏死", "栓塞", "水泡"],
    "course": ["疗程", "几次", "间隔多久", "总共", "多长时间", "规划", "时间表",
               "多少钱", "费用", "预算"],
    "anatomy_q": ["哪个部位", "额头", "苹果肌", "法令纹", "下颌线", "颈部",
                  "鼻子", "手背", "眼周", "泪沟"],
    "indication_q": ["松弛怎么", "干燥怎么", "毛孔怎么", "色斑怎么", "痘坑怎么",
                     "皱纹怎么", "用什么好", "适合什么", "推荐什么",
                     "毛孔粗大", "皮肤松弛", "皮肤干燥", "痘坑用", "暗沉用",
                     "怎么改善", "缺水怎么", "暗沉怎么", "粗糙怎么",
                     "方法改善", "什么方法"],
    "procedure_q": ["水光针是什么", "微针是什么", "光电是什么", "什么项目",
                    "项目有哪些", "操作流程", "有什么区别", "哪个好", "哪个更",
                    "有什么特点", "为什么选择", "原理是什么", "适合什么人",
                    "对标", "竞品", "替代",
                    # 项目实体短关键词（确保单独提及时能路由到共享知识库）
                    "水光", "水光针", "打水光", "微针", "MTS", "光电",
                    "光子嫩肤", "射频", "热玛吉", "热拉提", "超声刀",
                    "填充", "注射填充", "皮秒", "点阵激光", "IPL"],
    "equipment_q": ["什么仪器", "什么设备", "什么机器", "水光仪", "微针仪",
                    "射频仪", "仪器参数", "适配", "品牌的仪器",
                    "电动", "设备型号", "配置", "维护成本", "培训",
                    # 设备实体短关键词
                    "水光机", "水光注射仪", "德玛莎", "水光枪",
                    "微针笔", "电动微针", "飞针仪", "射频美容仪", "热玛吉仪"],
    "script": ["怎么解释", "怎么回答", "客户问", "话术", "怎么介绍",
               "客户担心", "怎么说", "向客户", "跟客户", "给客户",
               "风险话术", "期望管理", "合规表达", "医学严谨"],
}

# ===== 章节标题映射（主文档优先） =====
SECTION_RULES = {
    "anti_fake": {
        "titles": ["五、防伪鉴别方法", "防伪鉴别方法", "防伪"],
        "stops": ["六、", "七、", "术后护理", "禁忌人群"],
    },
    "aftercare": {
        "titles": ["六、术后护理与注意事项", "术后护理与注意事项", "术后护理"],
        "stops": ["七、", "五、", "禁忌人群", "防伪"],
    },
    "contraindication": {
        "titles": ["七、禁忌人群", "禁忌人群"],
        "stops": ["八、", "六、", "五、", "术后护理", "防伪"],
    },
    "operation": {
        "titles": ["四、操作方法与注射指南", "操作方法与注射指南", "操作方法", "注射指南"],
        "stops": ["五、", "六、", "七、", "防伪", "术后护理", "禁忌人群"],
    },
    "risk": {
        "titles": ["八、风险与不良反应", "风险与不良反应", "风险", "异常反应", "不良反应", "副作用"],
        "stops": ["九、", "十、", "联合方案"],
    },
    "combo": {
        "titles": ["九、联合方案与项目搭配", "联合方案与项目搭配", "联合方案", "联合", "搭配", "项目搭配"],
        "stops": ["八、", "十、", "风险"],
    },
    "ingredient": {
        "titles": ["二、核心成分与作用", "核心成分与作用", "核心成分", "成分与作用"],
        "stops": ["三、", "四、", "五、", "六、", "七、", "八、", "九、"],
    },
    "basic": {
        "titles": ["一、产品基础信息", "产品基础信息", "基础信息"],
        "stops": ["二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、", "十、"],
    },
    "effect": {
        "titles": ["十、效果与维持时间", "效果与维持时间"],
        "stops": ["十一、", "十二、", "十三、"],
    },
    "pre_care": {
        "titles": ["十一、术前准备", "术前准备"],
        "stops": ["十、", "十二、", "十三、"],
    },
    "design": {
        "titles": ["十二、方案设计与面部评估", "方案设计与面部评估"],
        "stops": ["十一、", "十三、", "十、"],
    },
    "repair": {
        "titles": ["十三、修复与补救方案", "修复与补救方案"],
        "stops": ["十二、", "十一、", "十、"],
    },
    # --- 共享知识路由的章节规则 ---
    "complication": {
        "titles": ["二、按症状分类的处理决策树", "按症状分类", "术后并发症"],
        "stops": ["五、紧急情况"],
    },
    "course": {
        "titles": ["一、疗程规划总则", "疗程规划", "疗程方案"],
        "stops": ["三、", "四、", "项目对比"],
    },
    "anatomy_q": {
        "titles": ["一、面部分区与常见治疗方案", "面部分区"],
        "stops": ["三、", "四、", "常见问题"],
    },
    "indication_q": {
        "titles": ["一、按皮肤状态分类的治疗推荐", "按皮肤状态"],
        "stops": ["三、", "四、", "常见问题"],
    },
    "procedure_q": {
        "titles": ["一、项目概述", "项目概述", "项目名称"],
        "stops": ["二、", "三、"],
    },
    "equipment_q": {
        "titles": ["一、设备概述", "设备概述", "设备名称"],
        "stops": ["二、", "三、"],
    },
    "script": {
        "titles": ["一、客户常见顾虑应答指南", "客户常见顾虑"],
        "stops": ["三、", "四、"],
    },
}

# ===== 别名与多实体 =====
PRODUCT_ALIASES = {
    "feiluoao": ["菲罗奥", "非罗奥", "菲洛奥", "赛罗菲", "赛罗菲提升", "CELLOFILL", "FILLOUP"],
    "sailuofei_vface": ["赛洛菲V脸溶脂", "赛洛菲V脸", "V脸溶脂", "赛洛菲溶脂", "赛洛菲"],
}
PROJECT_ALIASES = {
    "水光": ["水光", "水光针"],
    "微针": ["微针", "MTS"],
    "光电": ["光电", "光子", "射频"],
    "填充": ["填充", "填充剂"],
}

# ===== 共享知识实体（非产品级，跨产品通用） =====
SHARED_ENTITY_DIRS = {
    "procedure":    "procedures",
    "equipment":    "equipment",
    "anatomy":      "anatomy",
    "indication":   "indications",
    "complication":  "complications",
    "course":       "courses",
    "script":       "scripts",
    "material":     "materials",
}

# 项目/设备/适应症别名 → 对应目录名
PROCEDURE_ALIASES = {
    "water_light":   ["水光", "水光针", "水光注射", "打水光"],
    "microneedling": ["微针", "MTS", "MTS微针", "微针美塑", "滚针", "飞针"],
    "photoelectric": ["光电", "光子嫩肤", "IPL", "射频", "热玛吉", "热拉提",
                      "RF", "点阵激光", "皮秒", "超声刀", "光电项目"],
    "filling":       ["填充", "玻尿酸", "注射填充", "丰唇", "泪沟填充",
                      "苹果肌填充", "自体脂肪"],
}

EQUIPMENT_ALIASES = {
    "water_light_machine": ["水光仪", "水光机", "水光注射仪", "德玛莎", "水光枪"],
    "mts_device":          ["微针仪", "微针笔", "电动微针", "滚轮微针", "飞针仪"],
    "rf_device":           ["射频仪", "热玛吉仪", "射频美容仪"],
}

INDICATION_KEYWORDS = {
    "松弛":   ["松弛", "下垂", "轮廓模糊", "法令纹"],
    "干燥":   ["干燥", "缺水", "脱皮", "紧绷"],
    "毛孔":   ["毛孔", "毛孔粗大"],
    "色斑":   ["色斑", "斑", "肤色不均", "暗沉"],
    "痘坑":   ["痘坑", "痘印", "瘢痕", "痘疤"],
    "皱纹":   ["皱纹", "细纹", "抬头纹", "川字纹", "鱼尾纹", "颈纹"],
}

ANATOMY_KEYWORDS = {
    "额部":   ["额头", "额部", "抬头纹"],
    "眼周":   ["眼周", "眼袋", "泪沟", "黑眼圈", "鱼尾纹"],
    "苹果肌": ["苹果肌", "颧骨"],
    "法令纹": ["法令纹", "鼻唇沟"],
    "下颌线": ["下颌线", "轮廓线", "双下巴", "下颌"],
    "颈部":   ["颈部", "脖子", "颈纹"],
    "鼻部":   ["鼻子", "鼻部", "鼻梁", "鼻头"],
    "手部":   ["手部", "手背"],
}

TIME_TERMS = ["当天", "术后当天", "术后1天", "术后1-3天", "术后3天", "术后1周", "一周", "术后1个月"]
SYMPTOM_TERMS = ["红肿", "结节", "疼痛", "过敏", "硬块", "发热", "感染", "刺痛",
                 "肿", "淤青", "瘀青", "发紫", "红疹", "疹子", "痒", "化脓"]

# ===== 无知识兜底回复（价格、对比等无法回答的问题） =====
PRICE_REPLY = ("菲罗奥的具体价格因医院、区域和疗程方案不同而有差异，"
               "建议直接咨询所在医院或门诊获取准确报价。")
COMPARISON_REPLY = ("不同产品的适应症、成分和效果机制各有不同，"
                    "建议咨询医生根据您的皮肤状况选择最合适的方案，避免盲目对比。")
LOCATION_REPLY = "建议通过官方渠道或正规医美机构查询可做菲罗奥项目的医院。"

# ===== 实体关联 =====
RELATIONS_FILE = KNOWLEDGE_DIR / "relations.json"

# ===== 模型参数 =====
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_USE_FP16 = True
EMBED_BATCH_SIZE_BUILD = 8
EMBED_BATCH_SIZE_QUERY = 1
EMBED_MAX_LENGTH_BUILD = 8192
EMBED_MAX_LENGTH_QUERY = 1024
CHUNK_SIZE = 420
CHUNK_OVERLAP = 50

# ===== LLM 参数 =====
# LLM 查询改写开关：当静态同义词/别名无法识别用户术语时，调用 LLM 改写查询
# 依赖 USE_OPENAI=1，可独立关闭以节省 token
LLM_REWRITE_ENABLED = _os.environ.get("RAG_LLM_REWRITE", "1").strip().lower() in ("1", "true", "yes")

LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS_BRIEF = 1500
LLM_MAX_TOKENS_FULL = 2500

# 路由专属温度：风险/禁忌类需要确定性低温，效果/方案设计可稍高
ROUTE_LLM_TEMPERATURE = {
    "risk":              0.1,
    "contraindication":  0.1,
    "complication":      0.1,
    "repair":            0.1,
    "operation":         0.2,
    "pre_care":          0.2,
    "anti_fake":         0.2,
    "basic":             0.3,
    "ingredient":        0.3,
    "aftercare":         0.3,
    "combo":             0.3,
    "course":            0.3,
    "effect":            0.4,
    "design":            0.4,
    "script":            0.4,
    "anatomy_q":         0.3,
    "indication_q":      0.3,
    "procedure_q":       0.3,
    "equipment_q":       0.3,
}

# ===== 搜索调优 =====


def _safe_float(key: str, default: str) -> float:
    """安全读取环境变量并转为 float，非法值回退默认值并打印警告。"""
    raw = _os.environ.get(key, default)
    try:
        return float(raw)
    except (ValueError, TypeError):
        print(f"[WARN] 环境变量 {key}='{raw}' 无法转为 float，使用默认值 {default}")
        return float(default)


def _safe_int(key: str, default: str) -> int:
    """安全读取环境变量并转为 int，非法值回退默认值并打印警告。"""
    raw = _os.environ.get(key, default)
    try:
        return int(raw)
    except (ValueError, TypeError):
        print(f"[WARN] 环境变量 {key}='{raw}' 无法转为 int，使用默认值 {default}")
        return int(default)


BM25_K1 = _safe_float("RAG_BM25_K1", "1.5")
BM25_B = _safe_float("RAG_BM25_B", "0.75")
SIGMOID_SCALE = _safe_float("RAG_SIGMOID_SCALE", "5.0")
ROUTE_BOOST = _safe_float("RAG_ROUTE_BOOST", "0.12")
CACHE_MAX_PRODUCTS = 32

# ===== 回答构建 =====
MAX_SUB_QUESTIONS = 4    # 单次问答最多拆分的子问题数
MAX_EVIDENCE_CHUNKS = 6  # build_evidence / answer_formatter 保留的最大证据片段数

# ===== 检索 =====
VECTOR_TOP_K = _safe_int("RAG_VECTOR_TOP_K", "12")
KEYWORD_TOP_K = _safe_int("RAG_KEYWORD_TOP_K", "12")
HYBRID_VECTOR_WEIGHT = _safe_float("RAG_HYBRID_VW", "0.65")
HYBRID_KEYWORD_WEIGHT = _safe_float("RAG_HYBRID_KW", "0.35")

# ===== Reranker =====
# 混合检索结果经 reranker 重排序后再截断到 route_top_k
# rerank 使用 BGE-M3 的 compute_score (colbert+sparse+dense 融合)
RERANK_ENABLED = _os.environ.get("RAG_RERANK_ENABLED", "1").strip().lower() in ("1", "true", "yes")
RERANK_TOP_N = _safe_int("RAG_RERANK_TOP_N", "20")  # 送入 reranker 的候选数（需 > route_top_k）

# ===== 动态阈值 =====
# 启用后根据检索结果分数分布自适应调整过滤阈值，降低漏答率
DYNAMIC_THRESHOLD_ENABLED = _os.environ.get("RAG_DYN_THRESHOLD", "1").strip().lower() in ("1", "true", "yes")
# 动态阈值 = max(route_threshold * DYN_FLOOR_RATIO, top1_score * DYN_RATIO)
DYNAMIC_THRESHOLD_RATIO = _safe_float("RAG_DYN_RATIO", "0.40")
DYNAMIC_THRESHOLD_FLOOR_RATIO = _safe_float("RAG_DYN_FLOOR_RATIO", "0.70")

# ===== 中文分词 =====
# 启用 jieba 分词替代纯 bigram 切分，提升 BM25 精度
JIEBA_ENABLED = _os.environ.get("RAG_JIEBA_ENABLED", "1").strip().lower() in ("1", "true", "yes")

# ===== FAISS 索引类型 =====
# flat: IndexFlatIP (暴力搜索，适合小规模)
# hnsw: IndexHNSWFlat (近似最近邻，适合大规模)
FAISS_INDEX_TYPE = _os.environ.get("RAG_FAISS_INDEX", "hnsw").strip().lower()
FAISS_HNSW_M = _safe_int("RAG_HNSW_M", "32")         # HNSW 连接数
FAISS_HNSW_EF_CONSTRUCTION = _safe_int("RAG_HNSW_EFC", "200")  # 构建时搜索范围
FAISS_HNSW_EF_SEARCH = _safe_int("RAG_HNSW_EFS", "128")        # 查询时搜索范围

# ===== 按问题类型调整检索参数 =====
# vw/kw: 向量/关键词权重覆盖（可选）。精确参数类问题提高 kw，语义模糊问题提高 vw。
QUESTION_TYPE_CONFIG = {
    "ingredient":        {"k": 8,  "threshold": 0.25},
    "basic":             {"k": 8,  "threshold": 0.28},
    "operation":         {"k": 10, "threshold": 0.25, "vw": 0.55, "kw": 0.45},
    "aftercare":         {"k": 8,  "threshold": 0.25},
    "risk":              {"k": 8,  "threshold": 0.25},
    "anti_fake":         {"k": 8,  "threshold": 0.20, "vw": 0.45, "kw": 0.55},
    "contraindication":  {"k": 8,  "threshold": 0.25},
    "combo":             {"k": 10, "threshold": 0.30},
    "effect":            {"k": 8,  "threshold": 0.25},
    "pre_care":          {"k": 8,  "threshold": 0.25},
    "design":            {"k": 10, "threshold": 0.25},
    "repair":            {"k": 10, "threshold": 0.25},
    # 跨实体路由
    "complication":      {"k": 10, "threshold": 0.20, "vw": 0.60, "kw": 0.40},
    "course":            {"k": 10, "threshold": 0.25},
    "anatomy_q":         {"k": 8,  "threshold": 0.25, "vw": 0.50, "kw": 0.50},
    "indication_q":      {"k": 10, "threshold": 0.20, "vw": 0.55, "kw": 0.45},
    "procedure_q":       {"k": 10, "threshold": 0.25, "vw": 0.50, "kw": 0.50},
    "equipment_q":       {"k": 8,  "threshold": 0.25, "vw": 0.50, "kw": 0.50},
    "script":            {"k": 8,  "threshold": 0.25},
}

# ===== FAQ 快速路径阈值 =====
# score: 检索分数门槛（低于此值不走 FAQ 快速路径）
# ratio: bigram 重叠率门槛（低于此值不走 FAQ 快速路径）
# 安全相关路由要求更高的置信度
FAQ_FAST_PATH_THRESHOLDS = {
    "aftercare":        {"score": 0.35, "ratio": 0.40},
    "effect":           {"score": 0.35, "ratio": 0.45},
    "basic":            {"score": 0.38, "ratio": 0.45},
    "operation":        {"score": 0.38, "ratio": 0.45},
    "ingredient":       {"score": 0.38, "ratio": 0.45},
    "risk":             {"score": 0.45, "ratio": 0.55},
    "contraindication": {"score": 0.45, "ratio": 0.55},
    "complication":     {"score": 0.45, "ratio": 0.55},
    "repair":           {"score": 0.45, "ratio": 0.55},
}
FAQ_FAST_PATH_DEFAULT = {"score": 0.40, "ratio": 0.50}

# ===== 媒体 =====
# 每个产品的媒体文件位于 knowledge/{product_id}/media.json
# 由 media_router.py 按 product_id 加载，不再使用全局路径

# ===== 测试 =====
REGRESSION_CASES_FILE = BASE_DIR / "regression_cases.json"
