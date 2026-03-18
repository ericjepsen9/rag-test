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
             "不消", "越来越", "化脓", "溃烂", "坏死",
             "出问题", "出了问题", "出事", "有淤血", "青了", "紫了", "出血点",
             "发炎", "过敏", "安全", "疙瘩", "还疼", "一直疼", "咋整", "咋办"],
    "aftercare": ["术后", "护理", "恢复", "注意事项", "洗脸", "面膜", "保湿", "禁酒", "禁忌行为",
                  "运动", "健身", "出汗", "洗澡", "游泳", "桑拿", "汗蒸", "泡澡",
                  "化妆", "上妆", "卸妆", "防晒", "晒太阳", "紫外线", "日晒",
                  "喝酒", "饮酒", "辛辣", "吃辣", "海鲜", "咖啡", "喝茶",
                  "上班", "工作", "几天能", "多久能", "做完能", "做完可以",
                  "碰水", "沾水", "注意啥", "有啥讲究", "忌口", "能抽烟", "抽烟",
                  "做完", "打完", "请假", "营养", "补充", "火锅", "瑜伽"],
    "operation": ["注射", "深度", "参数", "微针", "水光", "操作", "针头", "0.8mm", "0.3ml", "2cm",
                  "疗程", "麻醉", "敷麻", "疼不疼", "痛不痛", "疼吗", "痛吗",
                  "好痛", "很痛", "不疼", "怕痛", "怕疼",
                  "单针", "涂抹", "每次"],
    "anti_fake": ["防伪", "真伪", "HiddenTag", "正品", "鉴别", "验真",
                  "厂家", "产地"],
    "contraindication": ["禁忌", "禁忌人群", "不适合", "过敏体质", "妊娠", "哺乳",
                         "孕妇", "怀孕", "备孕", "敏感肌", "皮肤敏感",
                         "阿司匹林", "抗凝", "免疫抑制",
                         "多大年纪", "年龄", "岁能", "18岁", "未成年",
                         "男性", "男生", "男人", "自己打", "自己注射", "在家打",
                         "做不了", "不让做", "不能做", "不可以做", "不建议做",
                         "不能打", "喂奶", "长痘", "起痘", "风湿", "红斑狼疮",
                         "疤痕体质", "哪些人不", "什么禁忌", "有什么禁忌",
                         "手术多久"],
    "combo": ["联合", "同做", "一起做", "间隔", "配合", "搭配",
             "同天", "还能做", "哪个先", "能同时", "多久可以打", "多久可以做"],
    "ingredient": ["成分是什么", "成分", "PCL", "聚己内酯", "透明质酸", "玻尿酸", "谷胱甘肽", "肽", "生长因子",
                    "HA", "胶原蛋白", "交联", "分子量", "蛋白", "胶原", "玻尿",
                    "IntoCell", "intocell", "聚乙二醇"],
    "basic": ["是什么", "作用", "适用", "产品", "功效", "备案", "规格"],
    "effect": ["效果", "维持", "见效", "多久", "持续", "变白", "变亮", "能维持", "几个月",
               "管用", "好使", "靠谱", "有没有效", "看得出", "明显",
               "提拉", "紧致", "去皱", "年轻", "能持续多久", "看到效果"],
    "pre_care": ["术前", "准备", "检查", "停药", "素颜", "到院", "之前"],
    "design": ["设计", "方案", "评估", "规划", "怎么打", "打哪里", "用量",
               "几支", "松弛程度", "面部分析"],
    "repair": ["修复", "补救", "不理想", "不均匀", "不对称", "矫正", "返修",
               "做坏了", "做失败", "效果差", "重新做", "之前做的",
               "垮了", "脸垮", "做完怎么办", "没效果", "没有效果"],
    # --- 跨实体路由（检索共享知识库） ---
    "complication": ["术后第", "术后当天", "天了还", "正常吗", "并发症", "急诊", "就医",
                     "肿了", "化脓", "坏死", "栓塞", "水泡",
                     "扎完针", "打完针", "做完后", "肿起来"],
    "course": ["疗程", "几次", "间隔多久", "总共", "多长时间", "规划", "时间表",
               "多少钱", "费用", "预算"],
    "anatomy_q": ["哪个部位", "额头", "苹果肌", "法令纹", "下颌线", "颈部",
                  "鼻子", "手背", "眼周", "泪沟", "脖子", "颈纹", "手部"],
    "indication_q": ["松弛怎么", "干燥怎么", "毛孔怎么", "色斑怎么", "痘坑怎么",
                     "皱纹怎么", "用什么好", "适合什么", "推荐什么",
                     "毛孔粗大", "皮肤松弛", "皮肤干燥", "痘坑用", "暗沉用",
                     "怎么改善", "缺水怎么", "暗沉怎么", "粗糙怎么",
                     "方法改善", "什么方法",
                     "老了", "显老", "抗衰", "有什么能做",
                     "痘坑", "暗黄", "无光泽", "医美", "做医美",
                     "油性", "做什么项目", "适合做"],
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
                    "微针笔", "电动微针", "飞针仪", "射频美容仪", "热玛吉仪",
                    "滚轮微针", "滚轮", "滚针"],
    "script": ["怎么解释", "怎么回答", "客户问", "话术", "怎么介绍",
               "客户担心", "怎么说", "向客户", "跟客户", "给客户",
               "风险话术", "期望管理", "合规表达", "医学严谨",
               "安慰", "客户怕"],
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
    "feiluoao": ["菲罗奥", "非罗奥", "菲洛奥", "非洛奥", "飞罗奥", "妃罗奥",
                 "赛罗菲", "赛罗菲提升", "CELLOFILL", "FILLOUP", "cellofill", "filloup"],
    "sailuofei_vface": ["赛洛菲V脸溶脂", "赛洛菲V脸", "V脸溶脂", "赛洛菲溶脂", "赛洛菲",
                        "塞罗菲", "塞洛菲"],
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
    "botox":         ["肉毒", "肉毒素", "肉毒杆菌", "瘦脸针", "瘦咬肌", "除皱针",
                      "botox", "保妥适", "衡力", "乐提葆", "肉毒素注射"],
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

MATERIAL_ALIASES = {
    "hyaluronic_acid": ["玻尿酸", "透明质酸", "HA", "hyaluronic acid", "玻尿"],
    "collagen": ["胶原蛋白", "胶原", "骨胶原", "collagen", "蛋白"],
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
                 "肿", "淤青", "瘀青", "发紫", "红疹", "疹子", "痒", "化脓",
                 "硬结", "肉芽肿", "血管栓塞", "表情不自然", "填充物移位"]

# ===== 无知识兜底回复（价格、对比等无法回答的问题） =====
PRICE_REPLY = ("具体价格因产品、医院、区域和疗程方案不同而有差异，"
               "建议直接咨询所在医院或门诊获取准确报价。")
COMPARISON_REPLY = ("不同产品的适应症、成分和效果机制各有不同，"
                    "建议咨询医生根据您的皮肤状况选择最合适的方案，避免盲目对比。")
LOCATION_REPLY = "建议通过官方渠道或正规医美机构查询可提供相关项目的医院。"

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
ROUTE_BOOST = _safe_float("RAG_ROUTE_BOOST", "0.25")
CACHE_MAX_PRODUCTS = 32

# ===== 回答构建 =====
MAX_SUB_QUESTIONS = 4    # 单次问答最多拆分的子问题数
MAX_EVIDENCE_CHUNKS = 6  # build_evidence / answer_formatter 保留的最大证据片段数

# ===== 检索 =====
VECTOR_TOP_K = _safe_int("RAG_VECTOR_TOP_K", "8")
KEYWORD_TOP_K = _safe_int("RAG_KEYWORD_TOP_K", "8")
HYBRID_VECTOR_WEIGHT = _safe_float("RAG_HYBRID_VW", "0.65")
HYBRID_KEYWORD_WEIGHT = _safe_float("RAG_HYBRID_KW", "0.35")

# ===== Reranker =====
# 混合检索结果经 reranker 重排序后再截断到 route_top_k
# rerank 使用 BGE-M3 的 compute_score (colbert+sparse+dense 融合)
RERANK_ENABLED = _os.environ.get("RAG_RERANK_ENABLED", "0").strip().lower() in ("1", "true", "yes")
RERANK_TOP_N = _safe_int("RAG_RERANK_TOP_N", "10")  # 送入 reranker 的候选数（需 > route_top_k）

# ===== 动态阈值 =====
# 启用后根据检索结果分数分布自适应调整过滤阈值，降低漏答率
DYNAMIC_THRESHOLD_ENABLED = _os.environ.get("RAG_DYN_THRESHOLD", "1").strip().lower() in ("1", "true", "yes")
# 动态阈值 = max(route_threshold * DYN_FLOOR_RATIO, top1_score * DYN_RATIO)
DYNAMIC_THRESHOLD_RATIO = _safe_float("RAG_DYN_RATIO", "0.40")
DYNAMIC_THRESHOLD_FLOOR_RATIO = _safe_float("RAG_DYN_FLOOR_RATIO", "0.70")

# ===== 消歧引导配置 =====
# 启用后，当用户查询模糊且缺乏上下文时，系统会在回答同时提供候选选项
CLARIFICATION_ENABLED = _os.environ.get("RAG_CLARIFICATION", "1").strip().lower() in ("1", "true", "yes")
# 触发消歧的最短查询长度（字符数，低于此值且无上下文时触发）
CLARIFICATION_MIN_QUERY_LEN = _safe_int("RAG_CLARIFY_MIN_LEN", "6")
# 触发消歧的最长查询长度（超过此值认为描述已足够详细）
CLARIFICATION_MAX_QUERY_LEN = _safe_int("RAG_CLARIFY_MAX_LEN", "15")

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

# ===== 服务器 / 域名访问配置 =====
SERVER_HOST = _os.environ.get("RAG_SERVER_HOST", "0.0.0.0")
SERVER_PORT = _safe_int("RAG_SERVER_PORT", "8000")

# ===== 运行时热更新支持 =====
# 允许通过 API 修改的参数白名单及其类型验证
import json as _json
import threading as _threading

_CONFIG_FILE = BASE_DIR / "data" / "runtime_overrides.json"
_SERVER_CONFIG_FILE = BASE_DIR / "data" / "server_config.json"
_config_lock = _threading.Lock()

# 模型提供商预设
MODEL_PRESETS = {
    "openai": {
        "label": "OpenAI",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        "default_model": "gpt-4o-mini",
        "api_base": "",
    },
    "deepseek": {
        "label": "DeepSeek",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat",
        "api_base": "https://api.deepseek.com/v1",
    },
    "minimax": {
        "label": "MiniMax",
        "models": ["MiniMax-Text-01", "abab6.5s-chat", "abab5.5-chat"],
        "default_model": "MiniMax-Text-01",
        "api_base": "https://api.minimax.chat/v1",
    },
    "custom": {
        "label": "Custom / 自定义",
        "models": [],
        "default_model": "",
        "api_base": "",
    },
}

# 可热更新参数定义：key -> (module_var_name, type, min, max, description)
TUNABLE_PARAMS = {
    # 搜索参数
    "bm25_k1":          ("BM25_K1",          float, 0.5, 5.0,   "BM25 词频饱和参数"),
    "bm25_b":           ("BM25_B",           float, 0.0, 1.0,   "BM25 文档长度归一化"),
    "sigmoid_scale":    ("SIGMOID_SCALE",    float, 1.0, 20.0,  "BM25 分数 sigmoid 缩放"),
    "route_boost":      ("ROUTE_BOOST",      float, 0.0, 0.5,   "路由匹配加分"),
    "vector_top_k":     ("VECTOR_TOP_K",     int,   1,   50,    "向量检索返回数"),
    "keyword_top_k":    ("KEYWORD_TOP_K",    int,   1,   50,    "关键词检索返回数"),
    "hybrid_vw":        ("HYBRID_VECTOR_WEIGHT",  float, 0.0, 1.0, "混合检索向量权重"),
    "hybrid_kw":        ("HYBRID_KEYWORD_WEIGHT", float, 0.0, 1.0, "混合检索关键词权重"),
    # Reranker
    "rerank_enabled":   ("RERANK_ENABLED",   bool, None, None,  "启用 Reranker 重排序"),
    "rerank_top_n":     ("RERANK_TOP_N",     int,   5,   50,    "Reranker 候选数"),
    # 动态阈值
    "dyn_threshold_enabled": ("DYNAMIC_THRESHOLD_ENABLED", bool, None, None, "启用动态阈值"),
    "dyn_ratio":        ("DYNAMIC_THRESHOLD_RATIO",       float, 0.1, 0.9, "动态阈值比率"),
    "dyn_floor_ratio":  ("DYNAMIC_THRESHOLD_FLOOR_RATIO", float, 0.3, 1.0, "动态阈值下限比率"),
    # LLM 参数
    "llm_temperature":  ("LLM_TEMPERATURE",       float, 0.0, 1.0,  "LLM 默认温度"),
    "llm_max_brief":    ("LLM_MAX_TOKENS_BRIEF",  int,   100, 4000, "LLM brief 最大 token"),
    "llm_max_full":     ("LLM_MAX_TOKENS_FULL",   int,   200, 8000, "LLM full 最大 token"),
    "llm_rewrite":      ("LLM_REWRITE_ENABLED",   bool,  None, None, "启用 LLM 查询改写"),
    # 分块参数
    "chunk_size":       ("CHUNK_SIZE",       int,   100, 2000, "文本分块大小（字符）"),
    "chunk_overlap":    ("CHUNK_OVERLAP",    int,   0,   500,  "分块重叠长度"),
    # FAISS
    "faiss_index_type": ("FAISS_INDEX_TYPE", str,   None, None, "FAISS 索引类型 (flat/hnsw)"),
    "hnsw_m":           ("FAISS_HNSW_M",    int,   4,   128,  "HNSW 连接数"),
    "hnsw_ef_construction": ("FAISS_HNSW_EF_CONSTRUCTION", int, 40, 800, "HNSW 构建搜索范围"),
    "hnsw_ef_search":   ("FAISS_HNSW_EF_SEARCH", int, 16, 512, "HNSW 查询搜索范围"),
    # 模型切换
    "use_openai":       ("USE_OPENAI",       bool, None, None, "启用 LLM（OpenAI 兼容）"),
    "openai_model":     ("OPENAI_MODEL",     str,  None, None, "LLM 模型名称"),
    "openai_api_base":  ("OPENAI_API_BASE",  str,  None, None, "LLM API 地址"),
    # 消歧引导
    "clarification_enabled": ("CLARIFICATION_ENABLED", bool, None, None, "启用模糊查询消歧引导"),
    "clarify_min_len":  ("CLARIFICATION_MIN_QUERY_LEN", int, 2, 20, "消歧触发最短查询长度"),
    "clarify_max_len":  ("CLARIFICATION_MAX_QUERY_LEN", int, 5, 50, "消歧触发最长查询长度"),
}


def get_tunable_config() -> dict:
    """获取所有可调参数的当前值"""
    import rag_runtime_config as _mod
    result = {}
    for key, (var_name, vtype, vmin, vmax, desc) in TUNABLE_PARAMS.items():
        val = getattr(_mod, var_name, None)
        result[key] = {
            "value": val,
            "type": vtype.__name__,
            "min": vmin,
            "max": vmax,
            "description": desc,
            "var_name": var_name,
        }
    return result


def update_tunable_config(updates: dict) -> dict:
    """热更新可调参数，返回实际更新的字段"""
    import rag_runtime_config as _mod
    changed = {}
    for key, new_val in updates.items():
        if key not in TUNABLE_PARAMS:
            continue
        var_name, vtype, vmin, vmax, desc = TUNABLE_PARAMS[key]
        # 类型转换
        try:
            if vtype == bool:
                if isinstance(new_val, str):
                    new_val = new_val.strip().lower() in ("1", "true", "yes", "on")
                else:
                    new_val = bool(new_val)
            elif vtype == int:
                new_val = int(new_val)
            elif vtype == float:
                new_val = float(new_val)
            else:
                new_val = str(new_val).strip()
        except (ValueError, TypeError):
            continue
        # 范围校验
        if vtype in (int, float) and vmin is not None and vmax is not None:
            new_val = max(vmin, min(vmax, new_val))
        # 特殊校验
        if key == "faiss_index_type" and new_val not in ("flat", "hnsw"):
            continue
        old_val = getattr(_mod, var_name, None)
        if old_val != new_val:
            setattr(_mod, var_name, new_val)
            changed[key] = {"old": old_val, "new": new_val}
    # 持久化覆盖值
    if changed:
        _persist_overrides(updates)
        # 如果模型相关参数变更，需要重置 LLM client 缓存
        _model_keys = {"use_openai", "openai_model", "openai_api_base"}
        if _model_keys & changed.keys():
            try:
                import rag_answer
                rag_answer._openai_client = None
                rag_answer._openai_client_checked = False
            except Exception:
                pass
            try:
                from llm_client import sync_from_legacy
                sync_from_legacy()
            except Exception:
                pass
    return changed


def _persist_overrides(updates: dict) -> None:
    """将运行时覆盖保存到文件，下次启动时自动加载"""
    with _config_lock:
        data_dir = _CONFIG_FILE.parent
        data_dir.mkdir(parents=True, exist_ok=True)
        existing = {}
        if _CONFIG_FILE.exists():
            try:
                existing = _json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[WARN] 读取运行时配置文件失败: {e}")
        for key, val in updates.items():
            if key in TUNABLE_PARAMS:
                existing[key] = val
        tmp = _CONFIG_FILE.with_suffix(".tmp")
        tmp.write_text(_json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(_CONFIG_FILE)


def load_persisted_overrides() -> dict:
    """启动时加载持久化的覆盖值"""
    if not _CONFIG_FILE.exists():
        return {}
    try:
        data = _json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        if data:
            changed = update_tunable_config(data)
            if changed:
                from rag_logger import log_event
                log_event("config", f"加载了 {len(changed)} 个运行时配置覆盖",
                          meta={"keys": list(changed.keys())})
            return changed
    except Exception as e:
        print(f"[WARN] 加载运行时配置覆盖失败: {e}")
    return {}


def get_model_config() -> dict:
    """获取当前模型配置"""
    import rag_runtime_config as _mod
    return {
        "use_openai": _mod.USE_OPENAI,
        "model": _mod.OPENAI_MODEL,
        "api_base": _mod.OPENAI_API_BASE or "",
        "api_key_set": bool(_os.environ.get("OPENAI_API_KEY", "").strip()),
        "llm_rewrite": _mod.LLM_REWRITE_ENABLED,
        "presets": MODEL_PRESETS,
    }


def switch_model_provider(provider: str, model: str = "", api_base: str = "",
                          api_key: str = "") -> dict:
    """切换模型提供商"""
    import rag_runtime_config as _mod
    preset = MODEL_PRESETS.get(provider)
    if not preset and provider != "custom":
        return {"error": f"未知提供商: {provider}"}
    if preset and not model:
        model = preset["default_model"]
    if preset and not api_base:
        api_base = preset["api_base"]
    _mod.USE_OPENAI = True
    _mod.OPENAI_MODEL = model
    _mod.OPENAI_API_BASE = api_base or None
    if api_key:
        _os.environ["OPENAI_API_KEY"] = api_key
    # 重置 OpenAI client 缓存（强制下次调用重新创建）
    try:
        from rag_answer import _get_openai_client
        import rag_answer
        rag_answer._openai_client = None
        rag_answer._openai_client_checked = False
    except Exception as e:
        print(f"[WARN] 重置 OpenAI client 缓存失败: {e}")
    # 同步到 llm_client（保持多 LLM 配置一致）
    try:
        from llm_client import sync_from_legacy
        sync_from_legacy()
    except Exception as e:
        print(f"[WARN] 同步 llm_client 配置失败: {e}")
    # 持久化
    _persist_overrides({
        "use_openai": True,
        "openai_model": model,
        "openai_api_base": api_base or "",
    })
    return {
        "ok": True,
        "provider": provider,
        "model": model,
        "api_base": api_base or "",
    }


# ===== 服务器 / 域名配置管理 =====

def get_server_config() -> dict:
    """获取服务器和域名访问配置"""
    data = _load_server_config_file()
    return {
        "host": SERVER_HOST,
        "port": SERVER_PORT,
        "domain": data.get("domain", ""),
        "ssl_enabled": data.get("ssl_enabled", False),
        "ssl_cert_path": data.get("ssl_cert_path", ""),
        "ssl_key_path": data.get("ssl_key_path", ""),
        "cors_origins": _os.environ.get("CORS_ORIGINS", "*"),
        "chat_path": "/chat",
        "admin_path": "/admin",
        "api_path": "/ask",
        "oai_path": "/v1/chat/completions",
        "nginx_config": data.get("nginx_config", ""),
        "auto_start": data.get("auto_start", True),
    }


def update_server_config(updates: dict) -> dict:
    """更新服务器和域名配置"""
    import rag_runtime_config as _mod
    data = _load_server_config_file()
    changed = {}
    allowed_keys = {"domain", "ssl_enabled", "ssl_cert_path", "ssl_key_path",
                    "cors_origins", "nginx_config", "auto_start"}
    for key, val in updates.items():
        if key not in allowed_keys:
            continue
        if key == "ssl_enabled":
            val = bool(val)
        elif key == "cors_origins":
            val = str(val).strip()
            _os.environ["CORS_ORIGINS"] = val
        else:
            val = str(val).strip()
        old = data.get(key, "")
        if old != val:
            data[key] = val
            changed[key] = {"old": old, "new": val}
    # host/port 可修改（下次重启生效）
    if "host" in updates:
        new_host = str(updates["host"]).strip()
        if new_host != SERVER_HOST:
            _mod.SERVER_HOST = new_host
            data["host"] = new_host
            changed["host"] = {"old": SERVER_HOST, "new": new_host}
    if "port" in updates:
        try:
            new_port = int(updates["port"])
            if 1 <= new_port <= 65535 and new_port != SERVER_PORT:
                _mod.SERVER_PORT = new_port
                data["port"] = new_port
                changed["port"] = {"old": SERVER_PORT, "new": new_port}
        except (ValueError, TypeError):
            pass
    if changed:
        _save_server_config_file(data)
    return changed


def generate_nginx_config(domain: str, port: int = 0, ssl: bool = False,
                          cert_path: str = "", key_path: str = "") -> str:
    """生成 nginx 反向代理配置"""
    port = port or SERVER_PORT
    if ssl and cert_path and key_path:
        return f"""server {{
    listen 80;
    server_name {domain};
    return 301 https://$host$request_uri;
}}

server {{
    listen 443 ssl http2;
    server_name {domain};

    ssl_certificate     {cert_path};
    ssl_certificate_key {key_path};
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    client_max_body_size 100m;

    location / {{
        proxy_pass http://127.0.0.1:{port};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }}

    location /v1/chat/completions {{
        proxy_pass http://127.0.0.1:{port};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_read_timeout 300s;
    }}
}}"""
    else:
        return f"""server {{
    listen 80;
    server_name {domain};

    client_max_body_size 100m;

    location / {{
        proxy_pass http://127.0.0.1:{port};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }}

    location /v1/chat/completions {{
        proxy_pass http://127.0.0.1:{port};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_read_timeout 300s;
    }}
}}"""


def _load_server_config_file() -> dict:
    if not _SERVER_CONFIG_FILE.exists():
        return {}
    try:
        return _json.loads(_SERVER_CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_server_config_file(data: dict) -> None:
    with _config_lock:
        d = _SERVER_CONFIG_FILE.parent
        d.mkdir(parents=True, exist_ok=True)
        tmp = _SERVER_CONFIG_FILE.with_suffix(".tmp")
        tmp.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(_SERVER_CONFIG_FILE)


# ===== BGE-M3 嵌入模型控制 =====

def get_embedding_status() -> dict:
    """获取 BGE-M3 嵌入模型当前状态"""
    try:
        import rag_answer
        model = getattr(rag_answer, "_model", None)
        return {
            "loaded": model is not None,
            "model_name": EMBED_MODEL_NAME,
            "use_fp16": EMBED_USE_FP16,
            "batch_size_build": EMBED_BATCH_SIZE_BUILD,
            "batch_size_query": EMBED_BATCH_SIZE_QUERY,
            "max_length_build": EMBED_MAX_LENGTH_BUILD,
            "max_length_query": EMBED_MAX_LENGTH_QUERY,
        }
    except Exception:
        return {"loaded": False, "model_name": EMBED_MODEL_NAME}


def start_embedding_model() -> dict:
    """手动加载 BGE-M3 嵌入模型"""
    try:
        from rag_answer import get_model, embed_query
        model = get_model()
        # 做一次预热编码确保完全就绪
        embed_query("预热")
        return {"ok": True, "message": "BGE-M3 模型已加载"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def stop_embedding_model() -> dict:
    """卸载 BGE-M3 嵌入模型释放显存/内存"""
    try:
        import rag_answer
        import gc
        rag_answer._model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return {"ok": True, "message": "BGE-M3 模型已卸载"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ===== LLM 服务控制 =====

def get_llm_status() -> dict:
    """获取 LLM 服务状态"""
    import rag_runtime_config as _mod
    try:
        import rag_answer
        client = getattr(rag_answer, "_openai_client", None)
        checked = getattr(rag_answer, "_openai_client_checked", False)
        return {
            "enabled": _mod.USE_OPENAI,
            "client_ready": client is not None,
            "client_checked": checked,
            "model": _mod.OPENAI_MODEL,
            "api_base": _mod.OPENAI_API_BASE or "",
            "api_key_set": bool(_os.environ.get("OPENAI_API_KEY", "").strip()),
            "rewrite_enabled": _mod.LLM_REWRITE_ENABLED,
            "temperature": _mod.LLM_TEMPERATURE,
        }
    except Exception:
        return {
            "enabled": _mod.USE_OPENAI,
            "client_ready": False,
            "client_checked": False,
            "model": _mod.OPENAI_MODEL,
            "api_base": _mod.OPENAI_API_BASE or "",
            "api_key_set": bool(_os.environ.get("OPENAI_API_KEY", "").strip()),
        }


def start_llm_service(api_key: str = "") -> dict:
    """启动/重连 LLM 服务"""
    import rag_runtime_config as _mod
    if api_key:
        _os.environ["OPENAI_API_KEY"] = api_key
    _mod.USE_OPENAI = True
    # 重置 client 缓存，强制重新创建
    try:
        import rag_answer
        rag_answer._openai_client = None
        rag_answer._openai_client_checked = False
        # 立即尝试创建 client
        client = rag_answer._get_openai_client()
        if client is None:
            return {"ok": False, "error": "LLM client 创建失败，请检查 API Key 和 API Base"}
        # 同步到 llm_client
        try:
            from llm_client import sync_from_legacy
            sync_from_legacy()
        except Exception:
            pass
        return {"ok": True, "message": f"LLM 服务已启动 (model={OPENAI_MODEL})"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def stop_llm_service() -> dict:
    """停止 LLM 服务"""
    import rag_runtime_config as _mod
    _mod.USE_OPENAI = False
    try:
        import rag_answer
        rag_answer._openai_client = None
        rag_answer._openai_client_checked = False
    except Exception:
        pass
    # 同步到 llm_client
    try:
        from llm_client import sync_from_legacy
        sync_from_legacy()
    except Exception:
        pass
    _persist_overrides({"use_openai": False})
    return {"ok": True, "message": "LLM 服务已停止"}


# 启动时自动加载持久化覆盖
load_persisted_overrides()

# 启动时加载服务器配置
_server_data = _load_server_config_file()
if _server_data.get("host"):
    SERVER_HOST = _server_data["host"]
if _server_data.get("port"):
    try:
        SERVER_PORT = int(_server_data["port"])
    except (ValueError, TypeError):
        pass
