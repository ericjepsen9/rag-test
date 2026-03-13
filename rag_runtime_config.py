from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ===== 基础路径 =====
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
STORE_ROOT = BASE_DIR / "stores"
OUT_PATH = BASE_DIR / "answer.txt"

# ===== OpenAI 开关 =====
USE_OPENAI = False
OPENAI_MODEL = "gpt-4o-mini"

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
                  "喝酒", "饮酒", "辛辣", "吃辣", "海鲜",
                  "上班", "工作", "几天能", "多久能"],
    "operation": ["注射", "深度", "参数", "微针", "水光", "操作", "针头", "0.8mm", "0.3ml", "2cm",
                  "疗程", "麻醉", "敷麻", "疼不疼", "痛不痛", "疼吗", "痛吗"],
    "anti_fake": ["防伪", "真伪", "HiddenTag", "正品", "鉴别", "验真"],
    "contraindication": ["禁忌", "禁忌人群", "不适合", "过敏体质", "妊娠", "哺乳",
                         "孕妇", "怀孕", "备孕", "敏感肌", "皮肤敏感",
                         "阿司匹林", "抗凝", "免疫抑制",
                         "多大年纪", "年龄", "岁能", "18岁", "未成年",
                         "男性", "男生", "男人", "自己打", "自己注射", "在家打"],
    "combo": ["联合", "同做", "一起做", "间隔", "配合", "搭配"],
    "ingredient": ["成分", "PCL", "聚己内酯", "透明质酸", "玻尿酸", "谷胱甘肽", "肽", "生长因子"],
    "basic": ["是什么", "作用", "适用", "产品", "功效", "备案", "规格"],
    "effect": ["效果", "维持", "见效", "多久", "持续", "变白", "变亮", "能维持", "几个月"],
    "pre_care": ["术前", "准备", "检查", "停药", "素颜", "到院"],
    "design": ["设计", "方案", "评估", "规划", "怎么打", "打哪里", "用量",
               "几支", "松弛程度", "面部分析"],
    "repair": ["修复", "补救", "不理想", "不均匀", "不对称", "矫正", "返修",
               "做坏了", "做失败", "效果差", "重新做", "之前做的"],
    # --- 跨实体路由（检索共享知识库） ---
    "complication": ["术后第", "天了还", "正常吗", "并发症", "急诊", "就医",
                     "肿了", "化脓", "坏死", "栓塞", "水泡"],
    "course": ["疗程", "几次", "间隔多久", "总共", "多长时间", "规划", "时间表",
               "多少钱", "费用", "预算"],
    "anatomy_q": ["哪个部位", "额头", "苹果肌", "法令纹", "下颌线", "颈部",
                  "鼻子", "手背", "眼周", "泪沟"],
    "indication_q": ["松弛怎么", "干燥怎么", "毛孔怎么", "色斑怎么", "痘坑怎么",
                     "皱纹怎么", "用什么好", "适合什么", "推荐什么",
                     "毛孔粗大", "皮肤松弛", "皮肤干燥"],
    "procedure_q": ["水光针是什么", "微针是什么", "光电是什么", "什么项目",
                    "有什么区别", "哪个好", "哪个更"],
    "equipment_q": ["什么仪器", "什么设备", "什么机器", "水光仪", "微针仪",
                    "射频仪", "仪器参数"],
    "script": ["怎么解释", "怎么回答", "客户问", "话术", "怎么介绍",
               "客户担心", "怎么说"],
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
        "stops": [],
    },
    "anatomy_q": {
        "titles": ["一、面部分区与常见治疗方案", "面部分区"],
        "stops": [],
    },
    "indication_q": {
        "titles": ["一、按皮肤状态分类的治疗推荐", "按皮肤状态"],
        "stops": [],
    },
    "procedure_q": {
        "titles": ["一、项目概述", "项目概述", "项目名称"],
        "stops": [],
    },
    "equipment_q": {
        "titles": ["一、设备概述", "设备概述", "设备名称"],
        "stops": [],
    },
    "script": {
        "titles": ["一、客户常见顾虑应答指南", "客户常见顾虑"],
        "stops": [],
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
    "填充": ["填充", "玻尿酸", "填充剂"],
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
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS_BRIEF = 1024
LLM_MAX_TOKENS_FULL = 2048

# ===== 搜索调优 =====
BM25_K1 = 1.5
BM25_B = 0.75
SIGMOID_SCALE = 5.0
ROUTE_BOOST = 0.05
CACHE_MAX_PRODUCTS = 32

# ===== 回答构建 =====
MAX_SUB_QUESTIONS = 4    # 单次问答最多拆分的子问题数
MAX_EVIDENCE_CHUNKS = 6  # build_evidence / answer_formatter 保留的最大证据片段数

# ===== 检索 =====
VECTOR_TOP_K = 12
KEYWORD_TOP_K = 12
HYBRID_VECTOR_WEIGHT = 0.65
HYBRID_KEYWORD_WEIGHT = 0.35

# ===== 按问题类型调整检索参数 =====
QUESTION_TYPE_CONFIG = {
    "ingredient":        {"k": 8,  "threshold": 0.25},
    "basic":             {"k": 6,  "threshold": 0.30},
    "operation":         {"k": 10, "threshold": 0.25},
    "aftercare":         {"k": 8,  "threshold": 0.25},
    "risk":              {"k": 8,  "threshold": 0.25},
    "anti_fake":         {"k": 8,  "threshold": 0.20},
    "contraindication":  {"k": 8,  "threshold": 0.25},
    "combo":             {"k": 10, "threshold": 0.30},
    "effect":            {"k": 8,  "threshold": 0.25},
    "pre_care":          {"k": 8,  "threshold": 0.25},
    "design":            {"k": 10, "threshold": 0.25},
    "repair":            {"k": 10, "threshold": 0.25},
    # 跨实体路由
    "complication":      {"k": 10, "threshold": 0.20},
    "course":            {"k": 10, "threshold": 0.25},
    "anatomy_q":         {"k": 8,  "threshold": 0.25},
    "indication_q":      {"k": 10, "threshold": 0.20},
    "procedure_q":       {"k": 10, "threshold": 0.25},
    "equipment_q":       {"k": 8,  "threshold": 0.25},
    "script":            {"k": 8,  "threshold": 0.25},
}

# ===== 媒体 =====
# 每个产品的媒体文件位于 knowledge/{product_id}/media.json
# 由 media_router.py 按 product_id 加载，不再使用全局路径

# ===== 测试 =====
REGRESSION_CASES_FILE = BASE_DIR / "regression_cases.json"
