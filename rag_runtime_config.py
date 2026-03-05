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
DEBUG = False
DEFAULT_MODE = "brief"   # brief / full
DEFAULT_TOP_K = 8

# ===== 回答模板 =====
REFERENCE_NOTE = "结果仅供参考，需询问专业医师。"
RISK_NOTE = "仅供参考，需医生评估。"

# ===== 问题路由 =====
QUESTION_ROUTES = {
    "risk": ["红肿", "结节", "疼痛", "感染", "硬块", "异常", "副作用", "不良反应", "发热", "肿胀"],
    "aftercare": ["术后", "护理", "恢复", "注意事项", "洗脸", "面膜", "保湿", "禁酒", "禁忌行为"],
    "operation": ["注射", "深度", "参数", "微针", "水光", "操作", "针头", "0.8mm", "0.3ml", "2cm"],
    "anti_fake": ["防伪", "真伪", "HiddenTag", "正品", "鉴别", "验真"],
    "contraindication": ["禁忌", "禁忌人群", "不适合", "过敏体质", "妊娠", "哺乳"],
    "combo": ["联合", "同做", "一起做", "间隔", "配合", "搭配"],
    "ingredient": ["成分", "PCL", "聚己内酯", "透明质酸", "玻尿酸", "谷胱甘肽", "肽", "生长因子"],
    "basic": ["是什么", "作用", "适用", "产品", "功效", "备案", "规格"],
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
        "stops": ["二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、"],
    },
}

# ===== 别名与多实体 =====
PRODUCT_ALIASES = {
    "feiluoao": ["菲罗奥", "非罗奥", "菲洛奥", "赛罗菲", "赛罗菲提升", "CELLOFILL", "FILLOUP"],
    "sailuofei_vface": ["赛洛菲V脸溶脂", "赛洛菲V脸", "V脸溶脂", "赛洛菲溶脂"],
}
PROJECT_ALIASES = {
    "水光": ["水光", "水光针"],
    "微针": ["微针", "MTS"],
    "光电": ["光电", "光子", "射频"],
    "填充": ["填充", "玻尿酸", "填充剂"],
}

TIME_TERMS = ["当天", "术后当天", "术后1天", "术后1-3天", "术后3天", "术后1周", "一周", "术后1个月"]
SYMPTOM_TERMS = ["红肿", "结节", "疼痛", "过敏", "硬块", "发热", "感染", "刺痛"]

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
}

# ===== 媒体 =====
MEDIA_FILE = BASE_DIR / "media.json"

# ===== 测试 =====
REGRESSION_CASES_FILE = BASE_DIR / "regression_cases.json"
