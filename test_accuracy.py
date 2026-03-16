"""全面准确度测试：路由检测、搜索准确性、同义词、消歧、边界场景、格式化、关联引擎

测试维度：
1. 路由检测准确度 — 覆盖所有 20+ 路由 + 消歧边界
2. 搜索工具链准确度 — BM25、向量搜索、混合排序
3. 同义词扩展准确度 — 口语映射、医美术语
4. 查询改写准确度 — 多轮对话上下文补全
5. 多问题拆分准确度 — 复合问题、选择问题、列举问题
6. 格式化输出准确度 — 结构化回答、安全提示
7. 产品/实体检测准确度 — 别名、错别字、英文
8. 关联引擎数据完整性
"""
import pytest
import re
from search_utils import (
    _extract_terms, _extract_terms_bigram, bm25_score,
    normalize_text, normalize_lines, uniq, section_block,
    split_multi_question, keyword_search, merge_hybrid, detect_terms,
    expand_synonyms, rerank_hits, compute_dynamic_threshold,
    _sigmoid_norm,
)
from query_rewrite import rewrite_query, _resolve_context, _extract_history_context
from rag_answer import (
    detect_route, detect_product, build_evidence, _truncate_to_sentence,
    _build_context, _detect_special_intent,
)
from rag_runtime_config import QUESTION_ROUTES, PRODUCT_ALIASES, PROJECT_ALIASES


# ============================================================
# 1. 路由检测准确度 — 全路由覆盖
# ============================================================

class TestRouteDetectionAccuracy:
    """全面测试路由检测的准确性，覆盖所有已知路由"""

    # ---- 基础路由 ----
    @pytest.mark.parametrize("q,expected", [
        ("菲罗奥是什么产品", "basic"),
        ("CELLOFILL是什么", "basic"),
        ("菲罗奥", "basic"),
        ("备案信息在哪查", "basic"),
    ])
    def test_basic_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("成分", "ingredient"),
        ("PCL是什么", "ingredient"),
        ("赛罗菲 成分", "ingredient"),
        ("含有什么成分", "ingredient"),
    ])
    def test_ingredient_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("效果维持多久", "effect"),
        ("打完多久见效", "effect"),
        ("能变白吗", "effect"),
        ("菲罗奥效果能维持多长时间", "effect"),
    ])
    def test_effect_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("注射深度 0.8mm", "operation"),
        ("注射量是多少", "operation"),
        ("MTS操作参数", "operation"),
        ("打菲罗奥疼不疼", "operation"),
    ])
    def test_operation_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("术后可以洗脸吗", "aftercare"),
        ("做完能运动吗", "aftercare"),
        ("术后能喝酒吗", "aftercare"),
        ("打完能晒太阳吗", "aftercare"),
        ("做完能洗澡吗", "aftercare"),
        ("术后能游泳吗", "aftercare"),
        ("做完能去桑拿吗", "aftercare"),
        ("术后能吃辣吗", "aftercare"),
        ("术后能吃海鲜吗", "aftercare"),
        ("做完能上班吗", "aftercare"),
        ("术后多久可以化妆", "aftercare"),
        ("术后多久可以敷面膜", "aftercare"),
    ])
    def test_aftercare_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("非罗奥 怎么验真伪", "anti_fake"),
        ("怎么下载 HiddenTag", "anti_fake"),
        ("正品标签长什么样", "anti_fake"),
    ])
    def test_anti_fake_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("注射后出现红肿怎么办", "risk"),
        ("术后有硬块正常吗", "risk"),
        ("感染了怎么处理", "risk"),
        ("术后红肿是正常反应吗", "risk"),
        ("打完淤青怎么办", "risk"),
        ("术后脸上起疹子怎么回事", "risk"),
        ("副作用", "risk"),
    ])
    def test_risk_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("禁忌人群", "contraindication"),
        ("哺乳期可以打吗", "contraindication"),
        ("过敏体质可以用吗", "contraindication"),
        ("孕妇能打吗", "contraindication"),
        ("怀孕可以打吗", "contraindication"),
        ("备孕期间能做吗", "contraindication"),
        ("可以自己在家打吗", "contraindication"),
        ("皮肤敏感能打吗", "contraindication"),
        ("多大年纪适合打菲罗奥", "contraindication"),
        ("男性可以做吗", "contraindication"),
        ("吃阿司匹林能打菲罗奥吗", "contraindication"),
    ])
    def test_contraindication_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("水光和微针可以一起做吗", "combo"),
        ("菲罗奥可以和光电一起做吗", "combo"),
        ("菲罗奥可以和什么项目一起做", "combo"),
        ("菲罗奥可以和水光一起做吗", "combo"),
        ("菲罗奥和水光针间隔多久做", "combo"),
    ])
    def test_combo_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("打菲罗奥之前要做什么准备", "pre_care"),
        ("术前需要停药吗", "pre_care"),
        ("做菲罗奥之前需要检查什么", "pre_care"),
        ("术前要不要停用维A酸", "pre_care"),
    ])
    def test_pre_care_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("菲罗奥怎么设计方案", "design"),
        ("脸部松弛需要打几支", "design"),
        ("法令纹怎么打", "design"),
        ("面部评估怎么做", "design"),
    ])
    def test_design_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("之前做的效果不理想怎么办", "repair"),
        ("做完不均匀怎么补救", "repair"),
        ("术后出现硬块需要修复吗", "repair"),
    ])
    def test_repair_route(self, q, expected):
        assert detect_route(q) == expected

    # ---- 跨实体路由 ----
    @pytest.mark.parametrize("q,expected", [
        ("术后第3天还肿正常吗", "complication"),
        ("术后第2天红肿正常吗", "complication"),
        ("做完脸肿了正常吗", "complication"),
        ("术后当天脸红是正常的吗", "complication"),
        ("做完第二天淤青正常吗", "complication"),
    ])
    def test_complication_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("疗程怎么安排", "course"),
        ("菲罗奥要做几次", "course"),
        ("总共需要多长时间", "course"),
        ("菲罗奥一个疗程几次", "course"),
        ("两次注射之间间隔多长时间", "course"),
    ])
    def test_course_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("苹果肌下垂用什么好", "anatomy_q"),
        ("颈部能打菲罗奥吗", "anatomy_q"),
        ("额头皱纹可以打菲罗奥吗", "anatomy_q"),
        ("手背可以做吗", "anatomy_q"),
    ])
    def test_anatomy_q_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("皮肤松弛怎么改善", "indication_q"),
        ("毛孔粗大打什么好", "indication_q"),
        ("痘坑用什么方法改善", "indication_q"),
        ("皮肤暗沉怎么改善", "indication_q"),
        ("松弛且缺水怎么改善", "indication_q"),
        ("皮肤干燥适合什么项目", "indication_q"),
    ])
    def test_indication_q_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("水光针是什么项目", "procedure_q"),
        ("微针和水光针有什么区别", "procedure_q"),
        ("水光和微针分别是什么", "procedure_q"),
        ("光电项目有哪些", "procedure_q"),
        ("微针的操作流程是什么", "procedure_q"),
    ])
    def test_procedure_q_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("水光仪的参数是什么", "equipment_q"),
        ("用什么仪器打菲罗奥", "equipment_q"),
        ("微针仪的针头深度怎么调", "equipment_q"),
        ("菲罗奥适配什么品牌的仪器", "equipment_q"),
    ])
    def test_equipment_q_route(self, q, expected):
        assert detect_route(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("客户担心疼怎么解释", "script"),
        ("怎么介绍菲罗奥优势", "script"),
        ("客户问价格太贵怎么回答", "script"),
        ("怎么向客户介绍菲罗奥和玻尿酸的区别", "script"),
    ])
    def test_script_route(self, q, expected):
        assert detect_route(q) == expected


class TestRouteDisambiguation:
    """消歧测试：当多个路由关键词交叉时，应选择正确路由"""

    def test_risk_vs_aftercare_symptom_priority(self):
        """症状关键词应优先匹配 risk 而非 aftercare"""
        assert detect_route("术后红肿是正常反应吗") == "risk"
        assert detect_route("术后有硬块正常吗") == "risk"

    def test_complication_vs_risk_temporal(self):
        """短期时间线→complication，长期→risk"""
        assert detect_route("术后第2天红肿正常吗") == "complication"
        assert detect_route("术后3个月还有硬块正常吗") == "risk"
        assert detect_route("术后1个月还红肿正常吗") == "risk"

    def test_contraindication_vs_risk(self):
        """禁忌信号词应优先命中 contraindication"""
        assert detect_route("过敏体质的人能不能做") == "contraindication"
        # 感染是风险，非禁忌
        assert detect_route("感染了怎么处理") == "risk"

    def test_combo_vs_operation(self):
        """'一起做' 应命中 combo 而非 operation"""
        assert detect_route("菲罗奥可以和水光一起做吗") == "combo"

    def test_equipment_vs_operation(self):
        """仪器关键词应优先匹配 equipment_q"""
        assert detect_route("用什么仪器打菲罗奥") == "equipment_q"

    def test_effect_vs_risk_maintain(self):
        """'维持' 应命中 effect 而非 risk"""
        assert detect_route("菲罗奥效果能维持多长时间") == "effect"

    def test_indication_vs_contraindication_recommend(self):
        """'适合' 偏向 indication_q 而非 contraindication"""
        assert detect_route("皮肤干燥适合什么项目") == "indication_q"

    def test_script_vs_ingredient(self):
        """客户沟通场景应命中 script"""
        assert detect_route("客户问价格太贵怎么回答") == "script"

    def test_repair_vs_risk(self):
        """'修复' 关键词应命中 repair"""
        assert detect_route("术后出现硬块需要修复吗") == "repair"

    def test_pain_inquiry_to_operation(self):
        """疼痛体感询问应命中 operation"""
        assert detect_route("打菲罗奥疼不疼") == "operation"

    def test_lifestyle_to_aftercare(self):
        """生活限制类应命中 aftercare"""
        for q in ["做完能运动吗", "术后能游泳吗", "做完能去桑拿吗",
                   "术后能吃辣吗", "做完能上班吗"]:
            assert detect_route(q) == "aftercare", f"Failed: {q}"


class TestRouteEdgeCases:
    """路由边界场景"""

    def test_empty_question(self):
        assert detect_route("") == "basic"

    def test_none_question(self):
        assert detect_route(None) == "basic"

    def test_single_keyword_routes(self):
        """单个关键词应能正确路由"""
        assert detect_route("成分") == "ingredient"
        assert detect_route("副作用") == "risk"
        assert detect_route("禁忌人群") == "contraindication"

    def test_product_name_only(self):
        """仅产品名应路由到 basic"""
        assert detect_route("菲罗奥") == "basic"

    def test_multi_keyword_conflict(self):
        """多关键词并存时有清晰的优先级"""
        result = detect_route("成分是什么？禁忌人群有哪些？")
        assert result == "contraindication"  # contraindication 优先级高

    def test_selection_question_not_split(self):
        """'还是' 选择问题不应被错误拆分"""
        # 选择问题不应检测 route，而是走对比意图
        q = "水光针好还是微针好"
        # 此处只验证不会崩溃，实际路由可能是 procedure_q
        result = detect_route(q)
        assert isinstance(result, str)


# ============================================================
# 2. 特殊意图检测准确度
# ============================================================

class TestSpecialIntentDetection:
    """测试价格/对比/地点特殊意图检测"""

    @pytest.mark.parametrize("q,expected", [
        ("菲罗奥多少钱一支", "price"),
        ("菲罗奥怎么收费", "price"),
        ("一支多少钱", "price"),
    ])
    def test_price_intent(self, q, expected):
        assert _detect_special_intent(q) == expected

    @pytest.mark.parametrize("q,expected", [
        ("北京哪里能打菲罗奥", "location"),
        ("哪家医院可以做", "location"),
        ("附近有做菲罗奥的吗", "location"),
    ])
    def test_location_intent(self, q, expected):
        assert _detect_special_intent(q) == expected

    def test_comparison_intent(self):
        # 菲罗奥和玻尿酸对比 → comparison（玻尿酸不在 _INTERNAL_COMPARE 中）
        assert _detect_special_intent("菲罗奥和玻尿酸有什么区别") == "comparison"
        # 含产品内部成分对比时不触发 comparison
        assert _detect_special_intent("PCL和透明质酸的区别") == ""

    def test_no_special_intent(self):
        assert _detect_special_intent("术后可以洗脸吗") == ""
        assert _detect_special_intent("菲罗奥成分是什么") == ""


# ============================================================
# 3. 搜索准确度 — BM25 + 混合排序
# ============================================================

class TestBM25Accuracy:
    """BM25 评分准确度"""

    def setup_method(self):
        self.docs = [
            "菲罗奥是一款含有PCL和透明质酸的医美产品",
            "术后护理需要注意保湿和防晒",
            "禁忌人群包括孕妇、哺乳期和免疫疾病患者",
            "注射深度建议0.8mm到1.0mm",
            "联合水光和微针可以提升效果",
        ]
        self.avg_dl = sum(len(d) for d in self.docs) / len(self.docs)
        self.n_docs = len(self.docs)

    def test_relevant_doc_scores_highest(self):
        """查询相关文档应得分最高"""
        doc_freqs = {"菲罗奥": 1, "PCL": 1}
        scores = [bm25_score("菲罗奥 PCL", d, self.avg_dl, self.n_docs, doc_freqs)
                  for d in self.docs]
        assert scores[0] == max(scores)

    def test_aftercare_query_matches_aftercare_doc(self):
        doc_freqs = {"术后": 1, "护理": 1, "保湿": 1}
        scores = [bm25_score("术后护理保湿", d, self.avg_dl, self.n_docs, doc_freqs)
                  for d in self.docs]
        assert scores[1] == max(scores)

    def test_contraindication_query_matches(self):
        doc_freqs = {"禁忌": 1, "孕妇": 1}
        scores = [bm25_score("禁忌人群 孕妇", d, self.avg_dl, self.n_docs, doc_freqs)
                  for d in self.docs]
        assert scores[2] == max(scores)

    def test_zero_score_for_no_match(self):
        s = bm25_score("完全不相关的词语", "菲罗奥术后", 10.0, 3, {})
        assert s == 0.0


class TestKeywordSearchAccuracy:
    """keyword_search 整体准确度"""

    def test_returns_relevant_results(self):
        docs = [
            {"text": "菲罗奥含有PCL聚己内酯和透明质酸成分"},
            {"text": "术后需要冰敷和保湿护理"},
            {"text": "注射深度建议0.8mm"},
        ]
        results = keyword_search("菲罗奥的成分", docs, top_k=3)
        assert len(results) > 0
        # 第一个结果应该是成分相关的文档
        assert "PCL" in results[0]["text"] or "成分" in results[0]["text"]

    def test_empty_docs(self):
        assert keyword_search("任何查询", [], top_k=5) == []

    def test_empty_query(self):
        docs = [{"text": "some content"}]
        assert keyword_search("", docs, top_k=5) == []


class TestMergeHybridAccuracy:
    """混合搜索合并准确度"""

    def test_both_sources_contribute(self):
        """向量和关键词搜索都应贡献分数"""
        v_hits = [{"text": "doc1", "score": 0.9, "meta": {"source_file": "a", "chunk_id": "1"}}]
        k_hits = [{"text": "doc2", "keyword_score": 0.8, "meta": {"source_file": "b", "chunk_id": "2"}}]
        result = merge_hybrid(v_hits, k_hits, 0.6, 0.4, 10)
        assert len(result) == 2

    def test_same_doc_merges_scores(self):
        """同一文档出现在两个来源时应合并分数"""
        v_hits = [{"text": "same", "score": 0.8, "meta": {"source_file": "a", "chunk_id": "1"}}]
        k_hits = [{"text": "same", "keyword_score": 0.7, "meta": {"source_file": "a", "chunk_id": "1"}}]
        result = merge_hybrid(v_hits, k_hits, 0.6, 0.4, 10)
        assert len(result) == 1
        expected = 0.8 * 0.6 + 0.7 * 0.4
        assert abs(result[0]["hybrid_score"] - expected) < 0.01

    def test_top_k_limit(self):
        v_hits = [{"text": f"doc{i}", "score": 0.1 * i, "meta": {"source_file": f"{i}", "chunk_id": "1"}}
                  for i in range(10)]
        result = merge_hybrid(v_hits, [], 1.0, 0.0, 3)
        assert len(result) == 3

    def test_route_boost_applied(self):
        """路由匹配的文档应获得加分"""
        v_hits = [
            {"text": "产品基础信息菲罗奥", "score": 0.5, "meta": {"source_file": "a", "chunk_id": "1"}},
            {"text": "其他内容无关", "score": 0.5, "meta": {"source_file": "b", "chunk_id": "2"}},
        ]
        result = merge_hybrid(v_hits, [], 1.0, 0.0, 10, route="basic")
        # 含有 "产品基础信息" 的文档应得到路由加分
        matched = [h for h in result if "产品基础信息" in h["text"]]
        unmatched = [h for h in result if "其他内容" in h["text"]]
        if matched and unmatched:
            assert matched[0]["hybrid_score"] >= unmatched[0]["hybrid_score"]


# ============================================================
# 4. 同义词扩展准确度
# ============================================================

class TestSynonymExpansionAccuracy:
    """同义词扩展完整性和准确度"""

    @pytest.mark.parametrize("query,expected_words", [
        ("打菲罗奥疼吗", ["注射", "疼痛"]),
        ("做完脸肿了", ["肿胀", "操作"]),
        ("瘀青怎么办", ["淤青"]),
        ("玻尿酸是什么", ["透明质酸"]),
        ("有副作用吗", ["不良反应"]),
        ("术后保养怎么做", ["护理"]),
        ("提拉效果好不好", ["紧致"]),
    ])
    def test_synonym_mapping(self, query, expected_words):
        result = expand_synonyms(query)
        for word in expected_words:
            assert word in result, f"'{word}' not found in expanded: {result}"

    def test_no_expansion_for_plain_text(self):
        """无同义词时应原样返回"""
        assert expand_synonyms("术后没什么问题") == "术后没什么问题"

    def test_time_pattern_expansion(self):
        """术后第N天 应追加恢复/消退"""
        result = expand_synonyms("术后第3天还肿")
        assert "恢复" in result
        assert "消退" in result

    def test_expansion_not_unbounded(self):
        """扩展结果不应超过2000字符"""
        long_q = "打菲罗奥疼吗" * 100
        result = expand_synonyms(long_q)
        assert len(result) <= 2000

    def test_medical_procedure_synonyms(self):
        """医美项目口语→规范名"""
        result = expand_synonyms("瘦脸针安全吗")
        assert "肉毒素注射" in result

    def test_skin_condition_synonyms(self):
        """皮肤问题口语→规范术语"""
        result = expand_synonyms("斑怎么去")
        assert "色斑" in result


# ============================================================
# 5. 查询改写准确度
# ============================================================

class TestQueryRewriteAccuracy:
    """多轮对话查询改写"""

    def test_pronoun_resolution(self):
        """代词应被替换为上文提到的实体"""
        result = rewrite_query("它的效果怎么样",
                               history=[{"role": "user", "content": "菲罗奥是什么"}])
        assert "菲罗奥" in result["original"]

    def test_context_inheritance(self):
        """短问题应继承上文产品名"""
        result = rewrite_query("安全吗",
                               history=[{"role": "user", "content": "菲罗奥成分是什么"}])
        assert "菲罗奥" in result["original"]

    def test_follow_up_context(self):
        """追问应补全上下文"""
        result = rewrite_query("还有别的吗",
                               history=[{"role": "user", "content": "菲罗奥术后注意什么"}])
        assert "菲罗奥" in result["original"]

    def test_no_contamination_for_unrelated(self):
        """不相关的问题不应被补全产品名"""
        result = rewrite_query("天气怎么样",
                               history=[{"role": "user", "content": "菲罗奥成分"}])
        assert "菲罗奥" not in result["original"]

    def test_empty_history(self):
        """无历史时应返回原始问题"""
        result = rewrite_query("菲罗奥成分", history=[])
        assert "菲罗奥" in result["original"]
        assert "成分" in result["original"]


# ============================================================
# 6. 多问题拆分准确度
# ============================================================

class TestMultiQuestionSplitAccuracy:
    """复合问题拆分准确度"""

    def test_question_mark_split(self):
        """问号分隔的多个问题应拆分"""
        result = split_multi_question("成分是什么？禁忌人群有哪些？")
        assert len(result) >= 2

    def test_selection_not_split(self):
        """'还是' 选择问题不应拆分"""
        result = split_multi_question("水光针好还是微针好")
        assert len(result) == 1

    def test_pair_split(self):
        """A和B分别是什么 应拆分"""
        result = split_multi_question("水光和微针分别是什么")
        assert len(result) == 2
        assert any("水光" in r for r in result)
        assert any("微针" in r for r in result)

    def test_enumeration_split(self):
        """顿号列举应拆分"""
        result = split_multi_question("水光、微针、光电怎么选")
        assert len(result) == 3

    def test_single_question_no_split(self):
        """单个问题不应拆分"""
        result = split_multi_question("菲罗奥术后护理")
        assert len(result) == 1

    def test_short_comma_not_split(self):
        """短逗号分隔不应拆分（如 '术后1天，可以洗脸'）"""
        result = split_multi_question("术后1天，可以洗脸")
        assert len(result) == 1


# ============================================================
# 7. 产品/实体检测准确度
# ============================================================

class TestProductDetectionAccuracy:
    """产品名称检测（含别名、错别字、英文）"""

    def test_standard_name(self):
        assert detect_product("菲罗奥是什么") == "feiluoao"

    def test_english_name(self):
        assert detect_product("CELLOFILL是什么") == "feiluoao"

    def test_misspelling(self):
        """常见错别字应能识别"""
        assert detect_product("菲洛奥的禁忌人群") == "feiluoao"

    def test_alias(self):
        """赛罗菲等别名应能识别"""
        assert detect_product("赛罗菲 成分") == "feiluoao"

    def test_default_when_no_product(self):
        """无产品名时应返回默认产品"""
        result = detect_product("术后可以洗脸吗")
        assert isinstance(result, str) and len(result) > 0


class TestEntityDetection:
    """实体（项目/设备）检测准确度"""

    def test_project_detection(self):
        """项目名应被检测"""
        found = detect_terms("水光针是什么", PROJECT_ALIASES)
        assert len(found) > 0

    def test_multiple_entities(self):
        """多个实体应都被检测到"""
        found = detect_terms("水光和微针有什么区别", PROJECT_ALIASES)
        assert len(found) >= 2

    def test_no_false_positive(self):
        """不相关文本不应检测到实体"""
        found = detect_terms("今天天气不错", PROJECT_ALIASES)
        assert len(found) == 0


# ============================================================
# 8. 格式化输出准确度
# ============================================================

class TestFormatAccuracy:
    """输出格式化准确度"""

    def test_all_routes_have_title(self):
        """所有已知路由都应有对应的标题"""
        from answer_formatter import _TITLE_MAP
        known_routes = [
            "basic", "operation", "aftercare", "risk", "combo", "anti_fake",
            "contraindication", "ingredient", "effect", "pre_care", "design",
            "repair", "complication", "course", "anatomy_q", "indication_q",
            "procedure_q", "equipment_q", "script",
        ]
        for route in known_routes:
            assert route in _TITLE_MAP, f"Route '{route}' missing from _TITLE_MAP"

    def test_safety_routes_get_risk_note(self):
        """安全路由应自动附带风险提示"""
        from answer_formatter import format_structured_answer
        from rag_runtime_config import RISK_NOTE
        for route in ("contraindication", "complication", "repair", "operation"):
            result = format_structured_answer(route, ["内容"])
            assert RISK_NOTE in result, f"{route} should include risk note"

    def test_reference_note_always_present(self):
        """所有回答应包含参考提示"""
        from answer_formatter import format_structured_answer
        from rag_runtime_config import REFERENCE_NOTE
        result = format_structured_answer("basic", ["测试"])
        assert REFERENCE_NOTE in result

    def test_bullet_format(self):
        """非格式化行应加上 bullet"""
        from answer_formatter import format_structured_answer
        result = format_structured_answer("basic", ["没有bullet的行"])
        assert "- 没有bullet的行" in result

    def test_existing_bullet_preserved(self):
        """已有 bullet 的行不应重复添加"""
        from answer_formatter import format_structured_answer
        result = format_structured_answer("basic", ["- 已有bullet"])
        assert result.count("- 已有bullet") == 1
        assert "- - 已有bullet" not in result


# ============================================================
# 9. 证据构建准确度
# ============================================================

class TestBuildEvidenceAccuracy:
    """证据构建和去重"""

    def test_dedup_by_source(self):
        hits = [
            {"meta": {"source_file": "a.txt", "chunk_id": "1"}, "text": "内容1"},
            {"meta": {"source_file": "a.txt", "chunk_id": "1"}, "text": "重复"},
            {"meta": {"source_file": "b.txt", "chunk_id": "2"}, "text": "内容2"},
        ]
        result = build_evidence(hits)
        assert len(result) == 2

    def test_respects_max_chunks(self):
        from rag_runtime_config import MAX_EVIDENCE_CHUNKS
        hits = [{"meta": {"source_file": f"f{i}.txt", "chunk_id": str(i)}, "text": f"内容{i}"}
                for i in range(MAX_EVIDENCE_CHUNKS + 5)]
        result = build_evidence(hits)
        assert len(result) <= MAX_EVIDENCE_CHUNKS

    def test_truncation(self):
        """长文本应被截断"""
        long_text = "这是很长的文本。" * 200
        hits = [{"meta": {"source_file": "a.txt", "chunk_id": "1"}, "text": long_text}]
        result = build_evidence(hits)
        assert len(result[0]["text"]) < len(long_text)


class TestTruncateToSentence:
    """文本截断应在句子边界"""

    def test_short_text_unchanged(self):
        assert _truncate_to_sentence("短文本", 450) == "短文本"

    def test_truncates_at_sentence_boundary(self):
        text = "第一句。第二句。第三句很长" + "很长的内容" * 100
        result = _truncate_to_sentence(text, 50)
        # 应在句子边界截断（。；等），且保留至少一半内容
        assert len(result) <= 50
        assert len(result) >= 25

    def test_preserves_half_content(self):
        text = "内容" * 100
        result = _truncate_to_sentence(text, 100)
        assert len(result) >= 50


# ============================================================
# 10. 文本处理工具准确度
# ============================================================

class TestNormalizeAccuracy:
    """文本标准化准确度"""

    def test_bom_removal(self):
        assert "\ufeff" not in normalize_text("\ufeff测试")

    def test_crlf_normalization(self):
        result = normalize_text("行1\r\n行2\r行3")
        assert "\r" not in result
        assert result == "行1\n行2\n行3"

    def test_unicode_normalization(self):
        """NFC 标准化"""
        # 全角字符应保留（NFC 标准化不改变全角）
        result = normalize_text("ＡＢＣ")
        assert len(result) > 0


class TestNormalizeLinesAccuracy:
    """行标准化准确度"""

    def test_empty_lines_removed(self):
        result = normalize_lines("行1\n\n\n行2")
        assert len(result) == 2

    def test_separator_lines_removed(self):
        result = normalize_lines("行1\n=====\n行2")
        assert len(result) == 2
        assert all("=" not in r for r in result)

    def test_whitespace_collapsed(self):
        result = normalize_lines("行1  有  多   空格")
        assert "  " not in result[0]


class TestSectionBlockAccuracy:
    """章节提取准确度"""

    def test_basic_extraction(self):
        text = "一、产品信息\n内容A\n二、操作方法\n内容B"
        result = section_block(text, ["一、产品信息"], ["二、"])
        assert "内容A" in result
        assert "内容B" not in result

    def test_multiple_titles_first_match(self):
        text = "一、产品信息\n内容A\n二、操作方法\n内容B"
        result = section_block(text, ["一、产品信息", "二、操作方法"], ["三、"])
        assert "内容A" in result

    def test_no_match_returns_empty(self):
        result = section_block("一些内容", ["不存在的标题"], [])
        assert result == ""


class TestUniqAccuracy:
    """去重准确度"""

    def test_basic_dedup(self):
        assert uniq(["a", "b", "a", "c"]) == ["a", "b", "c"]

    def test_whitespace_normalization(self):
        assert uniq(["a b", "a  b"]) == ["a b"]

    def test_empty_items_removed(self):
        assert uniq(["a", "", None, "b"]) == ["a", "b"]


# ============================================================
# 11. 动态阈值准确度
# ============================================================

class TestDynamicThresholdAccuracy:
    """动态阈值行为验证"""

    def test_monotonic_with_top1(self):
        """top1 分数越高，阈值应越高（在范围内）"""
        t1 = compute_dynamic_threshold([{"hybrid_score": 0.3}], 0.5, ratio=0.4, floor_ratio=0.7)
        t2 = compute_dynamic_threshold([{"hybrid_score": 0.9}], 0.5, ratio=0.4, floor_ratio=0.7)
        assert t2 >= t1

    def test_bounded_by_route_threshold(self):
        """动态阈值不应超过路由阈值"""
        t = compute_dynamic_threshold([{"hybrid_score": 1.0}], 0.3, ratio=0.6, floor_ratio=0.7)
        assert t <= 0.3


# ============================================================
# 12. Sigmoid 归一化准确度
# ============================================================

class TestSigmoidNormAccuracy:
    """sigmoid 归一化特性验证"""

    def test_zero_maps_to_half(self):
        assert abs(_sigmoid_norm(0.0) - 0.5) < 0.01

    def test_positive_above_half(self):
        assert _sigmoid_norm(5.0) > 0.5

    def test_negative_below_half(self):
        assert _sigmoid_norm(-5.0) < 0.5

    def test_extreme_values_clamped(self):
        """极端值应被钳位，不会溢出"""
        assert _sigmoid_norm(10000.0) < 1.0
        assert _sigmoid_norm(-10000.0) > 0.0


# ============================================================
# 13. Reranker 准确度
# ============================================================

class TestRerankAccuracy:
    """reranker 行为验证"""

    def test_preserves_order_without_model(self):
        """无模型时应保持原始排序"""
        hits = [
            {"text": "best", "hybrid_score": 0.9},
            {"text": "good", "hybrid_score": 0.7},
            {"text": "ok", "hybrid_score": 0.5},
        ]
        result = rerank_hits("query", hits, None, 3)
        assert result[0]["text"] == "best"

    def test_top_k_enforcement(self):
        hits = [{"text": f"doc{i}", "hybrid_score": 0.1 * i} for i in range(10)]
        result = rerank_hits("query", hits, None, 5)
        assert len(result) == 5


# ============================================================
# 14. 关联引擎数据完整性
# ============================================================

class TestRelationEngineIntegrity:
    """关联引擎数据加载和完整性"""

    def test_relations_file_loadable(self):
        from relation_engine import _load
        data = _load()
        assert isinstance(data, dict)

    def test_product_procedure_structure(self):
        from relation_engine import _load
        data = _load()
        if "product_procedure" in data:
            for rel in data["product_procedure"]:
                if isinstance(rel, dict):
                    assert "product" in rel
                    assert "procedure" in rel

    def test_indication_product_structure(self):
        from relation_engine import _load
        data = _load()
        if "indication_product" in data:
            for item in data["indication_product"]:
                if isinstance(item, dict):
                    assert "indication" in item


# ============================================================
# 15. 配置完整性验证
# ============================================================

class TestConfigIntegrity:
    """运行时配置完整性"""

    def test_all_routes_have_keywords(self):
        """所有路由都应有关键词配置"""
        from rag_runtime_config import QUESTION_TYPE_CONFIG
        for route in QUESTION_TYPE_CONFIG:
            config = QUESTION_TYPE_CONFIG[route]
            assert "top_k" in config or "threshold" in config, \
                f"Route '{route}' missing search config"

    def test_section_rules_match_routes(self):
        """SECTION_RULES 应覆盖主要路由"""
        from rag_runtime_config import SECTION_RULES
        critical_routes = ["basic", "ingredient", "operation", "aftercare",
                           "anti_fake", "risk", "contraindication", "combo"]
        for route in critical_routes:
            assert route in SECTION_RULES, f"Route '{route}' missing from SECTION_RULES"

    def test_route_order_covers_all_routes(self):
        """_ROUTE_ORDER 应覆盖所有 QUESTION_ROUTES 中的路由"""
        from rag_answer import _ROUTE_ORDER
        for route in QUESTION_ROUTES:
            assert route in _ROUTE_ORDER, f"Route '{route}' missing from _ROUTE_ORDER"

    def test_question_routes_not_empty(self):
        """每个路由至少有一个关键词"""
        for route, keywords in QUESTION_ROUTES.items():
            assert len(keywords) > 0, f"Route '{route}' has no keywords"


# ============================================================
# 16. 端到端路由覆盖统计
# ============================================================

class TestRouteCoverageStats:
    """验证回归测试覆盖了所有路由"""

    def test_regression_covers_all_routes(self):
        """regression_cases.json 应覆盖所有主要路由"""
        import json
        from pathlib import Path
        cases = json.loads((Path(__file__).parent / "regression_cases.json").read_text(encoding="utf-8"))
        covered_routes = set()
        for c in cases:
            if "route" in c and c["route"]:
                covered_routes.add(c["route"])
        # 验证关键路由都被覆盖
        expected_routes = {
            "basic", "ingredient", "effect", "operation", "aftercare",
            "anti_fake", "risk", "contraindication", "combo", "pre_care",
            "design", "repair", "complication", "course", "anatomy_q",
            "indication_q", "procedure_q", "equipment_q", "script",
        }
        missing = expected_routes - covered_routes
        assert not missing, f"Routes not covered by regression tests: {missing}"
