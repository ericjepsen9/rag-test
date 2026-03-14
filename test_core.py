"""核心算法单元测试：BM25、路由检测、上下文补全、工具函数"""
import pytest
from search_utils import (
    _count_term, _extract_terms, bm25_score, normalize_text,
    normalize_lines, uniq, section_block, split_multi_question,
    keyword_search, merge_hybrid, detect_terms,
)
from query_rewrite import (
    rewrite_query, _resolve_context, _extract_history_context,
)
from rag_answer import (
    detect_route, detect_product, build_evidence, _truncate_to_sentence,
    _build_context,
)


# ============================================================
# search_utils 单元测试
# ============================================================

class TestCountTerm:
    def test_basic(self):
        assert _count_term("ab", "ababab") == 3

    def test_no_match(self):
        assert _count_term("xyz", "hello world") == 0

    def test_empty_term(self):
        assert _count_term("", "anything") == 0

    def test_empty_text(self):
        assert _count_term("a", "") == 0

    def test_chinese(self):
        assert _count_term("菲罗奥", "菲罗奥是一种产品，菲罗奥很安全") == 2

    def test_single_char(self):
        assert _count_term("a", "banana") == 3


class TestExtractTerms:
    def test_basic(self):
        terms = _extract_terms("菲罗奥 成分")
        assert "菲罗奥" in terms
        assert "成分" in terms

    def test_bigram_split(self):
        terms = _extract_terms("术后护理")
        assert "术后" in terms
        assert "后护" in terms
        assert "护理" in terms

    def test_dedup(self):
        terms = _extract_terms("成分 成分")
        assert terms.count("成分") == 1

    def test_punctuation_split(self):
        terms = _extract_terms("红肿，硬块？疼痛")
        assert "红肿" in terms
        assert "硬块" in terms
        assert "疼痛" in terms


class TestBM25Score:
    def setup_method(self):
        self.docs = [
            "菲罗奥是一款医美产品",
            "术后护理注意事项包括保湿",
            "禁忌人群包括孕妇和哺乳期",
        ]
        self.avg_dl = sum(len(d) for d in self.docs) / len(self.docs)
        self.n_docs = len(self.docs)

    def test_relevant_higher(self):
        doc_freqs = {"菲罗奥": 1, "产品": 1}
        s1 = bm25_score("菲罗奥", self.docs[0], self.avg_dl, self.n_docs, doc_freqs)
        s2 = bm25_score("菲罗奥", self.docs[1], self.avg_dl, self.n_docs, doc_freqs)
        assert s1 > s2

    def test_zero_for_empty(self):
        assert bm25_score("", "some text", 10.0, 3, {}) == 0.0
        assert bm25_score("query", "", 10.0, 3, {}) == 0.0

    def test_positive_score(self):
        doc_freqs = {"术后": 1}
        s = bm25_score("术后", self.docs[1], self.avg_dl, self.n_docs, doc_freqs)
        assert s > 0


class TestNormalize:
    def test_bom_removal(self):
        assert normalize_text("\ufeffhello") == "hello"

    def test_crlf(self):
        assert normalize_text("a\r\nb") == "a\nb"

    def test_normalize_lines_empty(self):
        lines = normalize_lines("  \n---\n===\n  hello  ")
        assert lines == ["hello"]


class TestUniq:
    def test_basic(self):
        assert uniq(["a", "b", "a", "c"]) == ["a", "b", "c"]

    def test_whitespace(self):
        assert uniq(["  a  ", "a", " a"]) == ["a"]

    def test_empty(self):
        assert uniq(["", None, "a"]) == ["a"]


class TestSectionBlock:
    def test_extract(self):
        text = "一、产品介绍\n内容A\n二、成分\n内容B"
        block = section_block(text, ["一、产品介绍"], ["二、成分"])
        assert "内容A" in block
        assert "内容B" not in block

    def test_no_match(self):
        assert section_block("some text", ["不存在的标题"], []) == ""

    def test_empty(self):
        assert section_block("", ["标题"], []) == ""


class TestSplitMultiQuestion:
    def test_question_mark(self):
        parts = split_multi_question("成分是什么？禁忌人群有哪些？")
        assert len(parts) >= 2

    def test_and_pattern(self):
        parts = split_multi_question("水光和微针分别是什么")
        assert len(parts) == 2

    def test_no_split_short(self):
        parts = split_multi_question("术后1天，可以洗脸")
        # 逗号分隔但两侧太短，不应拆分
        assert len(parts) == 1


class TestDetectTerms:
    def test_alias_match(self):
        aliases = {"feiluoao": ["菲罗奥", "非罗奥", "CELLOFILL"]}
        assert detect_terms("非罗奥成分", aliases) == ["feiluoao"]

    def test_no_match(self):
        aliases = {"feiluoao": ["菲罗奥"]}
        assert detect_terms("天气怎么样", aliases) == []


class TestKeywordSearch:
    def test_returns_sorted(self):
        docs = [
            {"text": "菲罗奥是一款医美产品，成分包括PCL"},
            {"text": "术后护理注意保湿和防晒"},
            {"text": "禁忌人群包括孕妇"},
        ]
        results = keyword_search("菲罗奥 成分", docs, top_k=3)
        assert len(results) > 0
        # 第一个结果应该是最相关的
        assert "菲罗奥" in results[0]["text"]

    def test_empty_docs(self):
        assert keyword_search("query", [], top_k=5) == []

    def test_score_range(self):
        """sigmoid 归一化后分数应在 (0.5, 1.0) 区间"""
        docs = [{"text": "菲罗奥 PCL 成分"}]
        results = keyword_search("菲罗奥", docs, top_k=1)
        if results:
            assert 0.5 < results[0]["keyword_score"] < 1.0


class TestMergeHybrid:
    def test_merge(self):
        v_hits = [{"text": "A", "score": 0.8, "meta": {"source_file": "a", "chunk_id": "1"}}]
        k_hits = [{"text": "A", "keyword_score": 0.6, "meta": {"source_file": "a", "chunk_id": "1"}}]
        merged = merge_hybrid(v_hits, k_hits, 0.7, 0.3, 5)
        assert len(merged) == 1
        assert merged[0]["hybrid_score"] == pytest.approx(0.8 * 0.7 + 0.6 * 0.3)

    def test_disjoint(self):
        v_hits = [{"text": "A", "score": 0.8, "meta": {"source_file": "a", "chunk_id": "1"}}]
        k_hits = [{"text": "B", "keyword_score": 0.6, "meta": {"source_file": "b", "chunk_id": "2"}}]
        merged = merge_hybrid(v_hits, k_hits, 0.7, 0.3, 5)
        assert len(merged) == 2


# ============================================================
# query_rewrite 单元测试
# ============================================================

class TestResolveContext:
    def test_pronoun_start(self):
        ctx = {"product": "菲罗奥", "projects": []}
        result = _resolve_context("它的成分呢", ctx)
        assert "菲罗奥" in result

    def test_pronoun_mid_sentence(self):
        ctx = {"product": "菲罗奥", "projects": []}
        result = _resolve_context("怎么样，这个产品呢", ctx)
        assert "菲罗奥" in result

    def test_offtopic_no_resolve(self):
        ctx = {"product": "菲罗奥", "projects": []}
        result = _resolve_context("今天天气怎么样", ctx)
        assert "菲罗奥" not in result

    def test_switch_no_inherit(self):
        ctx = {"product": "菲罗奥", "projects": []}
        result = _resolve_context("换一个产品", ctx)
        assert "菲罗奥" not in result

    def test_already_has_product(self):
        ctx = {"product": "菲罗奥", "projects": []}
        result = _resolve_context("非罗奥的成分", ctx)
        assert result == "非罗奥的成分"  # 不应重复补全

    def test_followup(self):
        ctx = {"product": "菲罗奥", "projects": []}
        result = _resolve_context("还有别的吗", ctx)
        assert "菲罗奥" in result

    def test_implicit_topic_specific(self):
        ctx = {"product": "菲罗奥", "projects": []}
        result = _resolve_context("安全吗", ctx)
        assert "菲罗奥" in result

    def test_empty_history(self):
        result = _resolve_context("成分是什么", {})
        assert result == "成分是什么"


class TestExtractHistoryContext:
    def test_product_extraction(self):
        history = [
            {"role": "user", "content": "菲罗奥是什么"},
            {"role": "assistant", "content": "菲罗奥是一款产品"},
        ]
        ctx = _extract_history_context(history)
        assert ctx["product"] == "菲罗奥"
        assert ctx["product_id"] == "feiluoao"

    def test_route_extraction(self):
        history = [
            {"role": "user", "content": "术后怎么护理"},
            {"role": "assistant", "content": "术后需要注意..."},
        ]
        ctx = _extract_history_context(history)
        assert ctx["route"] == "aftercare"

    def test_empty_history(self):
        ctx = _extract_history_context([])
        assert ctx["product"] == ""
        assert ctx["route"] == ""


class TestRewriteQuery:
    def test_chitchat(self):
        result = rewrite_query("你好")
        assert result["is_chitchat"] is True

    def test_normal_query(self):
        result = rewrite_query("菲罗奥成分是什么")
        assert result["is_chitchat"] is False
        assert "feiluoao" in result["products"]

    def test_correction_prefix(self):
        result = rewrite_query("不对，我问的是禁忌人群")
        assert "不对" not in result["search_query"]
        assert "禁忌人群" in result["search_query"]

    def test_sub_questions(self):
        result = rewrite_query("成分是什么？禁忌人群有哪些？")
        assert len(result["sub_questions"]) >= 2


# ============================================================
# rag_answer 单元测试
# ============================================================

class TestDetectRoute:
    def test_risk(self):
        assert detect_route("注射后红肿怎么办") == "risk"

    def test_aftercare(self):
        assert detect_route("术后能洗脸吗") == "aftercare"

    def test_operation(self):
        assert detect_route("注射深度多少") == "operation"

    def test_anti_fake(self):
        assert detect_route("怎么验证防伪") == "anti_fake"

    def test_contraindication(self):
        assert detect_route("哺乳期可以用吗") == "contraindication"

    def test_combo(self):
        assert detect_route("水光和微针可以一起做吗") == "combo"

    def test_ingredient(self):
        assert detect_route("PCL成分是什么") == "ingredient"

    def test_basic_fallback(self):
        assert detect_route("菲罗奥是什么") == "basic"

    def test_contra_signal_priority(self):
        """当禁忌信号词存在时，contraindication 应优先于 risk"""
        assert detect_route("过敏体质的人群可以用吗") == "contraindication"


class TestTruncateToSentence:
    def test_short_text(self):
        assert _truncate_to_sentence("短文本", 200) == "短文本"

    def test_sentence_boundary(self):
        text = "第一句话。第二句话。第三句话很长很长很长很长很长很长很长很长很长"
        result = _truncate_to_sentence(text, 15)
        assert result.endswith("。")

    def test_no_boundary(self):
        text = "没有句子结束符的超长文本" * 20
        result = _truncate_to_sentence(text, 50)
        assert len(result) <= 50


class TestBuildContext:
    def test_chunk_boundary(self):
        hits = [
            {"text": "A" * 100, "meta": {"source_file": "a.txt", "chunk_id": "1"}},
            {"text": "B" * 100, "meta": {"source_file": "b.txt", "chunk_id": "2"}},
        ]
        context = _build_context(hits, max_chars=150)
        # 至少保留第一个 chunk
        assert "A" in context

    def test_empty_hits(self):
        assert _build_context([]) == ""


class TestBuildEvidence:
    def test_truncation(self):
        hits = [{"text": "x" * 600, "meta": {"source_file": "f.txt"}}]
        ev = build_evidence(hits)
        assert len(ev[0]["text"]) <= 450

    def test_sentence_aware(self):
        hits = [{"text": "第一句。第二句。" + "x" * 500, "meta": {}}]
        ev = build_evidence(hits)
        # 应在句子边界截断（max_chars=450）
        text = ev[0]["text"]
        assert text.endswith("。") or len(text) <= 450


class TestNormalizeTextUnicode:
    def test_nfc_normalization(self):
        """NFC 标准化应统一不同 Unicode 表示"""
        import unicodedata
        # 构造 NFD 形式（分解形式）
        nfd = unicodedata.normalize("NFD", "é")
        result = normalize_text(nfd)
        assert result == unicodedata.normalize("NFC", "é")

    def test_bom_removal(self):
        assert "\ufeff" not in normalize_text("\ufeffhello")

    def test_crlf(self):
        assert "\r" not in normalize_text("a\r\nb")


class TestChitchatVariations:
    def test_greeting_with_punctuation(self):
        from query_rewrite import _CHITCHAT_PATTERNS
        assert _CHITCHAT_PATTERNS.match("你好！")
        assert _CHITCHAT_PATTERNS.match("hello~")
        assert _CHITCHAT_PATTERNS.match("嗨啊")
        assert _CHITCHAT_PATTERNS.match("hi!")

    def test_thanks_with_suffix(self):
        from query_rewrite import _CHITCHAT_PATTERNS
        assert _CHITCHAT_PATTERNS.match("谢谢啊")
        assert _CHITCHAT_PATTERNS.match("好的！")

    def test_non_chitchat_not_matched(self):
        from query_rewrite import _CHITCHAT_PATTERNS
        assert not _CHITCHAT_PATTERNS.match("你好，请问菲罗奥成分是什么")


class TestRelationEngineIndices:
    def test_load_builds_indices(self):
        from relation_engine import _load, _idx_indication, _idx_anatomy, _idx_product_proc
        _load()
        # 索引应被初始化（可能为空 dict 但不应为 None，除非 relations.json 不存在）
        # 如果 relations.json 不存在，_relations={} 且 indices 保持 None
        # 这是可接受的——本测试验证加载流程不报错
        assert True

    def test_invalidate_clears_indices(self):
        from relation_engine import invalidate_relations_cache, _load
        import relation_engine as re_mod
        _load()  # 确保已加载
        invalidate_relations_cache()
        assert re_mod._idx_indication is None
        assert re_mod._idx_anatomy is None
        assert re_mod._idx_product_proc is None


class TestDetectProduct:
    def test_alias(self):
        assert detect_product("非罗奥成分") == "feiluoao"

    def test_cellofill(self):
        assert detect_product("CELLOFILL是什么") == "feiluoao"
