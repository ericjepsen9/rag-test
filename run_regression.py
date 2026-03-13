"""回归测试：验证路由检测 + 答案关键词匹配

用法：
  python run_regression.py              # 仅测试路由检测（不需要知识库）
  python run_regression.py --full       # 测试路由 + 答案内容（需要知识库和模型）
"""
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
_RAW_CASES = json.loads((BASE_DIR / "regression_cases.json").read_text(encoding="utf-8"))
# 过滤掉注释条目（只有 _comment 字段、没有 q 字段的条目）
CASES = [c for c in _RAW_CASES if "q" in c]


def test_route_detection():
    """测试路由检测是否准确（使用 rag_answer 中的实际 detect_route）"""
    from rag_answer import detect_route

    ok = 0
    total = 0
    for case in CASES:
        expected_route = case.get("route")
        if not expected_route:
            continue
        total += 1
        actual = detect_route(case["q"])
        passed = actual == expected_route
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] route: {case['q'][:30]}  expected={expected_route}  got={actual}")
        ok += 1 if passed else 0
    print(f"  Route detection: {ok}/{total}\n")
    return ok, total


def test_answer_content():
    """测试答案是否包含期望关键词"""
    from rag_answer import answer_question
    ok = 0
    total = len(CASES)
    for case in CASES:
        q = case["q"]
        # 使用 answer_question 而非 answer_one，以覆盖特殊意图（价格/对比/地点）路径
        ans = answer_question(q, "brief")
        passed = any(x in ans for x in case.get("expect_any", []))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] answer: {q}")
        if not passed:
            print(f"    Expected any of: {case['expect_any']}")
            print(f"    Got: {ans[:200]}")
        ok += 1 if passed else 0
    print(f"  Answer content: {ok}/{total}\n")
    return ok, total


def test_synonym_expansion():
    """测试同义词扩展是否正确"""
    from search_utils import expand_synonyms

    cases = [
        ("打菲罗奥疼吗", ["注射", "疼痛"]),
        ("做完脸肿了", ["肿胀", "操作"]),
        ("瘀青怎么办", ["淤青"]),
        ("玻尿酸是什么", ["透明质酸"]),
        ("有副作用吗", ["不良反应"]),
        ("术后保养怎么做", ["护理"]),
        ("提拉效果好不好", ["紧致"]),
        ("术后没什么问题", []),  # 无需扩展
        ("术后第3天还肿", ["恢复", "消退"]),  # 时间模式扩展
    ]
    ok = 0
    total = len(cases)
    for q, expected in cases:
        result = expand_synonyms(q)
        passed = all(e in result for e in expected)
        if not expected:
            passed = result == q  # 无扩展时应原样返回
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] synonym: {q} → {result[:60]}")
        ok += 1 if passed else 0
    print(f"  Synonym expansion: {ok}/{total}\n")
    return ok, total


def test_query_rewrite_context():
    """测试对话历史上下文补全"""
    from query_rewrite import rewrite_query

    cases = [
        # (问题, 历史, 预期包含的词)
        ("安全吗", [{"role": "user", "content": "菲罗奥成分是什么"}], "菲罗奥"),
        ("还有别的吗", [{"role": "user", "content": "菲罗奥术后注意什么"}], "菲罗奥"),
        ("它的效果怎么样", [{"role": "user", "content": "菲罗奥是什么"}], "菲罗奥"),
        ("天气怎么样", [{"role": "user", "content": "菲罗奥成分"}], None),  # 不应补全
    ]
    ok = 0
    total = len(cases)
    for q, history, expect_word in cases:
        result = rewrite_query(q, history=history)
        resolved = result["original"]
        if expect_word:
            passed = expect_word in resolved
        else:
            passed = "菲罗奥" not in resolved  # 不应包含产品名
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] rewrite: \"{q}\" → \"{resolved}\"")
        ok += 1 if passed else 0
    print(f"  Query rewrite context: {ok}/{total}\n")
    return ok, total


def main():
    full_mode = "--full" in sys.argv

    print("=" * 50)
    print("RAG Regression Test")
    print("=" * 50)

    total_ok, total_all = 0, 0

    print("\n[1] Route Detection Test")
    ok, total = test_route_detection()
    total_ok += ok
    total_all += total

    print("[2] Synonym Expansion Test")
    ok, total = test_synonym_expansion()
    total_ok += ok
    total_all += total

    print("[3] Query Rewrite Context Test")
    ok, total = test_query_rewrite_context()
    total_ok += ok
    total_all += total

    if full_mode:
        print("[4] Answer Content Test")
        ok, total = test_answer_content()
        total_ok += ok
        total_all += total
    else:
        print("[4] Answer Content Test (skipped, use --full to enable)")

    print("=" * 50)
    print(f"Total: {total_ok}/{total_all} passed")
    sys.exit(0 if total_ok == total_all else 1)


if __name__ == "__main__":
    main()
