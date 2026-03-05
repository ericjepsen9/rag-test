"""回归测试：验证路由检测 + 答案关键词匹配

用法：
  python run_regression.py              # 仅测试路由检测（不需要知识库）
  python run_regression.py --full       # 测试路由 + 答案内容（需要知识库和模型）
"""
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CASES = json.loads((BASE_DIR / "regression_cases.json").read_text(encoding="utf-8"))


def test_route_detection():
    """测试路由检测是否准确（仅依赖轻量模块）"""
    from rag_runtime_config import QUESTION_ROUTES

    def detect_route(question: str) -> str:
        q = (question or "").lower()
        order = ["risk", "combo", "aftercare", "operation", "anti_fake", "contraindication", "ingredient", "basic"]
        matched = {}
        for route in order:
            hits = [kw for kw in QUESTION_ROUTES.get(route, []) if kw.lower() in q]
            if hits:
                matched[route] = hits
        if not matched:
            return "basic"
        if "risk" in matched and "contraindication" in matched:
            contra_signals = ["体质", "人群", "可以用", "可以打", "适合", "能用", "能打"]
            if any(s in q for s in contra_signals):
                return "contraindication"
        for route in order:
            if route in matched:
                return route
        return "basic"
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
    from rag_answer import answer_one
    ok = 0
    total = len(CASES)
    for case in CASES:
        q = case["q"]
        ans = answer_one(q, "brief")
        passed = any(x in ans for x in case.get("expect_any", []))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] answer: {q}")
        if not passed:
            print(f"    Expected any of: {case['expect_any']}")
            print(f"    Got: {ans[:200]}")
        ok += 1 if passed else 0
    print(f"  Answer content: {ok}/{total}\n")
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

    if full_mode:
        print("[2] Answer Content Test")
        ok, total = test_answer_content()
        total_ok += ok
        total_all += total
    else:
        print("[2] Answer Content Test (skipped, use --full to enable)")

    print("=" * 50)
    print(f"Total: {total_ok}/{total_all} passed")
    sys.exit(0 if total_ok == total_all else 1)


if __name__ == "__main__":
    main()
