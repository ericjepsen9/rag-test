"""医美全场景测试：验证路由检测 + 答案关键词覆盖

用法：
  python run_medical_aesthetics_test.py              # 仅测试路由检测
  python run_medical_aesthetics_test.py --full       # 测试路由 + 答案内容（需要知识库和模型）
  python run_medical_aesthetics_test.py --category 成分  # 仅测试指定分类
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
_RAW_CASES = json.loads(
    (BASE_DIR / "test_medical_aesthetics.json").read_text(encoding="utf-8")
)
CASES = [c for c in _RAW_CASES if "q" in c]


def test_route_detection(cases, verbose=True):
    """测试路由检测准确率，按分类统计"""
    from rag_answer import detect_route

    stats = defaultdict(lambda: {"ok": 0, "total": 0, "fails": []})
    ok = 0
    total = 0

    for case in cases:
        expected_route = case.get("route")
        if not expected_route:
            continue
        total += 1
        category = case.get("category", "未分类")
        actual = detect_route(case["q"])
        passed = actual == expected_route
        stats[category]["total"] += 1
        if passed:
            ok += 1
            stats[category]["ok"] += 1
        else:
            stats[category]["fails"].append(
                f"    {case['q'][:40]}  expected={expected_route}  got={actual}"
            )
            if verbose:
                print(
                    f"  [FAIL] {case['q'][:40]}  "
                    f"expected={expected_route}  got={actual}"
                )

    # 分类统计输出
    print("\n  --- 路由检测分类统计 ---")
    for cat, s in sorted(stats.items()):
        rate = s["ok"] / s["total"] * 100 if s["total"] else 0
        marker = "✓" if s["ok"] == s["total"] else "✗"
        print(f"  {marker} {cat}: {s['ok']}/{s['total']} ({rate:.0f}%)")
        for f in s["fails"]:
            print(f)

    print(f"\n  路由检测总计: {ok}/{total} ({ok/total*100:.1f}%)\n")
    return ok, total


def test_answer_content(cases, verbose=True):
    """测试答案是否包含期望关键词"""
    from rag_answer import answer_question

    stats = defaultdict(lambda: {"ok": 0, "total": 0, "fails": []})
    ok = 0
    total = len(cases)

    for case in cases:
        q = case["q"]
        category = case.get("category", "未分类")
        ans = answer_question(q, "brief")
        passed = any(x in ans for x in case.get("expect_any", []))
        stats[category]["total"] += 1
        if passed:
            ok += 1
            stats[category]["ok"] += 1
        else:
            fail_msg = f"    {q}  expect_any={case.get('expect_any', [])}"
            stats[category]["fails"].append(fail_msg)
            if verbose:
                print(f"  [FAIL] {q}")
                print(f"    Expected any of: {case.get('expect_any', [])}")
                print(f"    Got: {ans[:200]}")

    # 分类统计输出
    print("\n  --- 答案内容分类统计 ---")
    for cat, s in sorted(stats.items()):
        rate = s["ok"] / s["total"] * 100 if s["total"] else 0
        marker = "✓" if s["ok"] == s["total"] else "✗"
        print(f"  {marker} {cat}: {s['ok']}/{s['total']} ({rate:.0f}%)")
        for f in s["fails"]:
            print(f)

    print(f"\n  答案内容总计: {ok}/{total} ({ok/total*100:.1f}%)\n")
    return ok, total


def main():
    full_mode = "--full" in sys.argv

    # 分类过滤
    category_filter = None
    for i, arg in enumerate(sys.argv):
        if arg == "--category" and i + 1 < len(sys.argv):
            category_filter = sys.argv[i + 1]

    cases = CASES
    if category_filter:
        cases = [c for c in CASES if category_filter in c.get("category", "")]
        print(f"过滤分类: '{category_filter}', 匹配 {len(cases)} 条用例\n")

    print("=" * 60)
    print("医美全场景测试 (Medical Aesthetics Comprehensive Test)")
    print(f"测试用例: {len(cases)} 条 (含路由测试 + 答案测试)")
    print("=" * 60)

    total_ok, total_all = 0, 0

    print("\n[1] 路由检测测试 (Route Detection)")
    print("-" * 40)
    ok, total = test_route_detection(cases)
    total_ok += ok
    total_all += total

    if full_mode:
        print("\n[2] 答案内容测试 (Answer Content)")
        print("-" * 40)
        ok, total = test_answer_content(cases)
        total_ok += ok
        total_all += total
    else:
        print("[2] 答案内容测试 (跳过, 使用 --full 启用)")

    print("=" * 60)
    rate = total_ok / total_all * 100 if total_all else 0
    print(f"总计: {total_ok}/{total_all} 通过 ({rate:.1f}%)")
    print("=" * 60)
    sys.exit(0 if total_ok == total_all else 1)


if __name__ == "__main__":
    main()
