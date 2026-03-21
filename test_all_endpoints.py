#!/usr/bin/env python3
"""
RAG 系统 API 端点综合测试脚本
Comprehensive test script for all key API endpoints of the RAG system.

用法 / Usage:
    python test_all_endpoints.py --key YOUR_ADMIN_KEY
    python test_all_endpoints.py --key YOUR_KEY --test-url "https://mp.weixin.qq.com/s/xxx"
    python test_all_endpoints.py --host 192.168.1.10 --port 9000 --key YOUR_KEY
"""

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

# ============================================================
# 终端颜色定义 - ANSI escape codes for colored terminal output
# ============================================================

class Colors:
    """终端颜色常量"""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def colored(text: str, color: str) -> str:
    """给文本添加终端颜色"""
    return f"{color}{text}{Colors.RESET}"


# ============================================================
# 测试结果数据结构 - Data structure for individual test results
# ============================================================

class TestResult:
    """单个测试用例的结果记录"""

    def __init__(self, step: int, name: str, method: str, endpoint: str):
        self.step = step              # 步骤编号
        self.name = name              # 测试名称
        self.method = method          # HTTP 方法 (GET/POST)
        self.endpoint = endpoint      # 请求路径
        self.status = "SKIP"          # 测试状态: PASS / FAIL / SKIP
        self.status_code: Optional[int] = None    # HTTP 响应码
        self.request_body: Optional[str] = None   # 请求体（截断后）
        self.response_body: Optional[str] = None  # 响应体（截断后）
        self.duration_ms: float = 0.0             # 耗时（毫秒）
        self.error_message: Optional[str] = None  # 错误信息
        self.skip_reason: Optional[str] = None    # 跳过原因

    def mark_pass(self, status_code: int, response_body: str, duration_ms: float):
        """标记测试通过"""
        self.status = "PASS"
        self.status_code = status_code
        self.response_body = _truncate(response_body, 2000)
        self.duration_ms = duration_ms

    def mark_fail(self, error_message: str, status_code: Optional[int] = None,
                  response_body: Optional[str] = None, duration_ms: float = 0.0):
        """标记测试失败"""
        self.status = "FAIL"
        self.error_message = error_message
        self.status_code = status_code
        self.response_body = _truncate(response_body, 2000) if response_body else None
        self.duration_ms = duration_ms

    def mark_skip(self, reason: str):
        """标记测试跳过"""
        self.status = "SKIP"
        self.skip_reason = reason

    @property
    def status_colored(self) -> str:
        """返回带颜色的状态文本"""
        if self.status == "PASS":
            return colored("PASS", Colors.GREEN)
        elif self.status == "FAIL":
            return colored("FAIL", Colors.RED)
        else:
            return colored("SKIP", Colors.YELLOW)

    @property
    def status_plain(self) -> str:
        """返回不带颜色的状态文本（用于写入文件）"""
        return self.status


def _truncate(text: str, max_len: int) -> str:
    """截断过长的文本，添加省略标记"""
    if text and len(text) > max_len:
        return text[:max_len] + f"\n... [truncated, total {len(text)} chars]"
    return text


# ============================================================
# API 测试运行器 - Main test runner class
# ============================================================

class RAGApiTester:
    """RAG 系统 API 综合测试器"""

    def __init__(self, host: str, port: int, api_key: str, test_url: Optional[str] = None):
        self.base_url = f"http://{host}:{port}"  # 基础 URL
        self.api_key = api_key                     # 管理员 API 密钥
        self.test_url = test_url                   # 可选的测试文章 URL
        self.results: List[TestResult] = []        # 测试结果列表
        self.step_counter = 0                      # 步骤计数器

        # 在测试过程中收集的共享数据，供后续测试使用
        self.discovered_products: List[str] = []   # 发现的产品列表
        self.fetched_media: List[Dict] = []        # fetch_url 返回的媒体列表

        # 通用请求头，包含认证信息
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # --------------------------------------------------------
    # 核心请求方法 - Core HTTP request helpers
    # --------------------------------------------------------

    def _next_step(self) -> int:
        """递增并返回下一个步骤编号"""
        self.step_counter += 1
        return self.step_counter

    def _request(self, method: str, path: str, json_body: Optional[Dict] = None,
                 params: Optional[Dict] = None, timeout: int = 60) -> TestResult:
        """
        发送 HTTP 请求并记录结果。
        这是所有测试用例的底层请求方法。
        """
        step = self._next_step()
        result = TestResult(step=step, name="", method=method, endpoint=path)

        # 记录请求体（截断）
        if json_body is not None:
            result.request_body = _truncate(json.dumps(json_body, ensure_ascii=False, indent=2), 1000)

        url = f"{self.base_url}{path}"

        try:
            start = time.time()
            resp = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                json=json_body,
                params=params,
                timeout=timeout,
            )
            elapsed_ms = (time.time() - start) * 1000

            # 尝试获取响应文本
            try:
                body_text = resp.text
            except Exception:
                body_text = "<unable to decode response body>"

            # 判断是否成功（2xx 状态码视为成功）
            if 200 <= resp.status_code < 300:
                result.mark_pass(resp.status_code, body_text, elapsed_ms)
            else:
                result.mark_fail(
                    error_message=f"HTTP {resp.status_code}",
                    status_code=resp.status_code,
                    response_body=body_text,
                    duration_ms=elapsed_ms,
                )
        except requests.exceptions.ConnectionError as exc:
            result.mark_fail(f"Connection error: {exc}")
        except requests.exceptions.Timeout:
            result.mark_fail(f"Request timed out after {timeout}s")
        except Exception as exc:
            result.mark_fail(f"Unexpected error: {type(exc).__name__}: {exc}")

        return result

    def _run_test(self, name: str, method: str, path: str,
                  json_body: Optional[Dict] = None,
                  params: Optional[Dict] = None,
                  skip_reason: Optional[str] = None,
                  timeout: int = 60) -> TestResult:
        """
        执行单个测试并打印实时结果。
        这是每个测试用例调用的统一入口。
        """
        # 如果需要跳过
        if skip_reason:
            step = self._next_step()
            result = TestResult(step=step, name=name, method=method, endpoint=path)
            result.mark_skip(skip_reason)
            self.results.append(result)
            self._print_result(result)
            return result

        # 打印正在测试的信息
        step_preview = self.step_counter + 1
        print(f"\n{Colors.CYAN}[{step_preview}]{Colors.RESET} {Colors.BOLD}{name}{Colors.RESET}")
        print(f"    {Colors.DIM}{method} {path}{Colors.RESET}", end="", flush=True)

        # 发送请求
        result = self._request(method, path, json_body=json_body, params=params, timeout=timeout)
        result.name = name
        self.results.append(result)

        # 打印结果
        self._print_result_inline(result)
        return result

    def _print_result_inline(self, result: TestResult):
        """在同一行追加打印结果状态"""
        duration_str = f"{result.duration_ms:.0f}ms"
        status_str = result.status_colored
        code_str = f"[{result.status_code}]" if result.status_code else ""
        print(f"  -> {status_str} {code_str} ({duration_str})")

        # 失败时打印错误详情
        if result.status == "FAIL" and result.error_message:
            print(f"    {Colors.RED}Error: {result.error_message}{Colors.RESET}")

    def _print_result(self, result: TestResult):
        """完整打印一个结果（用于 SKIP 等情况）"""
        print(f"\n{Colors.CYAN}[{result.step}]{Colors.RESET} {Colors.BOLD}{result.name}{Colors.RESET}")
        print(f"    {Colors.DIM}{result.method} {result.endpoint}{Colors.RESET}")
        if result.skip_reason:
            print(f"    -> {result.status_colored} ({result.skip_reason})")
        else:
            self._print_result_inline(result)

    def _get_json(self, result: TestResult) -> Optional[Any]:
        """尝试将响应体解析为 JSON"""
        if result.status != "PASS" or not result.response_body:
            return None
        try:
            return json.loads(result.response_body.split("\n... [truncated")[0])
        except (json.JSONDecodeError, ValueError):
            return None

    # --------------------------------------------------------
    # 测试用例组 - Individual test case methods
    # --------------------------------------------------------

    def test_health(self):
        """健康检查端点 - 验证服务是否存活"""
        self._run_test("Health Check", "GET", "/health")

    def test_stats(self):
        """管理员统计信息 - 获取系统整体统计数据"""
        self._run_test("Admin Stats", "GET", "/admin/stats")

    def test_products(self):
        """产品列表 - 获取所有已配置的产品，供后续测试使用"""
        result = self._run_test("Products List", "GET", "/admin/products")
        data = self._get_json(result)

        # 从响应中提取产品名称列表，供后续测试引用
        if data:
            if isinstance(data, list):
                # 响应直接是列表
                self.discovered_products = [
                    (p.get("name") or p.get("product") or p.get("id") or str(p))
                    if isinstance(p, dict) else str(p)
                    for p in data
                ]
            elif isinstance(data, dict):
                # 响应是字典，可能包含 products 或 items 字段
                items = data.get("products") or data.get("items") or data.get("data") or []
                if isinstance(items, list):
                    self.discovered_products = [
                        (p.get("name") or p.get("product") or p.get("id") or str(p))
                        if isinstance(p, dict) else str(p)
                        for p in items
                    ]

        if self.discovered_products:
            print(f"    {Colors.DIM}Discovered products: {self.discovered_products}{Colors.RESET}")

    def test_config(self):
        """配置端点 - 获取系统配置和模型配置"""
        self._run_test("Config (General)", "GET", "/admin/config")
        self._run_test("Config (Model)", "GET", "/admin/config/model")

    def test_services(self):
        """服务状态 - 检查嵌入服务和 LLM 服务的状态"""
        self._run_test("Service: Embedding", "GET", "/admin/service/embedding")
        self._run_test("Service: LLM", "GET", "/admin/service/llm")

    def test_llm_configs(self):
        """LLM 配置列表 - 获取所有可用的 LLM 配置"""
        self._run_test("LLM Configs", "GET", "/admin/llm/configs")

    def test_knowledge(self):
        """知识库管理 - 获取指定产品的知识库内容"""
        if not self.discovered_products:
            self._run_test(
                "Knowledge List", "GET", "/admin/knowledge/{product}",
                skip_reason="No products discovered from previous test",
            )
            return

        product = self.discovered_products[0]
        self._run_test(
            f"Knowledge List (product={product})",
            "GET",
            f"/admin/knowledge/{product}",
        )

    def test_synonyms(self):
        """同义词管理 - 获取所有同义词配置"""
        self._run_test("Synonyms (All)", "GET", "/admin/synonyms/all")

    def test_ask(self):
        """搜索/问答 - 向 RAG 系统发送一个简单的测试问题"""
        # 构建一个简单的测试问题
        body = {
            "question": "hello, this is a test question",
        }
        # 如果有已知产品，可以指定产品范围
        if self.discovered_products:
            body["product"] = self.discovered_products[0]

        self._run_test(
            "Ask (Search/QA)",
            "POST",
            "/ask",
            json_body=body,
            timeout=120,  # 问答可能较慢，给更长超时
        )

    def test_fetch_url(self):
        """URL 抓取 - 抓取指定文章 URL 的内容（需要 --test-url 参数）"""
        if not self.test_url:
            self._run_test(
                "Fetch URL", "POST", "/admin/fetch_url",
                skip_reason="No --test-url provided",
            )
            return

        body = {"url": self.test_url}
        result = self._run_test(
            "Fetch URL",
            "POST",
            "/admin/fetch_url",
            json_body=body,
            timeout=120,  # 网络抓取可能较慢
        )

        # 提取媒体列表供后续 proxy_media / save_media 测试使用
        data = self._get_json(result)
        if data and isinstance(data, dict):
            self.fetched_media = data.get("media") or data.get("images") or []

    def test_proxy_media(self):
        """媒体代理 - 通过代理获取媒体资源（需要 fetch_url 返回的媒体）"""
        if not self.fetched_media:
            self._run_test(
                "Proxy Media", "GET", "/admin/proxy_media",
                skip_reason="No media available from fetch_url result",
            )
            return

        # 取第一个媒体的 URL 进行测试
        first_media = self.fetched_media[0]
        media_url = first_media if isinstance(first_media, str) else (
            first_media.get("url") or first_media.get("src") or ""
        )

        if not media_url:
            self._run_test(
                "Proxy Media", "GET", "/admin/proxy_media",
                skip_reason="Could not extract media URL from fetch_url response",
            )
            return

        self._run_test(
            "Proxy Media",
            "GET",
            "/admin/proxy_media",
            params={"url": media_url},
        )

    def test_save_media(self):
        """保存媒体 - 测试保存媒体接口（使用空列表进行安全测试）"""
        body = {"media_list": [], "product": self.discovered_products[0] if self.discovered_products else "test"}
        self._run_test(
            "Save Media (empty list)",
            "POST",
            "/admin/save_media",
            json_body=body,
        )

    def test_import_knowledge(self):
        """知识导入（干跑模式） - 使用 dry_run=true 测试知识导入流程"""
        body = {
            "product": self.discovered_products[0] if self.discovered_products else "test",
            "content": "This is a test knowledge entry for dry-run import validation.",
            "title": "API Test - Dry Run",
            "dry_run": True,
        }
        self._run_test(
            "Import Knowledge (dry_run=true)",
            "POST",
            "/admin/import_knowledge",
            json_body=body,
            timeout=120,
        )

    def test_llm_test(self):
        """LLM 测试 - 验证 LLM 服务是否正常工作"""
        self._run_test(
            "LLM Test",
            "POST",
            "/admin/llm/test",
            json_body={},
            timeout=120,
        )

    def test_cache(self):
        """缓存管理 - 获取当前缓存状态"""
        self._run_test("Cache Status", "GET", "/admin/cache")

    def test_logs(self):
        """日志查询 - 获取 QA 日志、未命中日志和错误日志"""
        self._run_test("Logs: QA", "GET", "/admin/logs/qa")
        self._run_test("Logs: Miss", "GET", "/admin/logs/miss")
        self._run_test("Logs: Error", "GET", "/admin/logs/error")

    def test_rebuild(self):
        """索引重建 - 触发指定产品的索引重建（可选，可能耗时较长）"""
        if not self.discovered_products:
            self._run_test(
                "Index Rebuild", "POST", "/admin/rebuild",
                skip_reason="No products available; skipping rebuild",
            )
            return

        product = self.discovered_products[0]
        body = {"product": product}
        self._run_test(
            f"Index Rebuild (product={product})",
            "POST",
            "/admin/rebuild",
            json_body=body,
            timeout=300,  # 重建可能非常耗时
        )

    # --------------------------------------------------------
    # 主执行流程 - Main execution orchestrator
    # --------------------------------------------------------

    def run_all(self):
        """
        按照指定顺序依次执行所有测试组。
        每个测试失败不影响后续测试继续执行。
        """
        banner = "RAG System - Comprehensive API Endpoint Tests"
        print(f"\n{'=' * 60}")
        print(colored(f"  {banner}", Colors.BOLD))
        print(f"{'=' * 60}")
        print(f"  Base URL : {self.base_url}")
        print(f"  API Key  : {self.api_key[:8]}{'*' * (len(self.api_key) - 8) if len(self.api_key) > 8 else ''}")
        print(f"  Test URL : {self.test_url or '(not provided)'}")
        print(f"  Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")

        # 按顺序执行所有测试组
        test_groups = [
            ("Health Check", self.test_health),
            ("Stats", self.test_stats),
            ("Products", self.test_products),
            ("Config", self.test_config),
            ("Services", self.test_services),
            ("LLM Configs", self.test_llm_configs),
            ("Knowledge", self.test_knowledge),
            ("Synonyms", self.test_synonyms),
            ("Search/Ask", self.test_ask),
            ("Fetch URL", self.test_fetch_url),
            ("Proxy Media", self.test_proxy_media),
            ("Save Media", self.test_save_media),
            ("Import Knowledge", self.test_import_knowledge),
            ("LLM Test", self.test_llm_test),
            ("Cache", self.test_cache),
            ("Logs", self.test_logs),
            ("Index Rebuild", self.test_rebuild),
        ]

        for group_name, test_fn in test_groups:
            print(f"\n{Colors.BOLD}--- {group_name} ---{Colors.RESET}")
            try:
                test_fn()
            except Exception as exc:
                # 捕获所有异常，确保不会中断整个测试流程
                print(colored(f"  Unexpected error in test group '{group_name}': {exc}", Colors.RED))

        # 生成并输出报告
        self._generate_report()

    # --------------------------------------------------------
    # 报告生成 - Report generation and output
    # --------------------------------------------------------

    def _generate_report(self):
        """生成综合测试报告，输出到终端并保存到文件"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        skipped = sum(1 for r in self.results if r.status == "SKIP")

        # ---- 终端输出摘要 ----
        print(f"\n{'=' * 60}")
        print(colored("  TEST SUMMARY", Colors.BOLD))
        print(f"{'=' * 60}")
        print(f"  Total  : {total}")
        print(f"  Passed : {colored(str(passed), Colors.GREEN)}")
        print(f"  Failed : {colored(str(failed), Colors.RED)}")
        print(f"  Skipped: {colored(str(skipped), Colors.YELLOW)}")
        print(f"{'=' * 60}")

        if failed == 0:
            print(colored("\n  All executed tests passed!\n", Colors.GREEN))
        else:
            print(colored(f"\n  {failed} test(s) failed. See details below.\n", Colors.RED))

        # ---- 构建纯文本报告（写入文件） ----
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("  RAG SYSTEM API TEST REPORT")
        report_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"  Base URL : {self.base_url}")
        report_lines.append(f"  Test URL : {self.test_url or '(not provided)'}")
        report_lines.append("=" * 70)
        report_lines.append("")
        report_lines.append(f"SUMMARY: Total={total}  Pass={passed}  Fail={failed}  Skip={skipped}")
        report_lines.append("")
        report_lines.append("-" * 70)

        for r in self.results:
            report_lines.append("")
            report_lines.append(f"[{r.step}] {r.status_plain} | {r.name}")
            report_lines.append(f"    Method   : {r.method} {r.endpoint}")
            report_lines.append(f"    Status   : HTTP {r.status_code}" if r.status_code else f"    Status   : N/A")
            report_lines.append(f"    Duration : {r.duration_ms:.0f}ms")

            if r.skip_reason:
                report_lines.append(f"    Skipped  : {r.skip_reason}")

            if r.error_message:
                report_lines.append(f"    Error    : {r.error_message}")

            if r.request_body:
                report_lines.append(f"    Request  :")
                for line in r.request_body.splitlines():
                    report_lines.append(f"        {line}")

            if r.response_body:
                report_lines.append(f"    Response :")
                for line in r.response_body.splitlines():
                    report_lines.append(f"        {line}")

            report_lines.append("-" * 70)

        report_text = "\n".join(report_lines) + "\n"

        # ---- 保存报告到文件 ----
        report_path = "test_report.txt"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"  Report saved to: {Colors.CYAN}{report_path}{Colors.RESET}")
        except IOError as exc:
            print(colored(f"  Failed to save report: {exc}", Colors.RED))

        # ---- 打印详细报告到终端 ----
        print(f"\n{'=' * 60}")
        print(colored("  DETAILED RESULTS", Colors.BOLD))
        print(f"{'=' * 60}")

        for r in self.results:
            duration_str = f"{r.duration_ms:.0f}ms"
            print(f"\n  [{r.step}] {r.status_colored} | {r.name}")
            print(f"       {r.method} {r.endpoint}  ({duration_str})")
            if r.status_code:
                print(f"       HTTP {r.status_code}")
            if r.skip_reason:
                print(f"       {colored('Reason: ' + r.skip_reason, Colors.YELLOW)}")
            if r.error_message:
                print(f"       {colored('Error: ' + r.error_message, Colors.RED)}")

        print(f"\n{'=' * 60}\n")


# ============================================================
# 命令行参数解析 - CLI argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="RAG 系统 API 端点综合测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 / Examples:
  python test_all_endpoints.py --key sk-abc123
  python test_all_endpoints.py --key sk-abc123 --test-url "https://mp.weixin.qq.com/s/xxx"
  python test_all_endpoints.py --host 10.0.0.1 --port 9000 --key sk-abc123
        """,
    )
    parser.add_argument(
        "--host", default="localhost",
        help="API 服务器主机地址 (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="API 服务器端口 (default: 8000)",
    )
    parser.add_argument(
        "--key", required=True,
        help="管理员 API 密钥 (ADMIN_API_KEY)",
    )
    parser.add_argument(
        "--test-url", default=None,
        help="可选的文章 URL，用于测试 fetch_url 端点",
    )
    return parser.parse_args()


# ============================================================
# 入口点 - Script entry point
# ============================================================

def main():
    """主函数：解析参数并运行所有测试"""
    args = parse_args()

    tester = RAGApiTester(
        host=args.host,
        port=args.port,
        api_key=args.key,
        test_url=args.test_url,
    )

    tester.run_all()

    # 如果有任何测试失败，以非零退出码退出
    failed = sum(1 for r in tester.results if r.status == "FAIL")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
