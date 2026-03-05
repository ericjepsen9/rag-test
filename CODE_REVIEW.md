# RAG 知识库系统代码审查报告

## 一、逻辑 Bug（严重）

### 1. 向量检索结果被完全忽略（最核心 Bug）
**文件**: `rag_answer.py:330-351` `answer_one()`

虽然执行了 `vector_search` 和 `keyword_search` 混合检索，但 `hits` **只用于生成 evidence（来源引用）**，实际回答内容完全来自 `parse_answer()` 的硬编码章节正则解析（直接读 `main.txt`/`faq.txt`）。

> 整个 embedding/FAISS 向量索引形同虚设，检索结果从未影响最终答案内容。

### 2. `build_evidence()` 丢弃检索文本
**文件**: `rag_answer.py:139-145`

`build_evidence` 只提取 `meta`，完全丢弃 `h["text"]`，导致 evidence 中只有文件名和 chunk_id，没有实际内容。

### 3. 两个 `build_faiss` 版本不一致
- `build_faiss.py`: `chunk_size=600, overlap=80`，纯字符窗口
- `build_faiss_fixed.py`: `chunk_size=420, overlap=100`，按段落/标题智能切分
- 不确定实际使用哪个版本

### 4. `section_block()` 截取边界问题
**文件**: `search_utils.py:40-61`

截取从 title 位置开始（包含标题行本身），stops 从截取后的子串搜索。如果 stop 关键词出现在标题行本身（如 "五、防伪鉴别方法" 含 "防伪"），会导致截取内容为空。

### 5. `split_multi_question` 分隔符过于激进
**文件**: `search_utils.py:108`

使用 `"。"` 作为分隔符，会把正常中文句子（如"术后护理有哪些要点。"）错误拆分。

### 6. API 并发竞争
**文件**: `api_server.py:63-78`

`/ask` 通过 subprocess 调用 `rag_answer.py`，结果写入共享的 `answer.txt`。多个并发请求会互相覆盖。

---

## 二、功能 Bug

### 7. CLI 参数解析问题
**文件**: `rag_answer.py:371-376`

`sys.argv[2]` 的 top_k 参数从未被使用（`answer_question` 不接受 k 参数）。

### 8. `index.faiss` 为空文件
根目录下 `index.faiss` 大小为 0 字节，是无效索引文件。

### 9. 三套配置互相矛盾
- `rag_config.py`: product_id = `cellofill_lifting`
- `.rag_config.py`: product_id = `cellofill_lifting`
- `rag_runtime_config.py`: product_id = `feiluoao`

实际运行的 `rag_answer.py` 只 import `rag_runtime_config`，其余两个为死配置。

### 10. `QUESTION_TYPE_CONFIG` 未被使用
定义了每个 route 的 k 值和 threshold，但 `answer_one()` 从不根据 route 使用不同参数。

### 11. `rag_logger` 在 QA 流程中未被调用
`rag_answer.py` 从未 import 或调用 `log_qa()`。日志系统只在 admin rebuild 异常时使用。

---

## 三、切块（Chunking）评估

| 版本 | 切块策略 | 参数 | 状态 |
|------|---------|------|------|
| `build_faiss.py` | 纯字符窗口滑动 | 600/80 | **实际使用** |
| `build_faiss_fixed.py` | 段落感知+标题识别 | 420/100 | 未被采用 |
| `.section_parser.py` | 章节级切分+子块拆分 | 1200 | 未被采用 |

**评价**: 实际使用的是最原始的方案，会在句子/词语中间截断。但由于 Bug #1（检索结果未被使用），切块质量好坏目前不影响回答。

---

## 四、检索评估

- **向量检索**: FAISS IndexFlatIP + L2 normalize = cosine similarity，方向正确
- **关键词检索**: `keyword_score()` 用空格/逗号分隔中文查询，**等于没有分词**，对中文基本无效
- **混合检索**: score 融合无归一化；用 text 全文作为去重 key，可能错误合并不同来源的相同文本
- **核心问题**: 检索结果未被用于答案生成，整个检索管道无实际效果

---

## 五、改进建议

### P0 — 必须修复
1. 让检索结果真正驱动回答（拼接 context 送 LLM 或直接展示检索到的文本）
2. 消除 `answer.txt` 并发竞争（改为内存返回或 UUID 临时文件）
3. 统一配置文件（只保留 `rag_runtime_config.py`）

### P1 — 强烈建议
4. 采用章节感知切块（使用已有的 `section_parser.py`）
5. 根据 route 做检索后过滤（匹配 section_key）
6. 接入中文分词（jieba）改善关键词检索
7. 使用 `QUESTION_TYPE_CONFIG` 中定义的 k/threshold
8. 在 QA 流程中接入 `rag_logger`

### P2 — 锦上添花
9. 加入 cross-encoder Reranker
10. FAQ 精确/模糊匹配优先路径
11. 多轮对话上下文支持
12. 扩充回归测试用例（当前仅 4 个）
13. 清理死代码和空文件
