# RAG 知识库系统代码审查报告（修订版）

> 基于 `rag_final_complete_with_admin_knowledge_and_media_20260228.zip` 最终整合包
> 结合 5 份产品文档（框架结构、功能文档、接口说明、版本演进、替换说明）重新评估

---

## 总体评价

这是一个经过 **7 个迭代阶段** 的医美知识库问答系统，架构清晰、分层合理：
- 知识层（`knowledge/<product>/`）→ 索引层（`stores/<product>/`）→ 检索+规则层（`rag_answer.py`）→ API 层 → 媒体/后台层

**"规则优先 + 检索辅助"** 是有意的设计决策（第二阶段引入），目的是避免纯检索拼接导致的串段问题。对于固定资料型问答（防伪、禁忌、术后、操作），这个策略是合理的。

当前版本已经是一个 **能跑、能维护** 的系统。以下问题按"必须修"、"建议修"、"未来优化"分级。

---

## 一、必须修复的问题（P0）

### 1. API 并发竞争 — 多用户会互相覆盖答案
**文件**: `api_server.py:62-91`

`/ask` 通过 subprocess 调用 `rag_answer.py`，结果写入共享的 `answer.txt`。**两个用户同时提问，后者会覆盖前者的答案文件**，导致前者读到错误答案。

**修复方案**: 每次请求生成唯一临时文件（如 `answer_{uuid}.txt`），读取后删除；或改为 `rag_answer` 直接 import 调用（内存返回，不写文件）。

### 2. `rag_logger` 在 QA 流程中未接入
**文件**: `rag_answer.py`（全文未 import `rag_logger`）

`04_版本更新详细说明.md` 第六阶段明确说"增加 `rag_logger.py`，可查看 QA/miss/error 日志"，但实际 QA 主流程从未调用 `log_qa()`。日志系统目前 **只在 admin rebuild 异常时生效**。

**影响**: 后台 `/admin/logs/qa` 和 `/admin/logs/miss` 永远为空，后台管理形同虚设。

**修复方案**: 在 `rag_answer.py` 的 `answer_one()` 或 `answer_question()` 末尾调用 `log_qa()`，未命中时传 `hit=False`。

### 3. 三套配置文件互相矛盾
| 文件 | product_id | 状态 |
|------|-----------|------|
| `rag_runtime_config.py` | `feiluoao` | **实际使用** |
| `rag_config.py` | `cellofill_lifting` | 死配置 |
| `.rag_config.py` | `cellofill_lifting` | 死配置（隐藏文件） |

`rag_answer.py`、`build_faiss.py` 都只 import `rag_runtime_config`，另外两个文件不被任何代码引用。

**修复方案**: 删除 `rag_config.py` 和 `.rag_config.py`（或标注为归档），避免维护者混淆。

### 4. 根目录 `index.faiss` 为空文件（0 字节）
这个文件会误导使用者以为索引在根目录。实际索引应在 `stores/<product>/index.faiss`。

**修复方案**: 删除根目录的空 `index.faiss`。

---

## 二、建议修复的问题（P1）— ✅ 全部已修复

### 5. ✅ 向量检索结果仅用于 evidence，未参与答案生成
**文件**: `rag_answer.py:330-351`
**状态**: 已修复 — `_fallback_from_hits()` 在 `parse_answer` 返回空时从检索 chunks 提取内容作为 fallback。

### 6. ✅ `build_evidence()` 丢弃了检索到的实际文本
**文件**: `rag_answer.py:529-545`
**状态**: 已修复 — `build_evidence()` 现已保留 `text` 字段（经 `_truncate_to_sentence` 截断），支持答案溯源和调试。

### 7. ✅ `section_block()` 的边界条件
**文件**: `search_utils.py:138-162`
**状态**: 已修复 — 截取时跳过标题行本身（`title_end = start + len(chosen)`），从标题之后开始搜索 stops。

### 8. ✅ 中文关键词检索基本无效
**文件**: `search_utils.py:165-181`
**状态**: 已修复 — `_extract_terms()` 对纯中文长词（≥3字）做 bigram 切分，提升部分匹配召回率。同时配合同义词扩展提高中文检索效果。

### 9. ✅ `QUESTION_TYPE_CONFIG` 定义了但未使用
**文件**: `rag_answer.py:1244-1295`
**状态**: 已修复 — `answer_one()` 根据 `detect_route()` 动态选取 `route_cfg` 中的 k、threshold、vw、kw 参数。

### 10. ✅ `split_multi_question` 分隔符过于激进
**文件**: `search_utils.py:415-485`
**状态**: 已修复 — 分隔符列表不再包含 `"。"`，避免中文句号误拆。逗号拆分加入 ≥6 字符最小长度限制。

---

## 三、两个 `build_faiss` 的取舍 — ✅ 已合并

已统一为单一 `build_faiss.py`，`build_faiss_fixed.py` 和其他旧版本（`.build_faiss.py`、`.build_index_sectioned.py`、`.section_parser.py`）均已清理。

---

## 四、架构层面评估

### 做得好的方面
1. **知识库目录结构清晰**: `knowledge/<product>/` + `stores/<product>/` 分离原始资料和索引产物，扩展新产品只需加目录
2. **规则路由设计合理**: `QUESTION_ROUTES` + `SECTION_RULES` 让固定资料型问题有稳定、可控的回答
3. **分层部署灵活**: 基础版/API版/媒体版/后台版，按需叠加
4. **配置集中化**: `rag_runtime_config.py` 把路由、别名、权重等集中管理
5. **防伪回答逻辑完善**: `parse_anti_fake()` 的 STEP 解析、兜底默认值、FAQ 补缺都考虑周到

### 当前架构限制（已大幅改善）
1. ~~**subprocess 调用模式**~~: ✅ 已改为进程内直接调用（`import answer_question`）
2. ~~**无 score threshold 过滤**~~: ✅ 已通过 `QUESTION_TYPE_CONFIG` 的 `threshold` 按路由过滤低分结果
3. ~~**单进程单文件**~~: ✅ API 已直接返回字符串，无需 `answer.txt` 中间文件
4. **回归测试**: 171 个 test_core 用例，但 `regression_cases.json` 覆盖面仍可扩充

---

## 五、推荐的改进优先级

### 立即修复（影响正确性/可用性）— ✅ 全部完成
| # | 问题 | 状态 |
|---|------|------|
| 1 | API 并发竞争（answer.txt 覆盖） | ✅ 已改为进程内调用，无文件写入 |
| 2 | 接入 rag_logger 到 QA 流程 | ✅ log_qa() 覆盖所有路径 |
| 3 | 删除死配置文件，避免混淆 | ✅ 已清理 |
| 4 | 删除根目录空 index.faiss | ✅ 已删除 |

### 近期优化（提升回答质量）— ✅ 全部完成
| # | 问题 | 状态 |
|---|------|------|
| 5 | 规则提取失败时用检索结果做 fallback | ✅ _fallback_from_hits() |
| 6 | 采用 build_faiss_fixed 的切块策略 | ✅ 已合并为单一 build_faiss.py |
| 7 | 使用 QUESTION_TYPE_CONFIG 的 k/threshold | ✅ answer_one() 动态读取 |
| 8 | 改善中文关键词分词 | ✅ CJK bigram 切分 |

### 长远升级（架构演进）— 待规划
| # | 方向 | 说明 |
|---|------|------|
| 10 | 接入 LLM 做答案润色/整合 | 当前 OpenAI rewrite 默认关闭 |
| 11 | 多产品组合问答优化 | 拆分策略 + 产品消歧增强 |
| 12 | 时间轴/联合方案知识库 | `04_版本更新详细说明.md` 提到的下一步 |
| 13 | 扩充回归测试 | 覆盖所有 route 和边界情况 |

---

## 六、清理建议 — ✅ 全部完成

以下文件均已清理：

| 文件 | 状态 |
|------|------|
| `rag_config.py` / `.rag_config.py` | ✅ 已删除 |
| `.build_faiss.py` / `.build_index_sectioned.py` / `.section_parser.py` | ✅ 已删除 |
| `index.faiss`（根目录） | ✅ 已删除 |
| `PROJECT_STRUCTURE.md` / `PROJECT_STRUCTURE.txt` | ✅ 已删除 |
| `ARCHITECTURE.md` / `CHANGELOG.md` / `PROGRESS.md` | ✅ 已删除（由 `docs/` 下文档替代） |
