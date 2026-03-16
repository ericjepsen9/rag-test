# RAG 检索策略分析报告

## 1. 系统架构概览

本系统采用 **混合检索 (Hybrid Search)** 方案，包含三层回答策略：

```
用户问题
  │
  ├─ 查询改写 (query_rewrite.py)
  │    ├─ 指代消解 / 上下文补全
  │    ├─ 同义词扩展 (expand_synonyms)
  │    └─ 生成 search_query + expanded 两个版本
  │
  ├─ 路由检测 (detect_route)
  │    └─ 关键词匹配 + 消歧加分 → 选择最佳路由
  │
  ├─ 双路检索 (并行 ThreadPoolExecutor)
  │    ├─ 向量检索: FAISS IndexFlatIP + BGE-M3
  │    └─ 关键词检索: 自实现 BM25
  │
  ├─ 混合融合 (merge_hybrid)
  │    ├─ 加权合并 (默认 vw=0.65, kw=0.35)
  │    └─ 路由感知 boost
  │
  └─ 回答生成
       ├─ 策略0: FAQ 精确匹配快速路径
       ├─ 策略1: LLM RAG (检索 + LLM 生成)
       └─ 策略2: 规则章节提取 (Fallback)
```

## 2. 两种核心检索策略详解

### 2.1 向量检索 (FAISS + BGE-M3)

**实现位置**: `rag_answer.py:vector_search()`, `build_faiss.py:embed_texts()`

- **嵌入模型**: BAAI/bge-m3, FP16, max_length=1024 (查询) / 8192 (构建)
- **索引类型**: `IndexFlatIP` (内积 = 余弦相似度，因 L2 归一化)
- **查询流程**: 对 `search_query`（去噪版本）做嵌入 → FAISS 检索 top-12
- **得分范围**: [0, 1] (L2 归一化后余弦相似度)

**优势**:
- 语义匹配能力强，能捕捉"安全吗" → "不良反应"等语义关联
- BGE-M3 对中文支持优秀，多语言多粒度

**劣势**:
- 对精确术语（"0.8mm"、"HiddenTag"）匹配不敏感
- 需要 GPU/较多内存加载模型
- IndexFlatIP 为暴力搜索，大规模数据下 O(n) 复杂度

### 2.2 关键词检索 (BM25)

**实现位置**: `search_utils.py:keyword_search()`, `bm25_score()`

- **分词策略**: 标点分割 + 纯中文长词 bigram 切分 (`_extract_terms`)
- **同义词扩展**: 60+ 条医美术语同义词映射 (`_SYNONYM_MAP`)
- **时间模式扩展**: "术后第N天" → 自动追加"恢复"、"消退"等词
- **BM25 参数**: k1=1.5, b=0.75 (经典设置)
- **得分归一化**: sigmoid(raw_score / 5.0)，映射到 (0, 1) 区间
- **缓存优化**: 语料级缓存 + 文档频率缓存 + sigmoid LRU 缓存

**优势**:
- 精确术语匹配能力强
- 计算轻量（纯 CPU）
- 同义词扩展提升中文医美术语召回

**劣势**:
- 无语义理解能力
- 中文分词粗粒度（字符级 `text.count(term)` 而非真正分词）

## 3. 混合融合机制

### 3.1 加权合并 (`merge_hybrid`)

```python
hybrid_score = vector_score * vw + keyword_score * kw
```

- 默认权重: 向量 0.65 + 关键词 0.35
- 按路由动态调整（见下表）
- 同一文档在双路中出现时取最高分（非累加），避免重复文档分数膨胀

### 3.2 路由感知权重

| 路由 | 向量权重 (vw) | 关键词权重 (kw) | 设计理由 |
|------|:---:|:---:|------|
| 默认 | 0.65 | 0.35 | 语义检索为主 |
| operation | 0.55 | 0.45 | 操作参数需精确匹配 |
| anti_fake | 0.45 | 0.55 | 防伪步骤关键词密集 |
| anatomy_q | 0.50 | 0.50 | 部位名需精确+语义 |
| procedure_q | 0.50 | 0.50 | 项目名需精确+语义 |
| equipment_q | 0.50 | 0.50 | 设备名需精确+语义 |
| indication_q | 0.55 | 0.45 | 症状描述偏语义 |
| complication | 0.60 | 0.40 | 并发症偏语义描述 |

### 3.3 路由 Boost

匹配目标章节标题的 chunk 额外加 0.12 分 (`_apply_route_boost`)，基于 `_ROUTE_SECTION_MARKERS` 中的关键词匹配。

## 4. 辅助检索机制

### 4.1 查询改写 (`query_rewrite.py`)

- 生成两个查询版本：
  - `search_query`: 用于向量检索（去除"请问"等噪音前缀，语义更聚焦）
  - `expanded`: 用于关键词检索（包含同义词扩展）
- 指代消解：多轮对话中"它"→具体产品名
- 上下文补全：追问场景中补充产品/路由信息

### 4.2 多问题拆分 (`split_multi_question`)

- 自动拆分复合问题："A和B分别是什么" → ["A是什么", "B是什么"]
- 支持顿号、逗号、问号等多种分隔符
- 最多拆分 4 个子问题 (`MAX_SUB_QUESTIONS`)

### 4.3 共享知识库双库检索

- 产品级知识库：每个产品独立索引（如 `stores/feiluoao/`）
- 共享知识库：跨产品通用知识（`stores/_shared/`）
- 路由决定搜索范围：
  - `_SHARED_ROUTES`: 仅搜共享库（如 `script`、`procedure_q`）
  - `_HYBRID_ENTITY_ROUTES`: 同时搜产品库+共享库（如 `complication`、`course`）

## 5. 性能瓶颈分析：为什么 "水光" 查询耗时 40000ms

### 5.1 完整调用链路与耗时

```
POST /ask {question: "水光"}
  │
  ├── rewrite_query()                         ~10ms
  │
  └── answer_question() → answer_one()
      │
      ├── detect_route("水光") → procedure_q    ~1ms
      │   "水光" 在 procedure_q 关键词列表中
      │   procedure_q ∈ _HYBRID_ENTITY_ROUTES → 双库搜索
      │
      ├── ★ 瓶颈1: embed_query()               2-5s (CPU)
      │   BGE-M3 编码查询向量
      │
      ├── ★ 瓶颈2: ThreadPoolExecutor 并行搜索   5-15s
      │   ├── vector_search(product)    → embed_query + FAISS
      │   ├── keyword_search(product)   → BM25
      │   ├── vector_search(_shared)    → embed_query + FAISS
      │   └── keyword_search(_shared)   → BM25
      │   _search_lock 全局互斥锁导致 FAISS 搜索串行化！
      │
      ├── ★ 瓶颈3: rerank_hits()                5-15s (CPU)
      │   BGE-M3 compute_score 对 20 个 sentence_pairs
      │   做 colbert + sparse + dense 融合评分
      │   这是最大的性能杀手！
      │
      └── ★ 瓶颈4: llm_generate_answer()        3-10s
          外部 LLM API 调用
```

### 5.2 五大性能瓶颈详解

#### 瓶颈1: BGE-M3 向量编码 (embed_query)

**位置**: `rag_answer.py:178-194`

每次 `vector_search` 调用 `embed_query` 编码查询文本。"水光"触发双库搜索，`embed_query` 被调用 2 次，每次 CPU 上耗时 2-5 秒。

**建议**: 缓存相同查询的 embedding 结果，避免重复编码。

#### 瓶颈2: FAISS 搜索全局互斥锁

**位置**: `rag_answer.py:350-351`

`_search_lock` 是全局 `threading.Lock()`，所有 FAISS 搜索被串行化。即使 ThreadPoolExecutor 并行提交，实际 FAISS index.search 仍串行。

**建议**: FAISS 的只读搜索是线程安全的，可移除全局锁或改为每索引独立锁。

#### 瓶颈3: Rerank（最大瓶颈）

**位置**: `search_utils.py:626-676`

BGE-M3 `compute_score` 对 `RERANK_TOP_N=20` 个 sentence_pairs 做 colbert+sparse+dense 三路融合评分。CPU 上 20 对文本可能耗时 5-15 秒。

**建议**:
1. 设置 `RAG_RERANK_ENABLED=0` 关闭（立即见效）
2. 降低 `RAG_RERANK_TOP_N` 到 10 或 5
3. 使用 GPU 加速

#### 瓶颈4: LLM API 调用

**位置**: `rag_answer.py:1317-1326`

外部 API 网络延迟 + 模型推理延迟。如果走 fallback 路径，可能额外调用 `openai_rewrite_answer()`，一次请求最多 2 次 LLM 调用。

#### 瓶颈5: 共享知识库首次构建（一次性）

**位置**: `rag_answer.py:233-254`

`_shared` 索引不存在时自动构建，涉及 BGE-M3 编码所有共享知识 chunks，首次可能耗时 30s+。

### 5.3 为什么 "水光" 特别慢

1. **双库搜索**: procedure_q ∈ _HYBRID_ENTITY_ROUTES → 同时搜产品库 + 共享库（4 个任务 vs 普通 2 个）
2. **更多候选**: VECTOR_TOP_K=12 × 2 + KEYWORD_TOP_K=12 × 2 = 48 个初始候选
3. **更重 Rerank**: 20 个 sentence_pairs 的 compute_score
4. **更长 context**: 更多 hits → 传给 LLM 的 context 更长 → 生成更慢

### 5.4 优化建议（按效果排序）

| 优先级 | 优化措施 | 预估提速 | 实施方式 |
|--------|---------|---------|---------|
| P0 | 关闭 Rerank (`RAG_RERANK_ENABLED=0`) | -5~15s | 环境变量 |
| P0 | 使用 GPU 运行 BGE-M3 | -10~25s | 需 CUDA |
| P1 | 缓存 embed_query 结果（LRU） | -2~5s | 改代码 |
| P1 | 移除/细化 `_search_lock` | -1~3s | 改代码 |
| P2 | 降低 VECTOR_TOP_K/KEYWORD_TOP_K 到 8 | -1~2s | 环境变量 |
| P2 | 降低 RERANK_TOP_N 从 20 到 10 | -3~8s | 环境变量 |
| P3 | LLM 响应缓存（相似问题复用） | -3~10s | 改代码 |

### 5.5 快速验证方法

通过环境变量即可快速验证（无需改代码）：

```bash
# 关闭 Rerank（预计减少 5-15s）
export RAG_RERANK_ENABLED=0

# 减少检索候选数（预计减少 1-3s）
export RAG_VECTOR_TOP_K=8
export RAG_KEYWORD_TOP_K=8

# 或通过管理 API 热更新
curl -X POST http://localhost:8000/admin/config \
  -H "Content-Type: application/json" \
  -d '{"updates": {"rerank_enabled": false, "vector_top_k": 8, "keyword_top_k": 8}}'
```

## 6. 结论

对于当前医美垂直领域场景（中文短文本、术语密集、知识库规模约数百 chunk），**混合检索 + 路由感知方案是合理且高效的选择**。

40s 延迟的主要原因是 **Rerank（~10s）+ BGE-M3 编码（~5s）+ LLM API 调用（~5s）+ 双库搜索开销（~5s）** 的叠加效应。最快的优化方式是关闭 Rerank 和使用 GPU。
