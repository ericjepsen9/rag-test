# 知识库「审阅 → 反馈 → 修订」方案

## 目标

用户通过后台管理导入原始文档后，能够：
1. **完整审阅** LLM 整理的 main.txt / faq.txt / alias.txt
2. **下达修改指令**（如"FAQ 太少"、"缺少注射深度参数"、"第六章太笼统，需要更具体"）
3. **LLM 根据指令重新整理**，保留满意的部分，只修改不满意的部分
4. **可多轮迭代**，直到满意后再正式写入

## 设计

### 一、新增 API：`POST /admin/import_knowledge/refine`

**请求体：**
```json
{
  "type": "product",
  "id": "botox",
  "feedback": "1. FAQ太少，请增加到20个以上\n2. 第四章注射指南缺少具体深度参数\n3. alias缺少英文缩写",
  "current": {
    "main_txt": "（当前整理结果全文）",
    "faq_txt": "（当前FAQ全文）",
    "alias_txt": "（当前别名全文）"
  },
  "raw_text": "（可选，原始文档，供LLM参考补充细节）"
}
```

**响应：** 与 dry_run 相同格式，返回修订后的完整内容预览。

**LLM 修订 Prompt 设计：**
```
System: 你是医美行业知识库整理专家。用户对上一次的整理结果不满意，
请根据用户的修改意见进行修订。

规则：
1. 仅修改用户指出的问题部分，保留其他已满意的内容
2. 不要丢失原有的数字参数、具体数据
3. 输出完整的修订后 JSON（不是只输出修改部分）
4. 遵循与首次整理相同的格式规范

User:
【当前整理结果】
main_txt: ...
faq_txt: ...
alias_txt: ...

【用户修改意见】
...

【原始文档参考】（如有）
...
```

### 二、新增 API：`POST /admin/import_knowledge/commit`

用户在多轮修订满意后，将最终版本正式写入知识库文件并建索引。

**请求体：**
```json
{
  "type": "product",
  "id": "botox",
  "content": {
    "main_txt": "最终确认的main内容",
    "faq_txt": "最终确认的faq内容",
    "alias_txt": "最终确认的alias内容"
  },
  "build": true
}
```

### 三、前端 UI 改造（智能导入页面）

#### 流程变为：

```
原始文档 → [预览整理] → 审阅页面 → [下达修改指令] → LLM修订 → 审阅
                                         ↑_______________↓  (可多轮)
                                    满意后 → [正式导入]
```

#### 审阅页面设计（右侧结果区改造）：

1. **三个 Tab 全文查看**（main.txt / faq.txt / alias.txt）
   - 不再截断，显示完整内容
   - 每个 Tab 下方显示字数统计

2. **修改指令区域**
   - 一个文本框，用户写具体修改要求（支持多条，换行分隔）
   - 预置常用指令快捷按钮：
     - 「FAQ 数量不够」
     - 「缺少具体参数」
     - 「章节内容太笼统」
     - 「补充禁忌人群」
     - 「补充术后护理细节」

3. **操作按钮**
   - 「重新整理」→ 调用 /refine，LLM 根据指令修订
   - 「正式导入」→ 调用 /commit，将当前内容写入文件

4. **修订历史记录**
   - 页面内存保存每次修订的版本（用 JS 数组）
   - 可点击「上一版」「下一版」切换对比
   - 显示当前是第几次修订

5. **手动微调**
   - 每个 Tab 的内容区域可切换为「编辑模式」
   - 用户可以直接手动改几个字（小修补），不必再走 LLM

## 文件修改清单

| 文件 | 修改内容 |
|------|----------|
| `import_knowledge.py` | 新增 `refine_knowledge()` 函数，包含修订用 LLM prompt |
| `api_server.py` | 新增 `/admin/import_knowledge/refine` 和 `/commit` 接口 |
| `admin_page.html` | 改造智能导入页面右侧为审阅+反馈+修订 UI |

## 实现步骤

1. `import_knowledge.py` — 新增 `refine_knowledge(client, current, feedback, raw_text, entity_type)` 函数和修订 prompt
2. `api_server.py` — 新增 `/refine` 和 `/commit` 两个 API 接口
3. `admin_page.html` — 改造右侧结果区：全文查看、反馈输入、快捷指令、修订历史、手动编辑、正式导入
