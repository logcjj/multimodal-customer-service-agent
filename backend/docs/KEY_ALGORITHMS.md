# 关键算法说明

## 1. 动态意图路由

入口位于 `backend/app/runtime/dynamic_routing.py`。系统先用确定性规则识别产品事实、故障、安全、维修、售后和通用请求，再在边界不明确时调用 LLM 分类。对于产品、故障、安全和售后关键词形成强约束的问题，模型不能随意改写路由结果。

路由输出为 `RoutingIntent` 和 `RoutingDecision`，包含初始路线、最终路线、风险等级、是否需要知识检查、知识覆盖状态和澄清问题。

常见最终路线：

- `technical_knowledge`：进入说明书或产品知识库。
- `customer_service`：进入售后客服链路。
- `mixed`：技术事实与客服诉求合并处理。
- `evidence_clarification`：证据不足，先问清产品、型号或故障现象。
- `general_llm`：允许通用模型回答。
- `safe_handoff`：高风险或不适合自动回答，建议人工处理。

## 2. 知识覆盖门禁

`KnowledgeCoverageGate` 会在回答前判断证据是否足够。它综合考虑问题风险、目标产品、检索证据数量、缺失字段和通用回答安全边界。

该门禁解决两个问题：

- 证据不足时不强行生成说明书结论。
- 通用模型不能回答产品规格、维修、安全或售后承诺类事实。

当证据不足但可以补充时，系统返回澄清问题；当请求高风险且无法确认依据时，系统进入安全交接。

## 3. Parent/Child 混合检索

核心实现位于 `backend/app/knowledge/hybrid.py`。系统将文档拆成 Parent Chunk 与 Child Chunk：

- Child Chunk：用于精细召回，保留标题、正文、关键词、页码、产品和图片资产。
- Parent Chunk：用于生成回答时提供更完整上下文，避免步骤被切断。

在线检索流程：

1. 对查询做中文分词、英文 token、型号、数字和错误码归一化。
2. 使用 BM25 进行词法召回。
3. 如已配置 Embedding，则生成查询向量并执行向量召回。
4. 使用 RRF 融合词法和向量排名。
5. 如已配置 Rerank，则对候选文本做精排。
6. 将命中的 Child Chunk 聚合回 Parent Chunk。
7. 输出 `Evidence`，包含证据 ID、文档名、章节、页码、资产、得分拆解和置信度。

该设计兼顾召回精度和回答上下文完整性。

## 4. 图片 Chunk 检索

图片知识位于 `backend/app/knowledge/image_retrieval.py`。导入图片洞察时，系统会通过 `sha256`、文件名 stem 或图片 ID 匹配知识库资产，并把视觉摘要、可见文字、检索文本、适用问题、异常信号和相关 Parent/Child ID 规整为 Image Chunk。

图片 Chunk 可与文本证据共同参与检索。回答侧返回的是 `assets` 数组，前端通过 `/api/assets/{asset_id}` 获取图片文件并展示。

## 5. 多模态用户图片理解

用户上传图片由 `backend/app/agents/multimodal.py` 和 `backend/app/multimodal/visual_context.py` 处理。系统会把视觉模型输出规整为 `VisualContext`：

- `ocr_text`：图片可见文字。
- `detected_codes`：错误码、型号或序列标识。
- `detected_product`：疑似产品。
- `detected_components`：可见部件。
- `visible_objects`：可见对象。
- `visual_summary`：图片摘要。
- `provider_status`：VLM/OCR 调用状态。
- `confidence`：视觉上下文置信度。

视觉上下文会参与问题路由、查询改写、检索排序和回答约束。没有 VLM 时，系统会降级，不把图片内容写成确定事实。

## 6. 多智能体编排

主编排器位于 `backend/app/runtime/orchestrator.py`。一次请求中可能调用：

- `router`：判断问题类型和风险。
- `multimodal`：处理用户图片。
- `knowledge`：检索并生成技术证据回答。
- `customer_service`：处理售后诉求。
- `general`：处理安全边界内的通用问题。
- `evidence_gap`：生成澄清问题。
- `verifier`：核验最终结论。
- `memory_curator`：异步更新会话摘要和长期上下文。

编排器会把每个步骤写入 `trace`，流式接口也会输出相同事件，便于前端展示和运行问题定位。

## 7. Verifier 幻觉抑制

`backend/app/agents/verifier.py` 对回答做后置核验。重点检查：

- 事实性 claim 是否绑定有效 evidence ID。
- 型号、错误码、数字、单位是否能在证据中找到。
- 回答是否与“禁止、严禁、不得、切勿”等安全警告冲突。
- 操作步骤是否改变了证据中的安全顺序。
- 图片观察是否被表述为确定事实。
- 是否无依据承诺免费维修、全额退款、无条件退货、赔付等售后结论。

核验失败时，系统会选择修订、澄清、兜底或人工交接，而不是直接返回不可靠结论。

## 8. 会话记忆

会话能力分为两层：

- `SessionMemoryStore`：保存短期会话状态、产品、意图、证据引用、视觉上下文和缺失信息。
- `ConversationMemoryService`：在传入 `user_id` 时保存完整会话账本，维护滚动摘要、结构化上下文和待澄清状态。

这使系统能够处理“那还能继续用吗”“上一个型号也是这样吗”一类依赖前文的追问。记忆只作为上下文和检索线索，不会修改模型权重。

## 9. 离线索引与向量图

`backend/scripts/build_index_bundle.py` 为已发布知识库生成标准离线索引清单和可预加载数据。`backend/scripts/build_vector_map.py` 使用已有 Embedding 生成二维向量图，帮助前端展示知识分布和检索命中位置。

索引构建结果可通过 `/api/index-runtime` 和 `/api/datasets/{dataset_id}/index-manifest` 检查。
