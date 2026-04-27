# DataFountain 1165 多模态客服智能体

本项目用于参加 DataFountain 1165 客服智能体赛题。任务目标是基于商品说明书、售后政策和多模态图片信息，生成可提交评测的客服回答。当前方案以本地 RAG 检索与规则路由为主，优先保证答案相关性、格式正确性和图片引用一致性。

## 当前成绩

| 阶段 | 分数 | 说明 |
| --- | ---: | --- |
| 初始版本 | 0.33 左右 | 基础检索和模板回答，存在政策题误检索、英文手册解析错误、图片格式不一致等问题。 |
| 当前版本 | 0.45625000 | 已完成提交格式修复、英文汇总手册解析、售后题路由、tag 约束检索重排、图片 `<PIC>` 与图片数组一致性校验。 |
| 暂定目标 | 0.65 | 下一步重点提升检索精度、答案压缩质量和复杂多图说明题的覆盖率。 |

## 方案概览

系统将问题分为两类处理：

- **通用客服政策题**：退换货、退款、发票、物流、投诉、维修、临期商品、批量采购等问题走规则模板，避免误召回说明书内容。
- **产品说明书题**：根据问题识别目标产品和操作意图，使用多查询 BM25 检索、融合重排和证据压缩生成答案。

核心流程：

```text
用户问题
  -> 问题清洗与语言识别
  -> 政策题/说明书题路由
  -> 产品别名与意图扩展
  -> 多查询检索与重排
  -> 证据片段整理
  -> 答案与图片格式校验
  -> 生成 submission CSV
```

## 主要优化

### 1. 售后政策路由

早期版本中，部分售后题会被误判为说明书题，例如“保质期”“生产日期”“上门安装”“优惠券”等问题会检索到无关手册。当前版本补充了更细的政策模板，并对前置政策题强制走客服策略。

覆盖场景包括：

- 7 天无理由退换货、换尺寸、换款式；
- 退款到账、信用卡原路退回；
- 发票抬头、企业采购、发票重开；
- 包装破损、快递丢失、签收后发现损坏；
- 临期、过期、受潮、食品安全；
- 上门安装、上门检修、违规收费；
- 维修超时、质保期配件费、翻新机投诉。

### 2. 英文汇总手册解析

`汇总英文手册.txt` 实际由多段手册内容拼接而成。原始解析方式会把 Python/JSON 字面量残留一起索引，导致英文题大量误召回。当前版本按行解析多段列表字面量，将英文手册重新拆分为可检索片段。

已补充的英文产品别名包括：

- air fryer / airfryer / multi-use pressure cooker；
- coffee machine；
- earphones；
- eReader；
- fax；
- grill；
- landline；
- lawn mower；
- microwave；
- motherboard；
- vacuum；
- snowmobile；
- television；
- toothbrush。

### 3. 检索与重排

当前版本在 BM25 基础上增加了：

- 产品别名扩展；
- 操作意图和部件 tag 扩展，例如 charging、quick release、sealing ring、steering system、engine hood；
- 标题命中加权；
- 强意图词加权；
- 英文汇总手册的产品级过滤和错产品惩罚；
- 对 boat / jetski / camera / pressure cooker / microwave / air purifier 等高风险产品增加定向召回规则；
- 移除短图文题的泛化兜底，避免不同问题返回同一句“相关插图”式答案。

这可以减少“题目问步骤，答案命中安全声明”“题目问部件操作，答案命中产品泛化介绍”这类错误。

### 4. 提交格式与图片一致性

比赛要求 `ret` 字段采用如下形式：

```text
"答案正文<PIC>", ["image_id"]
```

当前生成器会统一做格式校验：

- 答案正文使用 JSON 字符串形式加引号；
- 图片 ID 只放在后面的数组中，不写入正文；
- 正文仍保留 `<PIC>` 占位；
- `<PIC>` 数量必须和图片数组数量一致；
- 图片数组中的 ID 必须能在 `手册/插图` 中找到对应文件；
- 手册中没有真实图片文件的 `<PIC>` 会被清理；
- 有真实图片文件的 `<PIC>` 会尽量补全图片数组，不再固定最多 4 张。

## 项目结构

```text
.
├── question_public.csv              # 公共测试问题
├── submission_example.csv           # 官方提交样例
├── requirements.txt                 # API 服务依赖
├── scripts/
│   ├── generate_submission.py       # 生成提交文件
│   └── inspect_retrieval.py         # 检查单题检索结果
├── src/df_kefu_baseline/
│   ├── answer.py                    # 回答引擎、检索后处理、答案生成
│   ├── data.py                      # 读写 CSV
│   ├── manuals.py                   # 手册解析与图片绑定
│   ├── policy.py                    # 通用客服政策回答
│   ├── query_planner.py             # 查询规划与产品识别
│   ├── retrieval.py                 # BM25 检索与融合
│   └── server.py                    # 可选 REST API 服务
├── submissions/
│   ├── submission_baseline.csv
│   ├── submission_rag_improved.csv
│   └── submission_optimized.csv     # 当前推荐提交文件
└── 手册/
    ├── *.txt                        # 产品说明书文本
    └── 插图/                        # 说明书图片资源
```

## 快速运行

macOS / Linux：

```bash
PYTHONPATH=src python3 scripts/generate_submission.py \
  --output submissions/submission_optimized.csv
```

Windows PowerShell：

```powershell
$env:PYTHONPATH = "src"
python .\scripts\generate_submission.py --output .\submissions\submission_optimized.csv
```

生成后建议检查：

```bash
PYTHONPATH=src python3 -m compileall -q src scripts
```

## 单题调试

按题号查看查询规划、召回证据和答案：

```bash
PYTHONPATH=src python3 scripts/inspect_retrieval.py --id 132
```

按自定义问题调试：

```bash
PYTHONPATH=src python3 scripts/inspect_retrieval.py \
  --question "如何给健身追踪器充电？"
```

## 可选 LLM 润色

默认流程不依赖大模型。若要接入 OpenAI 兼容接口，可设置环境变量：

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
PYTHONPATH=src python3 scripts/generate_submission.py --use-llm
```

注意：LLM 只应基于检索证据组织答案，不应编造手册中没有的参数、承诺或售后政策。

## API 服务

安装依赖：

```bash
pip install -r requirements.txt
```

启动服务：

```bash
PYTHONPATH=src uvicorn df_kefu_baseline.server:app --host 0.0.0.0 --port 8000
```

如评测接口需要鉴权：

```bash
export KAFU_API_TOKEN="评测方约定的 token"
```

## 下一步计划

暂定目标分数为 **0.65**。后续优先方向：

1. **建立本地抽样评测集**：按政策题、中文说明书题、英文说明书题、多图题拆分，避免只依赖线上反馈。
2. **提升复杂说明书题答案质量**：减少过长片段堆叠，按问题类型生成更紧凑的步骤式答案。
3. **加强图文对齐**：优先保留与问题相关的 `<PIC>`，减少无关图片拖累评分。
4. **细化英文产品分段**：将 `汇总英文手册` 进一步拆成产品级子文档，降低跨产品误召回。
5. **引入轻量 reranker 或 LLM judge**：用于离线筛查低相关答案，再进行针对性修正。

## 当前提交文件

推荐提交：

```text
submissions/submission_optimized.csv
```

该文件已按比赛提交格式生成，并经过基础一致性检查：

- 400 条答案完整；
- `ret` 字段为引号包裹答案；
- `<PIC>` 数量与图片数组数量一致；
- 图片数组 ID 均存在对应图片文件；
- 无 `相关插图:` / `Related images:` 等非比赛格式尾注；
- 当前线上反馈分数为 **0.45625000**。
