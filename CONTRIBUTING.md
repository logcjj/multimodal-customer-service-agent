# Contributing to Multimodal Customer Service Agent

感谢参与“基于自适应检索与自主学习的多模态客服智能体系统”。

## 开发流程

1. 从 `master` 创建短生命周期 topic branch。
2. 保持每个变更聚焦，并为行为变化补充测试。
3. 运行相关的后端和前端检查。
4. 提交 Pull Request，说明动机、实现、配置变化和验证结果。

## 环境准备

```bash
cp .env.example .env
cd backend && uv sync --dev
cd ../frontend && npm ci
```

仅使用本地或测试凭据。不要提交 `.env`、提供商密钥、数据库密码、回调 Secret、私有数据集或运行时输出。

## 必要检查

```bash
cd backend
uv run pytest -q
python -m compileall app

cd ../frontend
npm test -- --runInBand
npm run type-check
npm run build
```

## Pull Request 约定

- 说明用户可见行为和 API 契约变化。
- 在 `.env.example` 中以空值记录新增环境变量。
- 持久化结构变化必须附带迁移或升级说明。
- 生成数据、构建输出、缓存和本地模型资产不进入 Git。
- 索引、存储或部署行为变化要写明回滚步骤。
