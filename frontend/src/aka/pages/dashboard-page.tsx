import { api } from '@/aka/api/client';
import type {
  AgentTrace,
  Dataset,
  RuntimeCapabilities,
  RuntimeReadiness,
} from '@/aka/api/types';
import { PageError, PageLoading } from '@/aka/components/page-state';
import { useApiResource } from '@/aka/hooks/use-api-resource';
import { getBrowserClientId } from '@/aka/lib/conversations';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Activity,
  ArrowRight,
  Bot,
  Cable,
  Database,
  FileText,
  Image,
  Layers3,
  MessageSquareText,
  Network,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  Wrench,
} from 'lucide-react';
import React, { useCallback, useState } from 'react';

interface DashboardData {
  readiness: RuntimeReadiness;
  capabilities: RuntimeCapabilities;
  datasets: Dataset[];
  traces: AgentTrace[];
}

const number = new Intl.NumberFormat('zh-CN');

function Metric({
  label,
  value,
  icon: Icon,
}: {
  label: string;
  value: number;
  icon: React.ComponentType<{ className?: string }>;
}) {
  return (
    <div className="flex items-center gap-3 rounded-md border border-border-button bg-bg-card px-4 py-3">
      <div className="grid size-9 shrink-0 place-items-center rounded-md bg-bg-component text-text-secondary">
        <Icon className="size-4" />
      </div>
      <div className="min-w-0">
        <div className="text-xl font-semibold tabular-nums">
          {number.format(value)}
        </div>
        <div className="truncate text-xs text-text-secondary">{label}</div>
      </div>
    </div>
  );
}

function ModelRole({
  label,
  ready,
  detail,
}: {
  label: string;
  ready: boolean;
  detail: string;
}) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-border-button py-3 last:border-b-0">
      <div className="min-w-0">
        <div className="text-sm font-medium">{label}</div>
        <div className="truncate text-xs text-text-secondary">{detail}</div>
      </div>
      <Badge variant={ready ? 'success' : 'destructive'}>
        {ready ? 'Ready' : 'Degraded'}
      </Badge>
    </div>
  );
}

export default function DashboardPage() {
  const [clientId] = useState(() =>
    getBrowserClientId(
      typeof window === 'undefined' ? null : window.localStorage,
    ),
  );
  const load = useCallback(async (): Promise<DashboardData> => {
    if (!clientId) throw new Error('无法读取浏览器匿名用户标识');
    const [readiness, capabilities, datasets, traces] = await Promise.all([
      api.readiness(),
      api.capabilities(),
      api.datasets(),
      api.traces(clientId),
    ]);
    return { readiness, capabilities, datasets, traces };
  }, [clientId]);

  const { data, loading, error } = useApiResource<DashboardData | null>(
    load,
    null,
  );

  if (loading || !data) {
    if (error) return <PageError message={error} />;
    return (
      <div className="size-full overflow-auto p-5 lg:p-8">
        <PageLoading label="加载工作台" />
      </div>
    );
  }

  const { readiness, capabilities, datasets, traces } = data;
  const totals = datasets.reduce(
    (current, dataset) => ({
      documents: current.documents + dataset.document_count,
      parents: current.parents + dataset.parent_count,
      children: current.children + dataset.child_count,
      assets: current.assets + dataset.asset_count,
    }),
    { documents: 0, parents: 0, children: 0, assets: 0 },
  );
  const retrievalMode =
    readiness.embedding_configured && readiness.rerank_configured
      ? 'hybrid + rerank'
      : 'lexical-only';

  return (
    <div className="size-full overflow-auto px-5 py-6 lg:px-8 lg:py-8">
      <div className="mx-auto max-w-[1500px] space-y-6">
        <header className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="mb-1 flex items-center gap-2 text-xs text-text-secondary">
              <span className="size-2 rounded-full bg-state-success" />
              Agent runtime online
            </div>
            <h1 className="text-2xl font-semibold">工作台</h1>
            <p className="mt-1 text-sm text-text-secondary">
              RAG、模型、Agent 与知识资产的实时运行视图
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button asLink variant="outline" to="/datasets">
              <Database className="size-4" />
              管理知识库
            </Button>
            <Button asLink to="/chat">
              <MessageSquareText className="size-4" />
              发起对话
            </Button>
          </div>
        </header>

        <section
          className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4"
          aria-label="知识资产指标"
        >
          <Metric label="Documents" value={totals.documents} icon={FileText} />
          <Metric
            label="Parent Sections"
            value={totals.parents}
            icon={Layers3}
          />
          <Metric label="Child Chunks" value={totals.children} icon={Search} />
          <Metric label="Image Assets" value={totals.assets} icon={Image} />
        </section>

        <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
          <Card>
            <CardHeader className="flex-row items-start justify-between space-y-0 p-5">
              <div>
                <CardTitle as="h2" className="text-base">
                  模型与检索运行状态
                </CardTitle>
                <CardDescription>
                  只展示实际可调用能力，失败角色自动降级
                </CardDescription>
              </div>
              <Badge
                variant={
                  retrievalMode === 'lexical-only' ? 'destructive' : 'success'
                }
              >
                {retrievalMode}
              </Badge>
            </CardHeader>
            <CardContent className="grid gap-x-6 p-5 pt-0 md:grid-cols-2">
              <ModelRole
                label="LLM"
                ready={readiness.llm_configured}
                detail={readiness.llm_model ?? '未配置'}
              />
              <ModelRole
                label="Embedding"
                ready={readiness.embedding_configured}
                detail="Dense retrieval"
              />
              <ModelRole
                label="Rerank"
                ready={readiness.rerank_configured}
                detail="Child reranking"
              />
              <ModelRole
                label="Vision / OCR"
                ready={readiness.vlm_configured || readiness.ocr_configured}
                detail="Multimodal perception"
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="p-5">
              <CardTitle as="h2" className="text-base">
                系统能力
              </CardTitle>
              <CardDescription>后端注册表的真实数量</CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-3 gap-2 p-5 pt-0">
              {[
                {
                  icon: Bot,
                  value: `${capabilities.agent_count} Agents`,
                  label: '动态协作',
                },
                {
                  icon: Sparkles,
                  value: `${capabilities.skill_count} Skills`,
                  label: '技能注册',
                },
                {
                  icon: Network,
                  value: `${capabilities.tool_count} Tools`,
                  label: '工具契约',
                },
              ].map(({ icon: Icon, value, label }) => (
                <div
                  key={value}
                  className="rounded-md border border-border-button bg-bg-card p-3"
                >
                  <Icon className="mb-4 size-4 text-text-secondary" />
                  <div className="whitespace-nowrap text-sm font-medium">
                    {value}
                  </div>
                  <div className="mt-1 text-[11px] text-text-secondary">
                    {label}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
          <Card>
            <CardHeader className="flex-row items-center justify-between space-y-0 p-5">
              <div>
                <CardTitle as="h2" className="text-base">
                  知识库
                </CardTitle>
                <CardDescription>已发布 Dataset 与索引资产</CardDescription>
              </div>
              <Button asLink to="/datasets" variant="ghost" size="sm">
                查看全部 <ArrowRight className="size-3.5" />
              </Button>
            </CardHeader>
            <CardContent className="space-y-2 p-5 pt-0">
              {datasets.slice(0, 4).map((dataset) => (
                <Button
                  key={dataset.id}
                  asLink
                  variant="ghost"
                  size="auto"
                  to={`/dataset/${dataset.id}`}
                  className="h-auto w-full justify-between rounded-md border border-border-button px-3 py-3 text-left"
                >
                  <span className="min-w-0">
                    <span className="block truncate text-sm font-medium">
                      {dataset.name}
                    </span>
                    <span className="mt-1 block truncate text-xs text-text-secondary">
                      {dataset.document_count} 文档 ·{' '}
                      {number.format(dataset.child_count)} Chunks ·{' '}
                      {dataset.published_version ?? '未发布'}
                    </span>
                  </span>
                  <ArrowRight className="size-4 shrink-0" />
                </Button>
              ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="p-5">
              <CardTitle as="h2" className="text-base">
                最近运行
              </CardTitle>
              <CardDescription>Agent Trace 与质量验证</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 p-5 pt-0">
              {traces.length === 0 ? (
                <div className="grid min-h-36 place-items-center rounded-md border border-dashed border-border-button text-center text-sm text-text-secondary">
                  <div>
                    <Activity className="mx-auto mb-2 size-5" />
                    暂无运行记录
                  </div>
                </div>
              ) : (
                traces.slice(0, 4).map((trace) => (
                  <div
                    key={trace.request_id}
                    className="flex items-center gap-3 rounded-md border border-border-button p-3"
                  >
                    <ShieldCheck className="size-4 text-state-success" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium">
                        {trace.route}
                      </div>
                      <div className="text-xs text-text-secondary">
                        {trace.selected_agents.length} Agents ·{' '}
                        {trace.total_latency_ms} ms
                      </div>
                    </div>
                  </div>
                ))
              )}
            </CardContent>
          </Card>
        </div>

        <section aria-labelledby="operations-title">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <h2 id="operations-title" className="text-sm font-semibold">
                系统运维
              </h2>
              <p className="mt-1 text-xs text-text-secondary">
                检查技能、工具协议、运行轨迹与模型角色
              </p>
            </div>
          </div>
          <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
            {[
              { to: '/skills', label: 'Skills 注册表', icon: Sparkles },
              { to: '/mcp', label: 'MCP 调试台', icon: Cable },
              { to: '/traces', label: 'Agent Traces', icon: Wrench },
              { to: '/settings/models', label: '模型配置', icon: Settings },
            ].map(({ to, label, icon: Icon }) => (
              <Button
                key={to}
                asLink
                to={to}
                variant="outline"
                className="h-12 justify-between px-4"
              >
                <span className="inline-flex items-center gap-2">
                  <Icon className="size-4 text-text-secondary" />
                  {label}
                </span>
                <ArrowRight className="size-3.5 text-text-secondary" />
              </Button>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
