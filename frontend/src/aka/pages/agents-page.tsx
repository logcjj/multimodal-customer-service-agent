import { api } from '@/aka/api/client';
import type { AgentDefinition } from '@/aka/api/types';
import { PageError, PageLoading } from '@/aka/components/page-state';
import { useApiResource } from '@/aka/hooks/use-api-resource';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Bot, Boxes, Cable, ShieldCheck } from 'lucide-react';
import { useCallback } from 'react';

export default function AgentsPage() {
  const loader = useCallback(() => api.agents(), []);
  const {
    data: agents,
    loading,
    error,
  } = useApiResource<AgentDefinition[]>(loader, []);
  const online = agents.filter((item) =>
    item.execution_mode.startsWith('online'),
  ).length;
  const offline = agents.filter(
    (item) => item.execution_mode === 'offline',
  ).length;

  return (
    <div className="size-full overflow-auto px-5 py-6 lg:px-8 lg:py-8">
      <div className="mx-auto max-w-[1400px] space-y-5">
        <header className="flex items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold">Agent 团队</h1>
            <p className="mt-1 text-sm text-text-secondary">
              {online} 在线 · {offline} 离线 · 按请求动态组队
            </p>
          </div>
          <Badge variant="success">Orchestrator Ready</Badge>
        </header>
        {loading ? <PageLoading label="加载 Agent 定义" /> : null}
        {!loading && error ? <PageError message={error} /> : null}
        {!loading && !error ? (
          <div
            className="grid gap-4 md:grid-cols-2 xl:grid-cols-3"
            aria-label="Agent 拓扑"
          >
            {agents.map((agent, index) => (
              <Card key={agent.id} className="min-h-64">
                <CardHeader className="p-5">
                  <div className="mb-3 flex items-start justify-between gap-3">
                    <div className="grid size-10 place-items-center rounded-md bg-bg-component">
                      <Bot className="size-5" />
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs tabular-nums text-text-secondary">
                        {String(index + 1).padStart(2, '0')}
                      </span>
                      <Badge
                        variant={
                          agent.status === 'ready' ? 'success' : 'secondary'
                        }
                      >
                        {agent.status}
                      </Badge>
                    </div>
                  </div>
                  <CardTitle as="h2" className="text-base">
                    {agent.name}
                  </CardTitle>
                  <div className="text-xs text-accent-primary">
                    {agent.short_name}
                  </div>
                </CardHeader>
                <CardContent className="p-5 pt-0">
                  <p className="min-h-12 text-sm leading-6 text-text-secondary">
                    {agent.description}
                  </p>
                  <dl className="mt-5 grid grid-cols-3 gap-2">
                    <div className="rounded border border-border-button bg-bg-card p-2">
                      <dt className="flex items-center gap-1 text-[10px] text-text-secondary">
                        <Boxes className="size-3" />
                        Skills
                      </dt>
                      <dd className="mt-2 text-sm font-semibold">
                        {agent.skills.length}
                      </dd>
                    </div>
                    <div className="rounded border border-border-button bg-bg-card p-2">
                      <dt className="flex items-center gap-1 text-[10px] text-text-secondary">
                        <Cable className="size-3" />
                        Tools
                      </dt>
                      <dd className="mt-2 text-sm font-semibold">
                        {agent.tools.length}
                      </dd>
                    </div>
                    <div className="rounded border border-border-button bg-bg-card p-2">
                      <dt className="flex items-center gap-1 text-[10px] text-text-secondary">
                        <ShieldCheck className="size-3" />
                        Mode
                      </dt>
                      <dd className="mt-2 truncate text-xs font-medium">
                        {agent.execution_mode}
                      </dd>
                    </div>
                  </dl>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
