import { api } from '@/aka/api/client';
import type { AgentTrace, RuntimeReadiness } from '@/aka/api/types';
import { AgentRunInspector } from '@/aka/components/agent-run-inspector';
import { PageError } from '@/aka/components/page-state';
import { useApiResource } from '@/aka/hooks/use-api-resource';
import { getBrowserClientId } from '@/aka/lib/conversations';
import { Badge } from '@/components/ui/badge';
import { Activity, ChevronRight, Clock3, Route } from 'lucide-react';
import { useCallback, useState } from 'react';

export default function TracesPage() {
  const [clientId] = useState(() =>
    getBrowserClientId(
      typeof window === 'undefined' ? null : window.localStorage,
    ),
  );
  const loader = useCallback(async () => {
    if (!clientId) throw new Error('无法读取浏览器匿名用户标识');
    const [traces, readiness] = await Promise.all([
      api.traces(clientId),
      api.readiness(),
    ]);
    return { traces, readiness };
  }, [clientId]);
  const { data, loading, error } = useApiResource<{
    traces: AgentTrace[];
    readiness: RuntimeReadiness;
  }>(loader, {
    traces: [],
    readiness: {
      status: 'unknown',
      rollout_mode: 'unknown',
      legacy_available: false,
      model_registry: 'unknown',
      trace_store: 'unknown',
      llm_configured: false,
      llm_model: null,
      vlm_configured: false,
      embedding_configured: false,
      rerank_configured: false,
      ocr_configured: false,
      dynamic_routing: 'off',
      conversation_history: 'off',
      layered_memory: 'off',
      general_agent: 'off',
    },
  });
  const { traces, readiness } = data;
  const [selected, setSelected] = useState<AgentTrace | null>(null);
  const active = selected ?? traces[0] ?? null;

  return (
    <div className="grid size-full min-w-0 grid-cols-1 xl:grid-cols-[minmax(0,1fr)_380px]">
      <section className="min-h-0 overflow-auto px-5 py-6 lg:px-8 lg:py-8">
        <div className="mx-auto max-w-5xl space-y-5">
          <header className="flex items-end justify-between gap-4">
            <div>
              <h1 className="text-2xl font-semibold">运行追踪</h1>
              <p className="mt-1 text-sm text-text-secondary">
                {traces.length} 次 Agent 运行
              </p>
            </div>
            <Badge variant="success">Trace 持久化</Badge>
          </header>
          {error ? <PageError message={error} /> : null}
          {loading ? (
            <div className="p-8 text-center text-sm text-text-secondary">
              加载运行记录
            </div>
          ) : null}
          {!loading && !error ? (
            <div className="overflow-hidden rounded-md border border-border-button">
              <div className="hidden grid-cols-[minmax(200px,1fr)_130px_120px_120px_24px] gap-4 border-b border-border-button bg-bg-card px-4 py-3 text-xs text-text-secondary md:grid">
                <span>Request</span>
                <span>Route</span>
                <span>Team</span>
                <span>Latency</span>
                <span />
              </div>
              {traces.map((trace) => (
                <button
                  key={trace.request_id}
                  type="button"
                  onClick={() => setSelected(trace)}
                  className={`grid w-full gap-3 border-b border-border-button px-4 py-4 text-left last:border-b-0 md:grid-cols-[minmax(200px,1fr)_130px_120px_120px_24px] md:items-center md:gap-4 ${active?.request_id === trace.request_id ? 'bg-accent-primary/5' : 'hover:bg-bg-card'}`}
                >
                  <span className="flex min-w-0 items-center gap-2">
                    <Activity className="size-4 shrink-0" />
                    <code className="truncate text-xs">{trace.request_id}</code>
                  </span>
                  <span className="flex items-center gap-1 text-xs">
                    <Route className="size-3.5" />
                    {trace.route}
                  </span>
                  <span className="text-xs">
                    {trace.selected_agents.length} Agents
                  </span>
                  <span className="flex items-center gap-1 text-xs tabular-nums">
                    <Clock3 className="size-3.5" />
                    {trace.total_latency_ms} ms
                  </span>
                  <ChevronRight className="size-4 text-text-secondary" />
                </button>
              ))}
              {!traces.length ? (
                <div className="p-12 text-center text-sm text-text-secondary">
                  暂无运行记录
                </div>
              ) : null}
            </div>
          ) : null}
        </div>
      </section>
      <aside className="hidden min-h-0 border-l border-border-button bg-bg-card xl:block">
        <AgentRunInspector
          trace={active}
          events={[]}
          running={false}
          readiness={readiness}
        />
      </aside>
    </div>
  );
}
