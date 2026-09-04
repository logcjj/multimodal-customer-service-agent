import type {
  AgentTrace,
  RuntimeEvent,
  RuntimeReadiness,
} from '@/aka/api/types';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import {
  Bot,
  BrainCircuit,
  Check,
  CircleHelp,
  Clock3,
  Database,
  Eye,
  GitBranch,
  Headphones,
  MessageCircle,
  RotateCcw,
  Search,
  ShieldCheck,
} from 'lucide-react';

const agents = {
  orchestrator: { name: '主控智能体', icon: BrainCircuit },
  router: { name: '意图路由智能体', icon: GitBranch },
  'evidence-gap': { name: '证据补全智能体', icon: CircleHelp },
  general: { name: '通用对话智能体', icon: MessageCircle },
  'memory-curator': { name: '会话记忆整理智能体', icon: Database },
  multimodal: { name: '多模态感知智能体', icon: Eye },
  knowledge: { name: '知识检索智能体', icon: Search },
  'customer-service': { name: '客服业务智能体', icon: Headphones },
  verifier: { name: '质量监督智能体', icon: ShieldCheck },
  'legacy-champion': { name: '守护链路', icon: RotateCcw },
} as const;

type AgentId = keyof typeof agents;

function statusLabel(status: string, selected: boolean) {
  if (!selected) return '按需待命';
  if (status === 'running') return '执行中';
  if (status === 'completed') return '已完成';
  if (status === 'failed') return '失败';
  if (status === 'skipped') return '已跳过';
  return '已入队';
}

function AgentRow({
  id,
  selected,
  status,
}: {
  id: AgentId;
  selected: boolean;
  status: string;
}) {
  const meta = agents[id];
  const Icon = meta.icon;
  return (
    <div
      className={cn(
        'flex items-center gap-3 rounded-md border px-3 py-2.5',
        selected
          ? 'border-border-accent bg-accent-primary/5'
          : 'border-border-button bg-bg-card opacity-60',
      )}
    >
      <div className="grid size-8 shrink-0 place-items-center rounded bg-bg-component">
        <Icon className="size-4" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-medium">{meta.name}</div>
        <div className="text-xs text-text-secondary">
          {statusLabel(status, selected)}
        </div>
      </div>
      {selected && status === 'completed' ? (
        <Check className="size-4 text-state-success" />
      ) : null}
    </div>
  );
}

function latestStatus(events: RuntimeEvent[], agentId: string) {
  return (
    events.filter((event) => event.agent_id === agentId).at(-1)?.status ??
    'pending'
  );
}

function finalTraceStatus(trace: AgentTrace | null, agentId: string) {
  return [...(trace?.steps ?? [])]
    .reverse()
    .find((step) => step.agent_id === agentId)?.status;
}

export function actualModelName(
  trace: AgentTrace | null,
  events: RuntimeEvent[],
): string | undefined {
  const eventModel = [...events]
    .reverse()
    .find(
      (event) =>
        event.type === 'agent.completed' &&
        typeof event.payload.model_used === 'string' &&
        event.payload.model_used.trim(),
    )?.payload.model_used;
  if (typeof eventModel === 'string' && eventModel.trim()) {
    return eventModel;
  }

  const traceModel = [...(trace?.spans ?? [])]
    .reverse()
    .filter(
      (span) =>
        span.name !== 'legacy_champion' ||
        span.attributes.answer_adopted !== false,
    )
    .map((span) => span.attributes.model_used)
    .find((model) => typeof model === 'string' && model.trim());
  return typeof traceModel === 'string' ? traceModel : undefined;
}

export function AgentRunInspector({
  trace,
  events,
  running,
  readiness,
  reserveCloseButtonSpace = false,
}: {
  trace: AgentTrace | null;
  events: RuntimeEvent[];
  running: boolean;
  readiness: RuntimeReadiness | null;
  reserveCloseButtonSpace?: boolean;
}) {
  const plan = events.find((event) => event.type === 'plan.completed');
  const eventAgents = Array.from(
    new Set(
      events
        .map((event) => event.agent_id)
        .filter((id): id is AgentId => id in agents),
    ),
  );
  const selected = trace?.selected_agents ??
    Array.from(
      new Set([...(plan?.payload.selected_agents ?? []), ...eventAgents]),
    );
  const visibleEvents = events.filter(
    (event) => event.type !== 'run.completed',
  );
  const completedModel = actualModelName(trace, events);
  const plannedModel = plan?.payload.llm_model as string | undefined;
  const modelLabel = completedModel
    ? '实际回答模型'
    : plannedModel
      ? '本次计划模型'
      : trace
        ? '回答模型记录'
        : '当前默认模型';
  const model = completedModel
    ? completedModel
    : plannedModel
      ? plannedModel
      : trace
        ? '历史回答模型未记录'
        : readiness?.llm_model ?? '未配置默认 LLM';

  return (
    <div className="flex h-full min-h-0 flex-col" aria-label="Agent 执行轨迹">
      <div
        className={cn(
          'flex items-center justify-between border-b border-border-button py-3',
          reserveCloseButtonSpace ? 'pl-4 pr-14' : 'px-4',
        )}
      >
        <div className="flex items-center gap-2">
          <Bot className="size-4" />
          <span className="text-sm font-semibold">Multi-Agent 协作</span>
        </div>
        <Badge variant={running ? 'secondary' : trace ? 'success' : 'outline'}>
          {running ? '运行中' : trace ? `${trace.total_latency_ms} ms` : '待命'}
        </Badge>
      </div>

      <div className="min-h-0 flex-1 overflow-auto p-4">
        <div className="mb-4 rounded-md border border-border-accent bg-accent-primary/5 p-3">
          <div className="text-[11px] uppercase text-text-secondary">
            {modelLabel}
          </div>
          <div className="mt-1 truncate text-sm font-semibold text-accent-primary">
            {model}
          </div>
          <div className="mt-3 grid grid-cols-4 gap-1">
            {[
              ['Embedding', readiness?.embedding_configured],
              ['Rerank', readiness?.rerank_configured],
              ['VLM', readiness?.vlm_configured],
              ['OCR', readiness?.ocr_configured],
            ].map(([label, ready]) => (
              <div
                key={String(label)}
                className={cn(
                  'rounded border border-border-button px-1 py-1 text-center text-[10px]',
                  ready
                    ? 'text-state-success'
                    : 'text-text-secondary opacity-60',
                )}
              >
                {String(label)}
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-2" aria-label="动态 Agent 团队">
          {(Object.keys(agents) as AgentId[])
            .filter((id) => id !== 'legacy-champion' || selected.includes(id))
            .map((id) => (
              <AgentRow
                key={id}
                id={id}
                selected={
                  selected.includes(id) ||
                  (events.length === 0 && id === 'orchestrator')
                }
                status={
                  !running && trace && selected.includes(id)
                    ? finalTraceStatus(trace, id) ??
                      (events.some((event) => event.agent_id === id)
                        ? latestStatus(events, id)
                        : 'pending')
                    : latestStatus(events, id)
                }
              />
            ))}
        </div>

        <div className="mb-2 mt-5 flex items-center justify-between">
          <span className="text-xs font-semibold">协作事件</span>
          <span className="text-xs tabular-nums text-text-secondary">
            {visibleEvents.length}
          </span>
        </div>
        {visibleEvents.length === 0 ? (
          <div className="rounded-md border border-dashed border-border-button p-4 text-xs leading-5 text-text-secondary">
            Orchestrator 会按问题动态选择专业 Agent，由 Verifier 独立验收。
          </div>
        ) : (
          <div className="space-y-1">
            {visibleEvents.map((event) => (
              <div
                key={event.sequence}
                className="grid grid-cols-[24px_20px_1fr] gap-2 border-b border-border-button py-2.5 last:border-b-0"
              >
                <span className="pt-0.5 text-[10px] tabular-nums text-text-secondary">
                  {String(event.sequence).padStart(2, '0')}
                </span>
                <span className="pt-0.5 text-text-secondary">
                  {event.status === 'completed' ? (
                    <Check className="size-3.5 text-state-success" />
                  ) : event.status === 'running' ? (
                    <Clock3 className="size-3.5 text-accent-primary" />
                  ) : (
                    <RotateCcw className="size-3.5" />
                  )}
                  <span className="sr-only">
                    {statusLabel(event.status, true)}
                  </span>
                </span>
                <div className="min-w-0">
                  <div className="text-xs font-medium leading-5">
                    {event.label}
                  </div>
                  <div className="truncate text-[11px] text-text-secondary">
                    {event.summary || event.agent_id}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {trace?.spans?.length ? (
          <>
            <div className="mb-2 mt-5 flex items-center justify-between">
              <span className="text-xs font-semibold">Trace Spans</span>
              <span className="text-xs tabular-nums text-text-secondary">
                {trace.spans.length}
              </span>
            </div>
            <div className="space-y-1">
              {trace.spans.map((span) => (
                <div
                  key={span.span_id}
                  className="flex items-start justify-between gap-3 border-b border-border-button py-2 text-xs last:border-b-0"
                >
                  <span className="font-mono text-[11px]">{span.name}</span>
                  <span className="max-w-[55%] truncate text-right text-[11px] text-text-secondary">
                    {span.output_summary || `${span.latency_ms} ms`}
                  </span>
                </div>
              ))}
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
