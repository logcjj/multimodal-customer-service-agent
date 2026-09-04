import type {
  AgentResponse,
  FinalRoute,
  RoutingDecision,
  RuntimeEvent,
} from '@/aka/api/types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { cn } from '@/lib/utils';
import { ChevronDown, GitBranch, LoaderCircle, Route } from 'lucide-react';
import { useState } from 'react';

const routeLabels: Record<FinalRoute, string> = {
  technical_knowledge: '技术知识库',
  customer_service: '客服政策',
  mixed: '混合路由',
  evidence_clarification: '证据补全',
  general_llm: '通用大模型',
  safe_handoff: '安全转人工',
  general_unavailable: '模型不可用',
};

function compatibilityRoute(response: AgentResponse | null) {
  if (!response || response.routing) return null;
  if (response.used_legacy) {
    const fallback = response.trace.fallback_reason;
    return {
      label: '守护链路',
      reason: `该回答由守护链路生成，未经过现代动态路由${
        fallback ? `（兼容原因：${fallback}）` : ''
      }。`,
      variant: 'secondary' as const,
    };
  }
  return {
    label: '兼容路由记录',
    reason: `该历史回答未保存现代动态路由信息（原始 route：${response.route}）。`,
    variant: 'outline' as const,
  };
}

function eventRouting(events: RuntimeEvent[]): RoutingDecision | null {
  const resolved = [...events]
    .reverse()
    .find((item) => item.type === 'route.resolved');
  if (resolved) {
    const payload = resolved.payload as Partial<RoutingDecision>;
    if (payload.final_route) {
      return {
        initial_route: payload.initial_route ?? 'unknown',
        final_route: payload.final_route,
        route_label: routeLabels[payload.final_route],
        route_reason: payload.route_reason ?? resolved.summary,
        coverage_status: payload.coverage_status ?? 'covered',
        knowledge_covered: payload.knowledge_covered ?? false,
        risk_level: payload.risk_level ?? 'low',
        clarification: payload.clarification ?? null,
      };
    }
  }
  const detected = [...events]
    .reverse()
    .find((item) => item.type === 'route.detected');
  if (!detected) return null;
  const initialRoute = String(detected.payload.initial_route ?? 'technical_candidate');
  const candidateRoutes: Record<string, FinalRoute> = {
    technical_candidate: 'technical_knowledge',
    customer_service_candidate: 'customer_service',
    mixed_candidate: 'mixed',
    general_candidate: 'general_llm',
  };
  return {
    initial_route: initialRoute,
    final_route: candidateRoutes[initialRoute] ?? 'technical_knowledge',
    route_label: detected.summary || '路由识别完成',
    route_reason: String(detected.payload.reason_code ?? detected.label),
    coverage_status: 'general_allowed',
    knowledge_covered: false,
    risk_level: 'low',
    clarification: null,
  };
}

function routeVariant(route: FinalRoute) {
  if (route === 'safe_handoff' || route === 'general_unavailable') {
    return 'destructive' as const;
  }
  if (route === 'evidence_clarification') return 'secondary' as const;
  return 'success' as const;
}

export function RouteSummary({
  response,
  events,
  running = false,
}: {
  response: AgentResponse | null;
  events: RuntimeEvent[];
  running?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const routing = response?.routing ?? eventRouting(events);
  const compatibility = compatibilityRoute(response);
  if (!routing && !compatibility) {
    if (!running) return null;
    return (
      <div
        role="status"
        className="flex min-h-10 items-center gap-2 border-b border-border-button bg-bg-card/70 px-4 py-2 text-xs text-text-secondary lg:px-6"
      >
        <LoaderCircle className="size-4 animate-spin text-accent-primary" />
        <span className="font-medium text-text-primary">正在判断路由</span>
        <span className="truncate">Router 正在识别意图与风险边界</span>
      </div>
    );
  }
  const trace = response?.trace;
  const selectedAgents = trace?.selected_agents ?? [];
  const routeLabel = compatibility
    ? compatibility.label
    : response?.routing
      ? routeLabels[response.routing.final_route]
      : routing!.route_label || routeLabels[routing!.final_route];
  const routeReason = compatibility?.reason ?? routing!.route_reason;
  const badgeVariant = compatibility?.variant ?? routeVariant(routing!.final_route);

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div className="border-b border-border-button bg-bg-card/70 px-4 py-2.5 lg:px-6">
        <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-2">
          <div className="flex min-w-0 items-center gap-2">
            <Route className="size-4 shrink-0 text-text-secondary" />
            <Badge variant={badgeVariant}>{routeLabel}</Badge>
          </div>
          {selectedAgents.length ? (
            <span className="text-xs text-text-secondary">
              参与 {selectedAgents.length} 个 Agent
            </span>
          ) : null}
          {routing?.clarification ? (
            <span className="text-xs font-medium text-text-secondary">
              第 {routing.clarification.round}/{routing.clarification.max_rounds} 轮
            </span>
          ) : null}
          <span className="min-w-[12rem] flex-1 truncate text-xs text-text-secondary">
            {routeReason}
          </span>
          {trace ? (
            <CollapsibleTrigger asChild>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                aria-label="查看协作过程与路由依据"
                className="shrink-0"
              >
                <GitBranch className="size-3.5" />
                协作过程
                <ChevronDown
                  className={cn('size-3.5 transition-transform', open && 'rotate-180')}
                />
              </Button>
            </CollapsibleTrigger>
          ) : null}
        </div>

        <CollapsibleContent>
          <div className="mt-2 grid gap-3 border-t border-border-button pt-3 text-xs md:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
            <div>
              <div className="mb-1.5 font-medium">实际参与 Agent</div>
              <div className="flex flex-wrap gap-1.5">
                {selectedAgents.map((agent) => (
                  <span
                    key={agent}
                    className="rounded border border-border-button bg-bg-component px-2 py-1 font-mono text-[11px]"
                  >
                    {agent}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <div className="mb-1.5 font-medium">执行步骤</div>
              <div className="divide-y divide-border-button">
                {trace?.steps.map((step) => (
                  <div
                    key={`${step.agent_id}-${step.label}`}
                    className="flex items-start justify-between gap-3 py-1.5"
                  >
                    <span>{step.label}</span>
                    <span className="shrink-0 text-text-secondary">
                      {step.agent_id}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}
