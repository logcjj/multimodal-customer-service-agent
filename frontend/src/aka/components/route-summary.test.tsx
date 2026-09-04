import { fireEvent, render, screen } from '@testing-library/react';
import type { AgentResponse, FinalRoute } from '../api/types';
import { RouteSummary } from './route-summary';

const labels: Record<FinalRoute, string> = {
  technical_knowledge: '技术知识库',
  customer_service: '客服政策',
  mixed: '混合路由',
  evidence_clarification: '证据补全',
  general_llm: '通用大模型',
  safe_handoff: '安全转人工',
  general_unavailable: '模型不可用',
};

function response(finalRoute: FinalRoute): AgentResponse {
  return {
    request_id: 'req-1',
    session_id: 'c1',
    answer: 'answer',
    route: finalRoute,
    citations: [],
    assets: [],
    verification: { passed: true, action: 'accept', confidence: 0.8, issues: [] },
    trace: {
      request_id: 'req-1',
      session_id: 'c1',
      route: finalRoute,
      selected_agents: ['orchestrator', 'router', 'general', 'verifier'],
      steps: [
        {
          agent_id: 'router',
          label: '识别意图与风险',
          status: 'completed',
          latency_ms: 5,
          summary: '普通写作请求',
        },
      ],
      spans: [],
      fallback_reason: null,
      total_latency_ms: 120,
    },
    used_legacy: false,
    routing: {
      initial_route: 'general_candidate',
      final_route: finalRoute,
      route_label: labels[finalRoute],
      route_reason: '问题不依赖产品手册或客服政策',
      coverage_status:
        finalRoute === 'evidence_clarification' ? 'clarifiable' : 'general_allowed',
      knowledge_covered: finalRoute === 'technical_knowledge',
      risk_level: 'low',
      clarification:
        finalRoute === 'evidence_clarification'
          ? {
              case_id: 'case-1',
              field: 'model',
              question: '请提供产品型号。',
              round: 2,
              max_rounds: 3,
              accepted_input_types: ['text'],
            }
          : null,
    },
    timestamp: 1,
  };
}

describe('RouteSummary', () => {
  it.each(Object.entries(labels))('renders %s as %s', (route, label) => {
    render(<RouteSummary response={response(route as FinalRoute)} events={[]} />);
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it('shows reason, clarification round and an expandable actual collaboration trace', () => {
    render(
      <RouteSummary response={response('evidence_clarification')} events={[]} />,
    );

    expect(screen.getByText('问题不依赖产品手册或客服政策')).toBeInTheDocument();
    expect(screen.getByText('第 2/3 轮')).toBeInTheDocument();
    expect(screen.getByText('参与 4 个 Agent')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '查看协作过程与路由依据' }));
    expect(screen.getByText('识别意图与风险')).toBeInTheDocument();
    expect(screen.getAllByText('router').length).toBeGreaterThan(0);
    expect(document.body.textContent).not.toMatch(/chain_of_thought|reasoning/);
  });

  it('shows the detected candidate route before the final response arrives', () => {
    render(
      <RouteSummary
        response={null}
        events={[
          {
            sequence: 1,
            type: 'route.detected',
            agent_id: 'router',
            status: 'completed',
            label: '识别用户意图与风险边界',
            summary: '技术知识候选',
            payload: {
              initial_route: 'technical_candidate',
              reason_code: 'deterministic-technical',
            },
          },
        ]}
      />,
    );

    expect(screen.getByText('技术知识候选')).toBeInTheDocument();
    expect(screen.getByText('deterministic-technical')).toBeInTheDocument();
  });

  it('shows a compact route-detection state before the first routing event', () => {
    render(<RouteSummary response={null} events={[]} running />);

    expect(screen.getByRole('status')).toHaveTextContent('正在判断路由');
  });

  it('uses canonical final badge labels instead of stale backend display text', () => {
    const item = response('mixed');
    item.routing!.route_label = '技术与客服协同';

    render(<RouteSummary response={item} events={[]} />);

    expect(screen.getByText('混合路由')).toBeInTheDocument();
    expect(screen.queryByText('技术与客服协同')).not.toBeInTheDocument();
  });

  it('labels legacy and old routing-null records without inventing modern routing', () => {
    const legacy = response('technical_knowledge');
    legacy.used_legacy = true;
    legacy.routing = null;
    legacy.trace.fallback_reason = 'legacy_only';
    const { rerender } = render(<RouteSummary response={legacy} events={[]} />);

    expect(screen.getByText('守护链路')).toBeInTheDocument();
    expect(screen.getByText(/未经过现代动态路由/)).toBeInTheDocument();
    expect(screen.queryByText('技术知识库')).not.toBeInTheDocument();

    const oldRecord = response('technical_knowledge');
    oldRecord.routing = null;
    rerender(<RouteSummary response={oldRecord} events={[]} />);

    expect(screen.getByText('兼容路由记录')).toBeInTheDocument();
    expect(screen.getByText(/原始 route：technical_knowledge/)).toBeInTheDocument();
  });
});
