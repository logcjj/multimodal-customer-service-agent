import { render, screen, within } from '@testing-library/react';
import type { AgentTrace, RuntimeEvent, RuntimeReadiness } from '../api/types';
import { AgentRunInspector } from './agent-run-inspector';

const trace: AgentTrace = {
  request_id: 'req-1',
  session_id: 'c1',
  route: 'evidence_clarification',
  selected_agents: [
    'orchestrator',
    'router',
    'general',
    'evidence-gap',
    'memory-curator',
  ],
  steps: [
    {
      agent_id: 'orchestrator',
      label: '建立上下文',
      status: 'completed',
      latency_ms: 1,
      summary: '完成',
    },
    {
      agent_id: 'general',
      label: '调用通用模型',
      status: 'failed',
      latency_ms: 2,
      summary: '模型不可用',
    },
    {
      agent_id: 'evidence-gap',
      label: '判断补证字段',
      status: 'skipped',
      latency_ms: 0,
      summary: '无需继续追问',
    },
    {
      agent_id: 'memory-curator',
      label: '整理会话记忆',
      status: 'completed',
      latency_ms: 1,
      summary: '完成',
    },
  ],
  spans: [],
  fallback_reason: null,
  total_latency_ms: 20,
};

const events: RuntimeEvent[] = [
  {
    sequence: 1,
    type: 'plan.completed',
    agent_id: 'orchestrator',
    status: 'completed',
    label: '完成早期组队',
    summary: '早期计划尚未包含动态 Agent',
    payload: {
      selected_agents: ['orchestrator', 'router', 'knowledge', 'verifier'],
    },
  },
];

const readiness: RuntimeReadiness = {
  status: 'ready',
  rollout_mode: 'agent_first',
  legacy_available: true,
  model_registry: 'ready',
  trace_store: 'ready',
  llm_configured: true,
  llm_model: 'current-default-model',
  vlm_configured: false,
  embedding_configured: true,
  rerank_configured: true,
  ocr_configured: false,
  dynamic_routing: 'on',
  conversation_history: 'on',
  layered_memory: 'on',
  general_agent: 'on',
};

function agentContent(name: string) {
  const content = screen.getByText(name).parentElement;
  if (!content) throw new Error(`找不到 ${name} 状态区域`);
  return within(content);
}

describe('AgentRunInspector', () => {
  it('uses the final trace team and preserves completed, failed and skipped states', () => {
    render(
      <AgentRunInspector
        trace={trace}
        events={events}
        running={false}
        readiness={null}
      />,
    );

    expect(agentContent('通用对话智能体').getByText('失败')).toBeInTheDocument();
    expect(
      agentContent('证据补全智能体').getByText('已跳过'),
    ).toBeInTheDocument();
    expect(
      agentContent('会话记忆整理智能体').getByText('已完成'),
    ).toBeInTheDocument();
    expect(
      agentContent('知识检索智能体').getByText('按需待命'),
    ).toBeInTheDocument();
    expect(
      agentContent('意图路由智能体').getByText('已入队'),
    ).toBeInTheDocument();
  });

  it('prefers the actual completed Agent model and does not relabel readiness as history', () => {
    const modelEvents: RuntimeEvent[] = [
      {
        ...events[0],
        payload: {
          selected_agents: ['orchestrator', 'knowledge'],
          llm_model: 'planned-model',
        },
      },
      {
        sequence: 2,
        type: 'agent.completed',
        agent_id: 'knowledge',
        status: 'completed',
        label: '知识回答完成',
        summary: '实际模型完成',
        payload: { model_used: 'actual-answer-model' },
      },
    ];
    const { rerender } = render(
      <AgentRunInspector
        trace={trace}
        events={modelEvents}
        running={false}
        readiness={readiness}
      />,
    );

    expect(screen.getByText('实际回答模型')).toBeInTheDocument();
    expect(screen.getByText('actual-answer-model')).toBeInTheDocument();
    expect(screen.queryByText('planned-model')).not.toBeInTheDocument();
    expect(screen.queryByText('current-default-model')).not.toBeInTheDocument();

    rerender(
      <AgentRunInspector
        trace={trace}
        events={[]}
        running={false}
        readiness={readiness}
      />,
    );
    expect(screen.getByText('历史回答模型未记录')).toBeInTheDocument();
    expect(screen.queryByText('current-default-model')).not.toBeInTheDocument();

    rerender(
      <AgentRunInspector
        trace={null}
        events={[]}
        running={false}
        readiness={readiness}
      />,
    );
    expect(screen.getByText('当前默认模型')).toBeInTheDocument();
    expect(screen.getByText('current-default-model')).toBeInTheDocument();
  });

  it('recovers the actual model from a persisted trace after events are cleared', () => {
    render(
      <AgentRunInspector
        trace={{
          ...trace,
          spans: [
            {
              span_id: 'model-span',
              name: 'general_answer',
              status: 'completed',
              latency_ms: 10,
              input_summary: '',
              output_summary: '',
              attributes: { model_used: 'persisted-answer-model' },
            },
          ],
        }}
        events={[]}
        running={false}
        readiness={readiness}
      />,
    );

    expect(screen.getByText('实际回答模型')).toBeInTheDocument();
    expect(screen.getByText('persisted-answer-model')).toBeInTheDocument();
    expect(screen.queryByText('current-default-model')).not.toBeInTheDocument();
  });

  it('merges dynamically appearing event Agents into the running plan team', () => {
    const runningEvents: RuntimeEvent[] = [
      events[0],
      {
        sequence: 2,
        type: 'clarification.required',
        agent_id: 'evidence-gap',
        status: 'running',
        label: '收集缺失证据',
        summary: '等待型号',
        payload: {},
      },
      {
        sequence: 3,
        type: 'memory.updated',
        agent_id: 'memory-curator',
        status: 'completed',
        label: '更新会话记忆',
        summary: '已保存',
        payload: {},
      },
    ];
    render(
      <AgentRunInspector
        trace={null}
        events={runningEvents}
        running
        readiness={readiness}
      />,
    );

    expect(
      agentContent('证据补全智能体').getByText('执行中'),
    ).toBeInTheDocument();
    expect(
      agentContent('会话记忆整理智能体').getByText('已完成'),
    ).toBeInTheDocument();
  });
});
