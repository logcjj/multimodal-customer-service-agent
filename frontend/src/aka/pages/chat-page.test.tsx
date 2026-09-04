import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import type { AgentResponse, FinalRoute } from '../api/types';
import ChatPage, { answerSourceLabel } from './chat-page';

const agentResponse = {
  request_id: 'req-1',
  session_id: 'session-1',
  answer: '请先断开电源，再检查排水管和排水过滤器。',
  route: 'technical',
  citations: [
    {
      evidence_id: 'e1',
      source_type: 'manual',
      title: 'E03 排水故障',
      text: '请先断电并检查排水管。',
      dataset_id: 'kb-1',
      document_id: 'doc-1',
      file_id: 'file-1',
      document_name: '洗衣机使用手册.pdf',
      document_mime_type: 'application/pdf',
      document_version: 'idx-v1',
      section_id: 'parent-1',
      parent_id: 'parent-1',
      child_ids: ['child-1'],
      image_chunk_ids: ['image-chunk-1'],
      chapter_title: '排水故障处理',
      page_start: 12,
      page_end: 13,
      asset_ids: ['asset-1'],
      score: 0.024,
      retrieval_stage: 'hybrid-rerank',
      evidence_confidence: 0.91,
      score_breakdown: {
        lexical: 4.2,
        dense: 0.82,
        rrf: 0.0164,
        rerank: 0.91,
        parent: 0.024,
      },
    },
  ],
  assets: [],
  verification: {
    passed: true,
    action: 'accept',
    confidence: 0.91,
    issues: [],
  },
  trace: {
    request_id: 'req-1',
    session_id: 'session-1',
    route: 'technical',
    selected_agents: ['orchestrator', 'knowledge', 'verifier'],
    steps: [],
    spans: [
      {
        span_id: 's1',
        name: 'lexical_retrieval',
        status: 'completed',
        latency_ms: 3,
        input_summary: '',
        output_summary: '候选 1 条',
        attributes: {},
      },
    ],
    fallback_reason: null,
    total_latency_ms: 900,
  },
  used_legacy: false,
  timestamp: 1,
};

const streamEvents = [
  {
    sequence: 1,
    type: 'plan.completed',
    agent_id: 'orchestrator',
    status: 'completed',
    label: '完成任务拆分与动态组队',
    summary: '2 个子任务 · technical',
    payload: {
      selected_agents: ['orchestrator', 'knowledge', 'verifier'],
      route: 'technical',
      llm_configured: true,
      llm_model: 'deepseek-v4-flash',
    },
  },
  {
    sequence: 2,
    type: 'agent.completed',
    agent_id: 'knowledge',
    status: 'completed',
    label: '完成技术证据与回答',
    summary: 'deepseek-v4-flash · LLM 生成 · 680 ms',
    payload: {
      llm_generated: true,
      model_used: 'deepseek-v4-flash',
      evidence_count: 1,
    },
  },
  {
    sequence: 3,
    type: 'answer.delta',
    agent_id: 'knowledge',
    status: 'running',
    label: '生成已验证正文',
    summary: '',
    payload: { delta: '请先断开电源，再检查排水管和排水过滤器。' },
  },
  {
    sequence: 4,
    type: 'verification.completed',
    agent_id: 'verifier',
    status: 'completed',
    label: '验证完成',
    summary: '验证通过',
    payload: {},
  },
  {
    sequence: 5,
    type: 'run.completed',
    agent_id: 'orchestrator',
    status: 'completed',
    label: '回答生成完成',
    summary: '900 ms',
    payload: { response: agentResponse },
  },
];

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

describe('RAGFlow-style Multi-Agent chat', () => {
  beforeEach(() => {
    global.fetch = jest.fn(async (input) => {
      const url = String(input);
      if (url.endsWith('/api/readiness')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            status: 'ready',
            rollout_mode: 'agent_first',
            legacy_available: true,
            model_registry: 'ready',
            trace_store: 'ready',
            llm_configured: true,
            llm_model: 'deepseek-v4-flash',
            vlm_configured: false,
            embedding_configured: false,
            rerank_configured: false,
            ocr_configured: false,
          }),
        } as Response;
      }
      const encoded = new TextEncoder().encode(
        streamEvents.map((event) => JSON.stringify(event)).join('\n'),
      );
      let read = false;
      return {
        ok: true,
        status: 200,
        body: {
          getReader: () => ({
            read: async () => {
              if (read) return { done: true, value: undefined };
              read = true;
              return { done: false, value: encoded };
            },
          }),
        },
      } as Response;
    }) as jest.Mock;
  });

  it('renders the real model, selected agents, verifier, answer and evidence', async () => {
    render(
      <MemoryRouter>
        <ChatPage />
      </MemoryRouter>,
    );

    fireEvent.change(screen.getByRole('textbox', { name: '输入客服问题' }), {
      target: { value: '洗衣机 E03 怎么处理？' },
    });
    fireEvent.click(screen.getByRole('button', { name: '发送' }));

    expect(
      await screen.findByText('请先断开电源，再检查排水管和排水过滤器。'),
    ).toBeInTheDocument();
    expect(
      screen.getAllByText('请先断开电源，再检查排水管和排水过滤器。'),
    ).toHaveLength(1);
    expect(screen.getAllByText('deepseek-v4-flash').length).toBeGreaterThan(0);
    expect(screen.getByText('知识检索智能体')).toBeInTheDocument();
    expect(screen.getByText('验证通过')).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole('button', { name: '引用 1：E03 排水故障' }),
    );
    expect(
      screen.getByRole('dialog', { name: '引用证据' }),
    ).toBeInTheDocument();
    expect(screen.getByText('第 12-13 页')).toBeInTheDocument();
    expect(screen.getByText('洗衣机使用手册.pdf')).toBeInTheDocument();
    expect(screen.getByText('排水故障处理')).toBeInTheDocument();
    expect(screen.getByText('hybrid-rerank')).toBeInTheDocument();
    expect(screen.getByText('91%')).toBeInTheDocument();
    expect(screen.getByText('image-chunk-1')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: '定位到知识库' })).toHaveAttribute(
      'href',
      '/dataset/kb-1?tab=chunks&document=doc-1&child=child-1',
    );
    expect(screen.getByRole('link', { name: '查看原始文件' })).toHaveAttribute(
      'href',
      '/api/files/file-1/content#page=12',
    );
    expect(screen.getByText('0.0240')).toBeInTheDocument();
    expect(
      screen.getByRole('img', { name: 'E03 排水故障 关联图片' }),
    ).toHaveAttribute('src', '/api/assets/asset-1');
  });

  it('reserves close-button space only in the mobile Agent inspector Sheet header', async () => {
    render(
      <MemoryRouter>
        <ChatPage />
      </MemoryRouter>,
    );
    fireEvent.change(screen.getByRole('textbox', { name: '输入客服问题' }), {
      target: { value: '洗衣机 E03 怎么处理？' },
    });
    fireEvent.click(screen.getByRole('button', { name: '发送' }));
    await screen.findByText('请先断开电源，再检查排水管和排水过滤器。');

    const desktopInspector = screen.getByLabelText('Agent 执行轨迹');
    const desktopHeader = screen
      .getByText('Multi-Agent 协作')
      .parentElement?.parentElement;
    expect(desktopInspector).toContainElement(desktopHeader as HTMLElement);
    expect(desktopHeader).not.toHaveClass('pr-14');

    fireEvent.click(
      screen.getByRole('button', { name: '查看 Agent 执行轨迹' }),
    );
    const mobileSheet = await screen.findByRole('dialog', {
      name: 'Agent 执行轨迹',
    });
    const mobileTitle = within(mobileSheet).getByText('Multi-Agent 协作');
    const mobileHeader = mobileTitle.parentElement?.parentElement;

    expect(within(mobileSheet).getByText('900 ms')).toBeInTheDocument();
    expect(mobileHeader).toHaveClass('pr-14');
  });

  it('constrains the model badge so long names cannot squeeze the mobile header', async () => {
    render(
      <MemoryRouter>
        <ChatPage />
      </MemoryRouter>,
    );

    expect(await screen.findByText('deepseek-v4-flash · LLM')).toHaveClass(
      'max-w-[36vw]',
      'min-w-0',
      'truncate',
    );
  });

  it('updates the header badge to the model that actually answered', async () => {
    global.fetch = jest.fn(async (input) => {
      const url = String(input);
      if (url.endsWith('/api/readiness')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            llm_configured: true,
            llm_model: 'legacy-default-model',
          }),
        } as Response;
      }
      if (url.startsWith('/api/conversations?')) {
        return { ok: true, status: 200, json: async () => [] } as Response;
      }
      const actualEvents = streamEvents.map((event) =>
        event.type === 'agent.completed'
          ? {
              ...event,
              payload: {
                ...event.payload,
                model_used: 'actual-answer-model',
              },
            }
          : event,
      );
      const encoded = new TextEncoder().encode(
        actualEvents.map((event) => JSON.stringify(event)).join('\n'),
      );
      let read = false;
      return {
        ok: true,
        status: 200,
        body: {
          getReader: () => ({
            read: async () => {
              if (read) return { done: true, value: undefined };
              read = true;
              return { done: false, value: encoded };
            },
          }),
        },
      } as Response;
    }) as jest.Mock;

    render(
      <MemoryRouter>
        <ChatPage />
      </MemoryRouter>,
    );
    expect(
      await screen.findByText('legacy-default-model · LLM'),
    ).toBeInTheDocument();

    fireEvent.change(screen.getByRole('textbox', { name: '输入客服问题' }), {
      target: { value: '介绍一下北京交通大学' },
    });
    fireEvent.click(screen.getByRole('button', { name: '发送' }));

    expect(
      await screen.findByText('actual-answer-model · LLM'),
    ).toBeInTheDocument();
    expect(
      screen.queryByText('legacy-default-model · LLM'),
    ).not.toBeInTheDocument();
  });

  it('shows route detection immediately after send and before route.detected arrives', async () => {
    const pendingRead = deferred<ReadableStreamReadResult<Uint8Array>>();
    const delayedEncodedEvents = new TextEncoder().encode(
      streamEvents.map((event) => JSON.stringify(event)).join('\n'),
    );
    global.fetch = jest.fn(async (input) => {
      const url = String(input);
      if (url === '/api/readiness') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            status: 'ready',
            llm_configured: true,
            llm_model: 'deepseek-v4-flash',
          }),
        } as Response;
      }
      if (url.startsWith('/api/conversations?')) {
        return {
          ok: true,
          status: 200,
          json: async () => [],
        } as Response;
      }
      let started = false;
      return {
        ok: true,
        status: 200,
        body: {
          getReader: () => ({
            read: () => {
              if (started) return Promise.resolve({ done: true, value: undefined });
              started = true;
              return pendingRead.promise;
            },
          }),
        },
      } as Response;
    }) as jest.Mock;

    render(
      <MemoryRouter>
        <ChatPage />
      </MemoryRouter>,
    );
    fireEvent.change(screen.getByRole('textbox', { name: '输入客服问题' }), {
      target: { value: '洗衣机 E03 怎么处理？' },
    });
    fireEvent.click(screen.getByRole('button', { name: '发送' }));

    expect(await screen.findByText('正在判断路由')).toBeInTheDocument();
    await act(async () => {
      pendingRead.resolve({ done: false, value: delayedEncodedEvents });
      await pendingRead.promise;
    });
    expect(
      await screen.findByText('请先断开电源，再检查排水管和排水过滤器。'),
    ).toBeInTheDocument();
  });

  it('removes streamed assistant text when the run fails before verification completes', async () => {
    const failedEvents = [
      {
        sequence: 1,
        type: 'answer.delta',
        agent_id: 'knowledge',
        status: 'running',
        label: '生成候选正文',
        summary: '',
        payload: { delta: '这是一段尚未核验的临时正文' },
      },
      {
        sequence: 2,
        type: 'run.failed',
        agent_id: 'orchestrator',
        status: 'failed',
        label: '运行失败',
        summary: 'Verifier 前运行中断',
        payload: {},
      },
    ];
    global.fetch = jest.fn(async (input) => {
      const url = String(input);
      if (url === '/api/readiness') {
        return {
          ok: true,
          status: 200,
          json: async () => ({ llm_configured: true, llm_model: 'test-model' }),
        } as Response;
      }
      if (url.startsWith('/api/conversations?')) {
        return { ok: true, status: 200, json: async () => [] } as Response;
      }
      const encoded = new TextEncoder().encode(
        failedEvents.map((event) => JSON.stringify(event)).join('\n'),
      );
      let read = false;
      return {
        ok: true,
        status: 200,
        body: {
          getReader: () => ({
            read: async () => {
              if (read) return { done: true, value: undefined };
              read = true;
              return { done: false, value: encoded };
            },
          }),
        },
      } as Response;
    }) as jest.Mock;

    render(
      <MemoryRouter>
        <ChatPage />
      </MemoryRouter>,
    );
    fireEvent.change(screen.getByRole('textbox', { name: '输入客服问题' }), {
      target: { value: '测试失败流' },
    });
    fireEvent.click(screen.getByRole('button', { name: '发送' }));

    expect(await screen.findByRole('alert')).toHaveTextContent('Verifier 前运行中断');
    expect(screen.queryByText('这是一段尚未核验的临时正文')).not.toBeInTheDocument();
  });

  it.each<[FinalRoute, string]>([
    ['technical_knowledge', '知识库证据'],
    ['customer_service', '客服政策'],
    ['mixed', '多智能体证据'],
    ['evidence_clarification', 'Evidence Gap'],
    ['general_llm', 'General LLM'],
    ['safe_handoff', '安全转人工'],
    ['general_unavailable', '模型不可用'],
  ])('labels the %s answer source as %s', (finalRoute, expected) => {
    const response = {
      ...agentResponse,
      routing: {
        initial_route: 'technical_candidate',
        final_route: finalRoute,
        route_label: expected,
        route_reason: '测试路由',
        coverage_status:
          finalRoute === 'general_llm' ? 'general_allowed' : 'covered',
        knowledge_covered: finalRoute === 'technical_knowledge',
        risk_level: 'low',
        clarification: null,
      },
    } as AgentResponse;

    expect(answerSourceLabel(response)).toBe(expected);
  });
});
