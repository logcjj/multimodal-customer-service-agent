import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import type { AgentResponse, ConversationDetail, ConversationSummary } from '../api/types';
import {
  ACTIVE_CONVERSATION_STORAGE_KEY,
  CLIENT_ID_STORAGE_KEY,
} from '../lib/conversations';
import ChatPage from './chat-page';

function answer(sessionId: string, text: string): AgentResponse {
  return {
    request_id: `req-${sessionId}`,
    session_id: sessionId,
    answer: text,
    route: 'technical',
    citations: [],
    assets: [],
    verification: { passed: true, action: 'accept', confidence: 0.9, issues: [] },
    trace: {
      request_id: `req-${sessionId}`,
      session_id: sessionId,
      route: 'technical',
      selected_agents: ['orchestrator', 'router', 'knowledge', 'verifier'],
      steps: [],
      spans: [],
      fallback_reason: null,
      total_latency_ms: 100,
    },
    used_legacy: false,
    routing: {
      initial_route: 'technical_candidate',
      final_route: 'technical_knowledge',
      route_label: '技术知识库',
      route_reason: '检索到手册证据',
      coverage_status: 'covered',
      knowledge_covered: true,
      risk_level: 'low',
      clarification: null,
    },
    timestamp: 1,
  };
}

const conversations: ConversationSummary[] = ['c1', 'c2'].map((id, index) => ({
  id,
  owner_id: 'owner-a',
  title: index === 0 ? '洗衣机排障' : '净化器清洁',
  message_count: 2,
  last_message_preview: id,
  last_route: 'technical_knowledge',
  created_at: new Date(Date.now() - index * 1000).toISOString(),
  updated_at: new Date(Date.now() - index * 1000).toISOString(),
}));

function detail(id: string): ConversationDetail {
  const response = answer(id, id === 'c1' ? '恢复的 E03 回答' : '恢复的滤网回答');
  return {
    ...conversations.find((item) => item.id === id)!,
    turns: [
      {
        id: `turn-${id}`,
        conversation_id: id,
        ordinal: 1,
        request_id: response.request_id,
        user_text: id === 'c1' ? 'E03 怎么处理' : '滤网怎么清洁',
        attachment_metadata: [],
        assistant_text: response.answer,
        response,
        status: 'completed',
        error_code: null,
        initial_route: 'technical_candidate',
        final_route: 'technical_knowledge',
        route_reason: '检索到手册证据',
        coverage_status: 'covered',
        created_at: new Date().toISOString(),
        completed_at: new Date().toISOString(),
      },
    ],
    state: null,
  };
}

function jsonResponse(value: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => value,
  } as Response;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

describe('ChatPage persistent conversations', () => {
  let requestBodies: Array<Record<string, unknown>>;

  beforeAll(() => {
    window.PointerEvent = MouseEvent as typeof PointerEvent;
  });

  beforeEach(() => {
    window.localStorage.clear();
    window.localStorage.setItem(CLIENT_ID_STORAGE_KEY, 'owner-a');
    window.localStorage.setItem(ACTIVE_CONVERSATION_STORAGE_KEY, 'c1');
    requestBodies = [];
    global.fetch = jest.fn(async (input, init) => {
      const url = String(input);
      if (url === '/api/readiness') {
        return jsonResponse({
          status: 'ready',
          rollout_mode: 'agent_first',
          legacy_available: false,
          model_registry: 'ready',
          trace_store: 'ready',
          llm_configured: true,
          llm_model: 'qwen-plus',
          vlm_configured: false,
          embedding_configured: true,
          rerank_configured: true,
          ocr_configured: false,
          dynamic_routing: 'on',
          conversation_history: 'on',
          layered_memory: 'on',
          general_agent: 'on',
        });
      }
      if (url.startsWith('/api/conversations?')) return jsonResponse(conversations);
      if (url.includes('/api/conversations/c1')) return jsonResponse(detail('c1'));
      if (url.includes('/api/conversations/c2')) return jsonResponse(detail('c2'));
      if (url === '/api/chat/stream') {
        requestBodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
        const response = answer('c1', '新回答');
        const events = [
          {
            sequence: 1,
            type: 'answer.delta',
            agent_id: 'knowledge',
            status: 'running',
            label: '生成回答',
            summary: '',
            payload: { delta: '新回答' },
          },
          {
            sequence: 2,
            type: 'run.completed',
            agent_id: 'orchestrator',
            status: 'completed',
            label: '完成',
            summary: '',
            payload: { response },
          },
        ];
        const encoded = new TextEncoder().encode(
          events.map((event) => JSON.stringify(event)).join('\n'),
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
      }
      return jsonResponse(undefined, 204);
    }) as jest.Mock;
  });

  it('restores the last active conversation after refresh and isolates switching', async () => {
    render(
      <MemoryRouter>
        <ChatPage />
      </MemoryRouter>,
    );

    expect(await screen.findByText('恢复的 E03 回答')).toBeInTheDocument();
    const restoredAnswer = screen.getByText('恢复的 E03 回答').closest('article');
    expect(restoredAnswer).not.toBeNull();
    expect(within(restoredAnswer!).getByText('技术知识库')).toBeInTheDocument();
    expect(screen.getAllByText('技术知识库')).toHaveLength(1);
    fireEvent.click(screen.getByRole('button', { name: '净化器清洁' }));
    expect(await screen.findByText('恢复的滤网回答')).toBeInTheDocument();
    expect(screen.queryByText('恢复的 E03 回答')).not.toBeInTheDocument();
    expect(window.localStorage.getItem(ACTIVE_CONVERSATION_STORAGE_KEY)).toBe('c2');
  });

  it('starts a blank draft without creating an empty backend record', async () => {
    render(
      <MemoryRouter>
        <ChatPage />
      </MemoryRouter>,
    );
    await screen.findByText('恢复的 E03 回答');

    fireEvent.click(screen.getByRole('button', { name: '新建对话' }));

    expect(screen.queryByText('恢复的 E03 回答')).not.toBeInTheDocument();
    expect(window.localStorage.getItem(ACTIVE_CONVERSATION_STORAGE_KEY)).toBeNull();
    expect(
      (global.fetch as jest.Mock).mock.calls.some(
        ([url, init]) => String(url).includes('/api/conversations') && init?.method === 'POST',
      ),
    ).toBe(false);
  });

  it('sends the anonymous owner and active conversation id', async () => {
    render(
      <MemoryRouter>
        <ChatPage />
      </MemoryRouter>,
    );
    await screen.findByText('恢复的 E03 回答');
    fireEvent.change(screen.getByRole('textbox', { name: '输入客服问题' }), {
      target: { value: '继续排查' },
    });
    fireEvent.click(screen.getByRole('button', { name: '发送' }));

    await waitFor(() => expect(requestBodies).toHaveLength(1));
    expect(requestBodies[0]).toMatchObject({
      question: '继续排查',
      session_id: 'c1',
      user_id: 'owner-a',
    });
  });

  it('keeps a new blank draft when delayed initial restoration finishes later', async () => {
    const pendingDetail = deferred<Response>();
    const defaultFetch = global.fetch;
    global.fetch = jest.fn((input, init) => {
      if (String(input).includes('/api/conversations/c1')) {
        return pendingDetail.promise;
      }
      return defaultFetch(input, init);
    }) as jest.Mock;

    render(
      <MemoryRouter>
        <ChatPage />
      </MemoryRouter>,
    );
    await screen.findByRole('button', { name: '洗衣机排障' });
    fireEvent.click(screen.getByRole('button', { name: '新建对话' }));
    await act(async () => {
      pendingDetail.resolve(jsonResponse(detail('c1')));
      await pendingDetail.promise;
    });

    await waitFor(() => {
      expect(screen.queryByText('恢复的 E03 回答')).not.toBeInTheDocument();
      expect(window.localStorage.getItem(ACTIVE_CONVERSATION_STORAGE_KEY)).toBeNull();
    });
  });

  it('keeps the latest selected conversation when an older request finishes last', async () => {
    const pendingFirstDetail = deferred<Response>();
    const defaultFetch = global.fetch;
    global.fetch = jest.fn((input, init) => {
      if (String(input).includes('/api/conversations/c1')) {
        return pendingFirstDetail.promise;
      }
      return defaultFetch(input, init);
    }) as jest.Mock;

    render(
      <MemoryRouter>
        <ChatPage />
      </MemoryRouter>,
    );
    fireEvent.click(await screen.findByRole('button', { name: '净化器清洁' }));
    expect(await screen.findByText('恢复的滤网回答')).toBeInTheDocument();

    await act(async () => {
      pendingFirstDetail.resolve(jsonResponse(detail('c1')));
      await pendingFirstDetail.promise;
    });
    await waitFor(() => {
      expect(screen.getByText('恢复的滤网回答')).toBeInTheDocument();
      expect(screen.queryByText('恢复的 E03 回答')).not.toBeInTheDocument();
      expect(window.localStorage.getItem(ACTIVE_CONVERSATION_STORAGE_KEY)).toBe('c2');
    });
  });

  it('clears a stale active id when the owner has no conversations', async () => {
    const defaultFetch = global.fetch;
    global.fetch = jest.fn((input, init) => {
      if (String(input).startsWith('/api/conversations?')) {
        return Promise.resolve(jsonResponse([]));
      }
      return defaultFetch(input, init);
    }) as jest.Mock;

    render(
      <MemoryRouter>
        <ChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(window.localStorage.getItem(ACTIVE_CONVERSATION_STORAGE_KEY)).toBeNull();
    });
  });

  it('restores a legacy answer with an explicit compatibility route summary', async () => {
    const legacyDetail = detail('c1');
    legacyDetail.turns[0].response!.used_legacy = true;
    legacyDetail.turns[0].response!.routing = null;
    legacyDetail.turns[0].response!.trace.fallback_reason = 'legacy_only';
    const defaultFetch = global.fetch;
    global.fetch = jest.fn((input, init) => {
      if (String(input).includes('/api/conversations/c1')) {
        return Promise.resolve(jsonResponse(legacyDetail));
      }
      return defaultFetch(input, init);
    }) as jest.Mock;

    render(
      <MemoryRouter>
        <ChatPage />
      </MemoryRouter>,
    );

    const answerText = await screen.findByText('恢复的 E03 回答');
    const answerArticle = answerText.closest('article');
    expect(answerArticle).not.toBeNull();
    expect(within(answerArticle!).getAllByText('守护链路')).toHaveLength(2);
    expect(within(answerArticle!).getByText(/未经过现代动态路由/)).toBeInTheDocument();
  });

  it('locks conversation navigation while deleting and clears only the deleted active chat', async () => {
    const pendingDelete = deferred<Response>();
    const defaultFetch = global.fetch;
    global.fetch = jest.fn((input, init) => {
      if (
        String(input).includes('/api/conversations/c1') &&
        init?.method === 'DELETE'
      ) {
        return pendingDelete.promise;
      }
      return defaultFetch(input, init);
    }) as jest.Mock;

    render(
      <MemoryRouter>
        <ChatPage />
      </MemoryRouter>,
    );
    await screen.findByText('恢复的 E03 回答');
    fireEvent.pointerDown(
      screen.getByRole('button', { name: '更多操作：洗衣机排障' }),
      { button: 0, ctrlKey: false },
    );
    fireEvent.click(screen.getByText('删除'));
    fireEvent.click(screen.getByRole('button', { name: '确认删除' }));

    expect(screen.getByRole('button', { name: '新建对话' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '净化器清洁' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: '净化器清洁' }));

    await act(async () => {
      pendingDelete.resolve(jsonResponse(undefined, 204));
      await pendingDelete.promise;
    });
    await waitFor(() => {
      expect(screen.queryByText('恢复的 E03 回答')).not.toBeInTheDocument();
      expect(screen.queryByText('恢复的滤网回答')).not.toBeInTheDocument();
      expect(window.localStorage.getItem(ACTIVE_CONVERSATION_STORAGE_KEY)).toBeNull();
    });
  });
});
