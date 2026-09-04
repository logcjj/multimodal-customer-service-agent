import type { AgentResponse } from './types';
import { api } from './client';

describe('Trace API owner contract', () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
    jest.useRealTimers();
  });

  it('URL-encodes the explicit anonymous owner when listing traces', async () => {
    global.fetch = jest.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => [],
    })) as jest.Mock;

    await api.traces('anon-owner / 中文?');

    expect(global.fetch).toHaveBeenCalledWith(
      '/api/traces?user_id=anon-owner%20%2F%20%E4%B8%AD%E6%96%87%3F',
      expect.objectContaining({}),
    );
  });

  it('retries a transient stream gateway failure before returning the answer', async () => {
    jest.useFakeTimers();
    const finalResponse = {
      request_id: 'req-1',
      session_id: 'session-1',
      answer: '已恢复并返回答案。',
      route: 'technical_knowledge',
      citations: [],
      assets: [],
      verification: {
        passed: true,
        action: 'accept',
        confidence: 0.9,
        issues: [],
      },
      trace: {
        request_id: 'req-1',
        session_id: 'session-1',
        route: 'technical_knowledge',
        selected_agents: [],
        steps: [],
        spans: [],
        fallback_reason: null,
        total_latency_ms: 1,
      },
      used_legacy: false,
      timestamp: 1,
    } satisfies AgentResponse;
    const completedEvent = {
      sequence: 1,
      type: 'run.completed',
      agent_id: 'orchestrator',
      status: 'completed' as const,
      label: '回答生成完成',
      summary: '1 ms',
      payload: { response: finalResponse },
    };
    const encoded = new TextEncoder().encode(
      `${JSON.stringify(completedEvent)}\n`,
    );
    let calls = 0;
    global.fetch = jest.fn(async () => {
      calls += 1;
      if (calls === 1) return { ok: false, status: 503 } as Response;
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

    const events: string[] = [];
    const pending = api.streamChat(
      { question: '测试问题', images: [] },
      (event) => events.push(event.type),
    );
    await jest.runAllTimersAsync();

    await expect(pending).resolves.toEqual(finalResponse);
    expect(global.fetch).toHaveBeenCalledTimes(2);
    expect(events).toEqual(['run.completed']);
  });
});
