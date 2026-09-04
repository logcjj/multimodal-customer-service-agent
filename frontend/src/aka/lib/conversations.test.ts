import type { AgentResponse, ConversationSummary, ConversationTurn } from '../api/types';
import {
  CLIENT_ID_STORAGE_KEY,
  getBrowserClientId,
  getOrCreateClientId,
  groupConversations,
  turnsToEntries,
} from './conversations';

function summary(id: string, updatedAt: string): ConversationSummary {
  return {
    id,
    owner_id: 'owner-a',
    title: id,
    message_count: 2,
    last_message_preview: id,
    last_route: 'technical_knowledge',
    created_at: updatedAt,
    updated_at: updatedAt,
  };
}

describe('conversation helpers', () => {
  beforeEach(() => window.localStorage.clear());

  it('creates one persistent anonymous client id', () => {
    const first = getOrCreateClientId(window.localStorage);
    const second = getOrCreateClientId(window.localStorage);

    expect(first).toBe(second);
    expect(first).toMatch(/^anon-/);
    expect(window.localStorage.getItem(CLIENT_ID_STORAGE_KEY)).toBe(first);
  });

  it('reuses the same browser client id and safely returns null without storage', () => {
    window.localStorage.setItem(CLIENT_ID_STORAGE_KEY, 'anon-persisted-owner');

    expect(getBrowserClientId(window.localStorage)).toBe('anon-persisted-owner');
    expect(getBrowserClientId(window.localStorage)).toBe('anon-persisted-owner');
    expect(getBrowserClientId(null)).toBeNull();
  });

  it('groups conversations into today, yesterday and earlier', () => {
    const now = new Date('2026-07-25T12:00:00+08:00');
    const groups = groupConversations(
      [
        summary('today', '2026-07-25T02:00:00.000Z'),
        summary('yesterday', '2026-07-24T02:00:00.000Z'),
        summary('earlier', '2026-07-20T02:00:00.000Z'),
      ],
      now,
    );

    expect(groups.map((group) => group.label)).toEqual(['今天', '昨天', '更早']);
    expect(groups.map((group) => group.items[0].id)).toEqual([
      'today',
      'yesterday',
      'earlier',
    ]);
  });

  it('restores each completed turn as a user and assistant entry', () => {
    const response = {
      request_id: 'req-1',
      session_id: 'c1',
      answer: '请先断电。',
    } as AgentResponse;
    const turn = {
      id: 'turn-1',
      conversation_id: 'c1',
      ordinal: 1,
      request_id: 'req-1',
      user_text: 'E03 怎么处理',
      attachment_metadata: [],
      assistant_text: '请先断电。',
      response,
      status: 'completed',
      error_code: null,
      initial_route: 'technical_candidate',
      final_route: 'technical_knowledge',
      route_reason: '检索到手册证据',
      coverage_status: 'covered',
      created_at: '2026-07-25T02:00:00.000Z',
      completed_at: '2026-07-25T02:00:01.000Z',
    } satisfies ConversationTurn;

    expect(turnsToEntries([turn])).toEqual([
      { id: 'turn-1-user', role: 'user', text: 'E03 怎么处理' },
      {
        id: 'turn-1-assistant',
        role: 'assistant',
        text: '请先断电。',
        response,
      },
    ]);
  });
});
