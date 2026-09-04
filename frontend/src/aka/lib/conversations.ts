import type { AgentResponse, ConversationSummary, ConversationTurn } from '../api/types';

export const CLIENT_ID_STORAGE_KEY = 'aka-client-id';
export const ACTIVE_CONVERSATION_STORAGE_KEY = 'aka-active-conversation-id';

export interface ConversationEntry {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  response?: AgentResponse;
}

export interface ConversationGroup {
  key: 'today' | 'yesterday' | 'earlier';
  label: '今天' | '昨天' | '更早';
  items: ConversationSummary[];
}

export function getOrCreateClientId(storage: Storage): string {
  const existing = storage.getItem(CLIENT_ID_STORAGE_KEY)?.trim();
  if (existing) return existing;
  const randomPart =
    globalThis.crypto?.randomUUID?.() ??
    `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  const clientId = `anon-${randomPart}`;
  storage.setItem(CLIENT_ID_STORAGE_KEY, clientId);
  return clientId;
}

export function getBrowserClientId(
  storage: Storage | null | undefined,
): string | null {
  if (!storage) return null;
  try {
    return getOrCreateClientId(storage);
  } catch {
    return null;
  }
}

function localDay(value: Date): number {
  return new Date(value.getFullYear(), value.getMonth(), value.getDate()).getTime();
}

export function groupConversations(
  items: ConversationSummary[],
  now = new Date(),
): ConversationGroup[] {
  const today = localDay(now);
  const buckets: Record<ConversationGroup['key'], ConversationSummary[]> = {
    today: [],
    yesterday: [],
    earlier: [],
  };
  for (const item of items) {
    const difference = Math.round((today - localDay(new Date(item.updated_at))) / 86_400_000);
    if (difference <= 0) buckets.today.push(item);
    else if (difference === 1) buckets.yesterday.push(item);
    else buckets.earlier.push(item);
  }
  const labels: Array<[ConversationGroup['key'], ConversationGroup['label']]> = [
    ['today', '今天'],
    ['yesterday', '昨天'],
    ['earlier', '更早'],
  ];
  return labels
    .filter(([key]) => buckets[key].length > 0)
    .map(([key, label]) => ({ key, label, items: buckets[key] }));
}

export function turnsToEntries(turns: ConversationTurn[]): ConversationEntry[] {
  return turns.flatMap((turn) => {
    const entries: ConversationEntry[] = [
      {
        id: `${turn.id}-user`,
        role: 'user',
        text: turn.user_text,
      },
    ];
    if (turn.assistant_text.trim()) {
      entries.push({
        id: `${turn.id}-assistant`,
        role: 'assistant',
        text: turn.assistant_text,
        ...(turn.response ? { response: turn.response } : {}),
      });
    }
    return entries;
  });
}
