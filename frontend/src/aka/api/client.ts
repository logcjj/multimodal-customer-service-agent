import { apiUrl } from '@/lib/runtime-paths';
import type {
  AgentDefinition,
  AgentResponse,
  AgentTrace,
  ChildChunk,
  ChunkCollection,
  ConversationDetail,
  ConversationSummary,
  Dataset,
  EvalCase,
  EvalRun,
  FileAsset,
  ImageChunk,
  IndexManifest,
  KnowledgeDocument,
  McpServer,
  ModelConfig,
  ParsingJob,
  Provider,
  RetrievalExplanation,
  RetrievalProfile,
  RuntimeCapabilities,
  RuntimeEvent,
  RuntimeReadiness,
  SessionMemory,
  SkillDefinition,
  ToolDefinition,
  VectorMapResponse,
} from './types';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

const CHAT_STREAM_RETRY_DELAYS_MS = [500, 1_000, 2_000, 3_000];
const TRANSIENT_STREAM_STATUS_CODES = new Set([502, 503, 504]);

function waitForRetry(delayMs: number) {
  return new Promise<void>((resolve) => {
    globalThis.setTimeout(resolve, delayMs);
  });
}

async function openChatStream(payload: {
  question: string;
  images: string[];
  session_id?: string;
  user_id?: string;
}): Promise<Response> {
  let lastFailure: Error | undefined;

  for (
    let attempt = 0;
    attempt <= CHAT_STREAM_RETRY_DELAYS_MS.length;
    attempt += 1
  ) {
    try {
      const response = await fetch(apiUrl('/api/chat/stream'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (response.ok) return response;

      const failure = new Error(`请求失败 (${response.status})`);
      if (!TRANSIENT_STREAM_STATUS_CODES.has(response.status)) {
        throw failure;
      }
      lastFailure = failure;
    } catch (reason) {
      if (!(reason instanceof TypeError)) throw reason;
      lastFailure = reason;
    }

    const retryDelay = CHAT_STREAM_RETRY_DELAYS_MS[attempt];
    if (retryDelay !== undefined) await waitForRetry(retryDelay);
  }

  throw lastFailure ?? new Error('服务暂时不可用，请稍后重试');
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const isFormData = init?.body instanceof FormData;
  const response = await fetch(apiUrl(path), {
    ...init,
    headers:
      init?.body && !isFormData
        ? { 'Content-Type': 'application/json', ...init.headers }
        : init?.headers,
  });
  if (!response.ok) {
    let message = `请求失败 (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      // Keep the HTTP status message when a server does not return JSON.
    }
    throw new ApiError(message, response.status);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

async function streamChat(
  payload: {
    question: string;
    images: string[];
    session_id?: string;
    user_id?: string;
  },
  onEvent: (event: RuntimeEvent) => void,
): Promise<AgentResponse> {
  const response = await openChatStream(payload);
  if (!response.body) throw new Error('服务未返回运行事件流');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalResponse: AgentResponse | undefined;

  function consume(line: string) {
    const trimmed = line.trim();
    if (!trimmed) return;
    const event = JSON.parse(trimmed) as RuntimeEvent;
    onEvent(event);
    if (event.type === 'run.failed')
      throw new Error(event.summary || 'Agent 运行失败');
    if (event.type === 'run.completed' && event.payload.response)
      finalResponse = event.payload.response;
  }

  let streamDone = false;
  while (!streamDone) {
    const chunk = await reader.read();
    streamDone = chunk.done;
    buffer += decoder.decode(chunk.value, { stream: !streamDone });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    lines.forEach(consume);
  }
  consume(buffer);
  if (!finalResponse) throw new Error('Agent 运行结束，但没有最终回答');
  return finalResponse;
}

export const api = {
  providers: () => request<Provider[]>('/api/providers'),
  models: () => request<ModelConfig[]>('/api/models'),
  createModel: (payload: Record<string, unknown>) =>
    request<ModelConfig>('/api/models', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  updateModel: (id: string, payload: { kind: string }) =>
    request<ModelConfig>(`/api/models/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  setDefaultModel: (id: string) =>
    request<ModelConfig>(`/api/models/${id}/default`, { method: 'POST' }),
  testModel: (id: string) =>
    request<{ health: string; latency_ms: number | null; message: string }>(
      `/api/models/${id}/test`,
      { method: 'POST' },
    ),
  deleteModel: (id: string) =>
    request<void>(`/api/models/${id}`, { method: 'DELETE' }),
  chat: (payload: {
    question: string;
    images: string[];
    session_id?: string;
    user_id?: string;
  }) =>
    request<AgentResponse>('/api/chat', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  streamChat,
  conversations: (userId: string, offset = 0, limit = 100) =>
    request<ConversationSummary[]>(
      `/api/conversations?user_id=${encodeURIComponent(userId)}&offset=${offset}&limit=${limit}`,
    ),
  conversation: (conversationId: string, userId: string) =>
    request<ConversationDetail>(
      `/api/conversations/${encodeURIComponent(conversationId)}?user_id=${encodeURIComponent(userId)}`,
    ),
  createConversation: (
    userId: string,
    payload: { id?: string; title?: string } = {},
  ) =>
    request<ConversationSummary>(
      `/api/conversations?user_id=${encodeURIComponent(userId)}`,
      { method: 'POST', body: JSON.stringify(payload) },
    ),
  renameConversation: (conversationId: string, userId: string, title: string) =>
    request<ConversationSummary>(
      `/api/conversations/${encodeURIComponent(conversationId)}?user_id=${encodeURIComponent(userId)}`,
      { method: 'PATCH', body: JSON.stringify({ title }) },
    ),
  deleteConversation: (conversationId: string, userId: string) =>
    request<void>(
      `/api/conversations/${encodeURIComponent(conversationId)}?user_id=${encodeURIComponent(userId)}`,
      { method: 'DELETE' },
    ),
  readiness: () => request<RuntimeReadiness>('/api/readiness'),
  session: (sessionId: string, userId?: string) =>
    request<SessionMemory>(
      `/api/sessions/${encodeURIComponent(sessionId)}${
        userId ? `?user_id=${encodeURIComponent(userId)}` : ''
      }`,
    ),
  deleteSession: (sessionId: string, userId?: string) =>
    request<void>(
      `/api/sessions/${encodeURIComponent(sessionId)}${
        userId ? `?user_id=${encodeURIComponent(userId)}` : ''
      }`,
      { method: 'DELETE' },
    ),
  capabilities: () => request<RuntimeCapabilities>('/api/capabilities'),
  feedback: (payload: {
    request_id: string;
    rating: 'up' | 'down';
    category: string;
    comment?: string;
  }) =>
    request<Record<string, unknown>>('/api/feedback', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  agents: () => request<AgentDefinition[]>('/api/agents'),
  skills: () => request<SkillDefinition[]>('/api/skills'),
  tools: () => request<ToolDefinition[]>('/api/tools'),
  mcpServers: () => request<McpServer[]>('/api/mcp/servers'),
  mcpSearch: (payload: {
    dataset_ids: string[];
    query: string;
    top_n?: number;
    use_rerank?: boolean;
  }) =>
    request<{
      tool: string;
      is_error: boolean;
      content: Array<{ type: string; text: string }>;
      structured_content: RetrievalExplanation;
    }>('/api/mcp/tools/knowledge.search', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  traces: (userId: string) =>
    request<AgentTrace[]>(`/api/traces?user_id=${encodeURIComponent(userId)}`),
  files: () => request<FileAsset[]>('/api/files'),
  uploadFile: (file: File) => {
    const body = new FormData();
    body.append('file', file);
    return request<FileAsset>('/api/files', { method: 'POST', body });
  },
  datasets: () => request<Dataset[]>('/api/datasets'),
  dataset: (id: string) => request<Dataset>(`/api/datasets/${id}`),
  updateDataset: (id: string, payload: Record<string, unknown>) =>
    request<Dataset>(`/api/datasets/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  createDataset: (payload: {
    name: string;
    description?: string;
    parser_profile?: string;
  }) =>
    request<Dataset>('/api/datasets', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  documents: (datasetId: string) =>
    request<KnowledgeDocument[]>(`/api/datasets/${datasetId}/documents`),
  indexManifest: (datasetId: string) =>
    request<IndexManifest>(
      `/api/datasets/${encodeURIComponent(datasetId)}/index-manifest`,
    ),
  imageChunks: (datasetId: string, query?: string, limit = 100, offset = 0) =>
    request<ImageChunk[]>(
      `/api/datasets/${encodeURIComponent(datasetId)}/image-chunks?limit=${limit}&offset=${offset}${
        query ? `&query=${encodeURIComponent(query)}` : ''
      }`,
    ),
  imageChunk: (imageChunkId: string) =>
    request<ImageChunk>(
      `/api/image-chunks/${encodeURIComponent(imageChunkId)}`,
    ),
  vectorMap: (datasetId: string) =>
    request<VectorMapResponse>(
      `/api/datasets/${encodeURIComponent(datasetId)}/vector-map`,
    ),
  rebuildVectorMap: (datasetId: string) =>
    request<VectorMapResponse>(
      `/api/datasets/${encodeURIComponent(datasetId)}/vector-map/rebuild`,
      {
        method: 'POST',
      },
    ),
  linkDocument: (
    datasetId: string,
    payload: { file_id: string; parser_profile?: string },
  ) =>
    request<KnowledgeDocument>(`/api/datasets/${datasetId}/documents`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  parseDocument: (documentId: string) =>
    request<ParsingJob>(`/api/documents/${documentId}/parse`, {
      method: 'POST',
    }),
  publishDataset: (
    datasetId: string,
    indexVersion: string,
    evaluationRunId?: string,
  ) =>
    request<Dataset>(`/api/datasets/${datasetId}/publish`, {
      method: 'POST',
      body: JSON.stringify({
        index_version: indexVersion,
        evaluation_run_id: evaluationRunId,
      }),
    }),
  chunks: (documentId: string) =>
    request<ChunkCollection>(`/api/documents/${documentId}/chunks`),
  updateChunk: (chunkId: string, payload: Record<string, unknown>) =>
    request<ChildChunk>(`/api/chunks/${chunkId}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  retrievalTest: (payload: {
    dataset_ids: string[];
    query: string;
    top_n?: number;
    min_score?: number;
    use_rerank?: boolean;
    profile_id?: string;
  }) =>
    request<RetrievalExplanation>('/api/retrieval/test', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  retrievalProfiles: () =>
    request<RetrievalProfile[]>('/api/retrieval/profiles'),
  createRetrievalProfile: (payload: Record<string, unknown>) =>
    request<RetrievalProfile>('/api/retrieval/profiles', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  updateRetrievalProfile: (id: string, payload: Record<string, unknown>) =>
    request<RetrievalProfile>(`/api/retrieval/profiles/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  evalCases: () => request<EvalCase[]>('/api/evaluations/cases'),
  createEvalCase: (payload: Record<string, unknown>) =>
    request<EvalCase>('/api/evaluations/cases', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  evalRuns: () => request<EvalRun[]>('/api/evaluations/runs'),
  createEvalRun: (payload: { candidate_version: string; case_ids: string[] }) =>
    request<EvalRun>('/api/evaluations/runs', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  approveEvalRun: (id: string) =>
    request<EvalRun>(`/api/evaluations/runs/${id}/approve`, { method: 'POST' }),
};
