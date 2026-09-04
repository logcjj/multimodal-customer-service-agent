import { api } from '@/aka/api/client';
import type {
  AgentResponse,
  ConversationDetail,
  ConversationSummary,
  Evidence,
  RuntimeEvent,
  RuntimeReadiness,
} from '@/aka/api/types';
import {
  actualModelName,
  AgentRunInspector,
} from '@/aka/components/agent-run-inspector';
import { ConversationSidebar } from '@/aka/components/conversation-sidebar';
import { EvidenceDrawer } from '@/aka/components/evidence-drawer';
import { MessageContent } from '@/aka/components/message-content';
import { RouteSummary } from '@/aka/components/route-summary';
import {
  ACTIVE_CONVERSATION_STORAGE_KEY,
  ConversationEntry,
  getOrCreateClientId,
  turnsToEntries,
} from '@/aka/lib/conversations';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';
import {
  Bot,
  ImagePlus,
  LoaderCircle,
  PanelLeftOpen,
  PanelRightOpen,
  Send,
  ShieldCheck,
  ThumbsDown,
  ThumbsUp,
  User,
  X,
} from 'lucide-react';
import { ChangeEvent, FormEvent, useEffect, useRef, useState } from 'react';

async function readImages(files: FileList | null) {
  if (!files) return [];
  return Promise.all(
    Array.from(files)
      .slice(0, 3)
      .map(
        (file) =>
          new Promise<{ name: string; data: string }>((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () =>
              resolve({ name: file.name, data: String(reader.result) });
            reader.onerror = reject;
            reader.readAsDataURL(file);
          }),
      ),
  );
}

const quickQuestions = [
  '洗衣机出现 E03 怎么处理？',
  '空气净化器滤网如何清洁？',
  '设备故障，修不好可以退货吗？',
];

export function answerSourceLabel(response: AgentResponse): string {
  if (response.used_legacy) return '守护链路';
  switch (response.routing?.final_route) {
    case 'technical_knowledge':
      return '知识库证据';
    case 'customer_service':
      return '客服政策';
    case 'mixed':
      return '多智能体证据';
    case 'evidence_clarification':
      return 'Evidence Gap';
    case 'general_llm':
      return 'General LLM';
    case 'safe_handoff':
      return '安全转人工';
    case 'general_unavailable':
      return '模型不可用';
    default:
      return response.citations.length ? '知识库证据' : '智能体回答';
  }
}

export default function ChatPage() {
  const [question, setQuestion] = useState('');
  const [images, setImages] = useState<Array<{ name: string; data: string }>>(
    [],
  );
  const [entries, setEntries] = useState<ConversationEntry[]>([]);
  const [activeResponse, setActiveResponse] = useState<AgentResponse | null>(
    null,
  );
  const [events, setEvents] = useState<RuntimeEvent[]>([]);
  const [readiness, setReadiness] = useState<RuntimeReadiness | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [clientId] = useState(() =>
    globalThis.localStorage
      ? getOrCreateClientId(globalThis.localStorage)
      : `anon-${Date.now().toString(36)}`,
  );
  const [sending, setSending] = useState(false);
  const [conversationMutationPending, setConversationMutationPending] =
    useState(false);
  const [error, setError] = useState('');
  const [activeConversationId, setActiveConversationId] = useState<
    string | undefined
  >(
    () =>
      globalThis.localStorage?.getItem(ACTIVE_CONVERSATION_STORAGE_KEY) ||
      undefined,
  );
  const [historyOpen, setHistoryOpen] = useState(false);
  const [selectedEvidence, setSelectedEvidence] = useState<Evidence | null>(
    null,
  );
  const imageInput = useRef<HTMLInputElement>(null);
  const conversationRequestEpoch = useRef(0);
  const conversationInteractionDisabled =
    sending || conversationMutationPending;

  useEffect(() => {
    void api
      .readiness()
      .then(setReadiness)
      .catch(() => setReadiness(null));
  }, []);

  useEffect(() => {
    let cancelled = false;
    const requestEpoch = ++conversationRequestEpoch.current;
    void api
      .conversations(clientId)
      .then(async (items) => {
        if (cancelled || requestEpoch !== conversationRequestEpoch.current)
          return;
        setConversations(items);
        const stored = globalThis.localStorage?.getItem(
          ACTIVE_CONVERSATION_STORAGE_KEY,
        );
        const target =
          items.find((item) => item.id === stored)?.id ?? items[0]?.id;
        if (!target) {
          setActiveConversationId(undefined);
          setEntries([]);
          setActiveResponse(null);
          setEvents([]);
          globalThis.localStorage?.removeItem(ACTIVE_CONVERSATION_STORAGE_KEY);
          return;
        }
        const detail = await api.conversation(target, clientId);
        if (cancelled || requestEpoch !== conversationRequestEpoch.current)
          return;
        restoreConversation(detail);
      })
      .catch(() => {
        if (!cancelled && requestEpoch === conversationRequestEpoch.current) {
          setConversations([]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [clientId]);

  function restoreConversation(detail: ConversationDetail) {
    const restoredEntries = turnsToEntries(detail.turns);
    const latestResponse = [...restoredEntries]
      .reverse()
      .find((entry) => entry.response)?.response;
    setActiveConversationId(detail.id);
    setEntries(restoredEntries);
    setActiveResponse(latestResponse ?? null);
    setEvents([]);
    setError('');
    globalThis.localStorage?.setItem(
      ACTIVE_CONVERSATION_STORAGE_KEY,
      detail.id,
    );
  }

  async function refreshConversations() {
    try {
      setConversations(await api.conversations(clientId));
    } catch {
      // A completed answer remains usable even if the history refresh fails.
    }
  }

  function resetConversationDraft() {
    conversationRequestEpoch.current += 1;
    setActiveConversationId(undefined);
    setEntries([]);
    setActiveResponse(null);
    setEvents([]);
    setImages([]);
    setError('');
    setHistoryOpen(false);
    globalThis.localStorage?.removeItem(ACTIVE_CONVERSATION_STORAGE_KEY);
  }

  function startNewConversation() {
    if (conversationInteractionDisabled) return;
    resetConversationDraft();
  }

  async function selectConversation(conversationId: string) {
    if (
      conversationInteractionDisabled ||
      conversationId === activeConversationId
    )
      return;
    const requestEpoch = ++conversationRequestEpoch.current;
    try {
      const detail = await api.conversation(conversationId, clientId);
      if (requestEpoch !== conversationRequestEpoch.current) return;
      restoreConversation(detail);
      setHistoryOpen(false);
    } catch (reason) {
      if (requestEpoch !== conversationRequestEpoch.current) return;
      setError(reason instanceof Error ? reason.message : '对话加载失败');
    }
  }

  async function renameConversation(conversationId: string, title: string) {
    if (conversationInteractionDisabled) return;
    setConversationMutationPending(true);
    try {
      const updated = await api.renameConversation(
        conversationId,
        clientId,
        title,
      );
      setConversations((items) =>
        items.map((item) => (item.id === conversationId ? updated : item)),
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '重命名失败');
    } finally {
      setConversationMutationPending(false);
    }
  }

  async function deleteConversation(conversationId: string) {
    if (conversationInteractionDisabled) return;
    const deletingActiveConversation = conversationId === activeConversationId;
    setConversationMutationPending(true);
    try {
      await api.deleteConversation(conversationId, clientId);
      setConversations((items) =>
        items.filter((item) => item.id !== conversationId),
      );
      if (deletingActiveConversation) resetConversationDraft();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '删除失败');
    } finally {
      setConversationMutationPending(false);
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || conversationInteractionDisabled) return;
    conversationRequestEpoch.current += 1;

    setEntries((current) => [
      ...current,
      { id: `user-${Date.now()}`, role: 'user', text },
    ]);
    setQuestion('');
    setEvents([]);
    setActiveResponse(null);
    setSending(true);
    setError('');
    const temporaryAssistantId = `assistant-stream-${Date.now()}`;

    const updateTemporaryAssistant = (
      updater: (currentText: string) => string,
      response?: AgentResponse,
    ) => {
      setEntries((current) => {
        const index = current.findIndex(
          (item) => item.id === temporaryAssistantId,
        );
        if (index < 0) {
          return [
            ...current,
            {
              id: temporaryAssistantId,
              role: 'assistant',
              text: updater(''),
              response,
            },
          ];
        }
        return current.map((item, itemIndex) =>
          itemIndex === index
            ? {
                ...item,
                text: updater(item.text),
                response: response ?? item.response,
              }
            : item,
        );
      });
    };

    try {
      const response = await api.streamChat(
        {
          question: text,
          images: images.map((image) => image.data),
          session_id: activeConversationId,
          user_id: clientId,
        },
        (event) => {
          setEvents((current) => [...current, event]);
          if (event.type === 'answer.delta' && event.payload.delta) {
            updateTemporaryAssistant(
              (currentText) => currentText + event.payload.delta,
            );
          } else if (event.type === 'answer.revised' && event.payload.answer) {
            updateTemporaryAssistant(() => event.payload.answer ?? '');
          } else if (event.type === 'run.completed' && event.payload.response) {
            const finalResponse = event.payload.response;
            updateTemporaryAssistant(() => finalResponse.answer, finalResponse);
            setActiveResponse(finalResponse);
          }
        },
      );
      setActiveConversationId(response.session_id);
      globalThis.localStorage?.setItem(
        ACTIVE_CONVERSATION_STORAGE_KEY,
        response.session_id,
      );
      setActiveResponse(response);
      setEntries((current) => {
        const exists = current.some((item) => item.id === temporaryAssistantId);
        if (!exists) {
          return [
            ...current,
            {
              id: response.request_id,
              role: 'assistant',
              text: response.answer,
              response,
            },
          ];
        }
        return current.map((item) =>
          item.id === temporaryAssistantId
            ? {
                id: response.request_id,
                role: 'assistant' as const,
                text: response.answer,
                response,
              }
            : item,
        );
      });
      setImages([]);
      await refreshConversations();
    } catch (reason) {
      setEntries((current) =>
        current.filter((item) => item.id !== temporaryAssistantId),
      );
      setActiveResponse(null);
      setError(reason instanceof Error ? reason.message : '请求失败');
    } finally {
      setSending(false);
    }
  }

  async function selectImages(event: ChangeEvent<HTMLInputElement>) {
    try {
      setImages(await readImages(event.target.files));
    } catch {
      setError('图片读取失败');
    }
  }

  const currentEvent = events.at(-1);
  const inspector = (
    <AgentRunInspector
      trace={activeResponse?.trace ?? null}
      events={events}
      running={sending}
      readiness={readiness}
    />
  );
  const mobileInspector = (
    <AgentRunInspector
      trace={activeResponse?.trace ?? null}
      events={events}
      running={sending}
      readiness={readiness}
      reserveCloseButtonSpace
    />
  );
  const conversationSidebar = (
    <ConversationSidebar
      items={conversations}
      activeId={activeConversationId}
      disabled={conversationInteractionDisabled}
      onNew={startNewConversation}
      onSelect={(conversationId) => void selectConversation(conversationId)}
      onRename={renameConversation}
      onDelete={deleteConversation}
    />
  );
  const activeTitle =
    conversations.find((item) => item.id === activeConversationId)?.title ??
    '新对话';
  const displayedModel =
    actualModelName(activeResponse?.trace ?? null, events) ??
    readiness?.llm_model;

  return (
    <div className="aka-chat-layout grid size-full min-w-0 grid-cols-1 xl:grid-cols-[248px_minmax(0,1fr)_360px]">
      <aside className="aka-chat-history hidden min-h-0 border-r border-border-button xl:block">
        {conversationSidebar}
      </aside>

      <section className="aka-chat-panel grid min-h-0 min-w-0 grid-rows-[auto_auto_minmax(0,1fr)_auto] bg-bg-base">
        <header className="aka-chat-header flex min-w-0 items-center justify-between gap-3 border-b border-border-button px-4 py-3 lg:px-6">
          <div className="min-w-0">
            <h1 className="truncate text-base font-semibold">{activeTitle}</h1>
            <div className="mt-1 flex items-center gap-2 text-xs text-text-secondary">
              <span className="size-1.5 rounded-full bg-state-success" />
              在线编排
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="aka-chat-history-trigger xl:hidden">
              <Sheet open={historyOpen} onOpenChange={setHistoryOpen}>
                <SheetTrigger asChild>
                  <Button
                    variant="outline"
                    size="icon"
                    aria-label="查看对话历史"
                    disabled={conversationInteractionDisabled}
                  >
                    <PanelLeftOpen className="size-4" />
                  </Button>
                </SheetTrigger>
                <SheetContent
                  side="left"
                  className="w-[min(88vw,320px)] max-w-none rounded-none p-0"
                  closeIcon={false}
                >
                  <SheetTitle className="sr-only">对话历史</SheetTitle>
                  <SheetDescription className="sr-only">
                    新建、切换、重命名或删除对话
                  </SheetDescription>
                  <ConversationSidebar
                    items={conversations}
                    activeId={activeConversationId}
                    disabled={conversationInteractionDisabled}
                    onNew={startNewConversation}
                    onSelect={(conversationId) =>
                      void selectConversation(conversationId)
                    }
                    onRename={renameConversation}
                    onDelete={deleteConversation}
                    onClose={() => setHistoryOpen(false)}
                  />
                </SheetContent>
              </Sheet>
            </div>
            <Badge
              variant={displayedModel ? 'success' : 'destructive'}
              className="hidden max-w-[36vw] min-w-0 truncate sm:inline-flex sm:max-w-52 xl:max-w-64"
              title={displayedModel ?? undefined}
            >
              {displayedModel ? `${displayedModel} · LLM` : '确定性降级'}
            </Badge>
            <div className="xl:hidden">
              <Sheet>
                <SheetTrigger asChild>
                  <Button
                    variant="outline"
                    size="icon"
                    aria-label="查看 Agent 执行轨迹"
                  >
                    <PanelRightOpen className="size-4" />
                  </Button>
                </SheetTrigger>
                <SheetContent className="w-[min(94vw,420px)] max-w-none rounded-none p-0">
                  <SheetTitle className="sr-only">Agent 执行轨迹</SheetTitle>
                  <SheetDescription className="sr-only">
                    查看动态组队、模型、检索、验证与 Trace 事件
                  </SheetDescription>
                  {mobileInspector}
                </SheetContent>
              </Sheet>
            </div>
          </div>
        </header>

        <div className="min-w-0">
          {sending && !activeResponse ? (
            <RouteSummary response={null} events={events} running />
          ) : null}
        </div>

        <div className="aka-chat-messages min-h-0 overflow-auto">
          {entries.length === 0 ? (
            <div className="grid min-h-full place-items-center px-5 py-10 text-center">
              <div className="max-w-xl">
                <div className="mx-auto grid size-12 place-items-center rounded-xl border border-border-button bg-bg-card">
                  <Bot className="size-5" />
                </div>
                <h2 className="mt-4 text-xl font-semibold">多模态客服智能体系统</h2>
                <p className="mt-2 text-sm text-text-secondary">
                  输入产品、故障、错误码或售后问题，系统会动态选择 Agent
                  并引用说明书证据。
                </p>
                <div className="mt-5 flex flex-wrap justify-center gap-2 xl:hidden">
                  {quickQuestions.map((prompt) => (
                    <Button
                      key={prompt}
                      variant="outline"
                      size="sm"
                      onClick={() => setQuestion(prompt)}
                    >
                      {prompt}
                    </Button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-4xl px-4 py-6 lg:px-8">
              {entries.map((entry) => (
                <article
                  key={entry.id}
                  className="grid grid-cols-[32px_minmax(0,1fr)] gap-3 border-b border-border-button py-5 last:border-b-0"
                >
                  <div className="grid size-8 place-items-center rounded-md border border-border-button bg-bg-card">
                    {entry.role === 'assistant' ? (
                      <Bot className="size-4" />
                    ) : (
                      <User className="size-4" />
                    )}
                  </div>
                  <div className="min-w-0">
                    <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                      <strong>
                        {entry.role === 'assistant' ? '客服智能体' : '用户'}
                      </strong>
                      {entry.response ? (
                        <span className="inline-flex items-center gap-1 text-state-success">
                          <ShieldCheck className="size-3.5" />
                          置信度{' '}
                          {Math.round(
                            entry.response.verification.confidence * 100,
                          )}
                          %
                        </span>
                      ) : null}
                    </div>
                    <div
                      className="text-sm leading-7"
                      aria-live={
                        entry.id.startsWith('assistant-stream-')
                          ? 'polite'
                          : undefined
                      }
                      aria-atomic={
                        entry.id.startsWith('assistant-stream-') || undefined
                      }
                    >
                      <MessageContent assets={entry.response?.assets}>
                        {entry.text}
                      </MessageContent>
                    </div>

                    {entry.response ? (
                      <div className="mt-4 overflow-hidden rounded-md border border-border-button">
                        <RouteSummary response={entry.response} events={[]} />
                      </div>
                    ) : null}

                    {entry.response?.citations.length ? (
                      <div className="mt-4 space-y-2">
                        {entry.response.citations.map((citation, index) => (
                          <Button
                            key={citation.evidence_id}
                            variant="outline"
                            size="auto"
                            className="h-auto w-full justify-between whitespace-normal px-3 py-2.5 text-left"
                            aria-label={`引用 ${index + 1}：${citation.title}`}
                            onClick={() => setSelectedEvidence(citation)}
                          >
                            <span className="min-w-0 truncate text-xs">
                              <b className="mr-2 text-accent-primary">
                                [{index + 1}]
                              </b>
                              {citation.title}
                            </span>
                            <span className="ml-3 shrink-0 text-[10px] text-text-secondary">
                              {citation.locator_label ??
                                `第 ${citation.page_start ?? '--'} 页`}{' '}
                              · {citation.score?.toFixed(4) ?? '--'}
                            </span>
                          </Button>
                        ))}
                      </div>
                    ) : null}

                    {entry.response ? (
                      <div className="mt-3 flex items-center gap-1 text-xs text-text-secondary">
                        <Button
                          variant="ghost"
                          size="icon-xs"
                          aria-label="回答有帮助"
                          onClick={() =>
                            void api.feedback({
                              request_id: entry.response!.request_id,
                              rating: 'up',
                              category: 'answer',
                            })
                          }
                        >
                          <ThumbsUp className="size-3.5" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon-xs"
                          aria-label="回答需改进"
                          onClick={() =>
                            void api.feedback({
                              request_id: entry.response!.request_id,
                              rating: 'down',
                              category: 'answer',
                            })
                          }
                        >
                          <ThumbsDown className="size-3.5" />
                        </Button>
                        <span className="ml-2">
                          {answerSourceLabel(entry.response)}
                        </span>
                      </div>
                    ) : null}
                  </div>
                </article>
              ))}

              {sending ? (
                <div
                  role="status"
                  aria-live="polite"
                  aria-atomic="true"
                  className="mt-4 flex items-center gap-3 rounded-md border border-border-button bg-bg-card p-3"
                >
                  <LoaderCircle className="size-4 animate-spin text-accent-primary" />
                  <div className="min-w-0">
                    <span className="sr-only">
                      事件状态：
                      {currentEvent?.status === 'failed'
                        ? '失败'
                        : currentEvent?.status === 'completed'
                          ? '已完成'
                          : currentEvent?.status === 'skipped'
                            ? '已跳过'
                            : '执行中'}
                    </span>
                    <div className="truncate text-xs font-medium">
                      {currentEvent?.label ?? '建立 Agent 运行上下文'}
                    </div>
                    <div className="truncate text-[11px] text-text-secondary">
                      {currentEvent?.summary || '正在动态组建专业团队'}
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          )}
        </div>

        <form
          onSubmit={submit}
          className="aka-chat-composer border-t border-border-button bg-bg-base px-4 py-3 lg:px-6"
        >
          <div className="mx-auto max-w-4xl rounded-lg border border-border-button bg-bg-input p-2 shadow-sm focus-within:border-border-accent">
            {images.length ? (
              <div className="flex flex-wrap gap-2 border-b border-border-button p-2">
                {images.map((image, index) => (
                  <span
                    key={`${image.name}-${index}`}
                    className="inline-flex items-center gap-1 rounded bg-bg-card px-2 py-1 text-xs"
                  >
                    {image.name}
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon-xs"
                      aria-label={`移除 ${image.name}`}
                      onClick={() =>
                        setImages((current) =>
                          current.filter((_, itemIndex) => itemIndex !== index),
                        )
                      }
                    >
                      <X className="size-3" />
                    </Button>
                  </span>
                ))}
              </div>
            ) : null}
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              aria-label="输入客服问题"
              placeholder="输入产品、故障或售后问题"
              rows={3}
              className="aka-chat-textarea min-h-20 w-full resize-none bg-transparent px-2 py-2 text-sm outline-none placeholder:text-text-secondary"
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
            />
            <div className="flex items-center justify-between gap-2 px-1">
              <div className="flex items-center gap-2 text-xs text-text-secondary">
                <input
                  ref={imageInput}
                  hidden
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  multiple
                  onChange={(event) => void selectImages(event)}
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  aria-label="添加图片"
                  onClick={() => imageInput.current?.click()}
                >
                  <ImagePlus className="size-4" />
                </Button>
                <span>{images.length}/3</span>
              </div>
              <Button
                type="submit"
                size="icon"
                disabled={!question.trim() || conversationInteractionDisabled}
                aria-label="发送"
              >
                <Send className="size-4" />
              </Button>
            </div>
          </div>
          {error ? (
            <p
              className="mx-auto mt-2 max-w-4xl text-xs text-state-error"
              role="alert"
            >
              {error}
            </p>
          ) : null}
        </form>
      </section>

      <aside className="hidden min-h-0 border-l border-border-button bg-bg-card xl:block">
        {inspector}
      </aside>

      <EvidenceDrawer
        evidence={selectedEvidence}
        onClose={() => setSelectedEvidence(null)}
      />
    </div>
  );
}
