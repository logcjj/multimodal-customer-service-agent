import { api } from '@/aka/api/client';
import type { AgentResponse } from '@/aka/api/types';
import { MessageContent } from '@/aka/components/message-content';
import { getBrowserClientId } from '@/aka/lib/conversations';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import {
  Braces,
  CheckCircle2,
  Clock3,
  Database,
  ImagePlus,
  Play,
  RotateCcw,
  Route,
  X,
  XCircle,
} from 'lucide-react';
import { ChangeEvent, FormEvent, useRef, useState } from 'react';

const DEFAULT_QUESTION = '空气净化器滤网如何清洁？';
const MAX_IMAGE_COUNT = 3;

type SelectedImage = {
  name: string;
  data: string;
};

function readImages(files: File[]) {
  return Promise.all(
    files.map(
      (file) =>
        new Promise<SelectedImage>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () =>
            resolve({ name: file.name, data: String(reader.result) });
          reader.onerror = reject;
          reader.readAsDataURL(file);
        }),
    ),
  );
}

function formatLatency(value: number | null) {
  if (value === null) return '--';
  return `${value.toLocaleString('zh-CN')} ms`;
}

function ResponseMetric({
  label,
  value,
  icon: Icon,
}: {
  label: string;
  value: string;
  icon: typeof Clock3;
}) {
  return (
    <div className="min-w-0 rounded-md border border-border-button bg-bg-card px-3 py-2.5">
      <div className="flex items-center gap-1.5 text-[11px] text-text-secondary">
        <Icon className="size-3.5" />
        <span>{label}</span>
      </div>
      <div className="mt-1 truncate text-sm font-medium" title={value}>
        {value}
      </div>
    </div>
  );
}

export default function ApiTestPage() {
  const [userId] = useState(() =>
    getBrowserClientId(
      typeof window === 'undefined' ? null : window.localStorage,
    ),
  );
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [sessionId, setSessionId] = useState('');
  const [images, setImages] = useState<SelectedImage[]>([]);
  const [response, setResponse] = useState<AgentResponse | null>(null);
  const [error, setError] = useState('');
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const imageInput = useRef<HTMLInputElement>(null);

  async function selectImages(event: ChangeEvent<HTMLInputElement>) {
    const input = event.currentTarget;
    const files = Array.from(input.files ?? []);
    input.value = '';

    if (!files.length) return;

    const remaining = MAX_IMAGE_COUNT - images.length;
    if (remaining <= 0) {
      setError(`最多可添加 ${MAX_IMAGE_COUNT} 张图片`);
      return;
    }

    try {
      const selectedImages = await readImages(files.slice(0, remaining));
      setImages((current) => [...current, ...selectedImages].slice(0, MAX_IMAGE_COUNT));
      setError('');
    } catch {
      setError('图片读取失败，请重新选择');
    }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalizedQuestion = question.trim();
    if (!normalizedQuestion || busy) return;
    if (!userId) {
      setError('无法读取测试用户标识，请刷新后重试');
      return;
    }

    setBusy(true);
    setError('');
    setResponse(null);
    setLatencyMs(null);
    const startedAt = performance.now();

    try {
      const result = await api.chat({
        question: normalizedQuestion,
        images: images.map((image) => image.data),
        user_id: userId,
        ...(sessionId.trim() ? { session_id: sessionId.trim() } : {}),
      });
      setResponse(result);
      setSessionId(result.session_id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '接口调用失败');
    } finally {
      setLatencyMs(Math.round(performance.now() - startedAt));
      setBusy(false);
    }
  }

  function reset() {
    setQuestion(DEFAULT_QUESTION);
    setSessionId('');
    setImages([]);
    setResponse(null);
    setError('');
    setLatencyMs(null);
    if (imageInput.current) imageInput.current.value = '';
  }

  return (
    <div className="size-full overflow-auto px-5 py-6 lg:px-8 lg:py-8">
      <div className="mx-auto max-w-[1500px] space-y-5">
        <header className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="mb-1 flex items-center gap-2 text-xs text-text-secondary">
              <span className="size-2 rounded-full bg-state-success" />
              Customer service API
            </div>
            <h1 className="text-2xl font-semibold">接口测试</h1>
            <p className="mt-1 text-sm text-text-secondary">
              调用当前客服问答服务并查看真实响应
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="secondary">
              <Braces className="mr-1 size-3" />
              JSON
            </Badge>
            <Badge variant="success">
              <CheckCircle2 className="mr-1 size-3" />
              POST /api/chat
            </Badge>
          </div>
        </header>

        <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,0.85fr)_minmax(0,1.25fr)]">
          <Card>
            <CardHeader className="p-5">
              <CardTitle as="h2" className="text-base">
                请求参数
              </CardTitle>
              <CardDescription>客服问答接口</CardDescription>
            </CardHeader>
            <CardContent className="p-5 pt-0">
              <div className="mb-5 flex items-center gap-3 rounded-md border border-border-button bg-bg-card px-3 py-2.5">
                <Badge variant="success">POST</Badge>
                <code className="min-w-0 truncate text-sm">/api/chat</code>
              </div>

              <form className="space-y-4" onSubmit={(event) => void submit(event)}>
                <label className="block text-sm font-medium" htmlFor="api-test-question">
                  问题
                  <Textarea
                    id="api-test-question"
                    className="mt-2 text-sm leading-6"
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    placeholder="输入需要测试的问题"
                    rows={7}
                    resize="vertical"
                  />
                </label>

                <label className="block text-sm font-medium" htmlFor="api-test-session-id">
                  Session ID
                  <Input
                    id="api-test-session-id"
                    className="mt-2"
                    value={sessionId}
                    onChange={(event) => setSessionId(event.target.value)}
                    placeholder="留空时创建新的测试会话"
                  />
                </label>

                <section
                  className="rounded-md border border-border-button bg-bg-card p-3"
                  aria-labelledby="api-test-images-label"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div id="api-test-images-label" className="text-sm font-medium">
                        图片（可选）
                      </div>
                      <p className="mt-1 text-xs text-text-secondary">
                        支持 PNG、JPG/JPEG、WebP，最多 {MAX_IMAGE_COUNT} 张
                      </p>
                    </div>
                    <input
                      ref={imageInput}
                      hidden
                      type="file"
                      aria-label="选择图片文件"
                      accept="image/png,image/jpeg,image/webp"
                      multiple
                      onChange={(event) => void selectImages(event)}
                    />
                    <Button
                      type="button"
                      variant="outline"
                      size="icon"
                      aria-label="添加图片"
                      title="添加图片"
                      disabled={images.length >= MAX_IMAGE_COUNT || busy}
                      onClick={() => imageInput.current?.click()}
                    >
                      <ImagePlus className="size-4" />
                    </Button>
                  </div>

                  {images.length ? (
                    <div className="mt-3 grid gap-2 sm:grid-cols-2">
                      {images.map((image, index) => (
                        <div
                          key={`${image.name}-${index}`}
                          className="flex min-w-0 items-center gap-2 rounded-md border border-border-button bg-bg-base p-2"
                        >
                          <img
                            src={image.data}
                            alt={image.name}
                            className="size-12 shrink-0 rounded object-cover"
                          />
                          <span className="min-w-0 flex-1 truncate text-xs" title={image.name}>
                            {image.name}
                          </span>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon-xs"
                            aria-label={`移除 ${image.name}`}
                            title={`移除 ${image.name}`}
                            disabled={busy}
                            onClick={() =>
                              setImages((current) =>
                                current.filter((_, itemIndex) => itemIndex !== index),
                              )
                            }
                          >
                            <X className="size-3.5" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  ) : null}

                  <div className="mt-3 text-xs text-text-secondary">
                    已添加 {images.length}/{MAX_IMAGE_COUNT} 张
                  </div>
                </section>

                <div className="flex flex-wrap gap-2 pt-1">
                  <Button
                    type="submit"
                    loading={busy}
                    disabled={!question.trim() || !userId}
                  >
                    <Play className="size-4" />
                    发送测试
                  </Button>
                  <Button type="button" variant="outline" onClick={reset} disabled={busy}>
                    <RotateCcw className="size-4" />
                    重置
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex-row items-start justify-between space-y-0 p-5">
              <div>
                <CardTitle as="h2" className="text-base">
                  接口响应
                </CardTitle>
                <CardDescription>
                  {response
                    ? `请求 ${response.request_id}`
                    : error
                      ? '请求未成功完成'
                      : '等待请求'}
                </CardDescription>
              </div>
              {response ? (
                <Badge variant="success">
                  <CheckCircle2 className="mr-1 size-3" />
                  200 OK
                </Badge>
              ) : error ? (
                <Badge variant="destructive">
                  <XCircle className="mr-1 size-3" />
                  请求失败
                </Badge>
              ) : null}
            </CardHeader>
            <CardContent className="space-y-4 p-5 pt-0">
              {response ? (
                <>
                  <section className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
                    <ResponseMetric
                      label="调用耗时"
                      value={formatLatency(latencyMs)}
                      icon={Clock3}
                    />
                    <ResponseMetric
                      label="服务耗时"
                      value={formatLatency(response.trace.total_latency_ms)}
                      icon={Route}
                    />
                    <ResponseMetric
                      label="检索证据"
                      value={`${response.citations.length} 条`}
                      icon={Database}
                    />
                    <ResponseMetric
                      label="关联图片"
                      value={`${response.assets.length} 张`}
                      icon={Braces}
                    />
                  </section>

                  <section className="rounded-md border border-border-button bg-bg-card p-4">
                    <div className="mb-3 flex flex-wrap items-center gap-2">
                      <Badge variant="secondary">{response.route}</Badge>
                      <Badge
                        variant={
                          response.verification.passed ? 'success' : 'destructive'
                        }
                      >
                        {response.verification.passed ? '质量校验通过' : '质量校验待处理'}
                      </Badge>
                      <span className="text-xs text-text-secondary">
                        置信度 {Math.round(response.verification.confidence * 100)}%
                      </span>
                    </div>
                    <div className="text-sm leading-7">
                      <MessageContent assets={response.assets}>
                        {response.answer}
                      </MessageContent>
                    </div>
                  </section>

                  {response.citations.length ? (
                    <section className="rounded-md border border-border-button bg-bg-card p-4">
                      <div className="mb-2 text-xs font-semibold">检索证据</div>
                      <div className="space-y-2">
                        {response.citations.map((citation, index) => (
                          <div
                            key={citation.evidence_id}
                            className="flex min-w-0 items-center gap-2 text-xs"
                          >
                            <span className="shrink-0 text-accent-primary">[{index + 1}]</span>
                            <span className="truncate">{citation.title}</span>
                            <span className="ml-auto shrink-0 text-text-secondary">
                              {citation.locator_label ?? `第 ${citation.page_start ?? '--'} 页`}
                            </span>
                          </div>
                        ))}
                      </div>
                    </section>
                  ) : null}

                  <section className="overflow-hidden rounded-md border border-border-button bg-bg-card">
                    <div className="border-b border-border-button px-4 py-2 text-xs font-semibold">
                      原始 JSON
                    </div>
                    <pre className="max-h-96 overflow-auto p-4 text-xs leading-5">
                      {JSON.stringify(response, null, 2)}
                    </pre>
                  </section>
                </>
              ) : error ? (
                <div className="rounded-md border border-state-error/40 bg-state-error/5 p-4 text-sm text-state-error" role="alert">
                  {error}
                </div>
              ) : (
                <div className="grid min-h-96 place-items-center rounded-md border border-dashed border-border-button px-5 text-center text-sm text-text-secondary">
                  <div>
                    <Braces className="mx-auto mb-3 size-6" />
                    暂无接口响应
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
