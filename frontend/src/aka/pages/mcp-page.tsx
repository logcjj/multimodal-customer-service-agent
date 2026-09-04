import { api } from '@/aka/api/client';
import type { McpServer } from '@/aka/api/types';
import { PageError, PageLoading } from '@/aka/components/page-state';
import { useApiResource } from '@/aka/hooks/use-api-resource';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Cable,
  Database,
  LockKeyhole,
  Play,
  Radio,
  Wrench,
} from 'lucide-react';
import { FormEvent, useCallback, useState } from 'react';

export default function McpPage() {
  const loader = useCallback(() => api.mcpServers(), []);
  const {
    data: servers,
    loading,
    error,
  } = useApiResource<McpServer[]>(loader, []);
  const [query, setQuery] = useState('空气净化器滤网如何清洁？');
  const [result, setResult] = useState('');
  const [busy, setBusy] = useState(false);

  async function debug(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    try {
      const response = await api.mcpSearch({
        dataset_ids: ['v6-manuals'],
        query,
        top_n: 3,
        use_rerank: true,
      });
      setResult(response.content[0]?.text ?? '工具已执行');
    } catch (reason) {
      setResult(reason instanceof Error ? reason.message : '工具调用失败');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="size-full overflow-auto px-5 py-6 lg:px-8 lg:py-8">
      <div className="mx-auto max-w-[1400px] space-y-5">
        <header className="flex items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold">MCP 服务</h1>
            <p className="mt-1 text-sm text-text-secondary">
              JSON-RPC 2.0 · Streamable HTTP · 真实可执行工具
            </p>
          </div>
          <Badge variant="success">
            <Radio className="mr-1 size-3" />
            服务就绪
          </Badge>
        </header>
        {loading ? <PageLoading label="加载 MCP 服务" /> : null}
        {!loading && error ? <PageError message={error} /> : null}
        {!loading && !error ? (
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
            <div className="space-y-4">
              {servers.map((server) => (
                <Card key={server.id}>
                  <CardHeader className="flex-row items-start justify-between space-y-0 p-5">
                    <div className="flex min-w-0 gap-3">
                      <div className="grid size-10 shrink-0 place-items-center rounded-md bg-bg-component">
                        <Cable className="size-5" />
                      </div>
                      <div className="min-w-0">
                        <CardTitle as="h2" className="truncate text-base">
                          {server.name}
                        </CardTitle>
                        <code className="mt-1 block truncate text-[10px] text-text-secondary">
                          {server.id}
                        </code>
                      </div>
                    </div>
                    <Badge
                      variant={
                        server.status === 'ready' ? 'success' : 'secondary'
                      }
                    >
                      {server.status}
                    </Badge>
                  </CardHeader>
                  <CardContent className="p-5 pt-0">
                    <p className="text-sm text-text-secondary">
                      {server.description}
                    </p>
                    <div className="mt-4 flex flex-wrap gap-2 text-xs">
                      <Badge variant="secondary">
                        <LockKeyhole className="mr-1 size-3" />
                        {server.mode === 'read-only' ? '只读' : server.mode}
                      </Badge>
                      <Badge variant="secondary">{server.transport}</Badge>
                    </div>
                    <div className="mt-5 grid gap-4 sm:grid-cols-2">
                      <section>
                        <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold">
                          <Wrench className="size-3.5" />
                          Tools
                        </h3>
                        {server.tools.map((tool) => (
                          <code
                            key={tool}
                            className="block rounded border border-border-button bg-bg-card px-3 py-2 text-xs"
                          >
                            {tool}
                          </code>
                        ))}
                      </section>
                      <section>
                        <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold">
                          <Database className="size-3.5" />
                          Resources
                        </h3>
                        {server.resources.length ? (
                          server.resources.map((resource) => (
                            <code
                              key={resource}
                              className="block rounded border border-border-button bg-bg-card px-3 py-2 text-xs"
                            >
                              {resource}
                            </code>
                          ))
                        ) : (
                          <div className="rounded border border-dashed border-border-button px-3 py-2 text-xs text-text-secondary">
                            无静态资源
                          </div>
                        )}
                      </section>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            <Card>
              <CardHeader className="p-5">
                <CardTitle as="h2" className="text-base">
                  Tool 调试
                </CardTitle>
                <p className="text-xs text-text-secondary">
                  调用只读 `knowledge.search`，不会生成回答。
                </p>
              </CardHeader>
              <CardContent className="p-5 pt-0">
                <form onSubmit={(event) => void debug(event)}>
                  <label className="block text-xs">
                    <span className="mb-1.5 block">Query</span>
                    <textarea
                      value={query}
                      onChange={(event) => setQuery(event.target.value)}
                      rows={4}
                      className="w-full resize-y rounded border border-border-button bg-bg-input p-3 text-sm outline-none"
                    />
                  </label>
                  <Button
                    type="submit"
                    className="mt-3"
                    loading={busy}
                    disabled={!query.trim()}
                  >
                    <Play className="size-4" />
                    调用工具
                  </Button>
                </form>
                <div className="mt-5 rounded-md border border-border-button bg-bg-card p-4">
                  <div className="mb-2 text-[10px] uppercase text-text-secondary">
                    Input Schema
                  </div>
                  <pre className="overflow-auto whitespace-pre-wrap text-[11px] leading-5">
                    {JSON.stringify(
                      {
                        dataset_ids: ['string'],
                        query: 'string',
                        top_n: '1..20',
                        use_rerank: 'boolean',
                      },
                      null,
                      2,
                    )}
                  </pre>
                </div>
                {result ? (
                  <div
                    className="mt-3 rounded-md border border-border-accent bg-accent-primary/5 p-3 text-xs leading-5"
                    role="status"
                  >
                    {result}
                  </div>
                ) : null}
              </CardContent>
            </Card>
          </div>
        ) : null}
      </div>
    </div>
  );
}
