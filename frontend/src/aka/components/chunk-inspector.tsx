import { api } from '@/aka/api/client';
import type { ChunkCollection } from '@/aka/api/types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Check, FileSearch, Save } from 'lucide-react';
import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';

export function ChunkInspector({
  documentId,
  initialChildId,
  onDraftCreated,
}: {
  documentId: string;
  initialChildId?: string;
  onDraftCreated?: (indexVersion: string) => void;
}) {
  const [collection, setCollection] = useState<ChunkCollection>({
    parents: [],
    children: [],
  });
  const [selectedId, setSelectedId] = useState('');
  const [loading, setLoading] = useState(true);
  const [saved, setSaved] = useState(false);
  const parentButtonRefs = useRef(new Map<string, HTMLElement>());

  useEffect(() => {
    let active = true;
    setLoading(true);
    void api
      .chunks(documentId)
      .then((value) => {
        if (!active) return;
        setCollection(value);
        setSelectedId(
          value.children.some((item) => item.id === initialChildId)
            ? (initialChildId ?? '')
            : (value.children[0]?.id ?? ''),
        );
      })
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [documentId, initialChildId]);

  const selected =
    collection.children.find((item) => item.id === selectedId) ?? null;
  const parent =
    collection.parents.find((item) => item.id === selected?.parent_id) ?? null;
  const siblings = useMemo(
    () => collection.children.filter((item) => item.parent_id === parent?.id),
    [collection.children, parent?.id],
  );

  useEffect(() => {
    if (!initialChildId || selected?.id !== initialChildId || !parent) {
      return;
    }
    parentButtonRefs.current.get(parent.id)?.scrollIntoView({
      block: 'nearest',
      behavior: 'smooth',
    });
  }, [initialChildId, parent, selected?.id]);

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selected) return;
    const form = new FormData(event.currentTarget);
    const updated = await api.updateChunk(selected.id, {
      text: String(form.get('text') ?? ''),
      keywords: String(form.get('keywords') ?? '')
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean),
      tags: String(form.get('tags') ?? '')
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean),
    });
    const refreshed = await api.chunks(documentId);
    setCollection(refreshed);
    setSelectedId(
      refreshed.children.find((item) => item.id === updated.id)?.id ??
        refreshed.children[0]?.id ??
        '',
    );
    setSaved(true);
    onDraftCreated?.(updated.index_version);
  }

  if (loading)
    return (
      <div className="p-8 text-center text-sm text-text-secondary">
        加载 Chunks
      </div>
    );
  if (!selected || !parent)
    return (
      <div className="p-8 text-center text-sm text-text-secondary">
        该文档尚未生成 Chunk
      </div>
    );

  return (
    <div className="grid min-h-[620px] overflow-hidden rounded-md border border-border-button xl:grid-cols-[260px_minmax(0,1fr)_minmax(340px,0.8fr)]">
      <aside className="min-h-0 border-b border-border-button bg-bg-card xl:border-b-0 xl:border-r">
        <div className="flex items-center gap-2 border-b border-border-button px-4 py-3 text-xs font-semibold">
          <FileSearch className="size-4" />
          文档结构
        </div>
        <div className="max-h-72 overflow-auto p-2 xl:max-h-[570px]">
          {collection.parents.map((item) => {
            const count = collection.children.filter(
              (child) => child.parent_id === item.id,
            ).length;
            return (
              <Button
                ref={(node) => {
                  if (node) {
                    parentButtonRefs.current.set(item.id, node);
                  } else {
                    parentButtonRefs.current.delete(item.id);
                  }
                }}
                key={item.id}
                data-testid={`chunk-parent-${item.id}`}
                aria-current={item.id === parent.id}
                variant={item.id === parent.id ? 'secondary' : 'ghost'}
                size="auto"
                className="mb-1 h-auto w-full justify-start whitespace-normal px-3 py-2 text-left"
                onClick={() =>
                  setSelectedId(
                    collection.children.find(
                      (child) => child.parent_id === item.id,
                    )?.id ?? '',
                  )
                }
              >
                <span className="min-w-0">
                  <strong className="block truncate text-xs">
                    {item.title}
                  </strong>
                  <small className="mt-1 block text-[10px] text-text-secondary">
                    第 {item.page_start}-{item.page_end} 页 · {count} Child
                  </small>
                </span>
              </Button>
            );
          })}
        </div>
      </aside>

      <section className="min-w-0 border-b border-border-button p-5 xl:border-b-0 xl:border-r">
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <div className="text-[10px] uppercase text-text-secondary">
              Parent Chunk
            </div>
            <h3 className="mt-1 text-sm font-semibold">{parent.title}</h3>
          </div>
          <Badge variant="success">{parent.index_version}</Badge>
        </div>
        <p className="max-h-96 overflow-auto whitespace-pre-wrap text-sm leading-7 text-text-secondary">
          {parent.text}
        </p>
        <div className="mt-5 flex flex-wrap gap-1.5">
          {siblings.map((item, index) => (
            <Button
              key={item.id}
              type="button"
              data-testid={`chunk-child-${item.id}`}
              aria-pressed={item.id === selected.id}
              variant={item.id === selected.id ? 'default' : 'outline'}
              size="xs"
              onClick={() => {
                setSelectedId(item.id);
                setSaved(false);
              }}
            >
              Child {index + 1}
            </Button>
          ))}
        </div>
      </section>

      <form
        key={selected.id}
        onSubmit={(event) => void save(event)}
        className="flex min-w-0 flex-col p-5"
      >
        <div className="mb-4 flex items-start justify-between">
          <div>
            <div className="text-[10px] uppercase text-text-secondary">
              Child 编辑器
            </div>
            <div className="mt-1 text-sm font-semibold">
              第 {selected.page_start}-{selected.page_end} 页
            </div>
          </div>
          {selected.edited ? <Badge variant="secondary">草稿</Badge> : null}
        </div>
        <label className="mb-4 block min-h-0 flex-1 text-xs">
          <span className="mb-1.5 block">正文</span>
          <textarea
            aria-label="正文"
            name="text"
            defaultValue={selected.text}
            rows={14}
            className="h-full min-h-72 w-full resize-y rounded-md border border-border-button bg-bg-input p-3 text-sm leading-6 outline-none focus:border-border-accent"
          />
        </label>
        <label className="mb-3 block text-xs">
          <span className="mb-1.5 block">关键词</span>
          <input
            name="keywords"
            defaultValue={selected.keywords.join(', ')}
            className="h-9 w-full rounded border border-border-button bg-bg-input px-3 outline-none"
          />
        </label>
        <label className="mb-4 block text-xs">
          <span className="mb-1.5 block">标签</span>
          <input
            name="tags"
            defaultValue={selected.tags.join(', ')}
            className="h-9 w-full rounded border border-border-button bg-bg-input px-3 outline-none"
          />
        </label>
        <div className="flex items-center justify-between gap-3">
          <span className="text-[10px] text-text-secondary">
            {selected.token_count} 字符 · {selected.asset_ids.length} 资产
          </span>
          <Button type="submit">
            {saved ? <Check className="size-4" /> : <Save className="size-4" />}
            {saved ? '已保存草稿' : '保存新版本'}
          </Button>
        </div>
      </form>
    </div>
  );
}
