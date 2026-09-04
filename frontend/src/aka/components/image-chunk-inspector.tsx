import { api } from '@/aka/api/client';
import type { ImageChunk, IndexManifest } from '@/aka/api/types';
import { Button } from '@/components/ui/button';
import {
  ExternalLink,
  FileImage,
  Image as ImageIcon,
  LocateFixed,
  Search,
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

function approvalLabel(status: string) {
  if (status === 'approved') return '已批准';
  if (status === 'rejected') return '未通过评测';
  return '等待评测批准';
}

export function ImageChunkInspector({
  datasetId,
  initialImageChunkId = '',
}: {
  datasetId: string;
  initialImageChunkId?: string;
}) {
  const [manifest, setManifest] = useState<IndexManifest | null>(null);
  const [chunks, setChunks] = useState<ImageChunk[]>([]);
  const [selectedId, setSelectedId] = useState(initialImageChunkId);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    setLoading(true);
    Promise.all([api.indexManifest(datasetId), api.imageChunks(datasetId)])
      .then(([manifestValue, chunkValues]) => {
        if (!active) return;
        setManifest(manifestValue);
        setChunks(chunkValues);
        setSelectedId((current) => current || chunkValues[0]?.id || '');
        setError('');
      })
      .catch((reason: unknown) => {
        if (active)
          setError(reason instanceof Error ? reason.message : '图片知识加载失败');
      })
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [datasetId]);

  useEffect(() => {
    if (initialImageChunkId) setSelectedId(initialImageChunkId);
  }, [initialImageChunkId]);

  const filtered = useMemo(() => {
    const keyword = query.trim().toLowerCase();
    if (!keyword) return chunks;
    return chunks.filter((item) =>
      [
        item.manual_name,
        item.chapter_title,
        item.caption,
        item.ocr_text,
        item.retrieval_text,
        ...item.search_terms,
      ]
        .join(' ')
        .toLowerCase()
        .includes(keyword),
    );
  }, [chunks, query]);
  const selected = chunks.find((item) => item.id === selectedId) ?? filtered[0];

  async function searchAllImages() {
    setSearching(true);
    try {
      const values = await api.imageChunks(datasetId, query.trim() || undefined);
      setChunks(values);
      setSelectedId(values[0]?.id || '');
      setError('');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '图片知识搜索失败');
    } finally {
      setSearching(false);
    }
  }

  if (loading) {
    return <div className="py-12 text-center text-sm text-text-secondary">加载图片知识与索引清单</div>;
  }
  if (error) {
    return <div className="border border-state-error/30 px-4 py-3 text-sm text-state-error">{error}</div>;
  }

  return (
    <div className="space-y-4">
      {manifest ? (
        <section className="border-y border-border-button" aria-label="当前索引清单">
          <div className="grid grid-cols-2 divide-x divide-y divide-border-button text-xs md:grid-cols-6 md:divide-y-0">
            {[
              ['索引版本', manifest.index_version],
              ['向量模型', manifest.embedding_model ?? '未配置'],
              ['向量维度', manifest.vector_dimension.toLocaleString()],
              ['图片 Chunk', (manifest.counts.image_chunks ?? chunks.length).toLocaleString()],
              ['复用来源', manifest.incremental.reused.toLocaleString()],
              ['发布门禁', approvalLabel(manifest.approval_status)],
            ].map(([label, value]) => (
              <div key={label} className="min-w-0 px-3 py-3">
                <div className="text-[10px] text-text-secondary">{label}</div>
                <strong className="mt-1 block truncate" title={value}>{value}</strong>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      <div className="flex flex-wrap items-center justify-between gap-3">
        <form
          className="flex w-full max-w-xl gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            void searchAllImages();
          }}
        >
          <div className="relative min-w-0 flex-1">
            <Search className="pointer-events-none absolute left-3 top-2.5 size-4 text-text-secondary" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              aria-label="搜索图片知识"
              placeholder="搜索手册、章节、Caption 或 OCR"
              className="h-9 w-full rounded border border-border-button bg-bg-input pl-9 pr-3 text-sm outline-none focus:border-border-default"
            />
          </div>
          <Button type="submit" variant="outline" loading={searching} aria-label="执行图片搜索">
            <Search className="size-4" />搜索
          </Button>
        </form>
        <span className="text-xs text-text-secondary">
          当前显示 {filtered.length.toLocaleString()} / {(manifest?.counts.image_chunks ?? chunks.length).toLocaleString()}
        </span>
      </div>

      <div className="grid min-h-[560px] overflow-hidden border border-border-button lg:grid-cols-[minmax(300px,0.8fr)_minmax(0,1.2fr)]">
        <div className="min-h-0 overflow-auto border-b border-border-button lg:border-b-0 lg:border-r">
          {filtered.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setSelectedId(item.id)}
              aria-pressed={selected?.id === item.id}
              className={`grid w-full grid-cols-[72px_minmax(0,1fr)] gap-3 border-b border-border-button p-3 text-left last:border-b-0 ${
                selected?.id === item.id ? 'bg-bg-card' : 'hover:bg-bg-card/60'
              }`}
            >
              <img src={item.asset_url} alt="" className="aspect-square w-[72px] border border-border-button bg-white object-contain" />
              <span className="min-w-0">
                <strong className="line-clamp-2 text-sm leading-5">{item.caption || item.visual_summary || item.chapter_title}</strong>
                <span className="mt-1 block truncate text-xs text-text-secondary">{item.manual_name} · 第 {item.page_number} 页</span>
                <code className="mt-2 block truncate text-[10px] text-accent-primary">{item.id}</code>
              </span>
            </button>
          ))}
          {!filtered.length ? <div className="p-8 text-center text-sm text-text-secondary">没有匹配的图片知识</div> : null}
        </div>

        {selected ? (
          <article className="min-w-0 overflow-auto p-5">
            <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border-button pb-4">
              <div className="min-w-0">
                <div className="flex items-center gap-2 text-xs text-text-secondary"><FileImage className="size-4" />{selected.manual_name}</div>
                <h2 className="mt-2 text-base font-semibold">{selected.chapter_title}</h2>
                <p className="mt-1 text-xs text-text-secondary">第 {selected.page_number} 页 · 置信度 {Math.round(selected.confidence * 100)}%</p>
              </div>
              <div className="flex gap-2">
                {selected.related_child_ids[0] ? (
                  <Button asChild size="sm" variant="outline">
                    <a href={`/dataset/${encodeURIComponent(datasetId)}?tab=chunks&document=${encodeURIComponent(selected.document_id)}&child=${encodeURIComponent(selected.related_child_ids[0])}`}>
                      <LocateFixed className="size-4" />定位章节
                    </a>
                  </Button>
                ) : null}
                <Button asChild size="icon" variant="outline" title="打开原图">
                  <a href={selected.asset_url} target="_blank" rel="noreferrer" aria-label="打开原图"><ExternalLink className="size-4" /></a>
                </Button>
              </div>
            </div>

            <a href={selected.asset_url} target="_blank" rel="noreferrer" className="mt-5 block border border-border-button bg-white">
              <img src={selected.asset_url} alt={selected.caption || selected.chapter_title} className="mx-auto max-h-[360px] w-full object-contain" />
            </a>

            <dl className="mt-5 grid gap-x-5 gap-y-4 text-sm sm:grid-cols-2">
              {[
                ['Caption', selected.caption || '无'],
                ['OCR', selected.ocr_text || selected.visible_text.join('；') || '无'],
                ['视觉摘要', selected.visual_summary || '无'],
                ['适用问题', selected.applicable_questions.join('；') || '无'],
              ].map(([label, value]) => (
                <div key={label} className="min-w-0 border-t border-border-button pt-3">
                  <dt className="text-xs text-text-secondary">{label}</dt>
                  <dd className="mt-1 whitespace-pre-wrap leading-6">{value}</dd>
                </div>
              ))}
            </dl>
            <div className="mt-5 flex items-center gap-2 text-xs text-text-secondary"><ImageIcon className="size-4" />Embedding {selected.embedding_dimension.toLocaleString()} 维</div>
          </article>
        ) : null}
      </div>
    </div>
  );
}
