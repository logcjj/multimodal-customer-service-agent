import type { Evidence } from '@/aka/api/types';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';
import { Button } from '@/components/ui/button';
import {
  BookOpen,
  ExternalLink,
  FileText,
  Image,
  Layers3,
  LocateFixed,
} from 'lucide-react';

export function EvidenceDrawer({
  evidence,
  onClose,
}: {
  evidence: Evidence | null;
  onClose: () => void;
}) {
  const locator = evidence
    ? (evidence.locator_label ??
      `第 ${evidence.page_start ?? '--'}-${evidence.page_end ?? '--'} 页`)
    : '';
  const childId = evidence?.child_ids?.[0];
  const imageChunkId = evidence?.image_chunk_ids?.[0];
  const locateHref =
    evidence?.source_type === 'image' && evidence.dataset_id && imageChunkId
      ? `/dataset/${encodeURIComponent(evidence.dataset_id)}?tab=images&image=${encodeURIComponent(imageChunkId)}`
      : evidence?.dataset_id && evidence.document_id && childId
      ? `/dataset/${encodeURIComponent(evidence.dataset_id)}?tab=chunks&document=${encodeURIComponent(evidence.document_id)}&child=${encodeURIComponent(childId)}`
      : null;
  const originalFileHref = evidence?.file_id
    ? `/api/files/${encodeURIComponent(evidence.file_id)}/content${
        evidence.document_mime_type === 'application/pdf' && evidence.page_start
          ? `#page=${evidence.page_start}`
          : ''
      }`
    : null;

  return (
    <Sheet open={Boolean(evidence)} onOpenChange={(open) => !open && onClose()}>
      {evidence ? (
        <SheetContent
          side="right"
          className="flex w-[min(92vw,620px)] max-w-none flex-col gap-0 overflow-hidden rounded-none p-0"
          aria-label="引用证据"
        >
          <SheetHeader className="border-b border-border-button px-5 py-4 pr-12 text-left">
            <div className="text-[11px] uppercase text-accent-primary">
              Evidence
            </div>
            <SheetTitle className="sr-only">引用证据</SheetTitle>
            <h2 className="text-base font-semibold leading-6">
              {evidence.title}
            </h2>
            <SheetDescription className="sr-only">
              查看引用原文、图片资产和检索分数
            </SheetDescription>
          </SheetHeader>

          <div className="flex min-w-0 items-center gap-2 border-b border-border-button px-5 py-3 text-xs">
            <FileText className="size-4 shrink-0 text-text-secondary" />
            <span className="text-text-secondary">来源文件</span>
            <strong className="min-w-0 truncate" title={evidence.document_name ?? undefined}>
              {evidence.document_name ?? '未记录原始文件名'}
            </strong>
          </div>

          <div className="grid grid-cols-2 border-b border-border-button text-xs sm:grid-cols-4">
            {[
              {
                icon: FileText,
                value: evidence.document_version ?? '未标记版本',
              },
              { icon: BookOpen, value: locator },
              {
                icon: Layers3,
                value: `${evidence.child_ids?.length ?? 0} Child`,
              },
              { icon: Image, value: `${evidence.asset_ids.length} 资产` },
            ].map(({ icon: Icon, value }) => (
              <div
                key={value}
                className="flex min-w-0 items-center gap-2 border-b border-r border-border-button px-4 py-3 last:border-r-0 sm:border-b-0"
              >
                <Icon className="size-3.5 shrink-0 text-text-secondary" />
                <span className="truncate">{value}</span>
              </div>
            ))}
          </div>

          <dl className="grid grid-cols-2 border-b border-border-button text-xs sm:grid-cols-4">
            {[
              ['章节', evidence.chapter_title ?? evidence.title],
              ['检索阶段', evidence.retrieval_stage ?? '未记录'],
              [
                '证据置信度',
                evidence.evidence_confidence == null
                  ? '未计算'
                  : `${Math.round(evidence.evidence_confidence * 100)}%`,
              ],
              ['来源类型', evidence.source_type],
            ].map(([label, value]) => (
              <div key={label} className="min-w-0 border-r border-border-button px-4 py-3 last:border-r-0">
                <dt className="text-[10px] text-text-secondary">{label}</dt>
                <dd className="mt-1 truncate font-medium" title={value}>{value}</dd>
              </div>
            ))}
          </dl>

          <div className="min-h-0 flex-1 space-y-6 overflow-auto p-5">
            {evidence.asset_ids.length ? (
              <section aria-label="证据图片">
                <h3 className="mb-3 text-xs font-semibold">关联图片</h3>
                <div className="grid grid-cols-2 gap-3">
                  {evidence.asset_ids.map((assetId) => (
                    <a
                      key={assetId}
                      href={`/api/assets/${encodeURIComponent(assetId)}`}
                      target="_blank"
                      rel="noreferrer"
                      className="overflow-hidden rounded-md border border-border-button bg-white"
                    >
                      <img
                        src={`/api/assets/${encodeURIComponent(assetId)}`}
                        alt={`${evidence.title} 关联图片`}
                        loading="lazy"
                        className="aspect-video size-full object-contain"
                      />
                    </a>
                  ))}
                </div>
              </section>
            ) : null}

            <section>
              <h3 className="mb-3 text-xs font-semibold">原始证据</h3>
              <p className="whitespace-pre-wrap rounded-md border border-border-button bg-bg-card p-4 text-sm leading-7 text-text-secondary">
                {evidence.text}
              </p>
            </section>

            <section>
              <h3 className="mb-3 text-xs font-semibold">检索分数</h3>
              <dl className="grid grid-cols-2 gap-2 sm:grid-cols-5">
                {Object.entries(evidence.score_breakdown ?? {}).map(
                  ([name, value]) => (
                    <div
                      key={name}
                      className="rounded-md border border-border-button bg-bg-card p-3"
                    >
                      <dt className="text-[10px] uppercase text-text-secondary">
                        {name}
                      </dt>
                      <dd className="mt-2 font-mono text-xs font-semibold">
                        {value.toFixed(4)}
                      </dd>
                    </div>
                  ),
                )}
              </dl>
            </section>

            <section>
              <h3 className="mb-3 text-xs font-semibold">证据标识</h3>
              <dl className="space-y-2 border-y border-border-button py-3 text-xs">
                {[
                  ['Parent', evidence.parent_id ?? evidence.section_id ?? '--'],
                  ['Child', evidence.child_ids?.join(', ') || '--'],
                  ['ImageChunk', evidence.image_chunk_ids?.join(', ') || '--'],
                ].map(([label, value]) => (
                  <div key={label} className="grid grid-cols-[90px_minmax(0,1fr)] gap-3">
                    <dt className="text-text-secondary">{label}</dt>
                    <dd><code className="break-all text-[11px]">{value}</code></dd>
                  </div>
                ))}
              </dl>
            </section>
          </div>

          <footer className="border-t border-border-button px-5 py-3">
            <div className="mb-3 grid grid-cols-2 gap-2">
              {locateHref ? (
                <Button asChild variant="outline" size="sm">
                  <a href={locateHref} aria-label="定位到知识库">
                    <LocateFixed className="size-4" />
                    定位到知识库
                  </a>
                </Button>
              ) : null}
              {originalFileHref ? (
                <Button asChild variant="outline" size="sm">
                  <a
                    href={originalFileHref}
                    target="_blank"
                    rel="noreferrer"
                    aria-label="查看原始文件"
                  >
                    <ExternalLink className="size-4" />
                    查看原始文件
                  </a>
                </Button>
              ) : null}
            </div>
            <code className="block truncate text-[10px] text-accent-primary">
              {evidence.evidence_id}
            </code>
            <div className="mt-1 truncate text-[10px] text-text-secondary">
              {evidence.dataset_id ?? '--'} / {evidence.document_id ?? '--'}
            </div>
          </footer>
        </SheetContent>
      ) : null}
    </Sheet>
  );
}
