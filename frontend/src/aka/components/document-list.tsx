import type { KnowledgeDocument } from '@/aka/api/types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { FileText, Play, Rocket } from 'lucide-react';

function documentStatus(document: KnowledgeDocument) {
  if (document.latest_job_state === 'failed')
    return { label: '解析失败', variant: 'destructive' as const };
  if (document.latest_job_state === 'running')
    return {
      label: `解析中 ${document.latest_job_progress ?? 0}%`,
      variant: 'secondary' as const,
    };
  if (document.published_version)
    return { label: '已发布', variant: 'success' as const };
  if (document.active_version)
    return { label: '候选版本', variant: 'secondary' as const };
  return { label: '待解析', variant: 'outline' as const };
}

export function DocumentList({
  documents,
  busyId,
  onParse,
  onPublish,
}: {
  documents: KnowledgeDocument[];
  busyId: string;
  onParse: (document: KnowledgeDocument) => void;
  onPublish: (document: KnowledgeDocument) => void;
}) {
  return (
    <div className="overflow-hidden rounded-md border border-border-button">
      <div className="hidden grid-cols-[minmax(240px,1.5fr)_120px_120px_minmax(170px,1fr)_100px] gap-4 border-b border-border-button bg-bg-card px-4 py-3 text-xs text-text-secondary lg:grid">
        <span>文档</span>
        <span>模板</span>
        <span>状态</span>
        <span>索引版本</span>
        <span className="text-right">操作</span>
      </div>
      {documents.map((document) => {
        const status = documentStatus(document);
        return (
          <article
            key={document.id}
            className="grid gap-3 border-b border-border-button px-4 py-4 last:border-b-0 lg:grid-cols-[minmax(240px,1.5fr)_120px_120px_minmax(170px,1fr)_100px] lg:items-center lg:gap-4"
          >
            <div className="flex min-w-0 items-center gap-3">
              <div className="grid size-9 shrink-0 place-items-center rounded bg-bg-card">
                <FileText className="size-4" />
              </div>
              <div className="min-w-0">
                <div className="truncate text-sm font-medium">
                  {document.original_name}
                </div>
                <div className="truncate text-xs text-text-secondary">
                  {document.mime_type}
                </div>
              </div>
            </div>
            <div className="text-xs text-text-secondary">
              {document.parser_profile}
            </div>
            <div>
              <Badge variant={status.variant}>{status.label}</Badge>
            </div>
            <code className="truncate text-xs text-text-secondary">
              {document.active_version ?? '--'}
            </code>
            <div className="flex justify-end gap-1">
              <Button
                type="button"
                variant="ghost"
                size="icon"
                aria-label={`解析 ${document.original_name}`}
                title="重新解析"
                loading={busyId === document.id}
                onClick={() => onParse(document)}
              >
                <Play className="size-4" />
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                aria-label={`发布 ${document.original_name}`}
                title="发布候选版本"
                disabled={
                  !document.active_version ||
                  busyId === document.id ||
                  document.active_version === document.published_version
                }
                onClick={() => onPublish(document)}
              >
                <Rocket className="size-4" />
              </Button>
            </div>
          </article>
        );
      })}
    </div>
  );
}
