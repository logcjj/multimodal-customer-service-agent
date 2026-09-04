import { api } from '@/aka/api/client';
import type { FileAsset } from '@/aka/api/types';
import { PageError, PageLoading } from '@/aka/components/page-state';
import { useApiResource } from '@/aka/hooks/use-api-resource';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { File, HardDrive, UploadCloud } from 'lucide-react';
import { ChangeEvent, useCallback, useRef, useState } from 'react';

function sizeLabel(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default function FilesPage() {
  const loader = useCallback(() => api.files(), []);
  const {
    data: files,
    loading,
    error,
    setData,
  } = useApiResource<FileAsset[]>(loader, []);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? []);
    setBusy(true);
    try {
      for (const file of selected) await api.uploadFile(file);
      setData(await api.files());
    } finally {
      setBusy(false);
      event.target.value = '';
    }
  }

  const totalBytes = files.reduce((sum, item) => sum + item.size_bytes, 0);

  return (
    <div className="size-full overflow-auto px-5 py-6 lg:px-8 lg:py-8">
      <div className="mx-auto max-w-[1500px] space-y-5">
        <header className="flex items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold">文件</h1>
            <p className="mt-1 text-sm text-text-secondary">
              原始文件只存储一次，可链接到多个知识库
            </p>
          </div>
          <div>
            <input
              ref={inputRef}
              hidden
              aria-label="上传文件资产"
              type="file"
              multiple
              accept=".pdf,.docx,.md,.txt,.png,.jpg,.jpeg"
              onChange={(event) => void upload(event)}
            />
            <Button onClick={() => inputRef.current?.click()} loading={busy}>
              <UploadCloud className="size-4" />
              上传文件
            </Button>
          </div>
        </header>

        <div className="grid gap-3 sm:grid-cols-2">
          <div className="flex items-center gap-3 rounded-md border border-border-button bg-bg-card p-4">
            <File className="size-5 text-text-secondary" />
            <div>
              <strong className="block text-xl tabular-nums">
                {files.length}
              </strong>
              <span className="text-xs text-text-secondary">文件资产</span>
            </div>
          </div>
          <div className="flex items-center gap-3 rounded-md border border-border-button bg-bg-card p-4">
            <HardDrive className="size-5 text-text-secondary" />
            <div>
              <strong className="block text-xl tabular-nums">
                {sizeLabel(totalBytes)}
              </strong>
              <span className="text-xs text-text-secondary">已用存储</span>
            </div>
          </div>
        </div>

        {loading ? <PageLoading label="加载文件" /> : null}
        {!loading && error ? <PageError message={error} /> : null}
        {!loading && !error ? (
          <div className="overflow-hidden rounded-md border border-border-button">
            <div className="hidden grid-cols-[minmax(260px,1.5fr)_220px_120px_100px] gap-4 border-b border-border-button bg-bg-card px-4 py-3 text-xs text-text-secondary md:grid">
              <span>文件</span>
              <span>类型</span>
              <span>大小</span>
              <span>状态</span>
            </div>
            {files.map((item) => (
              <article
                key={item.id}
                className="grid gap-3 border-b border-border-button px-4 py-4 last:border-b-0 md:grid-cols-[minmax(260px,1.5fr)_220px_120px_100px] md:items-center"
              >
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium">
                    {item.original_name}
                  </div>
                  <code className="mt-1 block truncate text-[10px] text-text-secondary">
                    {item.content_hash}
                  </code>
                </div>
                <span className="truncate text-xs text-text-secondary">
                  {item.mime_type}
                </span>
                <span className="text-xs tabular-nums">
                  {sizeLabel(item.size_bytes)}
                </span>
                <div>
                  <Badge variant="success">{item.status}</Badge>
                </div>
              </article>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
