import { api } from '@/aka/api/client';
import type { Dataset, KnowledgeDocument } from '@/aka/api/types';
import { ChunkInspector } from '@/aka/components/chunk-inspector';
import { DocumentList } from '@/aka/components/document-list';
import { EvaluationPanel } from '@/aka/components/evaluation-panel';
import { ImageChunkInspector } from '@/aka/components/image-chunk-inspector';
import { PageLoading } from '@/aka/components/page-state';
import { RetrievalLab } from '@/aka/components/retrieval-lab';
import { RetrievalProfilePanel } from '@/aka/components/retrieval-profile-panel';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import {
  ArrowLeft,
  BookOpenCheck,
  Database,
  FileText,
  Images,
  Settings2,
  TestTube2,
  UploadCloud,
} from 'lucide-react';
import { ChangeEvent, useCallback, useEffect, useRef, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router';

type Tab = 'documents' | 'chunks' | 'images' | 'retrieval' | 'settings' | 'evaluation';

const tabs = [
  { id: 'documents' as const, label: '文档', icon: FileText },
  { id: 'chunks' as const, label: 'Chunks', icon: BookOpenCheck },
  { id: 'images' as const, label: '图片知识', icon: Images },
  { id: 'retrieval' as const, label: '检索测试', icon: TestTube2 },
  { id: 'settings' as const, label: '配置', icon: Settings2 },
  { id: 'evaluation' as const, label: '评测', icon: Database },
];

export default function DatasetDetailPage() {
  const { datasetId = '' } = useParams();
  const [searchParams] = useSearchParams();
  const requestedTab = searchParams.get('tab');
  const requestedDocumentId = searchParams.get('document') ?? '';
  const requestedChildId = searchParams.get('child') ?? '';
  const requestedImageChunkId = searchParams.get('image') ?? '';
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [tab, setTab] = useState<Tab>(
    tabs.some((item) => item.id === requestedTab)
      ? (requestedTab as Tab)
      : 'documents',
  );
  const [selectedDocumentId, setSelectedDocumentId] = useState(
    requestedDocumentId,
  );
  const [selectedChildId, setSelectedChildId] = useState(requestedChildId);
  const [busyId, setBusyId] = useState('');
  const [notice, setNotice] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    const [datasetValue, documentValues] = await Promise.all([
      api.dataset(datasetId),
      api.documents(datasetId),
    ]);
    setDataset(datasetValue);
    setDocuments(documentValues);
    setSelectedDocumentId((current) => current || documentValues[0]?.id || '');
  }, [datasetId]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (tabs.some((item) => item.id === requestedTab)) {
      setTab(requestedTab as Tab);
    }
    if (requestedDocumentId) {
      setSelectedDocumentId(requestedDocumentId);
    }
    setSelectedChildId(requestedChildId);
  }, [requestedChildId, requestedDocumentId, requestedTab]);

  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? []);
    setBusyId('upload');
    try {
      for (const file of selected) {
        const stored = await api.uploadFile(file);
        const document = await api.linkDocument(datasetId, {
          file_id: stored.id,
          parser_profile: dataset?.parser_profile,
        });
        await api.parseDocument(document.id);
      }
      setNotice(`${selected.length} 个文件已解析为候选版本，请检查后发布`);
      await load();
    } catch (reason) {
      setNotice(reason instanceof Error ? reason.message : '导入失败');
    } finally {
      setBusyId('');
      event.target.value = '';
    }
  }

  async function parse(document: KnowledgeDocument) {
    setBusyId(document.id);
    try {
      await api.parseDocument(document.id);
      setNotice(`${document.original_name} 解析完成`);
      await load();
    } finally {
      setBusyId('');
    }
  }

  async function publish(document: KnowledgeDocument) {
    if (!document.active_version) return;
    setBusyId(document.id);
    try {
      const updated = await api.publishDataset(
        datasetId,
        document.active_version,
      );
      setDataset(updated);
      setNotice(`${document.active_version} 已发布`);
      await load();
    } finally {
      setBusyId('');
    }
  }

  if (!dataset) {
    return (
      <div className="size-full overflow-auto p-6">
        <PageLoading label="加载 Dataset" />
      </div>
    );
  }

  return (
    <div className="grid size-full min-w-0 grid-cols-[minmax(0,1fr)] grid-rows-[auto_auto_1fr] overflow-hidden">
      <header className="flex flex-col gap-4 border-b border-border-button px-5 py-5 lg:flex-row lg:items-end lg:justify-between lg:px-8">
        <div className="min-w-0">
          <Link
            to="/datasets"
            className="mb-3 inline-flex items-center gap-1 text-xs text-text-secondary hover:text-text-primary"
          >
            <ArrowLeft className="size-3.5" />
            知识库
          </Link>
          <div className="flex min-w-0 items-center gap-3">
            <h1 className="truncate text-xl font-semibold">{dataset.name}</h1>
            <Badge
              variant={dataset.published_version ? 'success' : 'secondary'}
            >
              {dataset.published_version ?? '未发布'}
            </Badge>
          </div>
          <p className="mt-1 truncate text-sm text-text-secondary">
            {dataset.description || '未填写描述'}
          </p>
        </div>
        <div>
          <input
            ref={inputRef}
            hidden
            aria-label="向知识库导入文档"
            type="file"
            multiple
            accept=".pdf,.docx,.md,.txt,.png,.jpg,.jpeg"
            onChange={(event) => void upload(event)}
          />
          <Button
            onClick={() => inputRef.current?.click()}
            loading={busyId === 'upload'}
          >
            <UploadCloud className="size-4" />
            导入文档
          </Button>
        </div>
      </header>

      <div
        className="flex min-w-0 items-center gap-0 overflow-x-auto border-b border-border-button px-2 py-2 sm:gap-1 sm:px-5 lg:px-8"
        aria-label="知识库视图"
      >
        {tabs.map(({ id, label, icon: Icon }) => (
          <Button
            key={id}
            variant={tab === id ? 'secondary' : 'ghost'}
            size="sm"
            onClick={() => setTab(id)}
            aria-label={label}
          >
            <Icon className="hidden size-4 sm:block" />
            {label}
          </Button>
        ))}
      </div>

      <main className="min-h-0 min-w-0 overflow-auto px-5 py-5 lg:px-8">
        <div
          data-testid="dataset-content"
          data-wide-layout={tab === 'retrieval' || tab === 'images' ? 'true' : 'false'}
          className={cn(
            'mx-auto w-full space-y-4',
            tab === 'retrieval' || tab === 'images' ? 'max-w-none' : 'max-w-[1500px]',
          )}
        >
          {notice ? (
            <div
              className="rounded-md border border-border-button bg-bg-card px-4 py-3 text-sm"
              role="status"
            >
              {notice}
            </div>
          ) : null}

          {tab === 'documents' ? (
            <DocumentList
              documents={documents}
              busyId={busyId}
              onParse={(document) => void parse(document)}
              onPublish={(document) => void publish(document)}
            />
          ) : null}

          {tab === 'chunks' ? (
            <div className="space-y-4">
              <label className="flex items-center gap-3 text-xs">
                <span>文档</span>
                <select
                  value={selectedDocumentId}
                  onChange={(event) => {
                    setSelectedDocumentId(event.target.value);
                    setSelectedChildId('');
                  }}
                  className="h-9 min-w-72 rounded border border-border-button bg-bg-input px-3"
                >
                  {documents.map((document) => (
                    <option key={document.id} value={document.id}>
                      {document.original_name}
                    </option>
                  ))}
                </select>
              </label>
              {selectedDocumentId ? (
                <ChunkInspector
                  documentId={selectedDocumentId}
                  initialChildId={selectedChildId}
                  onDraftCreated={(indexVersion) => {
                    setDocuments((current) =>
                      current.map((item) =>
                        item.id === selectedDocumentId
                          ? { ...item, active_version: indexVersion }
                          : item,
                      ),
                    );
                    setNotice(`${indexVersion} 已生成，发布前不会影响在线检索`);
                  }}
                />
              ) : (
                <div className="p-8 text-center text-sm text-text-secondary">
                  先导入并解析文档
                </div>
              )}
            </div>
          ) : null}

          {tab === 'retrieval' ? (
            <RetrievalLab
              datasetId={datasetId}
              publishedVersion={dataset.published_version}
            />
          ) : null}

          {tab === 'images' ? (
            <ImageChunkInspector
              datasetId={datasetId}
              initialImageChunkId={requestedImageChunkId}
            />
          ) : null}

          {tab === 'settings' ? (
            <div className="space-y-4">
              <Card>
                <CardContent className="grid gap-3 p-5 sm:grid-cols-2 xl:grid-cols-4">
                  {[
                    ['默认解析模板', dataset.parser_profile],
                    ['已发布索引', dataset.published_version ?? '无'],
                    [
                      '文档 / Parent / Child',
                      `${dataset.document_count} / ${dataset.parent_count} / ${dataset.child_count}`,
                    ],
                    ['可见范围', dataset.visibility],
                  ].map(([label, value]) => (
                    <div
                      key={label}
                      className="rounded-md border border-border-button bg-bg-card p-3"
                    >
                      <div className="text-[10px] text-text-secondary">
                        {label}
                      </div>
                      <strong className="mt-2 block truncate text-sm">
                        {value}
                      </strong>
                    </div>
                  ))}
                </CardContent>
              </Card>
              <RetrievalProfilePanel
                dataset={dataset}
                onDatasetChange={setDataset}
              />
            </div>
          ) : null}

          {tab === 'evaluation' ? (
            <EvaluationPanel
              dataset={dataset}
              documents={documents}
              onDatasetChange={setDataset}
            />
          ) : null}
        </div>
      </main>
    </div>
  );
}
