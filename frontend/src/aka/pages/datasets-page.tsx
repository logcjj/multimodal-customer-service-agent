import { api } from '@/aka/api/client';
import type { Dataset } from '@/aka/api/types';
import { PageEmpty, PageError, PageLoading } from '@/aka/components/page-state';
import { useApiResource } from '@/aka/hooks/use-api-resource';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import {
  BookOpenCheck,
  FileText,
  Image,
  Layers3,
  Plus,
  Search,
  UploadCloud,
} from 'lucide-react';
import {
  ChangeEvent,
  FormEvent,
  useCallback,
  useMemo,
  useRef,
  useState,
} from 'react';

const number = new Intl.NumberFormat('zh-CN');

export default function DatasetsPage() {
  const loader = useCallback(() => api.datasets(), []);
  const {
    data: datasets,
    loading,
    error,
    setData,
  } = useApiResource<Dataset[]>(loader, []);
  const [query, setQuery] = useState('');
  const [creating, setCreating] = useState(false);
  const [busy, setBusy] = useState(false);
  const [targetDataset, setTargetDataset] = useState('');
  const [notice, setNotice] = useState('');
  const fileInput = useRef<HTMLInputElement>(null);

  const visible = useMemo(
    () =>
      datasets.filter((item) =>
        `${item.name} ${item.description}`
          .toLowerCase()
          .includes(query.toLowerCase()),
      ),
    [datasets, query],
  );
  const target =
    targetDataset ||
    datasets.find((item) => !item.is_system)?.id ||
    datasets[0]?.id ||
    '';

  async function createDataset(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setBusy(true);
    try {
      const created = await api.createDataset({
        name: String(form.get('name') ?? ''),
        description: String(form.get('description') ?? ''),
        parser_profile: String(form.get('parser_profile') ?? 'manual'),
      });
      setData((current) => [...current, created]);
      setTargetDataset(created.id);
      setCreating(false);
      setNotice(`${created.name} 已创建`);
    } catch (reason) {
      setNotice(reason instanceof Error ? reason.message : '创建失败');
    } finally {
      setBusy(false);
    }
  }

  async function importFiles(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    if (!files.length || !target) return;
    setBusy(true);
    setNotice(`正在导入 ${files.length} 个文件`);
    try {
      for (const file of files) {
        const stored = await api.uploadFile(file);
        const document = await api.linkDocument(target, { file_id: stored.id });
        await api.parseDocument(document.id);
      }
      setNotice(`${files.length} 个文件已生成候选版本，请检查后发布`);
      setData(await api.datasets());
    } catch (reason) {
      setNotice(reason instanceof Error ? reason.message : '导入失败');
    } finally {
      setBusy(false);
      event.target.value = '';
    }
  }

  return (
    <div className="size-full overflow-auto px-5 py-6 lg:px-8 lg:py-8">
      <div className="mx-auto max-w-[1500px] space-y-5">
        <header className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <h1 className="text-2xl font-semibold">知识库</h1>
            <p className="mt-1 text-sm text-text-secondary">
              版本化解析、Parent/Child 检索与图文证据
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {datasets.length ? (
              <select
                value={target}
                aria-label="导入到知识库"
                onChange={(event) => setTargetDataset(event.target.value)}
                className="h-8 min-w-44 rounded border border-border-button bg-bg-input px-3 text-sm outline-none"
              >
                {datasets.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.name}
                  </option>
                ))}
              </select>
            ) : null}
            <input
              ref={fileInput}
              hidden
              aria-label="导入文档"
              type="file"
              multiple
              accept=".pdf,.docx,.md,.txt,.png,.jpg,.jpeg"
              onChange={(event) => void importFiles(event)}
            />
            <Button
              variant="outline"
              disabled={!target || busy}
              onClick={() => fileInput.current?.click()}
            >
              <UploadCloud className="size-4" />
              导入文档
            </Button>
            <Button onClick={() => setCreating(true)}>
              <Plus className="size-4" />
              新建知识库
            </Button>
          </div>
        </header>

        {notice ? (
          <div
            className="rounded-md border border-border-button bg-bg-card px-4 py-3 text-sm"
            role="status"
          >
            {notice}
          </div>
        ) : null}

        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="text-sm text-text-secondary">
            {datasets.length} Datasets
          </div>
          <label className="flex h-9 w-full items-center gap-2 rounded-md border border-border-button bg-bg-input px-3 sm:w-72">
            <Search className="size-4 text-text-secondary" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="搜索知识库"
              aria-label="搜索知识库"
              className="min-w-0 flex-1 bg-transparent text-sm outline-none"
            />
          </label>
        </div>

        {loading ? <PageLoading label="加载知识库" /> : null}
        {!loading && error ? <PageError message={error} /> : null}
        {!loading && !error && visible.length === 0 ? (
          <PageEmpty
            title="暂无知识库"
            description="新建知识库后即可上传并解析文档。"
          />
        ) : null}
        {!loading && !error && visible.length ? (
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
            {visible.map((dataset) => (
              <Card key={dataset.id} className="group min-h-56 hover:shadow-md">
                <CardHeader className="p-5">
                  <div className="mb-3 flex items-start justify-between gap-3">
                    <div className="grid size-10 place-items-center rounded-md bg-bg-component">
                      <BookOpenCheck className="size-5 text-text-secondary" />
                    </div>
                    <Badge
                      variant={
                        dataset.published_version ? 'success' : 'secondary'
                      }
                    >
                      {dataset.published_version ? '已发布' : '草稿'}
                    </Badge>
                  </div>
                  <CardTitle as="h2" className="truncate text-base">
                    {dataset.name}
                  </CardTitle>
                  <CardDescription className="line-clamp-2 min-h-10">
                    {dataset.description || '未填写描述'}
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-5 pt-0">
                  <div className="grid grid-cols-2 gap-2 text-xs text-text-secondary">
                    <div className="flex items-center gap-1.5">
                      <FileText className="size-3.5" />
                      {dataset.document_count} Documents
                    </div>
                    <div className="flex items-center gap-1.5">
                      <Layers3 className="size-3.5" />
                      {number.format(dataset.parent_count)} Parent
                    </div>
                    <div>{number.format(dataset.child_count)} Chunks</div>
                    <div className="flex items-center gap-1.5">
                      <Image className="size-3.5" />
                      {number.format(dataset.asset_count)} Assets
                    </div>
                  </div>
                  <Button
                    asLink
                    to={`/dataset/${dataset.id}`}
                    aria-label={`打开 ${dataset.name}`}
                    variant="outline"
                    className="mt-5 w-full"
                  >
                    打开知识库
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : null}
      </div>

      <Dialog open={creating} onOpenChange={setCreating}>
        <DialogContent>
          <form onSubmit={(event) => void createDataset(event)}>
            <DialogHeader>
              <DialogTitle>新建知识库</DialogTitle>
              <DialogDescription>
                解析模板可在单个文档上覆盖。
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-5">
              <label className="block space-y-1.5 text-sm">
                <span>名称</span>
                <Input name="name" required autoFocus />
              </label>
              <label className="block space-y-1.5 text-sm">
                <span>描述</span>
                <Input name="description" />
              </label>
              <label className="block space-y-1.5 text-sm">
                <span>解析模板</span>
                <select
                  name="parser_profile"
                  defaultValue="manual"
                  className="h-9 w-full rounded border border-border-button bg-bg-input px-3"
                >
                  <option value="manual">Manual</option>
                  <option value="general">General</option>
                  <option value="qa">Q&amp;A</option>
                  <option value="table">Table</option>
                  <option value="picture">Picture</option>
                </select>
              </label>
            </div>
            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setCreating(false)}
              >
                取消
              </Button>
              <Button type="submit" loading={busy}>
                创建
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}
