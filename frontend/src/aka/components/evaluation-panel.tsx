import { api } from '@/aka/api/client';
import type {
  Dataset,
  EvalCase,
  EvalRun,
  KnowledgeDocument,
} from '@/aka/api/types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { CheckCircle2, Database, Play, Plus } from 'lucide-react';
import { FormEvent, useEffect, useMemo, useState } from 'react';

function splitFacts(value: string) {
  return value
    .split(/[，,]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

export function EvaluationPanel({
  dataset,
  documents,
  onDatasetChange,
}: {
  dataset: Dataset;
  documents: KnowledgeDocument[];
  onDatasetChange: (value: Dataset) => void;
}) {
  const [cases, setCases] = useState<EvalCase[]>([]);
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [candidate, setCandidate] = useState('');
  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState('');
  const versions = useMemo(
    () =>
      Array.from(
        new Set(
          documents
            .map((item) => item.active_version)
            .filter((item): item is string => Boolean(item)),
        ),
      ),
    [documents],
  );
  const latestRun =
    runs.find((item) => item.candidate_version === candidate) ?? runs[0];

  async function load() {
    const [caseValues, runValues] = await Promise.all([
      api.evalCases(),
      api.evalRuns(),
    ]);
    setCases(
      caseValues.filter((item) => item.dataset_ids.includes(dataset.id)),
    );
    setRuns(runValues);
    setCandidate(
      (current) => current || versions[0] || dataset.published_version || '',
    );
  }

  useEffect(() => {
    void load();
  }, [dataset.id]);

  async function createCase(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setBusy('case');
    try {
      await api.createEvalCase({
        question: String(form.get('question') ?? ''),
        dataset_ids: [dataset.id],
        target_parent_ids: splitFacts(String(form.get('parents') ?? '')),
        required_facts: splitFacts(String(form.get('required') ?? '')),
        forbidden_facts: splitFacts(String(form.get('forbidden') ?? '')),
        locked: form.get('locked') === 'on',
      });
      event.currentTarget.reset();
      setNotice('锁定用例已保存，在线反馈不会自动改写它');
      await load();
    } finally {
      setBusy('');
    }
  }

  async function runEvaluation() {
    if (!candidate || !cases.length) return;
    setBusy('run');
    try {
      const run = await api.createEvalRun({
        candidate_version: candidate,
        case_ids: cases.map((item) => item.id),
      });
      setRuns((current) => [run, ...current]);
      setNotice(
        run.passed
          ? '候选版本通过全部锁定用例'
          : '候选版本存在回归，已阻止发布',
      );
    } finally {
      setBusy('');
    }
  }

  async function approveAndPublish() {
    if (!latestRun?.passed || latestRun.candidate_version !== candidate) return;
    setBusy('publish');
    try {
      const approved =
        latestRun.status === 'approved'
          ? latestRun
          : await api.approveEvalRun(latestRun.id);
      const updated = await api.publishDataset(
        dataset.id,
        candidate,
        approved.id,
      );
      setRuns((current) =>
        current.map((item) => (item.id === approved.id ? approved : item)),
      );
      onDatasetChange(updated);
      setNotice(`${candidate} 已通过门禁并原子发布`);
    } finally {
      setBusy('');
    }
  }

  const field =
    'h-9 w-full rounded border border-border-button bg-bg-input px-3 text-sm outline-none focus:border-border-accent';

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 rounded-md border border-border-button bg-bg-card p-4">
        <Database className="size-5" />
        <div className="min-w-0 flex-1">
          <div className="text-sm font-semibold">离线评测与发布门禁</div>
          <div className="mt-1 text-xs text-text-secondary">
            {cases.length} 个锁定用例 · {runs.length} 次运行
          </div>
        </div>
      </div>
      {notice ? (
        <div
          className="rounded-md border border-border-button bg-bg-card p-3 text-sm"
          role="status"
        >
          {notice}
        </div>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-[360px_minmax(0,1fr)]">
        <Card>
          <CardHeader className="p-5">
            <CardTitle as="h3" className="flex items-center gap-2 text-base">
              <Plus className="size-4" />
              新增回归用例
            </CardTitle>
          </CardHeader>
          <CardContent className="p-5 pt-0">
            <form
              onSubmit={(event) => void createCase(event)}
              className="space-y-3"
            >
              <label className="block text-xs">
                <span className="mb-1.5 block">问题</span>
                <textarea
                  required
                  name="question"
                  rows={3}
                  placeholder="输入代表性客服问题"
                  className="w-full resize-y rounded border border-border-button bg-bg-input p-3 text-sm outline-none"
                />
              </label>
              <label className="block text-xs">
                <span className="mb-1.5 block">目标 Parent ID</span>
                <input
                  name="parents"
                  placeholder="多个 ID 用逗号分隔，可留空"
                  className={field}
                />
              </label>
              <label className="block text-xs">
                <span className="mb-1.5 block">必含事实</span>
                <input
                  name="required"
                  placeholder="例如：断电，排水过滤器"
                  className={field}
                />
              </label>
              <label className="block text-xs">
                <span className="mb-1.5 block">禁答事实</span>
                <input
                  name="forbidden"
                  placeholder="禁止出现在证据中的内容"
                  className={field}
                />
              </label>
              <label className="flex items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  name="locked"
                  defaultChecked
                  className="size-4"
                />
                锁定为发布门禁
              </label>
              <Button
                type="submit"
                className="w-full"
                loading={busy === 'case'}
              >
                <Plus className="size-4" />
                保存用例
              </Button>
            </form>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex-row items-start justify-between space-y-0 p-5">
            <div>
              <CardTitle as="h3" className="text-base">
                候选版本回归
              </CardTitle>
              <p className="mt-1 text-xs text-text-secondary">
                逐例计算 Recall@5、MRR、事实覆盖与禁答违规
              </p>
            </div>
            <select
              aria-label="候选索引版本"
              value={candidate}
              onChange={(event) => setCandidate(event.target.value)}
              className="h-8 max-w-52 rounded border border-border-button bg-bg-input px-3 text-xs"
            >
              {versions.map((version) => (
                <option key={version} value={version}>
                  {version}
                </option>
              ))}
            </select>
          </CardHeader>
          <CardContent className="p-5 pt-0">
            <div className="min-h-40 space-y-2 rounded-md border border-border-button bg-bg-card p-3">
              {cases.length ? (
                cases.map((item) => (
                  <div
                    key={item.id}
                    className="flex items-start gap-3 border-b border-border-button py-2 last:border-b-0"
                  >
                    <Badge variant={item.locked ? 'destructive' : 'secondary'}>
                      {item.locked ? 'LOCK' : 'OPEN'}
                    </Badge>
                    <div className="min-w-0 flex-1">
                      <div className="text-xs font-medium">{item.question}</div>
                      <div className="mt-1 text-[10px] text-text-secondary">
                        {item.required_facts.length} 必含 ·{' '}
                        {item.forbidden_facts.length} 禁答
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="grid min-h-32 place-items-center text-xs text-text-secondary">
                  先添加至少一个回归用例。
                </div>
              )}
            </div>

            {latestRun ? (
              <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
                {Object.entries(latestRun.metrics).map(([name, value]) => (
                  <div
                    key={name}
                    className="rounded-md border border-border-button bg-bg-card p-3"
                  >
                    <div className="text-[10px] text-text-secondary">
                      {name}
                    </div>
                    <div className="mt-2 font-mono text-sm font-semibold">
                      {value.toFixed(3)}
                    </div>
                  </div>
                ))}
              </div>
            ) : null}
            <div className="mt-4 flex justify-end gap-2">
              <Button
                variant="outline"
                onClick={() => void runEvaluation()}
                disabled={!candidate || !cases.length}
                loading={busy === 'run'}
              >
                <Play className="size-4" />
                运行评测
              </Button>
              <Button
                onClick={() => void approveAndPublish()}
                disabled={
                  !latestRun?.passed ||
                  latestRun.candidate_version !== candidate
                }
                loading={busy === 'publish'}
              >
                <CheckCircle2 className="size-4" />
                批准并发布
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
