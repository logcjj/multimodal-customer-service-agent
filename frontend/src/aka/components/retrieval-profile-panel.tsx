import { api } from '@/aka/api/client';
import type { Dataset, RetrievalProfile } from '@/aka/api/types';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Check, Save, SlidersHorizontal } from 'lucide-react';
import { FormEvent, useEffect, useState } from 'react';

const defaults = {
  name: '高精度检索',
  lexical_top_k: 20,
  dense_top_k: 20,
  rrf_k: 60,
  rerank_top_k: 12,
  final_top_n: 5,
  min_score: 0.012,
  min_coverage: 0.1,
  empty_response: '当前知识库中没有足够证据，请补充产品型号或问题细节。',
};

export function RetrievalProfilePanel({
  dataset,
  onDatasetChange,
}: {
  dataset: Dataset;
  onDatasetChange: (value: Dataset) => void;
}) {
  const [profiles, setProfiles] = useState<RetrievalProfile[]>([]);
  const [selectedId, setSelectedId] = useState(
    dataset.retrieval_profile_id ?? '',
  );
  const [draft, setDraft] = useState(defaults);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    void api.retrievalProfiles().then((items) => {
      setProfiles(items);
      const selected = items.find(
        (item) => item.id === dataset.retrieval_profile_id,
      );
      if (selected) setDraft(selected);
    });
  }, [dataset.retrieval_profile_id]);

  function selectProfile(id: string) {
    setSelectedId(id);
    setDraft(profiles.find((item) => item.id === id) ?? defaults);
    setSaved(false);
  }

  function setNumber(field: keyof typeof defaults, value: string) {
    setDraft((current) => ({ ...current, [field]: Number(value) }));
    setSaved(false);
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    try {
      const profile = selectedId
        ? await api.updateRetrievalProfile(selectedId, draft)
        : await api.createRetrievalProfile(draft);
      const updated = await api.updateDataset(dataset.id, {
        retrieval_profile_id: profile.id,
      });
      setProfiles((current) => [
        ...current.filter((item) => item.id !== profile.id),
        profile,
      ]);
      setSelectedId(profile.id);
      onDatasetChange(updated);
      setSaved(true);
    } finally {
      setSaving(false);
    }
  }

  const inputClass =
    'h-9 w-full rounded border border-border-button bg-bg-input px-3 text-sm outline-none focus:border-border-accent';

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0 p-5">
        <div>
          <CardTitle as="h3" className="flex items-center gap-2 text-base">
            <SlidersHorizontal className="size-4" />
            检索 Profile
          </CardTitle>
          <p className="mt-1 text-xs text-text-secondary">
            在线 Agent 与 Retrieval Lab 共用
          </p>
        </div>
        <select
          aria-label="检索配置"
          value={selectedId}
          onChange={(event) => selectProfile(event.target.value)}
          className="h-8 max-w-52 rounded border border-border-button bg-bg-input px-3 text-xs"
        >
          <option value="">新建配置</option>
          {profiles.map((profile) => (
            <option key={profile.id} value={profile.id}>
              {profile.name}
            </option>
          ))}
        </select>
      </CardHeader>
      <CardContent className="p-5 pt-0">
        <form onSubmit={(event) => void save(event)}>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <label className="text-xs">
              <span className="mb-1.5 block">名称</span>
              <input
                className={inputClass}
                value={draft.name}
                onChange={(event) =>
                  setDraft((current) => ({
                    ...current,
                    name: event.target.value,
                  }))
                }
              />
            </label>
            {[
              ['lexical_top_k', 'BM25 Top K', 1, 100, 1],
              ['dense_top_k', 'Dense Top K', 1, 100, 1],
              ['rrf_k', 'RRF K', 1, 500, 1],
              ['rerank_top_k', 'Rerank Top K', 1, 100, 1],
              ['final_top_n', '最终 Top N', 1, 20, 1],
              ['min_score', '最低分', 0, 1, 0.001],
              ['min_coverage', '最低覆盖率', 0, 1, 0.05],
            ].map(([field, label, min, max, step]) => (
              <label key={String(field)} className="text-xs">
                <span className="mb-1.5 block">{String(label)}</span>
                <input
                  className={inputClass}
                  type="number"
                  min={Number(min)}
                  max={Number(max)}
                  step={Number(step)}
                  value={Number(draft[field as keyof typeof defaults])}
                  onChange={(event) =>
                    setNumber(
                      field as keyof typeof defaults,
                      event.target.value,
                    )
                  }
                />
              </label>
            ))}
          </div>
          <label className="mt-4 block text-xs">
            <span className="mb-1.5 block">无证据回复</span>
            <textarea
              rows={3}
              value={draft.empty_response}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  empty_response: event.target.value,
                }))
              }
              className="w-full resize-y rounded border border-border-button bg-bg-input p-3 text-sm outline-none"
            />
          </label>
          <div className="mt-4 flex items-center justify-between gap-3">
            <span className="text-xs text-text-secondary">
              {dataset.retrieval_profile_id
                ? 'Dataset 已绑定可复用 Profile'
                : '当前使用系统默认参数'}
            </span>
            <Button
              type="submit"
              loading={saving}
              disabled={!draft.name.trim()}
            >
              {saved ? (
                <Check className="size-4" />
              ) : (
                <Save className="size-4" />
              )}
              {saved ? '已保存并应用' : '保存并应用'}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
