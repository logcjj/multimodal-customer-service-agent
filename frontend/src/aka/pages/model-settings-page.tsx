import { api } from '@/aka/api/client';
import type {
  ModelConfig,
  Provider,
  ProviderModelPreset,
} from '@/aka/api/types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Eye,
  FlaskConical,
  ListOrdered,
  Plus,
  ScanText,
  Search,
  Trash2,
} from 'lucide-react';
import { FormEvent, useEffect, useMemo, useState } from 'react';

const modelKinds = ['llm', 'embedding', 'rerank', 'vlm', 'ocr', 'asr', 'tts'];
const defaults: Record<string, string> = {
  deepseek: 'https://api.deepseek.com/v1',
  openai: 'https://api.openai.com/v1',
  anthropic: 'https://api.anthropic.com/v1',
  'tongyi-qianwen': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
  ollama: 'http://127.0.0.1:11434/v1',
  'openai-compatible': 'http://127.0.0.1:8002/v1',
};

function defaultBaseUrl(providerId: string, kind: string) {
  if (providerId === 'tongyi-qianwen' && kind === 'rerank') {
    return 'https://dashscope.aliyuncs.com/compatible-api/v1';
  }
  return defaults[providerId] ?? '';
}

function inferModelKind(name: string, capabilities: string[]) {
  const normalized = name.trim().toLowerCase();
  if (
    capabilities.includes('ocr') &&
    /(?:^|[-_.])ocr(?:$|[-_.])/.test(normalized)
  ) {
    return 'ocr';
  }
  if (
    capabilities.includes('vlm') &&
    /(?:^|[-_.])(?:vl|vision)(?:$|[-_.])/.test(normalized)
  ) {
    return 'vlm';
  }
  return null;
}

function ProviderDialog({
  provider,
  preferredKind,
  preferredName,
  onClose,
  onCreated,
}: {
  provider: Provider | null;
  preferredKind?: string;
  preferredName?: string;
  onClose: () => void;
  onCreated: (model: ModelConfig) => void;
}) {
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [name, setName] = useState('');
  const [kind, setKind] = useState('llm');
  const [baseUrl, setBaseUrl] = useState('');
  const [kindTouched, setKindTouched] = useState(false);

  useEffect(() => {
    if (!provider) return;
    const nextKind =
      preferredKind && provider.capabilities.includes(preferredKind)
        ? preferredKind
        : (provider.capabilities[0] ?? 'llm');
    setName(preferredName ?? '');
    setKind(nextKind);
    setBaseUrl(defaultBaseUrl(provider.id, nextKind));
    setKindTouched(false);
    setError('');
  }, [preferredKind, preferredName, provider]);

  if (!provider) return null;

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const selectedKind = String(
      form.get('kind') ?? provider!.capabilities[0] ?? 'llm',
    );
    setSaving(true);
    setError('');
    try {
      const created = await api.createModel({
        provider: provider!.id,
        name: String(form.get('name') ?? ''),
        kind: selectedKind,
        base_url: String(form.get('base_url') ?? ''),
        api_key: String(form.get('api_key') ?? '') || null,
        capabilities: [selectedKind],
        enabled: true,
      });
      onCreated(created);
      onClose();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '保存失败');
    } finally {
      setSaving(false);
    }
  }

  const field =
    'h-9 w-full rounded border border-border-button bg-bg-input px-3 text-sm outline-none';
  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent>
        <form onSubmit={(event) => void submit(event)}>
          <DialogHeader>
            <DialogTitle>添加模型配置</DialogTitle>
            <DialogDescription>
              {provider.name} · 密钥仅加密写入后端
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-5">
            <label className="block text-xs">
              <span className="mb-1.5 block">模型名称</span>
              <input
                name="name"
                required
                autoFocus
                value={name}
                onChange={(event) => {
                  const nextName = event.target.value;
                  setName(nextName);
                  if (!kindTouched) {
                    const inferred = inferModelKind(
                      nextName,
                      provider.capabilities,
                    );
                    if (inferred) setKind(inferred);
                  }
                }}
                placeholder="例如 deepseek-v4-flash"
                className={field}
              />
            </label>
            <label className="block text-xs">
              <span className="mb-1.5 block">模型类型</span>
              <select
                name="kind"
                value={kind}
                onChange={(event) => {
                  setKind(event.target.value);
                  setKindTouched(true);
                }}
                className={field}
              >
                {provider.capabilities.map((item) => (
                  <option key={item} value={item}>
                    {item.toUpperCase()}
                  </option>
                ))}
              </select>
            </label>
            <label className="block text-xs">
              <span className="mb-1.5 block">API 地址</span>
              <input
                name="base_url"
                required
                value={baseUrl}
                onChange={(event) => setBaseUrl(event.target.value)}
                className={field}
              />
            </label>
            <label className="block text-xs">
              <span className="mb-1.5 block">API 密钥</span>
              <input
                name="api_key"
                type="password"
                autoComplete="new-password"
                placeholder={
                  provider.id === 'ollama' ? '本地模型可留空' : '仅加密写入后端'
                }
                className={field}
              />
            </label>
            {error ? (
              <p className="text-xs text-state-error" role="alert">
                {error}
              </p>
            ) : null}
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={onClose}>
              取消
            </Button>
            <Button type="submit" loading={saving}>
              保存模型
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export default function ModelSettingsPage() {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<Provider | null>(
    null,
  );
  const [selectedPresetName, setSelectedPresetName] = useState('');
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [busyModel, setBusyModel] = useState('');

  async function load() {
    setLoading(true);
    try {
      const [providerData, modelData] = await Promise.all([
        api.providers(),
        api.models(),
      ]);
      setProviders(providerData);
      setModels(modelData);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '模型配置加载失败');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  const visibleProviders = useMemo(
    () =>
      providers.filter(
        (provider) =>
          provider.name.toLowerCase().includes(query.toLowerCase()) &&
          (filter === 'all' || provider.capabilities.includes(filter)),
      ),
    [filter, providers, query],
  );

  const hasPresetsForFilter = useMemo(
    () =>
      filter !== 'all' &&
      providers.some(
        (provider) => (provider.model_presets?.[filter]?.length ?? 0) > 0,
      ),
    [filter, providers],
  );

  const visibleModelPresets = useMemo(() => {
    if (!hasPresetsForFilter) return [];
    const normalizedQuery = query.trim().toLowerCase();
    return providers.flatMap((provider) =>
      (provider.model_presets?.[filter] ?? [])
        .filter(
          (preset) =>
            !normalizedQuery ||
            preset.name.toLowerCase().includes(normalizedQuery) ||
            preset.description.toLowerCase().includes(normalizedQuery) ||
            provider.name.toLowerCase().includes(normalizedQuery),
        )
        .map((preset) => ({ provider, preset })),
    );
  }, [filter, hasPresetsForFilter, providers, query]);

  const visibleProvidersWithoutPresets = useMemo(
    () =>
      hasPresetsForFilter
        ? visibleProviders.filter(
            (provider) =>
              (provider.model_presets?.[filter]?.length ?? 0) === 0,
          )
        : [],
    [filter, hasPresetsForFilter, visibleProviders],
  );

  async function setDefault(id: string) {
    if (!id) return;
    const updated = await api.setDefaultModel(id);
    setModels((current) =>
      current.map((item) =>
        item.kind === updated.kind
          ? { ...item, is_default: item.id === updated.id }
          : item,
      ),
    );
    setNotice(`${updated.name} 已设为默认模型`);
  }

  async function updateModelKind(model: ModelConfig, kind: string) {
    if (model.kind === kind) return;
    setBusyModel(model.id);
    try {
      const updated = await api.updateModel(model.id, { kind });
      setModels((current) =>
        current.map((item) => {
          if (item.id === updated.id) return updated;
          if (
            updated.is_default &&
            item.kind === updated.kind &&
            item.is_default
          ) {
            return { ...item, is_default: false };
          }
          return item;
        }),
      );
      setNotice(
        `${updated.name} 已调整为 ${updated.kind.toUpperCase()}，请重新测试连接`,
      );
    } catch (reason) {
      setNotice(reason instanceof Error ? reason.message : '模型类型调整失败');
    } finally {
      setBusyModel('');
    }
  }

  async function testModel(model: ModelConfig) {
    setBusyModel(model.id);
    try {
      const result = await api.testModel(model.id);
      setModels((current) =>
        current.map((item) =>
          item.id === model.id
            ? {
                ...item,
                health: result.health as ModelConfig['health'],
                latency_ms: result.latency_ms,
              }
            : item,
        ),
      );
      setNotice(
        `${result.message}${result.latency_ms !== null ? ` · ${result.latency_ms} ms` : ''}`,
      );
    } catch (reason) {
      setNotice(reason instanceof Error ? reason.message : '连接测试失败');
    } finally {
      setBusyModel('');
    }
  }

  async function remove(model: ModelConfig) {
    setBusyModel(model.id);
    try {
      await api.deleteModel(model.id);
      setModels((current) => current.filter((item) => item.id !== model.id));
      setNotice(`${model.name} 已删除`);
    } finally {
      setBusyModel('');
    }
  }

  const defaultLlm = models.find(
    (model) => model.kind === 'llm' && model.is_default && model.enabled,
  );

  return (
    <div className="grid size-full min-w-0 grid-cols-1 xl:grid-cols-[minmax(0,1fr)_400px]">
      <section className="min-h-0 overflow-auto px-5 py-6 lg:px-8 lg:py-8">
        <div className="mx-auto max-w-5xl space-y-5">
          <header className="flex items-end justify-between gap-4">
            <div>
              <h1 className="text-2xl font-semibold">模型配置</h1>
              <p className="mt-1 text-sm text-text-secondary">
                {models.length} 个实例 ·{' '}
                {models.filter((item) => item.is_default).length} 个默认槽位
              </p>
            </div>
            <Badge variant={defaultLlm ? 'success' : 'destructive'}>
              {defaultLlm ? `${defaultLlm.name} · LLM` : 'LLM 未配置'}
            </Badge>
          </header>
          {loading ? (
            <div className="p-8 text-center text-sm text-text-secondary">
              加载模型配置
            </div>
          ) : null}
          {error ? (
            <div className="rounded-md border border-state-error/40 p-3 text-sm text-state-error">
              {error}
            </div>
          ) : null}
          {!loading && !error ? (
            <>
              <Card>
                <CardHeader className="p-5">
                  <CardTitle as="h2" className="text-base">
                    默认模型角色
                  </CardTitle>
                </CardHeader>
                <CardContent className="grid gap-3 p-5 pt-0 sm:grid-cols-2">
                  {modelKinds.map((kind) => {
                    const options = models.filter(
                      (model) => model.kind === kind && model.enabled,
                    );
                    return (
                      <label key={kind} className="text-xs">
                        <span className="mb-1.5 block">
                          {kind.toUpperCase()}
                        </span>
                        <select
                          aria-label={`选择默认 ${kind.toUpperCase()} 模型`}
                          value={
                            options.find((model) => model.is_default)?.id ?? ''
                          }
                          onChange={(event) =>
                            void setDefault(event.target.value)
                          }
                          className="h-9 w-full rounded border border-border-button bg-bg-input px-3"
                        >
                          <option value="">选择模型</option>
                          {options.map((model) => (
                            <option key={model.id} value={model.id}>
                              {model.name} · {model.provider}
                            </option>
                          ))}
                        </select>
                      </label>
                    );
                  })}
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="p-5">
                  <CardTitle as="h2" className="text-base">
                    已配置模型
                  </CardTitle>
                  <p className="text-xs text-text-secondary">
                    凭据只显示掩码，不回传明文
                  </p>
                </CardHeader>
                <CardContent className="space-y-2 p-5 pt-0">
                  {models.map((model) => (
                    <article
                      key={model.id}
                      className="flex flex-col gap-3 rounded-md border border-border-button bg-bg-card p-3 sm:flex-row sm:items-center"
                    >
                      <span
                        className={`size-2 shrink-0 rounded-full ${model.health === 'healthy' ? 'bg-state-success' : model.health === 'unhealthy' ? 'bg-state-error' : 'bg-state-warning'}`}
                      />
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm font-medium">
                          {model.name}
                        </div>
                        <div className="mt-1 flex min-w-0 items-center gap-1 text-xs text-text-secondary">
                          <span className="truncate">{model.provider} ·</span>
                          <select
                            aria-label={`设置 ${model.name} 模型类型`}
                            value={model.kind}
                            disabled={busyModel === model.id}
                            onChange={(event) =>
                              void updateModelKind(model, event.target.value)
                            }
                            className="h-6 rounded border border-border-button bg-bg-input px-1 text-[10px] text-text-primary outline-none"
                          >
                            {modelKinds.map((item) => (
                              <option key={item} value={item}>
                                {item.toUpperCase()}
                              </option>
                            ))}
                          </select>
                          <span className="truncate">
                            · {model.secret_hint ?? '无密钥'}
                          </span>
                        </div>
                      </div>
                      {model.is_default ? (
                        <Badge variant="success">默认</Badge>
                      ) : null}
                      <Badge
                        variant={
                          model.health === 'healthy'
                            ? 'success'
                            : model.health === 'unhealthy'
                              ? 'destructive'
                              : 'outline'
                        }
                      >
                        {model.health === 'healthy'
                          ? '健康'
                          : model.health === 'unhealthy'
                            ? '不可用'
                            : '未测试'}
                      </Badge>
                      <div className="flex gap-1">
                        <Button
                          variant="ghost"
                          size="icon"
                          aria-label={`测试 ${model.name}`}
                          loading={busyModel === model.id}
                          onClick={() => void testModel(model)}
                        >
                          <FlaskConical className="size-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          aria-label={`删除 ${model.name}`}
                          disabled={model.is_default || busyModel === model.id}
                          onClick={() => void remove(model)}
                        >
                          <Trash2 className="size-4" />
                        </Button>
                      </div>
                    </article>
                  ))}
                  {!models.length ? (
                    <div className="p-8 text-center text-sm text-text-secondary">
                      尚未配置模型
                    </div>
                  ) : null}
                  {notice ? (
                    <p
                      className="rounded-md border border-border-button p-3 text-xs"
                      role="status"
                    >
                      {notice}
                    </p>
                  ) : null}
                </CardContent>
              </Card>
            </>
          ) : null}
        </div>
      </section>

      <aside className="min-h-0 overflow-auto border-l border-border-button bg-bg-card p-5">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold">
              {hasPresetsForFilter
                ? `${filter.toUpperCase()} 模型与提供商`
                : '模型提供商'}
            </h2>
            <p className="mt-1 text-xs text-text-secondary">
              {hasPresetsForFilter
                ? '选择模型预设或提供商后补充访问凭据'
                : '选择后添加实例'}
            </p>
          </div>
          <Badge variant="secondary">
            {hasPresetsForFilter
              ? visibleModelPresets.length +
                visibleProvidersWithoutPresets.length
              : visibleProviders.length}
          </Badge>
        </div>
        <label className="mb-3 flex h-9 items-center gap-2 rounded-md border border-border-button bg-bg-input px-3">
          <Search className="size-4 text-text-secondary" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={
              hasPresetsForFilter ? '搜索模型或提供商' : '搜索模型提供商'
            }
            aria-label={
              hasPresetsForFilter ? '搜索模型或提供商' : '搜索模型提供商'
            }
            className="min-w-0 flex-1 bg-transparent text-sm outline-none"
          />
        </label>
        <div className="mb-4 flex flex-wrap gap-1">
          {['all', ...modelKinds].map((item) => (
            <Button
              key={item}
              variant={filter === item ? 'default' : 'ghost'}
              size="xs"
              onClick={() => setFilter(item)}
            >
              {item === 'all' ? 'All' : item.toUpperCase()}
            </Button>
          ))}
        </div>
        <div className="space-y-2">
          {hasPresetsForFilter
            ? visibleModelPresets.map(({ provider, preset }) => (
                <ModelPresetCard
                  key={`${provider.id}:${filter}:${preset.name}`}
                  provider={provider}
                  preset={preset}
                  kind={filter}
                  onAdd={() => {
                    setSelectedPresetName(preset.name);
                    setSelectedProvider(provider);
                  }}
                />
              ))
            : null}
          {(hasPresetsForFilter
            ? visibleProvidersWithoutPresets
            : visibleProviders
          ).map((provider) => (
            <div
              key={provider.id}
              className="flex items-center gap-3 rounded-md border border-border-button bg-bg-input p-3"
            >
              <div className="grid size-9 place-items-center rounded bg-bg-component text-xs font-semibold">
                {provider.name.slice(0, 2).toUpperCase()}
              </div>
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium">
                  {provider.name}
                </div>
                <div className="mt-1 truncate text-[10px] text-text-secondary">
                  {provider.capabilities.join(' · ')}
                </div>
              </div>
              <Button
                variant="outline"
                size="sm"
                aria-label={`添加 ${provider.name}`}
                onClick={() => {
                  setSelectedPresetName('');
                  setSelectedProvider(provider);
                }}
              >
                <Plus className="size-3.5" />
                添加
              </Button>
            </div>
          ))}
        </div>
      </aside>

      <ProviderDialog
        provider={selectedProvider}
        preferredKind={filter === 'all' ? undefined : filter}
        preferredName={selectedPresetName || undefined}
        onClose={() => {
          setSelectedProvider(null);
          setSelectedPresetName('');
        }}
        onCreated={(created) => {
          setModels((current) => [...current, created]);
          setNotice(`${created.name} 已添加`);
        }}
      />
    </div>
  );
}

function ModelPresetCard({
  provider,
  preset,
  kind,
  onAdd,
}: {
  provider: Provider;
  preset: ProviderModelPreset;
  kind: string;
  onAdd: () => void;
}) {
  const PresetIcon =
    kind === 'vlm' ? Eye : kind === 'rerank' ? ListOrdered : ScanText;

  return (
    <div className="flex items-start gap-3 rounded-md border border-border-button bg-bg-input p-3">
      <div className="grid size-9 shrink-0 place-items-center rounded bg-bg-component text-text-secondary">
        <PresetIcon className="size-4" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate text-sm font-medium">{preset.name}</span>
        </div>
        <div className="mt-1 text-[10px] text-text-secondary">
          {provider.name} · {kind.toUpperCase()}
        </div>
        <p className="mt-1 line-clamp-2 text-[10px] leading-4 text-text-secondary">
          {preset.description}
        </p>
      </div>
      <Button
        variant="outline"
        size="sm"
        aria-label={`添加 ${preset.name}`}
        onClick={onAdd}
      >
        <Plus className="size-3.5" />
        添加
      </Button>
    </div>
  );
}
