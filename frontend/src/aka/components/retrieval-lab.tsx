import { api } from '@/aka/api/client';
import type {
  RetrievalExplanation,
  RetrievalStageItem,
  RetrievalVisualizationHit,
  VectorMapPoint,
  VectorMapResponse,
} from '@/aka/api/types';
import { VectorSpaceMap } from '@/aka/components/vector-space-map';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { queryClient, vectorMapQueryKey } from '@/query-client';
import {
  AlertTriangle,
  Clock3,
  Database,
  FlaskConical,
  Maximize2,
  Minimize2,
  RefreshCw,
  Search,
  SlidersHorizontal,
  Wrench,
} from 'lucide-react';
import {
  FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

const POLL_MS = 2000;
const VECTOR_MAP_CACHE_MS = 5 * 60 * 1000;

const stageLabels: Record<string, string> = {
  lexical: 'BM25',
  dense: 'Dense',
  rrf: 'RRF',
  rerank: 'Rerank',
  parent: 'Parent 聚合',
};

type MapRequestMode = 'load' | 'poll' | 'rebuild';
type TopHitStage = 'Dense' | 'Rerank' | 'RRF';
type TopHitView = TopHitStage;

interface TopHitRow {
  id: string;
  stage: TopHitStage;
  childId: string;
  rank: number;
  score: number;
  point: VectorMapPoint;
}

function formatScore(value: number | null | undefined, digits = 4) {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toFixed(digits)
    : Number(0).toFixed(digits);
}

function defaultMotionEnabled() {
  return !(
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  );
}

function pageLabel(point: Pick<VectorMapPoint, 'page_start' | 'page_end'>) {
  return point.page_start === point.page_end
    ? `第 ${point.page_start} 页`
    : `第 ${point.page_start}-${point.page_end} 页`;
}

function shouldPoll(response: VectorMapResponse | null) {
  return (
    response?.status === 'missing' ||
    response?.status === 'building' ||
    response?.status === 'stale'
  );
}

function canRebuild(response: VectorMapResponse | null) {
  return (
    response?.status === 'ready' ||
    response?.status === 'failed'
  );
}

function rebuildLabel(response: VectorMapResponse | null) {
  if (response?.status === 'failed') {
    return '重试投影';
  }
  if (canRebuild(response)) {
    return '重建投影';
  }
  return '不可重建';
}

function normalizeVectorMap(
  response: VectorMapResponse,
  datasetId: string,
): VectorMapResponse {
  if (response.status !== 'ready' && response.status !== 'stale') {
    return response;
  }

  const points = response.points ?? [];
  const pointsBelongToDataset = points.every(
    (point) => point.dataset_id === datasetId,
  );
  if (response.meta?.dataset_id === datasetId && pointsBelongToDataset) {
    return response;
  }

  return {
    status: 'failed',
    meta: null,
    points: [],
    error: {
      code: 'dataset_mismatch',
      message: '投影返回了其他知识库的数据，已忽略。',
    },
  };
}

function projectionState(
  response: VectorMapResponse | null,
  loading: boolean,
  error: string,
  pointCount: number,
  waitSeconds: number,
) {
  if (error) {
    return {
      badge: 'UMAP failed',
      title: '投影读取失败',
      message: error,
      variant: 'destructive' as const,
      icon: AlertTriangle,
    };
  }

  if (loading && !response) {
    return {
      badge: 'UMAP loading',
      title: '读取投影',
      message: '正在加载当前知识库向量空间',
      variant: 'secondary' as const,
      icon: Clock3,
    };
  }

  switch (response?.status) {
    case 'ready':
      return {
        badge: 'UMAP ready',
        title: '投影就绪',
        message: `${pointCount} 个点`,
        variant: 'success' as const,
        icon: Database,
      };
    case 'building':
      return {
        badge: 'UMAP building',
        title: response.message || 'UMAP 投影生成中',
        message: `每 2 秒自动检查，大型知识库首次构建可能需要 1-3 分钟 · 已等待 ${waitSeconds} 秒`,
        variant: 'secondary' as const,
        icon: Clock3,
      };
    case 'stale':
      return {
        badge: 'UMAP stale',
        title: response.message || 'UMAP 投影更新中',
        message: `每 2 秒自动检查，大型知识库首次构建可能需要 1-3 分钟 · 已等待 ${waitSeconds} 秒`,
        variant: 'secondary' as const,
        icon: RefreshCw,
      };
    case 'missing':
      return {
        badge: 'UMAP missing',
        title: response.message || '投影缺失，已触发构建',
        message: '等待后端生成当前知识库投影',
        variant: 'secondary' as const,
        icon: Clock3,
      };
    case 'failed':
      return {
        badge: 'UMAP failed',
        title: response.error?.message || response.message || 'UMAP 构建失败',
        message: response.error?.detail || '可重试当前知识库投影构建',
        variant: 'destructive' as const,
        icon: AlertTriangle,
      };
    case 'no_published_version':
      return {
        badge: 'UMAP blocked',
        title: response.message || '知识库尚未发布',
        message: '发布索引后才能生成投影',
        variant: 'secondary' as const,
        icon: Wrench,
      };
    case 'no_embeddings':
      return {
        badge: 'UMAP blocked',
        title: response.message || '当前知识库尚未完成向量化',
        message: '完成 Embedding 后才能生成投影',
        variant: 'secondary' as const,
        icon: Wrench,
      };
    default:
      return {
        badge: 'UMAP pending',
        title: '等待投影状态',
        message: '读取当前知识库投影',
        variant: 'secondary' as const,
        icon: Clock3,
      };
  }
}

function mapHitRows(
  hits: RetrievalVisualizationHit[],
  stage: TopHitStage,
  pointByChildId: Map<string, VectorMapPoint>,
): TopHitRow[] {
  return hits
    .slice(0, 10)
    .map((hit, index) => ({
      hit,
      rank: hit.rank || index + 1,
      point: pointByChildId.get(hit.child_id),
    }))
    .filter(
      (item): item is {
        hit: RetrievalVisualizationHit;
        rank: number;
        point: VectorMapPoint;
      } => Boolean(item.point),
    )
    .map(({ hit, rank, point }) => ({
      id: `${stage}-${hit.child_id}-${rank}`,
      stage,
      childId: hit.child_id,
      rank,
      score: hit.score,
      point,
    }));
}

function mapProjectionVersion(
  map: {
    meta: { published_version: string; content_digest: string };
  } | null,
) {
  if (!map) {
    return null;
  }
  return `${map.meta.published_version}:${map.meta.content_digest.slice(0, 12)}`;
}

function fallbackRowsFromStage(
  stages: Record<string, RetrievalStageItem[]> | undefined,
  pointByChildId: Map<string, VectorMapPoint>,
): TopHitRow[] {
  const source = stages?.rrf ?? [];
  return source
    .slice(0, 10)
    .flatMap((item, index): TopHitRow[] => {
      const childId = item.child_id || item.id;
      const point = childId ? pointByChildId.get(childId) : undefined;
      if (!childId || !point) {
        return [];
      }
      const rank = item.rank || index + 1;
      return [{
        id: `RRF-${childId}-${rank}`,
        stage: 'RRF',
        childId,
        rank,
        score: item.score ?? 0,
        point,
      }];
    });
}

export function RetrievalLab({
  datasetId,
  publishedVersion,
}: {
  datasetId: string;
  publishedVersion?: string | null;
}) {
  const [query, setQuery] = useState('');
  const [topN, setTopN] = useState(10);
  const [useRerank, setUseRerank] = useState(true);
  const [motionEnabled, setMotionEnabled] = useState(defaultMotionEnabled);
  const [result, setResult] = useState<RetrievalExplanation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [vectorMap, setVectorMap] = useState<VectorMapResponse | null>(null);
  const [mapLoading, setMapLoading] = useState(false);
  const [mapError, setMapError] = useState('');
  const [mapActionLoading, setMapActionLoading] = useState(false);
  const [mapWaitSeconds, setMapWaitSeconds] = useState(0);
  const [focusedChildId, setFocusedChildId] = useState<string | null>(null);
  const [topHitView, setTopHitView] = useState<TopHitView>('Rerank');
  const [nativeFullscreen, setNativeFullscreen] = useState(false);
  const [fallbackFullscreen, setFallbackFullscreen] = useState(false);

  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mapRequestIdRef = useRef(0);
  const retrievalRequestIdRef = useRef(0);
  const mapActionInFlightRef = useRef(false);
  const datasetIdRef = useRef(datasetId);
  const hitRowRefs = useRef(new Map<string, HTMLButtonElement>());
  const workspaceRef = useRef<HTMLDivElement | null>(null);
  const mapWaitStartedAtRef = useRef<number | null>(null);

  const workspaceFullscreen = nativeFullscreen || fallbackFullscreen;
  const vectorMapStatus = vectorMap?.status ?? null;

  datasetIdRef.current = datasetId;

  const clearPollTimer = useCallback(() => {
    if (pollTimerRef.current) {
      clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  const loadVectorMap = useCallback(
    async (mode: MapRequestMode) => {
      if (mode === 'rebuild' && mapActionInFlightRef.current) {
        return;
      }

      const requestId = mapRequestIdRef.current + 1;
      mapRequestIdRef.current = requestId;
      clearPollTimer();

      if (mode === 'rebuild') {
        mapActionInFlightRef.current = true;
        setMapActionLoading(true);
      }
      if (mode === 'load') {
        setMapLoading(true);
      }
      setMapError('');

      try {
        let response: VectorMapResponse;
        if (mode === 'rebuild') {
          response = normalizeVectorMap(
            await api.rebuildVectorMap(datasetId),
            datasetId,
          );
          queryClient.setQueryData(
            vectorMapQueryKey(datasetId, publishedVersion),
            response,
          );
        } else {
          response = await queryClient.fetchQuery({
            queryKey: vectorMapQueryKey(datasetId, publishedVersion),
            queryFn: async () =>
              normalizeVectorMap(await api.vectorMap(datasetId), datasetId),
            staleTime: mode === 'load' ? VECTOR_MAP_CACHE_MS : 0,
          });
        }
        if (
          requestId !== mapRequestIdRef.current ||
          datasetId !== datasetIdRef.current
        ) {
          return;
        }
        setVectorMap(response);
      } catch (reason) {
        if (
          requestId !== mapRequestIdRef.current ||
          datasetId !== datasetIdRef.current
        ) {
          return;
        }
        setVectorMap(null);
        setMapError(reason instanceof Error ? reason.message : '投影加载失败');
      } finally {
        if (
          requestId === mapRequestIdRef.current &&
          datasetId === datasetIdRef.current
        ) {
          if (mode === 'load') {
            setMapLoading(false);
          }
          if (mode === 'rebuild') {
            setMapActionLoading(false);
          }
        }
        if (mode === 'rebuild') {
          mapActionInFlightRef.current = false;
        }
      }
    },
    [clearPollTimer, datasetId, publishedVersion],
  );

  useEffect(() => {
    mapRequestIdRef.current += 1;
    retrievalRequestIdRef.current += 1;
    mapActionInFlightRef.current = false;
    clearPollTimer();
    hitRowRefs.current.clear();
    setVectorMap(null);
    setMapError('');
    setMapLoading(false);
    setMapActionLoading(false);
    setResult(null);
    setError('');
    setFocusedChildId(null);
    setTopHitView('Rerank');
    void loadVectorMap('load');

    return () => {
      mapRequestIdRef.current += 1;
      retrievalRequestIdRef.current += 1;
      clearPollTimer();
    };
  }, [clearPollTimer, datasetId, loadVectorMap, publishedVersion]);

  useEffect(() => {
    clearPollTimer();
    if (shouldPoll(vectorMap)) {
      pollTimerRef.current = setTimeout(() => {
        void loadVectorMap('poll');
      }, POLL_MS);
    }

    return clearPollTimer;
  }, [clearPollTimer, loadVectorMap, vectorMap]);

  useEffect(() => {
    if (!shouldPoll(vectorMap)) {
      mapWaitStartedAtRef.current = null;
      setMapWaitSeconds(0);
      return undefined;
    }
    if (mapWaitStartedAtRef.current === null) {
      mapWaitStartedAtRef.current = Date.now();
    }
    const updateElapsed = () => {
      const startedAt = mapWaitStartedAtRef.current ?? Date.now();
      setMapWaitSeconds(Math.max(0, Math.floor((Date.now() - startedAt) / 1000)));
    };
    updateElapsed();
    const timer = window.setInterval(updateElapsed, 1000);
    return () => window.clearInterval(timer);
  }, [vectorMap, vectorMapStatus]);

  useEffect(() => {
    const onFullscreenChange = () => {
      setNativeFullscreen(document.fullscreenElement === workspaceRef.current);
    };
    document.addEventListener('fullscreenchange', onFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', onFullscreenChange);
    };
  }, []);

  useEffect(() => {
    if (!workspaceFullscreen) {
      return undefined;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [workspaceFullscreen]);

  useEffect(() => {
    if (!fallbackFullscreen) {
      return undefined;
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setFallbackFullscreen(false);
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [fallbackFullscreen]);

  const toggleWorkspaceFullscreen = useCallback(async () => {
    const workspace = workspaceRef.current;
    if (!workspace) {
      return;
    }

    if (workspaceFullscreen) {
      if (
        document.fullscreenElement &&
        typeof document.exitFullscreen === 'function'
      ) {
        try {
          await document.exitFullscreen();
        } catch {
          // A stale browser fullscreen state should not trap the CSS fallback.
        }
      }
      setNativeFullscreen(false);
      setFallbackFullscreen(false);
      return;
    }

    if (typeof workspace.requestFullscreen === 'function') {
      try {
        await workspace.requestFullscreen();
        setNativeFullscreen(true);
        return;
      } catch {
        // Embedded browsers may expose but reject Fullscreen API.
      }
    }
    setFallbackFullscreen(true);
  }, [workspaceFullscreen]);

  const displayMap = useMemo(() => {
    if (
      vectorMap &&
      (vectorMap.status === 'ready' || vectorMap.status === 'stale') &&
      vectorMap.meta?.dataset_id === datasetId &&
      typeof vectorMap.meta.published_version === 'string' &&
      typeof vectorMap.meta.content_digest === 'string' &&
      Array.isArray(vectorMap.points)
    ) {
      return vectorMap as typeof vectorMap & {
        meta: {
          dataset_id: string;
          published_version: string;
          content_digest: string;
        };
        points: VectorMapPoint[];
      };
    }
    return null;
  }, [datasetId, vectorMap]);
  const mapPoints = useMemo(() => displayMap?.points ?? [], [displayMap]);
  const rawVisualization = result?.visualization ?? null;
  const currentProjectionVersion = mapProjectionVersion(displayMap);
  const visualization =
    rawVisualization?.projection_version === currentProjectionVersion
      ? rawVisualization
      : null;
  const visualizationVersionMismatch = Boolean(
    rawVisualization &&
      currentProjectionVersion &&
      rawVisualization.projection_version !== currentProjectionVersion,
  );
  const denseHits = useMemo(
    () => visualization?.dense_top10 ?? [],
    [visualization],
  );
  const rerankHits = useMemo(
    () => visualization?.rerank_top10 ?? [],
    [visualization],
  );
  const rrfHits = useMemo(
    () => visualization?.rrf_top10 ?? [],
    [visualization],
  );

  const pointByChildId = useMemo(() => {
    const byId = new Map<string, VectorMapPoint>();
    mapPoints.forEach((point) => {
      byId.set(point.child_id, point);
    });
    return byId;
  }, [mapPoints]);

  const rerankRows = useMemo(
    () => mapHitRows(rerankHits, 'Rerank', pointByChildId),
    [pointByChildId, rerankHits],
  );
  const denseRows = useMemo(
    () => mapHitRows(denseHits, 'Dense', pointByChildId),
    [denseHits, pointByChildId],
  );
  const rrfRows = useMemo(() => {
    const rows = mapHitRows(rrfHits, 'RRF', pointByChildId);
    return rows.length || !visualization
      ? rows
      : fallbackRowsFromStage(result?.stages, pointByChildId);
  }, [pointByChildId, result?.stages, rrfHits, visualization]);
  const rowsByView = useMemo(
    () => ({ Rerank: rerankRows, Dense: denseRows, RRF: rrfRows }),
    [denseRows, rerankRows, rrfRows],
  );
  const topHitRows = rowsByView[topHitView];
  const availableHitViews = useMemo(
    () =>
      (['Rerank', 'Dense', 'RRF'] as const).filter(
        (stage) => rowsByView[stage].length > 0,
      ),
    [rowsByView],
  );

  const projection = projectionState(
    vectorMap,
    mapLoading,
    mapError,
    mapPoints.length,
    mapWaitSeconds,
  );
  const ProjectionIcon = projection.icon;
  const rebuildEnabled = canRebuild(vectorMap) && !mapActionLoading;

  useEffect(() => {
    if (rerankRows.length) {
      setTopHitView('Rerank');
      return;
    }
    if (denseRows.length) {
      setTopHitView('Dense');
      return;
    }
    setTopHitView('RRF');
  }, [denseRows.length, rawVisualization, rerankRows.length]);

  const onMapFocusedChildChange = useCallback(
    (childId: string | null) => {
      setFocusedChildId(childId);
      if (!childId) {
        return;
      }
      if (rerankRows.some((row) => row.childId === childId)) {
        setTopHitView('Rerank');
        return;
      }
      if (denseRows.some((row) => row.childId === childId)) {
        setTopHitView('Dense');
        return;
      }
      if (rrfRows.some((row) => row.childId === childId)) {
        setTopHitView('RRF');
      }
    },
    [denseRows, rerankRows, rrfRows],
  );

  useEffect(() => {
    if (!focusedChildId) {
      return;
    }

    const preferredRow = topHitRows.find(
      (row) => row.childId === focusedChildId,
    );
    if (!preferredRow) {
      return;
    }

    hitRowRefs.current.get(preferredRow.id)?.scrollIntoView({
      block: 'nearest',
      behavior: motionEnabled ? 'smooth' : 'auto',
    });
  }, [focusedChildId, motionEnabled, topHitRows]);

  async function run(event: FormEvent) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) {
      return;
    }

    const requestId = retrievalRequestIdRef.current + 1;
    retrievalRequestIdRef.current = requestId;
    setLoading(true);
    setError('');
    try {
      const nextResult = await api.retrievalTest({
        dataset_ids: [datasetId],
        query: trimmed,
        top_n: topN,
        use_rerank: useRerank,
      });
      if (
        requestId !== retrievalRequestIdRef.current ||
        datasetId !== datasetIdRef.current
      ) {
        return;
      }
      setResult(nextResult);
      setFocusedChildId(null);
    } catch (reason) {
      if (
        requestId !== retrievalRequestIdRef.current ||
        datasetId !== datasetIdRef.current
      ) {
        return;
      }
      setError(reason instanceof Error ? reason.message : '检索失败');
    } finally {
      if (
        requestId === retrievalRequestIdRef.current &&
        datasetId === datasetIdRef.current
      ) {
        setLoading(false);
      }
    }
  }

  const onTopNChange = (value: string) => {
    const next = Number(value);
    if (!Number.isFinite(next)) {
      setTopN(10);
      return;
    }
    setTopN(Math.min(20, Math.max(1, Math.round(next))));
  };

  return (
    <div className="space-y-4">
      <section className="border-y border-border-button bg-bg-card/40 px-4 py-3">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex min-w-0 flex-wrap items-center gap-2 text-sm">
            <Database className="size-4 shrink-0 text-text-secondary" />
            <span className="text-xs text-text-secondary">知识库</span>
            <strong className="truncate font-mono text-xs">{datasetId}</strong>
            <span className="mx-1 h-4 w-px bg-border-button" />
            <ProjectionIcon className="size-4 shrink-0 text-text-secondary" />
            <Badge variant={projection.variant}>{projection.badge}</Badge>
            <span className="text-sm font-medium">{projection.title}</span>
            <span className="text-xs text-text-secondary">
              {projection.message}
            </span>
            {displayMap?.meta.published_version ? (
              <span className="rounded border border-border-button bg-bg-input px-2 py-1 font-mono text-[11px]">
                {displayMap.meta.published_version}
              </span>
            ) : null}
          </div>
          <Button
            type="button"
            size="sm"
            variant="outline"
            aria-label={rebuildLabel(vectorMap)}
            disabled={!rebuildEnabled}
            loading={mapActionLoading}
            onClick={() => void loadVectorMap('rebuild')}
          >
            <RefreshCw className="size-4" />
            {rebuildLabel(vectorMap)}
          </Button>
        </div>
      </section>

      <form
        onSubmit={(event) => void run(event)}
        className="grid gap-4 border-y border-border-button bg-bg-card/30 p-4 lg:grid-cols-[minmax(0,1fr)_360px]"
      >
        <div className="min-w-0">
          <label
            className="mb-2 block text-xs font-medium"
            htmlFor="retrieval-query"
          >
            测试问题
          </label>
          <textarea
            id="retrieval-query"
            aria-label="测试问题"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            rows={3}
            placeholder="输入一个问题，单独验证能否召回正确证据"
            className="w-full resize-y rounded-md border border-border-button bg-bg-input px-3 py-2 text-sm leading-6 outline-none focus:border-border-accent"
          />
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Button
              type="submit"
              disabled={!query.trim()}
              loading={loading}
              aria-label="运行检索"
            >
              <Search className="size-4" />
              运行检索
            </Button>
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
          <label className="block space-y-1.5 text-xs">
            <span className="flex items-center gap-1.5 text-text-secondary">
              <SlidersHorizontal className="size-3.5" />
              Top N
            </span>
            <input
              aria-label="Top N"
              type="number"
              min={1}
              max={20}
              value={topN}
              onChange={(event) => onTopNChange(event.target.value)}
              className="h-8 w-full rounded border border-border-button bg-bg-input px-3 outline-none"
            />
          </label>
          <label className="flex h-8 items-center justify-between gap-3 rounded border border-border-button bg-bg-input px-3 text-xs">
            <span className="font-medium">Rerank</span>
            <input
              aria-label="Rerank"
              type="checkbox"
              checked={useRerank}
              onChange={(event) => setUseRerank(event.target.checked)}
              className="size-4"
            />
          </label>
        </div>
      </form>

      {error ? (
        <div className="rounded-md border border-state-error/40 bg-state-error/5 p-3 text-sm text-state-error">
          {error}
        </div>
      ) : null}

      {result?.warnings?.length ? (
        <div className="space-y-1 rounded-md border border-state-warning/40 bg-state-warning/5 px-3 py-2 text-xs text-state-warning">
          {result.warnings.map((warning) => (
            <div key={warning}>{warning}</div>
          ))}
        </div>
      ) : null}

      {visualizationVersionMismatch ? (
        <div className="rounded-md border border-state-warning/40 bg-state-warning/5 px-3 py-2 text-xs text-state-warning">
          投影版本已更新，请重新运行检索以刷新高亮。
        </div>
      ) : null}

      <div
        ref={workspaceRef}
        data-testid="retrieval-visual-workspace"
        data-fullscreen={workspaceFullscreen ? 'true' : 'false'}
        className={cn(
          'relative min-w-0 bg-bg-base',
          workspaceFullscreen &&
            'z-50 h-screen w-screen overflow-auto p-3 sm:p-4 lg:overflow-hidden',
          fallbackFullscreen && 'fixed inset-0',
        )}
      >
        <div
          className={cn(
            'grid min-h-[620px] gap-4 lg:h-[clamp(620px,calc(100vh-260px),980px)] lg:grid-cols-[minmax(0,1fr)_minmax(320px,360px)]',
            workspaceFullscreen && 'min-h-0 lg:h-full',
          )}
        >
        <section className="h-[clamp(460px,64vh,680px)] min-h-0 min-w-0 lg:h-full">
          {displayMap ? (
            <VectorSpaceMap
              points={mapPoints}
              queryPoint={visualization?.query ?? null}
              denseHits={denseHits}
              rerankHits={rerankHits}
              rrfHits={rrfHits}
              onMotionEnabledChange={setMotionEnabled}
              focusedChildId={focusedChildId}
              onFocusedChildChange={onMapFocusedChildChange}
            />
          ) : (
            <div className="grid h-full min-h-[420px] place-items-center rounded-md border border-dashed border-border-button bg-bg-card/40 p-6 text-center">
              <div>
                <ProjectionIcon className="mx-auto size-6 text-text-secondary" />
                <div className="mt-3 text-sm font-medium">
                  向量图暂不可用
                </div>
                <p className="mt-1 text-xs text-text-secondary">
                  {projection.message}
                </p>
              </div>
            </div>
          )}
        </section>

        <section className="min-w-0 space-y-3 lg:flex lg:min-h-0 lg:flex-col lg:gap-3 lg:space-y-0">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-sm font-semibold">
              <FlaskConical className="size-4" />
              Top 10 命中
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs tabular-nums text-text-secondary">
                {topHitRows.length}
              </span>
              <Button
                type="button"
                variant="outline"
                size="icon"
                title={workspaceFullscreen ? '退出全屏' : '全屏查看'}
                aria-label={workspaceFullscreen ? '退出全屏' : '全屏查看'}
                onClick={() => void toggleWorkspaceFullscreen()}
              >
                {workspaceFullscreen ? (
                  <Minimize2 className="size-4" />
                ) : (
                  <Maximize2 className="size-4" />
                )}
              </Button>
            </div>
          </div>
          {availableHitViews.length > 1 ? (
            <div
              role="group"
              aria-label="命中阶段"
              className="grid grid-cols-3 gap-1 rounded-md border border-border-button bg-bg-input p-1"
            >
              {availableHitViews.map((stage) => (
                <button
                  key={stage}
                  type="button"
                  aria-label={`${stage} Top 10`}
                  aria-pressed={topHitView === stage}
                  onClick={() => setTopHitView(stage)}
                  className={cn(
                    'h-7 rounded px-2 text-xs transition-colors',
                    topHitView === stage
                      ? 'bg-bg-card font-medium text-text-primary shadow-sm'
                      : 'text-text-secondary hover:text-text-primary',
                  )}
                >
                  {stage} Top 10
                </button>
              ))}
            </div>
          ) : null}
          <div
            role="list"
            aria-label="Top 10 命中"
            className="space-y-2 lg:min-h-0 lg:flex-1 lg:overflow-y-auto lg:pr-1"
          >
            {topHitRows.length ? (
              topHitRows.map((row, index) => {
                const selected = focusedChildId === row.childId;
                return (
                  <div key={row.id} role="listitem">
                    <button
                      ref={(node) => {
                        if (node) {
                          hitRowRefs.current.set(row.id, node);
                        } else {
                          hitRowRefs.current.delete(row.id);
                        }
                      }}
                      type="button"
                      data-testid={`retrieval-hit-row-${row.stage}-${row.childId}-${row.rank}`}
                      aria-pressed={selected}
                      onClick={() => setFocusedChildId(row.childId)}
                      className={cn(
                        'w-full rounded-md border bg-bg-card p-3 text-left transition-colors',
                        motionEnabled &&
                          'animate-in fade-in-0 slide-in-from-right-2 duration-300',
                        selected
                          ? 'border-accent-primary ring-1 ring-accent-primary/50'
                          : 'border-border-button hover:border-border-default',
                      )}
                      style={
                        motionEnabled
                          ? {
                              animationDelay: `${index * 60}ms`,
                              animationFillMode: 'both',
                            }
                          : undefined
                      }
                    >
                      <div className="flex min-w-0 items-center gap-2">
                        <Badge
                          variant={
                            row.stage === 'Rerank' ? 'success' : 'secondary'
                          }
                        >
                          {row.stage}
                        </Badge>
                        <span className="font-mono text-xs text-text-secondary">
                          #{row.rank}
                        </span>
                        <strong className="ml-auto font-mono text-xs">
                          {formatScore(row.score)}
                        </strong>
                      </div>
                      <div className="mt-2 truncate text-sm font-medium">
                        {row.point.title}
                      </div>
                      <p className="mt-1 line-clamp-2 text-xs leading-5 text-text-secondary">
                        {row.point.excerpt}
                      </p>
                      <div className="mt-2 flex min-w-0 flex-wrap gap-x-2 gap-y-1 text-[11px] text-text-secondary">
                        <span className="truncate">{row.point.document_name}</span>
                        <span>{pageLabel(row.point)}</span>
                        <span className="min-w-0 max-w-full truncate font-mono">
                          {row.childId}
                        </span>
                      </div>
                    </button>
                  </div>
                );
              })
            ) : (
              <div className="rounded-md border border-dashed border-border-button p-4 text-sm text-text-secondary">
                {result
                  ? '当前投影没有可映射的 Top 10 命中'
                  : '等待检索结果'}
              </div>
            )}
          </div>
        </section>
        </div>
      </div>

      {!result ? (
        <div className="grid min-h-40 place-items-center rounded-md border border-dashed border-border-button text-center">
          <div>
            <FlaskConical className="mx-auto size-6 text-text-secondary" />
            <div className="mt-3 text-sm font-medium">
              先验证检索，再验证生成
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <section className="flex flex-col gap-3 border-y border-border-button bg-bg-card/40 p-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-2">
              <FlaskConical className="size-4" />
              <Badge
                variant={
                  result.mode === 'lexical-only' ? 'destructive' : 'success'
                }
              >
                {result.mode}
              </Badge>
              <span className="text-xs text-text-secondary">
                {result.results.length} 个 Parent
              </span>
            </div>
            <div className="flex flex-wrap gap-2">
              {Object.entries(stageLabels).map(([key, label]) => (
                <span
                  key={key}
                  className="rounded border border-border-button bg-bg-input px-2 py-1 text-[11px]"
                >
                  {label}{' '}
                  <b className="ml-1 tabular-nums">
                    {result.stages[key]?.length ?? 0}
                  </b>
                </span>
              ))}
            </div>
          </section>

          {result.rejected_reason ? (
            <div className="rounded-md border border-state-warning/40 bg-state-warning/5 p-4 text-sm">
              {result.rejected_reason}
            </div>
          ) : (
            <div className="space-y-3">
              {result.results.map((item, index) => (
                <article
                  key={item.parent_id}
                  className="rounded-md border border-border-button bg-bg-card p-4"
                >
                  <header className="flex items-start gap-3">
                    <span className="grid size-7 shrink-0 place-items-center rounded bg-bg-component text-xs tabular-nums">
                      {String(index + 1).padStart(2, '0')}
                    </span>
                    <div className="min-w-0 flex-1">
                      <h3 className="truncate text-sm font-semibold">
                        {item.title}
                      </h3>
                      <div className="mt-1 text-xs text-text-secondary">
                        第 {item.page_start}-{item.page_end} 页 ·{' '}
                        {item.matched_children.length} Child
                      </div>
                    </div>
                    <strong className="font-mono text-xs">
                      {item.scores.parent.toFixed(4)}
                    </strong>
                  </header>
                  <p className="mt-3 line-clamp-4 whitespace-pre-wrap text-sm leading-6 text-text-secondary">
                    {item.text}
                  </p>
                  <dl className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-5">
                    {[
                      ['BM25', item.scores.lexical, 3],
                      ['Dense', item.scores.dense, 3],
                      ['RRF', item.scores.rrf, 4],
                      ['Rerank', item.scores.rerank, 3],
                      ['Parent', item.scores.parent, 4],
                    ].map(([name, value, digits]) => (
                      <div
                        key={String(name)}
                        className="rounded border border-border-button bg-bg-card p-2"
                      >
                        <dt className="text-[10px] text-text-secondary">
                          {String(name)}
                        </dt>
                        <dd className="mt-1 font-mono text-xs">
                          {Number(value).toFixed(Number(digits))}
                        </dd>
                      </div>
                    ))}
                  </dl>
                </article>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
