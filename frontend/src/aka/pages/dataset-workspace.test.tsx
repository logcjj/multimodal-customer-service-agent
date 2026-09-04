import { api } from '@/aka/api/client';
import { RetrievalLab } from '@/aka/components/retrieval-lab';
import { queryClient } from '@/query-client';
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router';
import DatasetDetailPage from './dataset-detail-page';
import DatasetsPage from './datasets-page';

type RetrievalExplanationType = import('@/aka/api/types').RetrievalExplanation;
type VectorMapPointType = import('@/aka/api/types').VectorMapPoint;
type VectorMapResponseType = import('@/aka/api/types').VectorMapResponse;
type VectorSpaceMapPropsType =
  import('@/aka/components/vector-space-map').VectorSpaceMapProps;

const mockVectorSpaceMapCalls: VectorSpaceMapPropsType[] = [];
const mockScrollIntoView = jest.fn();

class MockResizeObserver {
  observe = jest.fn();
  unobserve = jest.fn();
  disconnect = jest.fn();
}

jest.mock('@/aka/components/vector-space-map', () => {
  // Jest executes this factory after hoisting, so load React inside the factory.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const React = require('react');

  return {
    VectorSpaceMap: (props: VectorSpaceMapPropsType) => {
      mockVectorSpaceMapCalls.push(props);

      return React.createElement(
        'button',
        {
          type: 'button',
          'aria-label': '知识库 UMAP 向量空间',
          'data-testid': 'vector-space-map',
          'data-point-count': String(props.points.length),
          'data-point-ids': props.points
            .map((point) => point.child_id)
            .join('|'),
          'data-query': props.queryPoint
            ? `${props.queryPoint.x},${props.queryPoint.y}`
            : '',
          'data-dense': props.denseHits
            .map((hit) => `${hit.child_id}:${hit.rank}:${hit.score}`)
            .join('|'),
          'data-rerank': props.rerankHits
            .map((hit) => `${hit.child_id}:${hit.rank}:${hit.score}`)
            .join('|'),
          'data-rrf': props.rrfHits
            .map((hit) => `${hit.child_id}:${hit.rank}:${hit.score}`)
            .join('|'),
          'data-focused': props.focusedChildId ?? '',
          'data-motion': String(props.motionEnabled),
          onClick: () => props.onFocusedChildChange?.('child-2'),
          onDoubleClick: () => props.onFocusedChildChange?.('child-1'),
        },
        '知识库 UMAP 向量空间',
      );
    },
  };
});

jest.mock('@/aka/api/client', () => ({
  api: {
    datasets: jest.fn(),
    dataset: jest.fn(),
    documents: jest.fn(),
    vectorMap: jest.fn(),
    rebuildVectorMap: jest.fn(),
    retrievalTest: jest.fn(),
    chunks: jest.fn(),
    updateChunk: jest.fn(),
    retrievalProfiles: jest.fn(),
    evalCases: jest.fn(),
    evalRuns: jest.fn(),
    indexManifest: jest.fn(),
    imageChunks: jest.fn(),
  },
}));

const mockedApi = api as Record<string, jest.Mock>;

const dataset = {
  id: 'v6-manuals',
  name: 'V6 图文说明书知识库',
  description: '39 类产品说明书',
  parser_profile: 'manual',
  visibility: 'private',
  published_version: 'v6-import-v1',
  status: 'ready',
  is_system: true,
  retrieval_profile_id: null,
  document_count: 39,
  parent_count: 1943,
  child_count: 6570,
  asset_count: 2563,
  failed_job_count: 0,
  created_at: '2026-07-23T00:00:00',
  updated_at: '2026-07-23T00:00:00',
};

const document = {
  id: 'doc-air',
  dataset_id: dataset.id,
  file_id: 'file-air',
  original_name: '空气净化器手册.json',
  mime_type: 'application/json',
  parser_profile: 'manual',
  enabled: true,
  active_version: 'v6-import-v1',
  published_version: 'v6-import-v1',
  latest_job_state: null,
  latest_job_progress: null,
  created_at: '2026-07-23T00:00:00',
  updated_at: '2026-07-23T00:00:00',
};

const indexManifest = {
  schema_version: 'aka-index-bundle-v1',
  dataset_id: dataset.id,
  index_version: 'idx-bundle-v1',
  parent_index_version: null,
  built_at: '2026-07-24T08:00:00Z',
  parser_version: 'parser-v3',
  embedding_model: 'text-embedding-v4',
  vector_dimension: 1024,
  sources: [],
  artifacts: {},
  counts: { text_chunks: 8513, image_chunks: 2563, assets: 2563 },
  incremental: { reused: 39, added: 0, updated: 0, deleted: 0 },
  validation_status: 'valid',
  evaluation_status: 'not_run',
  approval_status: 'awaiting_approval',
};

const imageChunks = [
  {
    id: 'image-chunk-1',
    dataset_id: dataset.id,
    document_id: document.id,
    index_version: 'idx-bundle-v1',
    asset_id: 'asset-1',
    asset_url: '/api/assets/asset-1',
    image_id: 'manual-image-1',
    manual_name: '空气净化器手册',
    chapter_title: '滤网清洁',
    page_number: 11,
    caption: '预过滤网拆卸位置示意图',
    ocr_text: 'PRE-FILTER',
    visible_text: ['PRE-FILTER'],
    visual_summary: '图中标出预过滤网位置',
    visual_meaning: '帮助用户定位滤网',
    retrieval_text: '空气净化器 预过滤网 清洁 定位',
    search_terms: ['预过滤网', '清洁'],
    applicable_questions: ['滤网在哪里'],
    issue_signals: [],
    related_parent_ids: ['parent-2'],
    related_child_ids: ['child-2'],
    confidence: 0.96,
    content_hash: 'hash-1',
    embedding_dimension: 1024,
    enabled: true,
  },
];

const chunkCollection = {
  parents: [
    {
      id: 'parent-1',
      document_id: document.id,
      index_version: 'v6-import-v1',
      title: '安装准备',
      heading_path: ['安装准备'],
      text: '准备安装滤网。',
      page_start: 10,
      page_end: 10,
      token_count: 8,
      enabled: true,
      edited: false,
    },
    {
      id: 'parent-2',
      document_id: document.id,
      index_version: 'v6-import-v1',
      title: '滤网清洁定位章节',
      heading_path: ['维护', '滤网清洁'],
      text: '使用吸尘器清理预过滤网。',
      page_start: 11,
      page_end: 12,
      token_count: 15,
      enabled: true,
      edited: false,
    },
  ],
  children: [
    {
      id: 'child-1',
      parent_id: 'parent-1',
      document_id: document.id,
      index_version: 'v6-import-v1',
      title: '安装准备',
      text: '准备安装滤网。',
      page_start: 10,
      page_end: 10,
      token_count: 8,
      keywords: [],
      questions: [],
      tags: [],
      asset_ids: [],
      enabled: true,
      edited: false,
    },
    {
      id: 'child-2',
      parent_id: 'parent-2',
      document_id: document.id,
      index_version: 'v6-import-v1',
      title: '滤网清洁',
      text: '目标 Child：使用吸尘器清理预过滤网。',
      page_start: 11,
      page_end: 11,
      token_count: 18,
      keywords: ['滤网'],
      questions: [],
      tags: ['维护'],
      asset_ids: [],
      enabled: true,
      edited: false,
    },
  ],
};

function vectorPoint(
  childId: string,
  overrides: Partial<VectorMapPointType> = {},
): VectorMapPointType {
  return {
    child_id: childId,
    dataset_id: dataset.id,
    document_id: document.id,
    document_name: document.original_name,
    title: `标题 ${childId}`,
    excerpt: `摘要 ${childId}`,
    page_start: 10,
    page_end: 10,
    product: '空气净化器手册',
    x: 0.25,
    y: 0.5,
    ...overrides,
  };
}

const readyVectorMap: VectorMapResponseType = {
  status: 'ready',
  meta: {
    dataset_id: dataset.id,
    published_version: 'v6-import-v1',
    embedding_model: 'text-embedding-v4',
    vector_dimension: 3,
    point_count: 2,
    bounds: {
      x_min: 0,
      x_max: 1,
      y_min: 0,
      y_max: 1,
    },
    content_digest: 'vector-map-test-digest',
    built_at: '2026-07-24T00:00:00Z',
    umap: {
      n_components: 2,
      metric: 'cosine',
      n_neighbors: 1,
      min_dist: 0.1,
      random_state: 42,
      transform_seed: 42,
      low_memory: true,
      reducer_available: false,
      small_sample: true,
    },
  },
  points: [
    vectorPoint('child-1', {
      title: '滤网清洁、更换与重置',
      excerpt: '可使用吸尘器或软刷清洁预过滤网。',
      page_start: 11,
      page_end: 11,
      x: 0.25,
      y: 0.5,
    }),
    vectorPoint('child-2', {
      title: '滤网复位',
      excerpt: '更换滤网后按住复位键三秒。',
      page_start: 12,
      page_end: 12,
      x: 0.75,
      y: 0.5,
    }),
  ],
};

const buildingVectorMap: VectorMapResponseType = {
  status: 'building',
  meta: null,
  points: [],
  message: 'UMAP 投影生成中',
};

const staleVectorMap: VectorMapResponseType = {
  status: 'stale',
  meta: {
    ...readyVectorMap.meta,
    stale: true,
    target_published_version: 'idx-new',
  },
  points: readyVectorMap.points,
  message: '正在为当前发布版本重建，暂时显示上一版向量图。',
};

const failedVectorMap: VectorMapResponseType = {
  status: 'failed',
  meta: null,
  points: [],
  error: {
    code: 'umap_failed',
    message: 'UMAP 构建失败',
  },
};

const retrievalExplanation: RetrievalExplanationType = {
  query: '空气净化器滤网清洁',
  mode: 'lexical-only',
  rejected_reason: null,
  stages: {
    lexical: Array.from({ length: 20 }, (_, index) => ({
      id: `lexical-${index}`,
      title: '滤网清洁候选',
      score: 38.73 - index,
      document_id: document.id,
      page_start: 11,
    })),
    dense: [],
    rrf: Array.from({ length: 20 }, (_, index) => ({
      id: `rrf-${index}`,
      title: 'RRF 候选',
      score: 0.0164,
      document_id: document.id,
      page_start: 11,
    })),
    rerank: [],
    parent: [
      {
        id: 'parent-1',
        title: '滤网清洁、更换与重置',
        score: 0.0184,
      },
    ],
  },
  visualization: {
    projection_version: 'v6-import-v1:vector-map-t',
    query: { x: 0.61, y: 0.42 },
    dense_top10: [{ child_id: 'child-1', rank: 1, score: 0.91 }],
    rerank_top10: [{ child_id: 'child-2', rank: 1, score: 0.88 }],
    rrf_top10: [{ child_id: 'child-1', rank: 1, score: 0.0164 }],
  },
  results: [
    {
      parent_id: 'parent-1',
      dataset_id: dataset.id,
      document_id: document.id,
      document_version: 'v6-import-v1',
      title: '滤网清洁、更换与重置',
      text: '可使用吸尘器或软刷清洁预过滤网，切勿用水清洗滤网。',
      product: '空气净化器手册',
      page_start: 11,
      page_end: 11,
      asset_ids: [],
      matched_children: ['child-1', 'child-2'],
      scores: {
        lexical: 38.73,
        dense: 0,
        rrf: 0.0164,
        rerank: 0,
        parent: 0.0184,
      },
    },
  ],
};

describe('RAGFlow Dataset workspace', () => {
  beforeEach(() => {
    jest.useRealTimers();
    queryClient.clear();
    Object.values(mockedApi).forEach((mock) => mock.mockReset());
    mockVectorSpaceMapCalls.length = 0;
    mockScrollIntoView.mockClear();
    global.ResizeObserver =
      MockResizeObserver as unknown as typeof ResizeObserver;
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
      configurable: true,
      value: mockScrollIntoView,
    });
    mockedApi.datasets.mockResolvedValue([dataset]);
    mockedApi.dataset.mockResolvedValue(dataset);
    mockedApi.documents.mockResolvedValue([document]);
    mockedApi.vectorMap.mockResolvedValue(readyVectorMap);
    mockedApi.rebuildVectorMap.mockResolvedValue({
      ...readyVectorMap,
      status: 'building',
      meta: null,
      points: [],
    });
    mockedApi.retrievalProfiles.mockResolvedValue([]);
    mockedApi.evalCases.mockResolvedValue([]);
    mockedApi.evalRuns.mockResolvedValue([]);
    mockedApi.retrievalTest.mockResolvedValue(retrievalExplanation);
    mockedApi.chunks.mockResolvedValue(chunkCollection);
    mockedApi.indexManifest.mockResolvedValue(indexManifest);
    mockedApi.imageChunks.mockResolvedValue(imageChunks);
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('shows real Dataset metrics in RAGFlow cards', async () => {
    render(
      <MemoryRouter>
        <DatasetsPage />
      </MemoryRouter>,
    );

    expect(
      (await screen.findAllByText('V6 图文说明书知识库')).length,
    ).toBeGreaterThan(0);
    expect(screen.getByText('6,570 Chunks')).toBeInTheDocument();
    expect(screen.getByText('1,943 Parent')).toBeInTheDocument();
    expect(
      screen.getByRole('link', { name: /V6 图文说明书知识库/ }),
    ).toHaveAttribute('href', '/dataset/v6-manuals');
  });

  it('runs the real retrieval contract and exposes stage scores', async () => {
    render(
      <MemoryRouter initialEntries={['/dataset/v6-manuals']}>
        <Routes>
          <Route path="/dataset/:datasetId" element={<DatasetDetailPage />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText('空气净化器手册.json')).toBeInTheDocument();
    expect(screen.getByText('已发布')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '检索测试' }));
    fireEvent.change(screen.getByRole('textbox', { name: '测试问题' }), {
      target: { value: '空气净化器滤网怎么清洁？' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行检索' }));

    expect(await screen.findByText('lexical-only')).toBeInTheDocument();
    expect(mockedApi.retrievalTest).toHaveBeenCalledWith(
      expect.objectContaining({
        dataset_ids: ['v6-manuals'],
        top_n: 10,
        use_rerank: true,
      }),
    );
    expect(screen.getAllByText('滤网清洁、更换与重置').length).toBeGreaterThan(0);
    expect(screen.getByText('38.730')).toBeInTheDocument();
    expect(screen.getAllByText('0.0184').length).toBeGreaterThan(0);
  });

  it('shows the active index manifest and independent image chunks', async () => {
    render(
      <MemoryRouter initialEntries={['/dataset/v6-manuals']}>
        <Routes>
          <Route path="/dataset/:datasetId" element={<DatasetDetailPage />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText('空气净化器手册.json')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '图片知识' }));

    expect(
      (await screen.findAllByText('预过滤网拆卸位置示意图')).length,
    ).toBeGreaterThan(0);
    expect(screen.getByText('idx-bundle-v1')).toBeInTheDocument();
    expect(screen.getAllByText('1,024').length).toBeGreaterThan(0);
    expect(screen.getByText('等待评测批准')).toBeInTheDocument();
    expect(mockedApi.indexManifest).toHaveBeenCalledWith('v6-manuals');
    expect(mockedApi.imageChunks).toHaveBeenCalledWith('v6-manuals');
  });

  it('requests the current dataset vector map only after entering retrieval tab', async () => {
    render(
      <MemoryRouter initialEntries={['/dataset/v6-manuals']}>
        <Routes>
          <Route path="/dataset/:datasetId" element={<DatasetDetailPage />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText('空气净化器手册.json')).toBeInTheDocument();
    expect(mockedApi.vectorMap).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: '检索测试' }));

    await waitFor(() => {
      expect(mockedApi.vectorMap).toHaveBeenCalledTimes(1);
    });
    expect(mockedApi.vectorMap).toHaveBeenCalledWith('v6-manuals');
  });

  it('widens only the retrieval tab content for the vector workspace', async () => {
    render(
      <MemoryRouter initialEntries={['/dataset/v6-manuals']}>
        <Routes>
          <Route path="/dataset/:datasetId" element={<DatasetDetailPage />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText('空气净化器手册.json')).toBeInTheDocument();
    expect(screen.getByTestId('dataset-content')).toHaveAttribute(
      'data-wide-layout',
      'false',
    );

    fireEvent.click(screen.getByRole('button', { name: '检索测试' }));

    expect(screen.getByTestId('dataset-content')).toHaveAttribute(
      'data-wide-layout',
      'true',
    );
    expect(
      await screen.findByTestId('retrieval-visual-workspace'),
    ).toBeInTheDocument();
  });

  it('opens the requested document and highlights the linked Parent and Child from URL params', async () => {
    render(
      <MemoryRouter
        initialEntries={[
          '/dataset/v6-manuals?tab=chunks&document=doc-air&child=child-2',
        ]}
      >
        <Routes>
          <Route path="/dataset/:datasetId" element={<DatasetDetailPage />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText('文档结构')).toBeInTheDocument();
    expect(screen.getByRole('combobox')).toHaveValue('doc-air');
    expect(screen.getByTestId('chunk-parent-parent-2')).toHaveAttribute(
      'aria-current',
      'true',
    );
    expect(screen.getByTestId('chunk-child-child-2')).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    expect(screen.getByRole('textbox', { name: '正文' })).toHaveValue(
      '目标 Child：使用吸尘器清理预过滤网。',
    );
    expect(mockScrollIntoView).toHaveBeenCalled();
  });

  it('enters and exits native fullscreen for the whole vector workspace', async () => {
    let fullscreenElement: Element | null = null;
    const requestFullscreen = jest.fn(async () => {
      fullscreenElement = screen.getByTestId('retrieval-visual-workspace');
      global.document.dispatchEvent(new Event('fullscreenchange'));
    });
    const exitFullscreen = jest.fn(async () => {
      fullscreenElement = null;
      global.document.dispatchEvent(new Event('fullscreenchange'));
    });
    Object.defineProperty(global.document, 'fullscreenElement', {
      configurable: true,
      get: () => fullscreenElement,
    });
    Object.defineProperty(HTMLElement.prototype, 'requestFullscreen', {
      configurable: true,
      value: requestFullscreen,
    });
    Object.defineProperty(global.document, 'exitFullscreen', {
      configurable: true,
      value: exitFullscreen,
    });

    render(<RetrievalLab datasetId="v6-manuals" />);

    await screen.findByTestId('retrieval-visual-workspace');
    fireEvent.click(screen.getByRole('button', { name: '全屏查看' }));

    await waitFor(() => expect(requestFullscreen).toHaveBeenCalledTimes(1));
    expect(screen.getByTestId('retrieval-visual-workspace')).toHaveAttribute(
      'data-fullscreen',
      'true',
    );
    fireEvent.click(screen.getByRole('button', { name: '退出全屏' }));

    await waitFor(() => expect(exitFullscreen).toHaveBeenCalledTimes(1));
    expect(screen.getByTestId('retrieval-visual-workspace')).toHaveAttribute(
      'data-fullscreen',
      'false',
    );
  });

  it('falls back to a viewport overlay when Fullscreen API is unavailable', async () => {
    Object.defineProperty(HTMLElement.prototype, 'requestFullscreen', {
      configurable: true,
      value: undefined,
    });

    render(<RetrievalLab datasetId="v6-manuals" />);

    await screen.findByTestId('retrieval-visual-workspace');
    fireEvent.click(screen.getByRole('button', { name: '全屏查看' }));

    expect(screen.getByTestId('retrieval-visual-workspace')).toHaveAttribute(
      'data-fullscreen',
      'true',
    );
    act(() => {
      global.document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    });
    expect(screen.getByTestId('retrieval-visual-workspace')).toHaveAttribute(
      'data-fullscreen',
      'false',
    );
  });

  it('reuses the React Query vector map cache when the retrieval view remounts', async () => {
    const first = render(<RetrievalLab datasetId="v6-manuals" />);

    expect(await screen.findByText('2 个点')).toBeInTheDocument();
    first.unmount();
    render(<RetrievalLab datasetId="v6-manuals" />);

    expect(await screen.findByText('2 个点')).toBeInTheDocument();
    expect(mockedApi.vectorMap).toHaveBeenCalledTimes(1);
  });

  it('uses the dataset published revision in the vector map cache key', async () => {
    const first = render(
      <RetrievalLab datasetId="v6-manuals" publishedVersion="v6-import-v1" />,
    );
    expect(await screen.findByText('2 个点')).toBeInTheDocument();
    first.unmount();

    render(<RetrievalLab datasetId="v6-manuals" publishedVersion="idx-new" />);

    await waitFor(() => expect(mockedApi.vectorMap).toHaveBeenCalledTimes(2));
  });

  it('renders a ready vector map with true point count and canvas props', async () => {
    render(<RetrievalLab datasetId="v6-manuals" />);

    expect(await screen.findByText('2 个点')).toBeInTheDocument();
    expect(screen.getByText('UMAP ready')).toBeInTheDocument();
    expect(screen.getByLabelText('知识库 UMAP 向量空间')).toBeInTheDocument();
    expect(screen.getByTestId('vector-space-map')).toHaveAttribute(
      'data-point-count',
      '2',
    );
    expect(screen.getByTestId('vector-space-map')).toHaveAttribute(
      'data-point-ids',
      'child-1|child-2',
    );
  });

  it('polls every two seconds while the vector map is building and stops when ready', async () => {
    jest.useFakeTimers();
    mockedApi.vectorMap
      .mockResolvedValueOnce(buildingVectorMap)
      .mockResolvedValueOnce(readyVectorMap);

    render(<RetrievalLab datasetId="v6-manuals" />);

    expect(await screen.findByText('UMAP 投影生成中')).toBeInTheDocument();
    expect(
      screen.getAllByText(/每 2 秒自动检查，大型知识库首次构建可能需要 1-3 分钟/),
    ).not.toHaveLength(0);
    expect(mockedApi.vectorMap).toHaveBeenCalledTimes(1);

    act(() => {
      jest.advanceTimersByTime(1999);
    });
    expect(mockedApi.vectorMap).toHaveBeenCalledTimes(1);

    act(() => {
      jest.advanceTimersByTime(1);
    });

    await waitFor(() => {
      expect(mockedApi.vectorMap).toHaveBeenCalledTimes(2);
    });
    expect(await screen.findByText('2 个点')).toBeInTheDocument();

    act(() => {
      jest.advanceTimersByTime(4000);
    });
    expect(mockedApi.vectorMap).toHaveBeenCalledTimes(2);
  });

  it('keeps the previous projection visible while the new revision is building', async () => {
    mockedApi.vectorMap.mockResolvedValueOnce(staleVectorMap);

    render(
      <RetrievalLab datasetId="v6-manuals" publishedVersion="idx-new" />,
    );

    expect(await screen.findByText('UMAP stale')).toBeInTheDocument();
    expect(screen.getByTestId('vector-space-map')).toHaveAttribute(
      'data-point-count',
      '2',
    );
    expect(screen.getByText(/暂时显示上一版向量图/)).toBeInTheDocument();
  });

  it('shows failed projection state and prevents duplicate rebuild clicks', async () => {
    let resolveRebuild: (value: VectorMapResponseType) => void = () => undefined;
    const rebuildPromise = new Promise<VectorMapResponseType>((resolve) => {
      resolveRebuild = resolve;
    });
    mockedApi.vectorMap.mockResolvedValueOnce(failedVectorMap);
    mockedApi.rebuildVectorMap.mockReturnValueOnce(rebuildPromise);

    render(<RetrievalLab datasetId="v6-manuals" />);

    expect(await screen.findByText('UMAP 构建失败')).toBeInTheDocument();
    const retryButton = screen.getByRole('button', { name: '重试投影' });
    fireEvent.click(retryButton);
    fireEvent.click(retryButton);

    expect(mockedApi.rebuildVectorMap).toHaveBeenCalledTimes(1);
    expect(mockedApi.rebuildVectorMap).toHaveBeenCalledWith('v6-manuals');

    await act(async () => {
      resolveRebuild(readyVectorMap);
      await rebuildPromise;
    });

    expect(await screen.findByTestId('vector-space-map')).toHaveAttribute(
      'data-point-count',
      '2',
    );
  });

  it('passes both highlight stages to canvas and exposes Rerank and Dense Top 10 views', async () => {
    render(<RetrievalLab datasetId="v6-manuals" />);

    await screen.findByTestId('vector-space-map');
    fireEvent.change(screen.getByRole('textbox', { name: '测试问题' }), {
      target: { value: '空气净化器滤网怎么清洁？' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行检索' }));

    expect(await screen.findByText('lexical-only')).toBeInTheDocument();
    expect(screen.getByTestId('vector-space-map')).toHaveAttribute(
      'data-query',
      '0.61,0.42',
    );
    expect(screen.getByTestId('vector-space-map')).toHaveAttribute(
      'data-dense',
      'child-1:1:0.91',
    );
    expect(screen.getByTestId('vector-space-map')).toHaveAttribute(
      'data-rerank',
      'child-2:1:0.88',
    );
    expect(screen.getByTestId('vector-space-map')).toHaveAttribute(
      'data-rrf',
      'child-1:1:0.0164',
    );

    const topList = screen.getByRole('list', { name: 'Top 10 命中' });
    expect(screen.getByRole('button', { name: 'Rerank Top 10' })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    expect(screen.getByRole('button', { name: 'Dense Top 10' })).toHaveAttribute(
      'aria-pressed',
      'false',
    );
    expect(
      within(topList).queryByTestId('retrieval-hit-row-Dense-child-1-1'),
    ).not.toBeInTheDocument();
    expect(
      within(topList).getByTestId('retrieval-hit-row-Rerank-child-2-1'),
    ).toHaveTextContent('Rerank');
    expect(
      within(topList).getByTestId('retrieval-hit-row-Rerank-child-2-1'),
    ).toHaveTextContent('0.8800');
    expect(
      within(topList).getByTestId('retrieval-hit-row-Rerank-child-2-1'),
    ).toHaveTextContent('滤网复位');
    expect(
      within(topList).getByTestId('retrieval-hit-row-Rerank-child-2-1'),
    ).toHaveTextContent('更换滤网后按住复位键三秒。');
    expect(
      within(topList).getByTestId('retrieval-hit-row-Rerank-child-2-1'),
    ).toHaveTextContent('空气净化器手册.json');
    expect(
      within(topList).getByTestId('retrieval-hit-row-Rerank-child-2-1'),
    ).toHaveTextContent('第 12 页');

    fireEvent.click(screen.getByRole('button', { name: 'Dense Top 10' }));

    expect(
      within(topList).getByTestId('retrieval-hit-row-Dense-child-1-1'),
    ).toHaveTextContent('0.9100');
  });

  it('switches to the Dense list when the canvas selects a Dense-only point', async () => {
    render(<RetrievalLab datasetId="v6-manuals" />);

    const map = await screen.findByTestId('vector-space-map');
    fireEvent.change(screen.getByRole('textbox', { name: '测试问题' }), {
      target: { value: '空气净化器滤网怎么清洁？' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行检索' }));
    await screen.findByTestId('retrieval-hit-row-Rerank-child-2-1');

    fireEvent.doubleClick(map);

    const denseRow = await screen.findByTestId(
      'retrieval-hit-row-Dense-child-1-1',
    );
    expect(denseRow).toHaveAttribute('aria-pressed', 'true');
    expect(mockScrollIntoView).toHaveBeenCalled();
  });

  it('hides stale query highlights when retrieval and projection versions differ', async () => {
    mockedApi.retrievalTest.mockResolvedValueOnce({
      ...retrievalExplanation,
      visualization: {
        ...retrievalExplanation.visualization,
        projection_version: 'v6-import-v1:obsolete-map',
      },
    });
    render(<RetrievalLab datasetId="v6-manuals" />);

    await screen.findByTestId('vector-space-map');
    fireEvent.change(screen.getByRole('textbox', { name: '测试问题' }), {
      target: { value: '空气净化器滤网怎么清洁？' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行检索' }));

    expect(
      await screen.findByText('投影版本已更新，请重新运行检索以刷新高亮。'),
    ).toBeInTheDocument();
    expect(screen.getByTestId('vector-space-map')).toHaveAttribute(
      'data-query',
      '',
    );
    expect(screen.getByTestId('vector-space-map')).toHaveAttribute(
      'data-dense',
      '',
    );
    expect(screen.getByTestId('vector-space-map')).toHaveAttribute(
      'data-rerank',
      '',
    );
  });

  it('falls back to Dense Top 10 when Rerank has no hits', async () => {
    mockedApi.retrievalTest.mockResolvedValueOnce({
      ...retrievalExplanation,
      visualization: {
        ...retrievalExplanation.visualization,
        rerank_top10: [],
      },
    });
    render(<RetrievalLab datasetId="v6-manuals" />);

    await screen.findByTestId('vector-space-map');
    fireEvent.change(screen.getByRole('textbox', { name: '测试问题' }), {
      target: { value: '空气净化器滤网怎么清洁？' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行检索' }));

    const denseRow = await screen.findByTestId(
      'retrieval-hit-row-Dense-child-1-1',
    );
    expect(denseRow).toHaveTextContent('Dense');
    expect(denseRow).toHaveTextContent('0.9100');
    expect(denseRow).toHaveTextContent('滤网清洁、更换与重置');
  });

  it('focuses the canvas point when a Top 10 row is clicked', async () => {
    render(<RetrievalLab datasetId="v6-manuals" />);

    await screen.findByTestId('vector-space-map');
    fireEvent.change(screen.getByRole('textbox', { name: '测试问题' }), {
      target: { value: '空气净化器滤网怎么清洁？' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行检索' }));

    const rerankRow = await screen.findByTestId(
      'retrieval-hit-row-Rerank-child-2-1',
    );
    fireEvent.click(rerankRow);

    expect(screen.getByTestId('vector-space-map')).toHaveAttribute(
      'data-focused',
      'child-2',
    );
    expect(rerankRow).toHaveAttribute('aria-pressed', 'true');
  });

  it('selects and scrolls the matching Top 10 row when the canvas reports a clicked point', async () => {
    render(<RetrievalLab datasetId="v6-manuals" />);

    await screen.findByTestId('vector-space-map');
    fireEvent.change(screen.getByRole('textbox', { name: '测试问题' }), {
      target: { value: '空气净化器滤网怎么清洁？' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行检索' }));

    const rerankRow = await screen.findByTestId(
      'retrieval-hit-row-Rerank-child-2-1',
    );
    fireEvent.click(screen.getByTestId('vector-space-map'));

    await waitFor(() => {
      expect(rerankRow).toHaveAttribute('aria-pressed', 'true');
    });
    expect(mockScrollIntoView).toHaveBeenCalledWith(
      expect.objectContaining({ block: 'nearest' }),
    );
  });

  it('clears old map data, query highlights and focused state when dataset changes', async () => {
    const nextVectorMap: VectorMapResponseType = {
      ...readyVectorMap,
      meta: {
        ...readyVectorMap.meta,
        dataset_id: 'kb-next',
        point_count: 1,
      },
      points: [
        vectorPoint('next-child', {
          dataset_id: 'kb-next',
          title: '下一知识库 Chunk',
        }),
      ],
    };
    mockedApi.vectorMap
      .mockResolvedValueOnce(readyVectorMap)
      .mockResolvedValueOnce(nextVectorMap);

    const { rerender } = render(<RetrievalLab datasetId="v6-manuals" />);

    await screen.findByTestId('vector-space-map');
    fireEvent.change(screen.getByRole('textbox', { name: '测试问题' }), {
      target: { value: '空气净化器滤网怎么清洁？' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行检索' }));
    const rerankRow = await screen.findByTestId(
      'retrieval-hit-row-Rerank-child-2-1',
    );
    fireEvent.click(rerankRow);
    expect(screen.getByTestId('vector-space-map')).toHaveAttribute(
      'data-query',
      '0.61,0.42',
    );
    expect(screen.getByTestId('vector-space-map')).toHaveAttribute(
      'data-focused',
      'child-2',
    );

    rerender(<RetrievalLab datasetId="kb-next" />);

    await waitFor(() => {
      expect(mockedApi.vectorMap).toHaveBeenLastCalledWith('kb-next');
    });
    expect(
      screen.queryByTestId('retrieval-hit-row-Rerank-child-2-1'),
    ).not.toBeInTheDocument();
    expect(await screen.findByText('1 个点')).toBeInTheDocument();
    expect(screen.getByTestId('vector-space-map')).toHaveAttribute(
      'data-point-ids',
      'next-child',
    );
    expect(screen.getByTestId('vector-space-map')).toHaveAttribute(
      'data-query',
      '',
    );
    expect(screen.getByTestId('vector-space-map')).toHaveAttribute(
      'data-focused',
      '',
    );
  });

  it('keeps retrieval working when the vector map failed', async () => {
    mockedApi.vectorMap.mockResolvedValueOnce(failedVectorMap);

    render(<RetrievalLab datasetId="v6-manuals" />);

    expect(await screen.findByText('UMAP 构建失败')).toBeInTheDocument();
    fireEvent.change(screen.getByRole('textbox', { name: '测试问题' }), {
      target: { value: '空气净化器滤网怎么清洁？' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行检索' }));

    expect(await screen.findByText('lexical-only')).toBeInTheDocument();
    expect(screen.getByText('滤网清洁、更换与重置')).toBeInTheDocument();
    expect(mockedApi.retrievalTest).toHaveBeenCalledWith(
      expect.objectContaining({ dataset_ids: ['v6-manuals'] }),
    );
  });
});

describe('vector map API client contract', () => {
  const originalFetch = global.fetch;
  const fetchMock = jest.fn();

  beforeEach(() => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => readyVectorMap,
    });
    global.fetch = fetchMock as unknown as typeof fetch;
  });

  afterEach(() => {
    fetchMock.mockReset();
    global.fetch = originalFetch;
  });

  it('requests the encoded dataset vector map URL', async () => {
    const { api: actualApi } = jest.requireActual(
      '@/aka/api/client',
    ) as typeof import('@/aka/api/client');

    await actualApi.vectorMap('v6/manuals α');

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/datasets/v6%2Fmanuals%20%CE%B1/vector-map',
    );
  });

  it('posts to the encoded vector map rebuild URL', async () => {
    const { api: actualApi } = jest.requireActual(
      '@/aka/api/client',
    ) as typeof import('@/aka/api/client');

    await actualApi.rebuildVectorMap('v6/manuals α');

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/datasets/v6%2Fmanuals%20%CE%B1/vector-map/rebuild',
    );
    expect(fetchMock.mock.calls[0][1]).toEqual(
      expect.objectContaining({ method: 'POST' }),
    );
  });
});
