import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type {
  RetrievalVisualizationHit,
  RetrievalVisualizationPoint,
  VectorMapPoint,
} from '@/aka/api/types';
import { VectorSpaceMap } from './vector-space-map';

const BLUE = '#2563eb';
const PURPLE = '#8b5cf6';
const GREEN = '#16a34a';
const ORANGE = '#f59e0b';
const RED = '#ef4444';

type MockContext = CanvasRenderingContext2D & {
  arcCalls: Array<[number, number, number, number, number]>;
  fillCalls: string[];
  lineToCalls: Array<[number, number]>;
  moveToCalls: Array<[number, number]>;
  setTransformCalls: unknown[][];
  strokeCalls: string[];
};

class MockResizeObserver {
  static instances: MockResizeObserver[] = [];

  observe = jest.fn();
  unobserve = jest.fn();
  disconnect = jest.fn();

  constructor(private readonly callback: ResizeObserverCallback) {
    MockResizeObserver.instances.push(this);
  }

  emit(width = 800, height = 500) {
    this.callback(
      [
        {
          contentRect: {
            width,
            height,
            x: 0,
            y: 0,
            top: 0,
            left: 0,
            right: width,
            bottom: height,
            toJSON: () => ({}),
          },
        } as ResizeObserverEntry,
      ],
      this as unknown as ResizeObserver,
    );
  }
}

class MockPointerEvent extends MouseEvent {
  readonly pointerId: number;
  readonly pointerType: string;

  constructor(type: string, init: PointerEventInit = {}) {
    super(type, init);
    this.pointerId = init.pointerId ?? 0;
    this.pointerType = init.pointerType ?? 'touch';
  }
}

function createContext(): MockContext {
  let fillStyle = '';
  let strokeStyle = '';
  const context = {
    arcCalls: [],
    fillCalls: [],
    lineToCalls: [],
    moveToCalls: [],
    setTransformCalls: [],
    strokeCalls: [],
    canvas: document.createElement('canvas'),
    save: jest.fn(),
    restore: jest.fn(),
    clearRect: jest.fn(),
    setTransform: jest.fn((...args: unknown[]) => {
      context.setTransformCalls.push(args);
    }),
    beginPath: jest.fn(),
    closePath: jest.fn(),
    moveTo: jest.fn((x: number, y: number) => {
      context.moveToCalls.push([x, y]);
    }),
    lineTo: jest.fn((x: number, y: number) => {
      context.lineToCalls.push([x, y]);
    }),
    rect: jest.fn(),
    clip: jest.fn(),
    fill: jest.fn(() => {
      context.fillCalls.push(fillStyle);
    }),
    stroke: jest.fn(() => {
      context.strokeCalls.push(strokeStyle);
    }),
    arc: jest.fn((x: number, y: number, radius: number, start: number, end: number) => {
      context.arcCalls.push([x, y, radius, start, end]);
    }),
    fillRect: jest.fn(),
    strokeRect: jest.fn(),
    translate: jest.fn(),
    scale: jest.fn(),
  } as unknown as MockContext;

  Object.defineProperty(context, 'fillStyle', {
    get: () => fillStyle,
    set: (value: string) => {
      fillStyle = value;
    },
  });
  Object.defineProperty(context, 'strokeStyle', {
    get: () => strokeStyle,
    set: (value: string) => {
      strokeStyle = value;
    },
  });

  return context;
}

function point(overrides: Partial<VectorMapPoint> = {}): VectorMapPoint {
  return {
    child_id: 'child-a',
    dataset_id: 'dataset-a',
    document_id: 'doc-a',
    document_name: 'manual-a.pdf',
    title: '真实标题 A',
    excerpt: '真实摘要 A',
    page_start: 1,
    page_end: 2,
    product: 'AKA',
    x: 0,
    y: 0,
    ...overrides,
  };
}

function hit(childId: string, rank: number, score: number): RetrievalVisualizationHit {
  return { child_id: childId, rank, score };
}

function queryPoint(x = 0.5, y = 0.5): RetrievalVisualizationPoint {
  return { x, y };
}

function setDocumentVisibility(state: DocumentVisibilityState) {
  Object.defineProperty(document, 'visibilityState', {
    configurable: true,
    value: state,
  });
  document.dispatchEvent(new Event('visibilitychange'));
}

function renderMap(
  overrides: Partial<React.ComponentProps<typeof VectorSpaceMap>> = {},
) {
  const onMotionEnabledChange = jest.fn();
  const onFocusedChildChange = jest.fn();
  const result = render(
    <VectorSpaceMap
      points={[point()]}
      queryPoint={null}
      denseHits={[]}
      rerankHits={[]}
      rrfHits={[]}
      focusedChildId={null}
      onFocusedChildChange={onFocusedChildChange}
      onMotionEnabledChange={onMotionEnabledChange}
      {...overrides}
    />,
  );

  act(() => {
    MockResizeObserver.instances.at(-1)?.emit();
  });

  return { ...result, onMotionEnabledChange, onFocusedChildChange };
}

let context: MockContext;
let reducedMotion = false;
let rafCallbacks: Map<number, FrameRequestCallback>;
let nextRafId = 1;

beforeEach(() => {
  context = createContext();
  reducedMotion = false;
  rafCallbacks = new Map();
  nextRafId = 1;
  MockResizeObserver.instances = [];

  jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context);
  jest.spyOn(HTMLCanvasElement.prototype, 'getBoundingClientRect').mockReturnValue({
    x: 0,
    y: 0,
    width: 800,
    height: 500,
    top: 0,
    left: 0,
    right: 800,
    bottom: 500,
    toJSON: () => ({}),
  });

  Object.defineProperty(window, 'devicePixelRatio', {
    configurable: true,
    value: 2,
  });
  Object.defineProperty(document, 'visibilityState', {
    configurable: true,
    value: 'visible',
  });
  Object.defineProperty(window, 'PointerEvent', {
    configurable: true,
    value: MockPointerEvent,
  });
  global.PointerEvent = MockPointerEvent as unknown as typeof PointerEvent;

  window.matchMedia = jest.fn((query: string) => ({
    matches: query.includes('prefers-reduced-motion') ? reducedMotion : false,
    media: query,
    onchange: null,
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    addListener: jest.fn(),
    removeListener: jest.fn(),
    dispatchEvent: jest.fn(),
  }));

  global.ResizeObserver = MockResizeObserver as unknown as typeof ResizeObserver;
  global.requestAnimationFrame = jest.fn((callback: FrameRequestCallback) => {
    const id = nextRafId;
    nextRafId += 1;
    rafCallbacks.set(id, callback);
    return id;
  });
  global.cancelAnimationFrame = jest.fn((id: number) => {
    rafCallbacks.delete(id);
  });
});

afterEach(() => {
  jest.restoreAllMocks();
});

describe('VectorSpaceMap', () => {
  it('renders an accessible canvas, viewport buttons and the default motion switch', () => {
    renderMap();

    expect(screen.getByLabelText('知识库 UMAP 向量空间')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '放大' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '缩小' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '恢复初始视图' })).toBeInTheDocument();
    expect(screen.getByRole('switch', { name: '粒子动效' })).toBeChecked();
    expect(screen.getByTestId('vector-zoom-level')).toHaveTextContent('100%');
  });

  it('reports the live zoom percentage after toolbar zooming', () => {
    renderMap({ motionEnabled: false });

    fireEvent.click(screen.getByRole('button', { name: '放大' }));

    expect(screen.getByTestId('vector-zoom-level')).toHaveTextContent('118%');
  });

  it('renders a compact data legend for base, RRF, Dense, Rerank and query markers', () => {
    renderMap();

    const legend = screen.getByRole('list', { name: '向量图图例' });
    expect(within(legend).getByText('Chunk')).toBeInTheDocument();
    expect(within(legend).getByText('Dense')).toBeInTheDocument();
    expect(within(legend).getByText('Rerank')).toBeInTheDocument();
    expect(within(legend).getByText('RRF')).toBeInTheDocument();
    expect(within(legend).getByText('问题')).toBeInTheDocument();
    expect(within(legend).getByTestId('legend-base')).toHaveStyle({
      backgroundColor: BLUE,
    });
    expect(within(legend).getByTestId('legend-dense')).toHaveStyle({
      backgroundColor: PURPLE,
    });
    expect(within(legend).getByTestId('legend-rerank')).toHaveStyle({
      backgroundColor: GREEN,
    });
    expect(within(legend).getByTestId('legend-rrf')).toHaveStyle({
      backgroundColor: ORANGE,
    });
    expect(within(legend).getByTestId('legend-query')).toHaveStyle({
      backgroundColor: RED,
    });
  });

  it('keeps the canvas discoverable by name without adding an inert tab stop', async () => {
    const user = userEvent.setup();
    renderMap();

    const canvas = screen.getByRole('img', { name: '知识库 UMAP 向量空间' });
    expect(canvas).toHaveAccessibleName('知识库 UMAP 向量空间');
    expect(canvas).not.toHaveAttribute('tabindex');

    await user.tab();

    expect(screen.getByRole('button', { name: '放大' })).toHaveFocus();
    expect(canvas).not.toHaveFocus();
  });

  it('starts motion disabled when prefers-reduced-motion is active', () => {
    reducedMotion = true;

    renderMap();

    expect(screen.getByRole('switch', { name: '粒子动效' })).not.toBeChecked();
    expect(requestAnimationFrame).not.toHaveBeenCalled();
  });

  it('respects external motion state and reports switch changes', () => {
    const { onMotionEnabledChange } = renderMap({ motionEnabled: false });

    const motionSwitch = screen.getByRole('switch', { name: '粒子动效' });
    expect(motionSwitch).not.toBeChecked();

    fireEvent.click(motionSwitch);

    expect(onMotionEnabledChange).toHaveBeenCalledWith(true);
  });

  it('uses a unique generated id for each motion switch label association', () => {
    const sharedProps = {
      points: [point()],
      queryPoint: null,
      denseHits: [],
      rerankHits: [],
      rrfHits: [],
      focusedChildId: null,
      onFocusedChildChange: jest.fn(),
      onMotionEnabledChange: jest.fn(),
    } satisfies React.ComponentProps<typeof VectorSpaceMap>;
    const { container } = render(
      <>
        <VectorSpaceMap {...sharedProps} />
        <VectorSpaceMap {...sharedProps} />
      </>,
    );

    act(() => {
      MockResizeObserver.instances.forEach((observer) => observer.emit());
    });

    const switches = screen.getAllByRole('switch', { name: '粒子动效' });
    const labels = Array.from(container.querySelectorAll<HTMLLabelElement>('label[for]'));
    const ids = switches.map((switchControl) => switchControl.getAttribute('id'));

    expect(switches).toHaveLength(2);
    expect(labels).toHaveLength(2);
    expect(new Set(ids).size).toBe(2);
    labels.forEach((label, index) => {
      expect(label.htmlFor).toBe(ids[index]);
      expect(label.control).toBe(switches[index]);
    });
  });

  it('draws all real points with blue, Dense purple, Rerank green and query red priority', () => {
    renderMap({
      points: [
        point({ child_id: 'child-a', x: 0, y: 0 }),
        point({ child_id: 'child-b', x: 0.5, y: 0.5 }),
        point({ child_id: 'child-c', x: 1, y: 1 }),
      ],
      queryPoint: queryPoint(),
      rrfHits: [hit('child-a', 1, 0.0164)],
      denseHits: [hit('child-b', 1, 0.91)],
      rerankHits: [hit('child-b', 1, 0.88), hit('child-c', 2, 0.72)],
      motionEnabled: false,
    });

    expect(context.fillCalls.filter((style) => style === BLUE).length).toBeGreaterThanOrEqual(3);
    expect(context.fillCalls).toContain(ORANGE);
    expect(context.fillCalls).toContain(PURPLE);
    expect(context.fillCalls).toContain(GREEN);
    expect(context.fillCalls).toContain(RED);
    expect(context.fillCalls.lastIndexOf(GREEN)).toBeGreaterThan(
      context.fillCalls.lastIndexOf(PURPLE),
    );
  });

  it('draws the red query marker as a diamond path', () => {
    renderMap({ queryPoint: queryPoint(0.5, 0.5), motionEnabled: false });

    expect(context.fillCalls).toContain(RED);
    expect(context.moveToCalls).toContainEqual([400, 244]);
    expect(context.lineToCalls).toEqual(
      expect.arrayContaining([
        [406, 250],
        [400, 256],
        [394, 250],
      ]),
    );
  });

  it('resizes the canvas backing store for DPR and resets the drawing transform', () => {
    renderMap({ motionEnabled: false });
    const canvas = screen.getByLabelText('知识库 UMAP 向量空间') as HTMLCanvasElement;

    act(() => {
      MockResizeObserver.instances.at(-1)?.emit(320, 180);
    });

    expect(canvas.width).toBe(640);
    expect(canvas.height).toBe(360);
    expect(canvas.style.width).toBe('320px');
    expect(canvas.style.height).toBe('180px');
    expect(context.setTransformCalls).toContainEqual([2, 0, 0, 2, 0, 0]);
  });

  it('cancels RAF while hidden and schedules a new frame when visible again', () => {
    renderMap({ motionEnabled: true });
    const scheduledFrame = nextRafId - 1;

    expect(rafCallbacks.has(scheduledFrame)).toBe(true);

    setDocumentVisibility('hidden');

    expect(cancelAnimationFrame).toHaveBeenCalledWith(scheduledFrame);
    expect(rafCallbacks.has(scheduledFrame)).toBe(false);

    const requestCountWhileHidden = (requestAnimationFrame as jest.Mock).mock.calls.length;

    setDocumentVisibility('visible');

    expect(requestAnimationFrame).toHaveBeenCalledTimes(requestCountWhileHidden + 1);
    expect(rafCallbacks.has(nextRafId - 1)).toBe(true);
  });

  it('reuses projected points and the spatial grid across animation frames for 6570 chunks', () => {
    const largePointSet = Array.from({ length: 6570 }, (_, index) =>
      point({
        child_id: `child-${index}`,
        x: (index % 100) / 99,
        y: (Math.floor(index / 100) % 66) / 65,
      }),
    );
    renderMap({ points: largePointSet, motionEnabled: true });
    const canvas = screen.getByLabelText('知识库 UMAP 向量空间');
    const buildsAfterProjection = Number(
      canvas.getAttribute('data-projection-build-count'),
    );

    expect(buildsAfterProjection).toBeGreaterThan(0);

    for (let index = 0; index < 3; index += 1) {
      const callback = Array.from(rafCallbacks.values()).at(-1);
      expect(callback).toBeDefined();
      act(() => callback?.(1000 + index * 16));
    }

    expect(Number(canvas.getAttribute('data-projection-build-count'))).toBe(
      buildsAfterProjection,
    );
  });

  it('shows true chunk metadata and hit ranking in a single tooltip on hover', async () => {
    renderMap({
      points: [
        point({
          child_id: 'chunk-real',
          document_name: 'source-document.pdf',
          title: '真实 Chunk 标题',
          excerpt: '真实 Chunk 摘要内容',
          page_start: 3,
          page_end: 5,
          x: 0,
          y: 0,
        }),
      ],
      denseHits: [hit('chunk-real', 2, 0.7654)],
      rerankHits: [hit('chunk-real', 1, 0.9876)],
      motionEnabled: false,
    });

    fireEvent.pointerMove(screen.getByLabelText('知识库 UMAP 向量空间'), {
      clientX: 32,
      clientY: 32,
    });

    expect(await screen.findByRole('tooltip')).toHaveTextContent('真实 Chunk 标题');
    expect(screen.getByRole('tooltip')).toHaveTextContent('真实 Chunk 摘要内容');
    expect(screen.getByRole('tooltip')).toHaveTextContent('source-document.pdf');
    expect(screen.getByRole('tooltip')).toHaveTextContent('第 3-5 页');
    expect(screen.getByRole('tooltip')).toHaveTextContent('chunk-real');
    expect(screen.getByRole('tooltip')).toHaveTextContent('Rerank');
    expect(screen.getByRole('tooltip')).toHaveTextContent('第 1 名');
    expect(screen.getByRole('tooltip')).toHaveTextContent('0.9876');
    expect(screen.getAllByRole('tooltip')).toHaveLength(1);
  });

  it('shows query semantic tooltip for the red query point', async () => {
    renderMap({ queryPoint: queryPoint(0.5, 0.5), motionEnabled: false });

    fireEvent.pointerMove(screen.getByLabelText('知识库 UMAP 向量空间'), {
      clientX: 400,
      clientY: 250,
    });

    expect(await screen.findByRole('tooltip')).toHaveTextContent('查询语义');
  });

  it('zooms around the wheel pointer and pans by dragging without mutating data coordinates', () => {
    const originalPoint = point({ x: 0, y: 0 });
    renderMap({ points: [originalPoint], motionEnabled: false });
    const canvas = screen.getByLabelText('知识库 UMAP 向量空间');
    const initialX = context.arcCalls.at(-1)?.[0];

    context.arcCalls = [];
    const wheelEvent = new WheelEvent('wheel', {
      bubbles: true,
      cancelable: true,
      clientX: 400,
      clientY: 250,
      deltaY: -120,
    });
    const preventDefault = jest.spyOn(wheelEvent, 'preventDefault');
    act(() => {
      canvas.dispatchEvent(wheelEvent);
    });
    const zoomedX = context.arcCalls.at(-1)?.[0];

    context.arcCalls = [];
    fireEvent.pointerDown(canvas, { pointerId: 1, clientX: 100, clientY: 100 });
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 130, clientY: 110 });
    fireEvent.pointerUp(canvas, { pointerId: 1, clientX: 130, clientY: 110 });
    const pannedX = context.arcCalls.at(-1)?.[0];

    expect(preventDefault).toHaveBeenCalled();
    expect(zoomedX).not.toBe(initialX);
    expect(pannedX).not.toBe(zoomedX);
    expect(originalPoint).toMatchObject({ x: 0, y: 0 });
  });

  it('zooms in and out with a two-pointer pinch while keeping its center anchored', () => {
    renderMap({
      points: [point({ child_id: 'center', x: 0.5, y: 0.5 })],
      motionEnabled: false,
    });
    const canvas = screen.getByLabelText('知识库 UMAP 向量空间');
    const initialCenter = context.arcCalls.at(-1)?.slice(0, 2);

    fireEvent.pointerDown(canvas, { pointerId: 1, clientX: 300, clientY: 250 });
    fireEvent.pointerDown(canvas, { pointerId: 2, clientX: 500, clientY: 250 });
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 250, clientY: 250 });
    fireEvent.pointerMove(canvas, { pointerId: 2, clientX: 550, clientY: 250 });

    expect(screen.getByTestId('vector-zoom-level')).toHaveTextContent('150%');
    expect(context.arcCalls.at(-1)?.slice(0, 2)).toEqual(initialCenter);

    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 325, clientY: 250 });
    fireEvent.pointerMove(canvas, { pointerId: 2, clientX: 475, clientY: 250 });

    expect(screen.getByTestId('vector-zoom-level')).toHaveTextContent('75%');
    expect(context.arcCalls.at(-1)?.slice(0, 2)).toEqual(initialCenter);
  });

  it('pans with the moving pinch center and suppresses the following click', () => {
    const { onFocusedChildChange } = renderMap({
      points: [point({ child_id: 'child-a', x: 0.5, y: 0.5 })],
      motionEnabled: false,
    });
    const canvas = screen.getByLabelText('知识库 UMAP 向量空间');

    fireEvent.pointerDown(canvas, { pointerId: 1, clientX: 300, clientY: 250 });
    fireEvent.pointerDown(canvas, { pointerId: 2, clientX: 500, clientY: 250 });
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 320, clientY: 260 });
    fireEvent.pointerMove(canvas, { pointerId: 2, clientX: 520, clientY: 260 });
    fireEvent.pointerUp(canvas, { pointerId: 1, clientX: 320, clientY: 260 });
    fireEvent.pointerUp(canvas, { pointerId: 2, clientX: 520, clientY: 260 });

    expect(context.arcCalls.at(-1)?.[0]).toBeCloseTo(420, 0);
    expect(context.arcCalls.at(-1)?.[1]).toBeCloseTo(260, 0);

    fireEvent.click(canvas, { clientX: 420, clientY: 260 });
    expect(onFocusedChildChange).not.toHaveBeenCalled();
  });

  it('cleans up a cancelled pinch and allows a new single-pointer drag', () => {
    renderMap({ motionEnabled: false });
    const canvas = screen.getByLabelText('知识库 UMAP 向量空间');

    fireEvent.pointerDown(canvas, { pointerId: 1, clientX: 300, clientY: 250 });
    fireEvent.pointerDown(canvas, { pointerId: 2, clientX: 500, clientY: 250 });
    fireEvent.pointerCancel(canvas, { pointerId: 1, clientX: 300, clientY: 250 });
    fireEvent.pointerCancel(canvas, { pointerId: 2, clientX: 500, clientY: 250 });

    context.arcCalls = [];
    fireEvent.pointerDown(canvas, { pointerId: 3, clientX: 100, clientY: 100 });
    fireEvent.pointerMove(canvas, { pointerId: 3, clientX: 120, clientY: 100 });
    fireEvent.pointerUp(canvas, { pointerId: 3, clientX: 120, clientY: 100 });

    expect(context.arcCalls.length).toBeGreaterThan(0);
    expect(canvas).toHaveStyle({ cursor: '' });
  });

  it('does not pan for a 3px pointer jitter and still lets the click select the hit', () => {
    const { onFocusedChildChange } = renderMap({
      points: [point({ child_id: 'child-a', x: 0, y: 0 })],
      motionEnabled: false,
    });
    const canvas = screen.getByLabelText('知识库 UMAP 向量空间');

    context.arcCalls = [];
    fireEvent.pointerDown(canvas, { pointerId: 1, clientX: 32, clientY: 32 });
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 35, clientY: 32 });
    fireEvent.pointerUp(canvas, { pointerId: 1, clientX: 35, clientY: 32 });

    expect(context.arcCalls).toHaveLength(0);

    fireEvent.click(canvas, { clientX: 35, clientY: 32 });

    expect(onFocusedChildChange).toHaveBeenCalledWith('child-a');
  });

  it('applies cumulative pan when drag crosses 3px and suppresses the click', () => {
    const { onFocusedChildChange } = renderMap({
      points: [point({ child_id: 'child-a', x: 0, y: 0 })],
      motionEnabled: false,
    });
    const canvas = screen.getByLabelText('知识库 UMAP 向量空间');

    context.arcCalls = [];
    fireEvent.pointerDown(canvas, { pointerId: 1, clientX: 32, clientY: 32 });
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 34, clientY: 32 });
    expect(context.arcCalls).toHaveLength(0);

    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 36, clientY: 32 });
    expect(context.arcCalls.at(-1)?.[0]).toBeCloseTo(36, 0);

    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 41, clientY: 32 });
    expect(context.arcCalls.at(-1)?.[0]).toBeCloseTo(41, 0);

    fireEvent.pointerUp(canvas, { pointerId: 1, clientX: 41, clientY: 32 });
    fireEvent.click(canvas, { clientX: 41, clientY: 32 });

    expect(onFocusedChildChange).not.toHaveBeenCalled();
  });

  it('keeps drag interaction usable when pointer capture APIs reject synthetic pointers', () => {
    renderMap({ motionEnabled: false });
    const canvas = screen.getByLabelText('知识库 UMAP 向量空间') as HTMLCanvasElement;
    canvas.setPointerCapture = jest.fn(() => {
      throw new DOMException('no active pointer', 'NotFoundError');
    });
    canvas.releasePointerCapture = jest.fn(() => {
      throw new DOMException('no active pointer', 'NotFoundError');
    });

    expect(() => {
      fireEvent.pointerDown(canvas, { pointerId: 17, clientX: 40, clientY: 40 });
      fireEvent.pointerMove(canvas, { pointerId: 17, clientX: 60, clientY: 50 });
      fireEvent.pointerUp(canvas, { pointerId: 17, clientX: 60, clientY: 50 });
    }).not.toThrow();
    expect(canvas.style.cursor).toBe('');
  });

  it('restores the initial view with the reset button', () => {
    renderMap({ points: [point({ x: 0, y: 0 })], motionEnabled: false });
    const canvas = screen.getByLabelText('知识库 UMAP 向量空间');
    const initialX = context.arcCalls.at(-1)?.[0];

    context.arcCalls = [];
    fireEvent.pointerDown(canvas, { pointerId: 1, clientX: 100, clientY: 100 });
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 140, clientY: 100 });
    fireEvent.pointerUp(canvas, { pointerId: 1, clientX: 140, clientY: 100 });
    expect(context.arcCalls.at(-1)?.[0]).not.toBe(initialX);

    context.arcCalls = [];
    fireEvent.click(screen.getByRole('button', { name: '恢复初始视图' }));

    expect(context.arcCalls.at(-1)?.[0]).toBe(initialX);
  });

  it('clicks canvas hits and recenters when focusedChildId changes externally', async () => {
    const points = [
      point({ child_id: 'child-a', x: 0, y: 0 }),
      point({ child_id: 'child-b', x: 1, y: 1 }),
    ];
    const { onFocusedChildChange, rerender } = renderMap({
      points,
      motionEnabled: false,
    });
    const canvas = screen.getByLabelText('知识库 UMAP 向量空间');

    fireEvent.click(canvas, { clientX: 32, clientY: 32 });
    expect(onFocusedChildChange).toHaveBeenCalledWith('child-a');

    context.arcCalls = [];
    rerender(
      <VectorSpaceMap
        points={points}
        queryPoint={null}
        denseHits={[]}
        rerankHits={[]}
        rrfHits={[]}
        motionEnabled={false}
        focusedChildId="child-b"
        onFocusedChildChange={onFocusedChildChange}
        onMotionEnabledChange={jest.fn()}
      />,
    );

    await waitFor(() => {
      const lastTwo = context.arcCalls.slice(-2);
      expect(lastTwo[1]?.[0]).toBeCloseTo(400, 0);
      expect(lastTwo[1]?.[1]).toBeCloseTo(250, 0);
    });
  });

  it('cleans up animation, observers and DOM listeners on unmount', () => {
    const removeDocumentListener = jest.spyOn(document, 'removeEventListener');
    const removeCanvasListener = jest.spyOn(HTMLCanvasElement.prototype, 'removeEventListener');
    const { unmount } = renderMap({ motionEnabled: true });

    expect(requestAnimationFrame).toHaveBeenCalled();
    unmount();

    expect(cancelAnimationFrame).toHaveBeenCalled();
    expect(MockResizeObserver.instances.at(-1)?.disconnect).toHaveBeenCalled();
    expect(removeDocumentListener).toHaveBeenCalledWith(
      'visibilitychange',
      expect.any(Function),
    );
    expect(removeCanvasListener).toHaveBeenCalledWith('wheel', expect.any(Function));
    expect(removeCanvasListener).toHaveBeenCalledWith('pointermove', expect.any(Function));
  });
});
