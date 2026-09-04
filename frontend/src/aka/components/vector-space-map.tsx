import type {
  RetrievalVisualizationHit,
  RetrievalVisualizationPoint,
  VectorMapPoint,
} from '@/aka/api/types';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { LocateFixed, Sparkles, ZoomIn, ZoomOut } from 'lucide-react';
import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import {
  HIT_PRIORITY,
  INITIAL_VIEW,
  buildSpatialGrid,
  findNearestPoint,
  panBy,
  projectPoint,
  resetView,
  zoomAt,
  type SpatialGrid,
  type SpatialPoint,
  type VectorPoint,
  type VectorView,
} from './vector-space-math';

export interface VectorSpaceMapProps {
  points: VectorMapPoint[];
  queryPoint: RetrievalVisualizationPoint | null;
  rrfHits: RetrievalVisualizationHit[];
  denseHits: RetrievalVisualizationHit[];
  rerankHits: RetrievalVisualizationHit[];
  motionEnabled?: boolean;
  onMotionEnabledChange?: (enabled: boolean) => void;
  focusedChildId?: string | null;
  onFocusedChildChange?: (childId: string | null) => void;
}

type HitStage = 'RRF' | 'Dense' | 'Rerank';

interface HitMarker {
  stage: HitStage;
  rank: number;
  score: number;
}

interface HitLookup {
  rrf: Map<string, HitMarker>;
  dense: Map<string, HitMarker>;
  rerank: Map<string, HitMarker>;
}

interface ChunkScreenPoint extends SpatialPoint {
  kind: 'chunk';
  point: VectorMapPoint;
  hit: HitMarker | null;
}

interface QueryScreenPoint extends SpatialPoint {
  kind: 'query';
  point: RetrievalVisualizationPoint;
}

type ScreenPoint = ChunkScreenPoint | QueryScreenPoint;

interface ProjectedChunk {
  screen: VectorPoint;
  index: number;
}

interface ProjectedScene {
  projectedPoints: ProjectedChunk[];
  projectedById: Map<string, VectorPoint>;
  queryScreen: VectorPoint | null;
  grid: SpatialGrid<ScreenPoint>;
}

interface PinchGesture {
  active: boolean;
  lastCenter: VectorPoint;
  lastDistance: number;
}

type HoverState =
  | {
      kind: 'chunk';
      screen: VectorPoint;
      point: VectorMapPoint;
      hit: HitMarker | null;
    }
  | {
      kind: 'query';
      screen: VectorPoint;
      point: RetrievalVisualizationPoint;
    };

const BLUE = '#2563eb';
const PURPLE = '#8b5cf6';
const GREEN = '#16a34a';
const ORANGE = '#f59e0b';
const RED = '#ef4444';
const GRID = '#d9e2ec';
const LINE = '#94a3b8';
const RING = '#0f172a';
const FOCUS = '#f59e0b';
const PADDING = 32;
const HIT_RADIUS = 12;
const CLIP_MARGIN = 36;
const DRAG_THRESHOLD_PX = 3;
const FALLBACK_WIDTH = 800;
const FALLBACK_HEIGHT = 500;
const QUERY_ID = '__vector-query__';
const QUERY_SCAN_MS = 1100;
const HIT_REVEAL_DELAY_MS = 90;
const HIT_REVEAL_MS = 260;

function pinchGeometry(points: Map<number, VectorPoint>) {
  const [first, second] = Array.from(points.values());
  if (!first || !second) {
    return null;
  }

  return {
    center: {
      x: (first.x + second.x) / 2,
      y: (first.y + second.y) / 2,
    },
    distance: Math.hypot(second.x - first.x, second.y - first.y),
  };
}

function prefersReducedMotion() {
  return (
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  );
}

function animationNow() {
  return typeof performance !== 'undefined' ? performance.now() : 0;
}

function finiteOr(value: number | undefined, fallback: number) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function roundedScore(score: number) {
  return Number.isFinite(score) ? score.toFixed(4) : '0.0000';
}

function pageLabel(point: VectorMapPoint) {
  if (point.page_start === point.page_end) {
    return `第 ${point.page_start} 页`;
  }
  return `第 ${point.page_start}-${point.page_end} 页`;
}

function safelyTogglePointerCapture(
  canvas: HTMLCanvasElement,
  action: 'set' | 'release',
  pointerId: number,
) {
  const method =
    action === 'set' ? canvas.setPointerCapture : canvas.releasePointerCapture;
  if (typeof method !== 'function') {
    return;
  }
  try {
    method.call(canvas, pointerId);
  } catch {
    return;
  }
}

function hitKey(items: RetrievalVisualizationHit[]) {
  return items
    .slice(0, 10)
    .map((item) => `${item.child_id}:${item.rank}:${item.score}`)
    .join('|');
}

function queryKey(point: RetrievalVisualizationPoint | null) {
  return point ? `${point.x}:${point.y}` : 'none';
}

function normalizeHits(
  hits: RetrievalVisualizationHit[],
  stage: HitStage,
) {
  return hits
    .slice(0, 10)
    .filter((hit) => hit.child_id)
    .sort((left, right) => left.rank - right.rank)
    .map((hit) => ({
      child_id: hit.child_id,
      marker: {
        stage,
        rank: hit.rank,
        score: hit.score,
      } satisfies HitMarker,
    }));
}

function buildHitLookup(
  rrfHits: RetrievalVisualizationHit[],
  denseHits: RetrievalVisualizationHit[],
  rerankHits: RetrievalVisualizationHit[],
): HitLookup {
  const rrf = new Map<string, HitMarker>();
  const dense = new Map<string, HitMarker>();
  const rerank = new Map<string, HitMarker>();

  normalizeHits(rrfHits, 'RRF').forEach(({ child_id, marker }) => {
    rrf.set(child_id, marker);
  });
  normalizeHits(denseHits, 'Dense').forEach(({ child_id, marker }) => {
    dense.set(child_id, marker);
  });
  normalizeHits(rerankHits, 'Rerank').forEach(({ child_id, marker }) => {
    rerank.set(child_id, marker);
  });

  return { rrf, dense, rerank };
}

function markerFor(point: VectorMapPoint, lookup: HitLookup) {
  return (
    lookup.rerank.get(point.child_id) ??
    lookup.dense.get(point.child_id) ??
    lookup.rrf.get(point.child_id) ??
    null
  );
}

function pointPriority(marker: HitMarker | null) {
  if (!marker) {
    return HIT_PRIORITY.normal;
  }
  if (marker.stage === 'Rerank') {
    return HIT_PRIORITY.highlight + 3;
  }
  return marker.stage === 'Dense' ? HIT_PRIORITY.highlight + 2 : HIT_PRIORITY.highlight + 1;
}

function viewportContains(point: VectorPoint, width: number, height: number) {
  return (
    point.x >= -CLIP_MARGIN &&
    point.x <= width + CLIP_MARGIN &&
    point.y >= -CLIP_MARGIN &&
    point.y <= height + CLIP_MARGIN
  );
}

function clientPoint(event: MouseEvent | PointerEvent | WheelEvent, canvas: HTMLCanvasElement) {
  const rect = canvas.getBoundingClientRect();
  const left = finiteOr(rect.left, 0);
  const top = finiteOr(rect.top, 0);

  return {
    x: finiteOr(event.clientX, 0) - left,
    y: finiteOr(event.clientY, 0) - top,
  };
}

function drawCircle(
  context: CanvasRenderingContext2D,
  point: VectorPoint,
  radius: number,
  color: string,
  alpha = 1,
) {
  context.globalAlpha = alpha;
  context.fillStyle = color;
  context.beginPath();
  context.arc(point.x, point.y, radius, 0, Math.PI * 2);
  context.fill();
  context.globalAlpha = 1;
}

function drawDiamond(
  context: CanvasRenderingContext2D,
  point: VectorPoint,
  radius: number,
  color: string,
  alpha = 1,
) {
  context.globalAlpha = alpha;
  context.fillStyle = color;
  context.beginPath();
  context.moveTo(point.x, point.y - radius);
  context.lineTo(point.x + radius, point.y);
  context.lineTo(point.x, point.y + radius);
  context.lineTo(point.x - radius, point.y);
  context.closePath();
  context.fill();
  context.globalAlpha = 1;
}

function drawRing(
  context: CanvasRenderingContext2D,
  point: VectorPoint,
  radius: number,
  color: string,
  width = 2,
) {
  context.strokeStyle = color;
  context.lineWidth = width;
  context.beginPath();
  context.arc(point.x, point.y, radius, 0, Math.PI * 2);
  context.stroke();
}

function drawGrid(context: CanvasRenderingContext2D, width: number, height: number) {
  context.save();
  context.strokeStyle = GRID;
  context.lineWidth = 0.5;
  context.globalAlpha = 0.55;

  for (let x = PADDING; x < width; x += 64) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }
  for (let y = PADDING; y < height; y += 64) {
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }

  context.restore();
}

function revealFactor(marker: HitMarker, startTime: number, time: number, motion: boolean) {
  if (!motion) {
    return 1;
  }

  const elapsed = time - startTime - Math.max(0, marker.rank - 1) * HIT_REVEAL_DELAY_MS;
  return clamp(elapsed / HIT_REVEAL_MS, 0, 1);
}

function getTooltipStyle(screen: VectorPoint, width: number, height: number) {
  const tooltipWidth = 280;
  const tooltipHeight = 190;
  const left = clamp(screen.x + 14, 8, Math.max(8, width - tooltipWidth - 8));
  const preferredTop = screen.y + 14;
  const flippedTop = screen.y - tooltipHeight - 14;
  const top = clamp(
    preferredTop + tooltipHeight > height ? flippedTop : preferredTop,
    8,
    Math.max(8, height - tooltipHeight - 8),
  );

  return {
    left,
    top,
    width: tooltipWidth,
  };
}

export function VectorSpaceMap({
  points,
  queryPoint,
  rrfHits,
  denseHits,
  rerankHits,
  motionEnabled,
  onMotionEnabledChange,
  focusedChildId = null,
  onFocusedChildChange,
}: VectorSpaceMapProps) {
  const motionSwitchId = useId();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const contextRef = useRef<CanvasRenderingContext2D | null>(null);
  const animationRef = useRef<number | null>(null);
  const loopRef = useRef<FrameRequestCallback>(() => undefined);
  const queryAnimationKeyRef = useRef(queryKey(queryPoint));
  const queryScanStartedAtRef = useRef(animationNow());
  const hitAnimationKeyRef = useRef(
    `${hitKey(rrfHits)}::${hitKey(denseHits)}::${hitKey(rerankHits)}`,
  );
  const hitRevealStartedAtRef = useRef(animationNow());
  const dragRef = useRef({
    active: false,
    moved: false,
    pointerId: -1,
    startX: 0,
    startY: 0,
    lastX: 0,
    lastY: 0,
  });
  const activePointersRef = useRef(new Map<number, VectorPoint>());
  const pinchRef = useRef<PinchGesture>({
    active: false,
    lastCenter: { x: 0, y: 0 },
    lastDistance: 0,
  });
  const gridRef = useRef<SpatialGrid<ScreenPoint>>(buildSpatialGrid([]));
  const projectedSceneRef = useRef<ProjectedScene | null>(null);
  const projectionBuildCountRef = useRef(0);
  const viewRef = useRef<VectorView>({ ...INITIAL_VIEW });
  const sizeRef = useRef({ width: FALLBACK_WIDTH, height: FALLBACK_HEIGHT });
  const pointsRef = useRef(points);
  const queryPointRef = useRef(queryPoint);
  const hitLookupRef = useRef<HitLookup>(
    buildHitLookup(rrfHits, denseHits, rerankHits),
  );
  const rrfHitsRef = useRef(rrfHits);
  const denseHitsRef = useRef(denseHits);
  const rerankHitsRef = useRef(rerankHits);
  const focusedChildIdRef = useRef<string | null>(focusedChildId);
  const hoverRef = useRef<HoverState | null>(null);
  const motionRef = useRef(false);
  const onFocusedChildChangeRef = useRef(onFocusedChildChange);
  const projectionInputsRef = useRef({ points, queryPoint, hitLookup: hitLookupRef.current });

  const [view, setView] = useState<VectorView>(() => ({ ...INITIAL_VIEW }));
  const [size, setSize] = useState({ width: FALLBACK_WIDTH, height: FALLBACK_HEIGHT });
  const [hover, setHover] = useState<HoverState | null>(null);
  const [internalMotionEnabled, setInternalMotionEnabled] = useState(
    () => !prefersReducedMotion(),
  );

  const hitLookup = useMemo(
    () => buildHitLookup(rrfHits, denseHits, rerankHits),
    [denseHits, rerankHits, rrfHits],
  );

  const pointById = useMemo(() => {
    const byId = new Map<string, VectorMapPoint>();
    points.forEach((point) => {
      byId.set(point.child_id, point);
    });
    return byId;
  }, [points]);

  const effectiveMotionEnabled = motionEnabled ?? internalMotionEnabled;

  if (
    projectionInputsRef.current.points !== points ||
    projectionInputsRef.current.queryPoint !== queryPoint ||
    projectionInputsRef.current.hitLookup !== hitLookup
  ) {
    projectionInputsRef.current = { points, queryPoint, hitLookup };
    projectedSceneRef.current = null;
  }

  viewRef.current = view;
  sizeRef.current = size;
  pointsRef.current = points;
  queryPointRef.current = queryPoint;
  hitLookupRef.current = hitLookup;
  rrfHitsRef.current = rrfHits;
  denseHitsRef.current = denseHits;
  rerankHitsRef.current = rerankHits;
  focusedChildIdRef.current = focusedChildId;
  hoverRef.current = hover;
  motionRef.current = effectiveMotionEnabled;
  onFocusedChildChangeRef.current = onFocusedChildChange;

  const renderFrame = useCallback((time = performance.now()) => {
    const context = contextRef.current;
    const canvas = canvasRef.current;
    const { width, height } = sizeRef.current;
    if (!context || !canvas || width <= 0 || height <= 0) {
      return;
    }

    const lookup = hitLookupRef.current;
    const currentMotion = motionRef.current;

    let scene = projectedSceneRef.current;
    if (!scene) {
      const currentView = viewRef.current;
      const screenPoints: ScreenPoint[] = [];
      const projectedPoints: ProjectedChunk[] = [];
      const projectedById = new Map<string, VectorPoint>();

      pointsRef.current.forEach((point, index) => {
        const projected = projectPoint(point, {
          width,
          height,
          padding: PADDING,
          view: currentView,
        });
        if (!viewportContains(projected, width, height)) {
          return;
        }
        const marker = markerFor(point, lookup);
        projectedById.set(point.child_id, projected);
        projectedPoints.push({ screen: projected, index });
        screenPoints.push({
          id: point.child_id,
          x: projected.x,
          y: projected.y,
          kind: 'chunk',
          point,
          hit: marker,
          priority: pointPriority(marker),
        });
      });

      const query = queryPointRef.current;
      const queryScreen = query
        ? projectPoint(query, { width, height, padding: PADDING, view: currentView })
        : null;
      if (query && queryScreen && viewportContains(queryScreen, width, height)) {
        screenPoints.push({
          id: QUERY_ID,
          x: queryScreen.x,
          y: queryScreen.y,
          kind: 'query',
          point: query,
          priority: HIT_PRIORITY.query,
        });
      }

      scene = {
        projectedPoints,
        projectedById,
        queryScreen,
        grid: buildSpatialGrid(screenPoints),
      };
      projectedSceneRef.current = scene;
      gridRef.current = scene.grid;
      projectionBuildCountRef.current += 1;
      canvas.dataset.projectionBuildCount = String(
        projectionBuildCountRef.current,
      );
    }

    const { projectedPoints, projectedById, queryScreen } = scene;
    const query = queryPointRef.current;

    context.clearRect(0, 0, width, height);
    drawGrid(context, width, height);

    if (query && queryScreen) {
      context.save();
      context.strokeStyle = LINE;
      context.lineWidth = 1;
      context.globalAlpha = 0.28;
      [
        ...rrfHitsRef.current.slice(0, 10),
        ...denseHitsRef.current.slice(0, 10),
        ...rerankHitsRef.current.slice(0, 10),
      ].forEach(
        (hit) => {
          const target = projectedById.get(hit.child_id);
          if (
            target &&
            viewportContains(target, width, height) &&
            viewportContains(queryScreen, width, height)
          ) {
            context.beginPath();
            context.moveTo(queryScreen.x, queryScreen.y);
            context.lineTo(target.x, target.y);
            context.stroke();
          }
        },
      );
      context.restore();
    }

    projectedPoints.forEach(({ screen, index }) => {
      if (viewportContains(screen, width, height)) {
        const pulse = currentMotion ? Math.sin(time / 900 + index * 0.071) : 0;
        drawCircle(
          context,
          screen,
          2.2 + pulse * 0.38,
          BLUE,
          currentMotion ? 0.58 + pulse * 0.12 : 0.7,
        );
      }
    });

    normalizeHits(rrfHitsRef.current, 'RRF').forEach(({ child_id, marker }) => {
      const projected = projectedById.get(child_id);
      if (!projected || !viewportContains(projected, width, height)) {
        return;
      }
      const reveal = revealFactor(
        marker,
        hitRevealStartedAtRef.current,
        time,
        currentMotion,
      );
      drawCircle(context, projected, 4 + reveal, ORANGE, 0.35 + reveal * 0.65);
    });

    normalizeHits(denseHitsRef.current, 'Dense').forEach(({ child_id, marker }) => {
      const projected = projectedById.get(child_id);
      if (!projected || !viewportContains(projected, width, height)) {
        return;
      }
      const reveal = revealFactor(
        marker,
        hitRevealStartedAtRef.current,
        time,
        currentMotion,
      );
      drawCircle(context, projected, 4.4 + reveal * 1.2, PURPLE, 0.35 + reveal * 0.65);
    });

    normalizeHits(rerankHitsRef.current, 'Rerank').forEach(({ child_id, marker }) => {
      const projected = projectedById.get(child_id);
      if (!projected || !viewportContains(projected, width, height)) {
        return;
      }
      const reveal = revealFactor(
        marker,
        hitRevealStartedAtRef.current,
        time,
        currentMotion,
      );
      drawCircle(context, projected, 4.8 + reveal * 1.4, GREEN, 0.35 + reveal * 0.65);
    });

    const hovered = hoverRef.current;
    if (hovered?.kind === 'chunk' && viewportContains(hovered.screen, width, height)) {
      drawRing(context, hovered.screen, hovered.hit ? 9 : 7, RING, 2);
    }

    const focused = focusedChildIdRef.current
      ? projectedById.get(focusedChildIdRef.current)
      : null;
    if (focused && viewportContains(focused, width, height)) {
      drawRing(context, focused, 11, FOCUS, 2.5);
    }

    if (queryScreen && viewportContains(queryScreen, width, height)) {
      const pulse = currentMotion ? Math.sin(time / 520) : 0;
      const scanElapsed = time - queryScanStartedAtRef.current;
      if (currentMotion && scanElapsed >= 0 && scanElapsed <= QUERY_SCAN_MS) {
        drawRing(
          context,
          queryScreen,
          10 + (scanElapsed / QUERY_SCAN_MS) * 42,
          RED,
          1.4,
        );
      }
      drawDiamond(context, queryScreen, 6 + pulse * 0.6, RED, currentMotion ? 0.9 : 1);
      if (hovered?.kind === 'query') {
        drawRing(context, queryScreen, 12, RED, 2);
      }
    }

  }, []);

  const applyView = useCallback(
    (nextView: VectorView) => {
      viewRef.current = nextView;
      projectedSceneRef.current = null;
      if (hoverRef.current) {
        hoverRef.current = null;
        setHover(null);
      }
      setView(nextView);
      renderFrame();
    },
    [renderFrame],
  );

  const updateHover = useCallback(
    (nextHover: HoverState | null) => {
      hoverRef.current = nextHover;
      setHover(nextHover);
      renderFrame();
    },
    [renderFrame],
  );

  const hitAt = useCallback((point: VectorPoint) => {
    return findNearestPoint(gridRef.current, point, {
      radius: HIT_RADIUS,
      priorityDistanceTolerance: 8,
    });
  }, []);

  useEffect(() => {
    const key = queryKey(queryPoint);
    if (queryAnimationKeyRef.current !== key) {
      queryAnimationKeyRef.current = key;
      queryScanStartedAtRef.current = animationNow();
    }
    renderFrame();
  }, [queryPoint, renderFrame]);

  useEffect(() => {
    const key = `${hitKey(rrfHits)}::${hitKey(denseHits)}::${hitKey(rerankHits)}`;
    if (hitAnimationKeyRef.current !== key) {
      hitAnimationKeyRef.current = key;
      hitRevealStartedAtRef.current = animationNow();
    }
    renderFrame();
  }, [denseHits, rerankHits, renderFrame, rrfHits]);

  useEffect(() => {
    renderFrame();
  }, [view, size, points, focusedChildId, hitLookup, hover, effectiveMotionEnabled, renderFrame]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const root = rootRef.current;
    if (!canvas || !root) {
      return undefined;
    }

    const context = canvas.getContext('2d');
    if (!context) {
      return undefined;
    }
    contextRef.current = context;

    const updateSize = (widthValue: number, heightValue: number) => {
      const width = Math.max(1, Math.round(finiteOr(widthValue, FALLBACK_WIDTH)));
      const height = Math.max(1, Math.round(finiteOr(heightValue, FALLBACK_HEIGHT)));
      const ratio = Math.max(1, finiteOr(window.devicePixelRatio, 1));

      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);

      const nextSize = { width, height };
      sizeRef.current = nextSize;
      projectedSceneRef.current = null;
      setSize(nextSize);
      renderFrame();
    };

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      const rect = entry?.contentRect;
      updateSize(rect?.width ?? FALLBACK_WIDTH, rect?.height ?? FALLBACK_HEIGHT);
    });
    observer.observe(root);

    const initialRect = root.getBoundingClientRect();
    updateSize(initialRect.width || FALLBACK_WIDTH, initialRect.height || FALLBACK_HEIGHT);

    return () => {
      observer.disconnect();
      contextRef.current = null;
    };
  }, [renderFrame]);

  useEffect(() => {
    loopRef.current = (time: number) => {
      animationRef.current = null;
      if (!motionRef.current || document.visibilityState === 'hidden') {
        renderFrame(time);
        return;
      }

      renderFrame(time);
      animationRef.current = requestAnimationFrame(loopRef.current);
    };
  }, [renderFrame]);

  useEffect(() => {
    const stop = () => {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    };
    const start = () => {
      if (
        animationRef.current === null &&
        motionRef.current &&
        document.visibilityState !== 'hidden'
      ) {
        animationRef.current = requestAnimationFrame(loopRef.current);
      }
    };
    const onVisibilityChange = () => {
      if (document.visibilityState === 'hidden') {
        stop();
        return;
      }
      renderFrame();
      start();
    };

    document.addEventListener('visibilitychange', onVisibilityChange);

    if (effectiveMotionEnabled) {
      start();
    } else {
      stop();
      renderFrame();
    }

    return () => {
      stop();
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, [effectiveMotionEnabled, renderFrame]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return undefined;
    }
    const activePointers = activePointersRef.current;

    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const anchor = clientPoint(event, canvas);
      const factor = event.deltaY < 0 ? 1.16 : 1 / 1.16;
      applyView(zoomAt(viewRef.current, anchor, factor));
    };

    const onPointerDown = (event: PointerEvent) => {
      activePointers.set(event.pointerId, {
        x: event.clientX,
        y: event.clientY,
      });
      safelyTogglePointerCapture(canvas, 'set', event.pointerId);

      const pinch = pinchGeometry(activePointers);
      if (pinch) {
        pinchRef.current = {
          active: true,
          lastCenter: pinch.center,
          lastDistance: pinch.distance,
        };
        dragRef.current = {
          ...dragRef.current,
          active: false,
          moved: true,
        };
        canvas.style.cursor = 'grabbing';
        return;
      }

      dragRef.current = {
        active: true,
        moved: false,
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        lastX: event.clientX,
        lastY: event.clientY,
      };
      canvas.style.cursor = 'grabbing';
    };

    const onPointerMove = (event: PointerEvent) => {
      if (activePointers.has(event.pointerId)) {
        activePointers.set(event.pointerId, {
          x: event.clientX,
          y: event.clientY,
        });
      }

      const pinch = pinchGeometry(activePointers);
      if (pinch && pinchRef.current.active) {
        event.preventDefault();
        const previous = pinchRef.current;
        const rect = canvas.getBoundingClientRect();
        const anchor = {
          x: pinch.center.x - finiteOr(rect.left, 0),
          y: pinch.center.y - finiteOr(rect.top, 0),
        };
        const distanceFactor =
          previous.lastDistance > 0 ? pinch.distance / previous.lastDistance : 1;
        const centeredView = panBy(
          viewRef.current,
          pinch.center.x - previous.lastCenter.x,
          pinch.center.y - previous.lastCenter.y,
        );

        pinchRef.current = {
          active: true,
          lastCenter: pinch.center,
          lastDistance: pinch.distance,
        };
        dragRef.current = { ...dragRef.current, moved: true };
        applyView(zoomAt(centeredView, anchor, distanceFactor));
        return;
      }

      const drag = dragRef.current;
      if (drag.active && drag.pointerId === event.pointerId) {
        const totalX = event.clientX - drag.startX;
        const totalY = event.clientY - drag.startY;
        const crossedThreshold = Math.hypot(totalX, totalY) > DRAG_THRESHOLD_PX;

        if (!drag.moved && !crossedThreshold) {
          dragRef.current = {
            ...drag,
            lastX: event.clientX,
            lastY: event.clientY,
          };
          return;
        }

        const deltaX = drag.moved ? event.clientX - drag.lastX : totalX;
        const deltaY = drag.moved ? event.clientY - drag.lastY : totalY;
        dragRef.current = {
          ...drag,
          moved: true,
          lastX: event.clientX,
          lastY: event.clientY,
        };
        applyView(panBy(viewRef.current, deltaX, deltaY));
        return;
      }

      const hit = hitAt(clientPoint(event, canvas));
      if (!hit) {
        if (hoverRef.current) {
          updateHover(null);
        }
        return;
      }

      if (hit.point.kind === 'chunk') {
        updateHover({
          kind: 'chunk',
          screen: { x: hit.point.x, y: hit.point.y },
          point: hit.point.point,
          hit: hit.point.hit,
        });
        return;
      }

      updateHover({
        kind: 'query',
        screen: { x: hit.point.x, y: hit.point.y },
        point: hit.point.point,
      });
    };

    const stopPointer = (event: PointerEvent) => {
      const wasPinching = pinchRef.current.active;
      activePointers.delete(event.pointerId);
      safelyTogglePointerCapture(canvas, 'release', event.pointerId);

      const remainingPinch = pinchGeometry(activePointers);
      if (wasPinching && remainingPinch) {
        pinchRef.current = {
          active: true,
          lastCenter: remainingPinch.center,
          lastDistance: remainingPinch.distance,
        };
        dragRef.current = { ...dragRef.current, active: false, moved: true };
        return;
      }

      const remainingPointer = Array.from(activePointers.entries())[0];
      if (wasPinching && remainingPointer) {
        const [pointerId, point] = remainingPointer;
        pinchRef.current = { ...pinchRef.current, active: false };
        dragRef.current = {
          active: true,
          moved: true,
          pointerId,
          startX: point.x,
          startY: point.y,
          lastX: point.x,
          lastY: point.y,
        };
        return;
      }

      pinchRef.current = { ...pinchRef.current, active: false };
      if (dragRef.current.pointerId === event.pointerId || wasPinching) {
        dragRef.current = {
          ...dragRef.current,
          active: false,
          moved: dragRef.current.moved || wasPinching,
        };
      }
      canvas.style.cursor = '';
    };

    const onPointerLeave = () => {
      if (!dragRef.current.active && hoverRef.current) {
        updateHover(null);
      }
    };

    const onClick = (event: MouseEvent) => {
      if (dragRef.current.moved) {
        dragRef.current = { ...dragRef.current, moved: false };
        return;
      }

      const hit = hitAt(clientPoint(event, canvas));
      if (!hit) {
        onFocusedChildChangeRef.current?.(null);
        return;
      }

      if (hit.point.kind === 'chunk') {
        onFocusedChildChangeRef.current?.(hit.point.point.child_id);
      }
    };

    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', stopPointer);
    canvas.addEventListener('pointercancel', stopPointer);
    canvas.addEventListener('pointerleave', onPointerLeave);
    canvas.addEventListener('click', onClick);

    return () => {
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', stopPointer);
      canvas.removeEventListener('pointercancel', stopPointer);
      canvas.removeEventListener('pointerleave', onPointerLeave);
      canvas.removeEventListener('click', onClick);
      activePointers.clear();
    };
  }, [applyView, hitAt, updateHover]);

  useEffect(() => {
    if (!focusedChildId) {
      return;
    }

    const target = pointById.get(focusedChildId);
    const { width, height } = sizeRef.current;
    if (!target || width <= 0 || height <= 0) {
      return;
    }

    const projected = projectPoint(target, {
      width,
      height,
      padding: PADDING,
      view: viewRef.current,
    });
    applyView(
      panBy(viewRef.current, width / 2 - projected.x, height / 2 - projected.y),
    );
  }, [applyView, focusedChildId, pointById, size.width, size.height]);

  const onZoomIn = () => {
    applyView(
      zoomAt(viewRef.current, { x: sizeRef.current.width / 2, y: sizeRef.current.height / 2 }, 1.18),
    );
  };
  const onZoomOut = () => {
    applyView(
      zoomAt(viewRef.current, { x: sizeRef.current.width / 2, y: sizeRef.current.height / 2 }, 1 / 1.18),
    );
  };
  const onReset = () => {
    applyView(resetView());
  };
  const onSwitchMotion = (enabled: boolean) => {
    if (motionEnabled === undefined) {
      setInternalMotionEnabled(enabled);
    }
    onMotionEnabledChange?.(enabled);
  };

  const tooltipStyle = hover
    ? getTooltipStyle(hover.screen, size.width, size.height)
    : null;

  return (
    <div
      ref={rootRef}
      className="relative h-full min-h-[420px] w-full overflow-hidden rounded-md border border-border-button bg-bg-card"
    >
      <canvas
        ref={canvasRef}
        aria-label="知识库 UMAP 向量空间"
        className="absolute inset-0 size-full touch-none outline-none"
        role="img"
      />

      <div
        role="list"
        aria-label="向量图图例"
        className="pointer-events-none absolute left-3 top-3 flex max-w-[calc(100%-1.5rem)] flex-wrap items-center gap-x-3 gap-y-1 rounded-md border border-border-button bg-bg-card/95 px-2.5 py-1.5 text-[11px] text-text-secondary shadow-sm backdrop-blur"
      >
        {[
          ['base', 'Chunk', BLUE, false],
          ['rrf', 'RRF', ORANGE, false],
          ['dense', 'Dense', PURPLE, false],
          ['rerank', 'Rerank', GREEN, false],
          ['query', '问题', RED, true],
        ].map(([id, label, color, diamond]) => (
          <span key={String(id)} role="listitem" className="flex items-center gap-1.5">
            <span
              data-testid={`legend-${id}`}
              className={`size-2 shrink-0 ${diamond ? 'rotate-45 rounded-[1px]' : 'rounded-full'}`}
              style={{ backgroundColor: String(color) }}
            />
            <span>{label}</span>
          </span>
        ))}
      </div>

      <div className="absolute bottom-3 left-3 right-3 flex max-w-[calc(100%-1.5rem)] flex-wrap items-center justify-end gap-1 rounded-md border border-border-button bg-bg-card/95 p-1 shadow-sm backdrop-blur sm:left-auto sm:gap-2">
        <output
          data-testid="vector-zoom-level"
          aria-label="当前缩放比例"
          className="min-w-12 px-1 text-center font-mono text-[11px] tabular-nums text-text-secondary"
        >
          {Math.round(view.scale * 100)}%
        </output>
        <Button
          type="button"
          variant="outline"
          size="icon"
          title="放大"
          aria-label="放大"
          onClick={onZoomIn}
        >
          <ZoomIn className="size-4" />
        </Button>
        <Button
          type="button"
          variant="outline"
          size="icon"
          title="缩小"
          aria-label="缩小"
          onClick={onZoomOut}
        >
          <ZoomOut className="size-4" />
        </Button>
        <Button
          type="button"
          variant="outline"
          size="icon"
          title="恢复初始视图"
          aria-label="恢复初始视图"
          onClick={onReset}
        >
          <LocateFixed className="size-4" />
        </Button>
        <label
          htmlFor={motionSwitchId}
          className="ml-1 flex h-8 items-center gap-2 rounded border border-border-button bg-bg-input px-2 text-xs text-text-secondary"
        >
          <Sparkles className="size-3.5" />
          <span>粒子动效</span>
          <Switch
            id={motionSwitchId}
            aria-label="粒子动效"
            checked={effectiveMotionEnabled}
            onCheckedChange={onSwitchMotion}
          />
        </label>
      </div>

      {hover && tooltipStyle ? (
        <div
          role="tooltip"
          className="pointer-events-none absolute z-10 rounded-md border border-border-button bg-bg-card p-3 text-xs leading-5 text-text-primary shadow-lg"
          style={tooltipStyle}
        >
          {hover.kind === 'chunk' ? (
            <div className="space-y-2">
              <div className="font-semibold">{hover.point.title}</div>
              <p className="line-clamp-3 text-text-secondary">{hover.point.excerpt}</p>
              <dl className="grid grid-cols-[72px_minmax(0,1fr)] gap-x-2 gap-y-1">
                <dt className="text-text-secondary">文档</dt>
                <dd className="truncate">{hover.point.document_name}</dd>
                <dt className="text-text-secondary">页码</dt>
                <dd>{pageLabel(hover.point)}</dd>
                <dt className="text-text-secondary">Child ID</dt>
                <dd className="truncate font-mono">{hover.point.child_id}</dd>
                {hover.hit ? (
                  <>
                    <dt className="text-text-secondary">阶段</dt>
                    <dd>{hover.hit.stage}</dd>
                    <dt className="text-text-secondary">排名</dt>
                    <dd>第 {hover.hit.rank} 名</dd>
                    <dt className="text-text-secondary">分数</dt>
                    <dd className="font-mono">{roundedScore(hover.hit.score)}</dd>
                  </>
                ) : null}
              </dl>
            </div>
          ) : (
            <div className="space-y-1">
              <div className="font-semibold">查询语义</div>
              <div className="font-mono text-text-secondary">
                x {hover.point.x.toFixed(4)} · y {hover.point.y.toFixed(4)}
              </div>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}

export default VectorSpaceMap;
