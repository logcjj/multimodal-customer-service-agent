export interface VectorView {
  scale: number;
  offsetX: number;
  offsetY: number;
}

export interface VectorPoint {
  x: number;
  y: number;
}

export interface ProjectPointOptions {
  width: number;
  height: number;
  padding?: number;
  view?: Partial<VectorView>;
}

export interface SpatialPoint extends VectorPoint {
  id: string;
  priority?: number;
  hittable?: boolean;
}

export interface SpatialGrid<TPoint extends SpatialPoint = SpatialPoint> {
  cellSize: number;
  points: readonly TPoint[];
  cells: Map<string, number[]>;
}

export interface HitTestOptions<TPoint extends SpatialPoint = SpatialPoint> {
  radius: number;
  getPriority?: (point: TPoint, index: number) => number;
  priorityDistanceTolerance?: number;
}

export interface HitTestResult<TPoint extends SpatialPoint = SpatialPoint> {
  point: TPoint;
  index: number;
  distance: number;
  priority: number;
}

export const MIN_SCALE = 0.55;
export const MAX_SCALE = 24;
export const DEFAULT_PADDING = 24;
export const SPATIAL_GRID_CELL_SIZE = 24;
export const INITIAL_VIEW: VectorView = { scale: 1, offsetX: 0, offsetY: 0 };
export const HIT_PRIORITY = {
  normal: 0,
  highlight: 10,
  query: 20,
} as const;

const EPSILON = 1e-9;

function finiteOr(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function nonNegativeFinite(value: number | undefined, fallback = 0): number {
  return Math.max(0, finiteOr(value, fallback));
}

function positiveFinite(value: number | undefined, fallback: number): number {
  const finite = finiteOr(value, fallback);
  return finite > 0 ? finite : fallback;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function normalizeScale(scale: number | undefined): number {
  return clamp(finiteOr(scale, INITIAL_VIEW.scale), MIN_SCALE, MAX_SCALE);
}

function normalizeView(view?: Partial<VectorView>): VectorView {
  return {
    scale: normalizeScale(view?.scale),
    offsetX: finiteOr(view?.offsetX, INITIAL_VIEW.offsetX),
    offsetY: finiteOr(view?.offsetY, INITIAL_VIEW.offsetY),
  };
}

function normalizePadding(padding: number | undefined, size: number): number {
  return Math.min(nonNegativeFinite(padding, DEFAULT_PADDING), size / 2);
}

function toFinitePoint(point: VectorPoint): VectorPoint {
  return {
    x: finiteOr(point.x, 0),
    y: finiteOr(point.y, 0),
  };
}

function cellKey(cellX: number, cellY: number): string {
  return `${cellX},${cellY}`;
}

function cellFor(point: VectorPoint, cellSize: number): VectorPoint {
  return {
    x: Math.floor(point.x / cellSize),
    y: Math.floor(point.y / cellSize),
  };
}

export function projectPoint(point: VectorPoint, options: ProjectPointOptions): VectorPoint {
  const width = nonNegativeFinite(options.width);
  const height = nonNegativeFinite(options.height);
  const paddingX = normalizePadding(options.padding, width);
  const paddingY = normalizePadding(options.padding, height);
  const innerWidth = Math.max(0, width - paddingX * 2);
  const innerHeight = Math.max(0, height - paddingY * 2);
  const normalizedPoint = toFinitePoint(point);
  const view = normalizeView(options.view);
  const x = (paddingX + normalizedPoint.x * innerWidth) * view.scale + view.offsetX;
  const y = (paddingY + normalizedPoint.y * innerHeight) * view.scale + view.offsetY;

  return {
    x: finiteOr(x, 0),
    y: finiteOr(y, 0),
  };
}

export function zoomAt(view: VectorView, anchor: VectorPoint, scaleFactor: number): VectorView {
  const current = normalizeView(view);
  const safeAnchor = toFinitePoint(anchor);
  const factor = positiveFinite(scaleFactor, 1);
  const nextScale = clamp(current.scale * factor, MIN_SCALE, MAX_SCALE);
  const worldX = (safeAnchor.x - current.offsetX) / current.scale;
  const worldY = (safeAnchor.y - current.offsetY) / current.scale;

  return {
    scale: nextScale,
    offsetX: finiteOr(safeAnchor.x - worldX * nextScale, 0),
    offsetY: finiteOr(safeAnchor.y - worldY * nextScale, 0),
  };
}

export function panBy(view: VectorView, deltaX: number, deltaY: number): VectorView {
  const current = normalizeView(view);

  return {
    scale: current.scale,
    offsetX: finiteOr(current.offsetX + finiteOr(deltaX, 0), 0),
    offsetY: finiteOr(current.offsetY + finiteOr(deltaY, 0), 0),
  };
}

export function resetView(): VectorView {
  return { ...INITIAL_VIEW };
}

export function buildSpatialGrid<TPoint extends SpatialPoint>(
  points: readonly TPoint[],
  cellSize = SPATIAL_GRID_CELL_SIZE,
): SpatialGrid<TPoint> {
  const safeCellSize = positiveFinite(cellSize, SPATIAL_GRID_CELL_SIZE);
  const cells = new Map<string, number[]>();

  points.forEach((point, index) => {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      return;
    }

    const cell = cellFor(point, safeCellSize);
    const key = cellKey(cell.x, cell.y);
    const bucket = cells.get(key) ?? [];
    bucket.push(index);
    cells.set(key, bucket);
  });

  return {
    cellSize: safeCellSize,
    points,
    cells,
  };
}

export function getSpatialGridCandidateIndices<TPoint extends SpatialPoint>(
  grid: SpatialGrid<TPoint>,
  pointer: VectorPoint,
): number[] {
  if (!Number.isFinite(pointer.x) || !Number.isFinite(pointer.y)) {
    return [];
  }

  const cell = cellFor(pointer, grid.cellSize);
  const candidates = new Set<number>();

  for (let deltaX = -1; deltaX <= 1; deltaX += 1) {
    for (let deltaY = -1; deltaY <= 1; deltaY += 1) {
      const bucket = grid.cells.get(cellKey(cell.x + deltaX, cell.y + deltaY));
      bucket?.forEach((index) => candidates.add(index));
    }
  }

  return Array.from(candidates).sort((left, right) => left - right);
}

export function findNearestPoint<TPoint extends SpatialPoint>(
  grid: SpatialGrid<TPoint>,
  pointer: VectorPoint,
  options: HitTestOptions<TPoint>,
): HitTestResult<TPoint> | null {
  if (!Number.isFinite(pointer.x) || !Number.isFinite(pointer.y)) {
    return null;
  }

  const radius = finiteOr(options.radius, 0);
  if (radius <= 0) {
    return null;
  }

  const radiusSquared = radius * radius;
  const tolerance = nonNegativeFinite(options.priorityDistanceTolerance, 0);
  let best: HitTestResult<TPoint> | null = null;

  getSpatialGridCandidateIndices(grid, pointer).forEach((index) => {
    const point = grid.points[index];
    if (!point || point.hittable === false) {
      return;
    }

    const deltaX = point.x - pointer.x;
    const deltaY = point.y - pointer.y;
    const distanceSquared = deltaX * deltaX + deltaY * deltaY;
    if (!Number.isFinite(distanceSquared) || distanceSquared - radiusSquared > EPSILON) {
      return;
    }

    const distance = Math.sqrt(distanceSquared);
    const rawPriority = options.getPriority?.(point, index) ?? point.priority ?? HIT_PRIORITY.normal;
    const priority = finiteOr(rawPriority, HIT_PRIORITY.normal);
    const candidate: HitTestResult<TPoint> = { point, index, distance, priority };

    if (!best) {
      best = candidate;
      return;
    }

    const distanceGap = candidate.distance - best.distance;
    const sameDistanceBand = Math.abs(distanceGap) <= tolerance + EPSILON;
    if (
      distanceGap < -tolerance - EPSILON ||
      (sameDistanceBand &&
        (candidate.priority > best.priority ||
          (candidate.priority === best.priority && candidate.distance < best.distance - EPSILON) ||
          (candidate.priority === best.priority &&
            Math.abs(candidate.distance - best.distance) <= EPSILON &&
            candidate.index < best.index)))
    ) {
      best = candidate;
    }
  });

  return best;
}
