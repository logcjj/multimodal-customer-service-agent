import {
  HIT_PRIORITY,
  INITIAL_VIEW,
  buildSpatialGrid,
  findNearestPoint,
  getSpatialGridCandidateIndices,
  panBy,
  projectPoint,
  resetView,
  zoomAt,
} from './vector-space-math';

const canvas = { width: 200, height: 100, padding: 10 };

describe('vector-space-math', () => {
  it('projects normalized UMAP coordinates into padded canvas space with scale and offset', () => {
    const projected = projectPoint(
      { x: 0.25, y: 0.75 },
      {
        ...canvas,
        view: { scale: 2, offsetX: -5, offsetY: 7 },
      },
    );

    expect(projected).toEqual({ x: 105, y: 147 });
  });

  it('projects boundary and negative coordinates without clamping valid finite values', () => {
    expect(projectPoint({ x: 0, y: 1 }, { ...canvas, view: INITIAL_VIEW })).toEqual({
      x: 10,
      y: 90,
    });
    expect(projectPoint({ x: -0.5, y: 1.5 }, { ...canvas, view: INITIAL_VIEW })).toEqual({
      x: -80,
      y: 130,
    });
  });

  it('keeps projected coordinates finite for zero-size canvas and NaN or Infinity inputs', () => {
    const projected = projectPoint(
      { x: Number.NaN, y: Number.POSITIVE_INFINITY },
      {
        width: 0,
        height: Number.NaN,
        padding: Number.POSITIVE_INFINITY,
        view: {
          scale: Number.NaN,
          offsetX: Number.POSITIVE_INFINITY,
          offsetY: Number.NEGATIVE_INFINITY,
        },
      },
    );

    expect(Number.isFinite(projected.x)).toBe(true);
    expect(Number.isFinite(projected.y)).toBe(true);
  });

  it('zooms around an anchor without drifting the data point under the cursor', () => {
    const point = { x: 0.4, y: 0.2 };
    const before = projectPoint(point, {
      width: 300,
      height: 200,
      padding: 20,
      view: INITIAL_VIEW,
    });

    const view = zoomAt(INITIAL_VIEW, before, 1.75);
    const after = projectPoint(point, {
      width: 300,
      height: 200,
      padding: 20,
      view,
    });

    expect(view.scale).toBe(1.75);
    expect(after.x).toBeCloseTo(before.x, 8);
    expect(after.y).toBeCloseTo(before.y, 8);
  });

  it('clamps zoom to the supported scale range', () => {
    expect(zoomAt(INITIAL_VIEW, { x: 50, y: 50 }, 100).scale).toBe(24);
    expect(zoomAt(INITIAL_VIEW, { x: 50, y: 50 }, 0.01).scale).toBe(0.55);
    expect(zoomAt({ scale: Number.NaN, offsetX: 0, offsetY: 0 }, { x: 50, y: 50 }, 2)).toEqual({
      scale: 2,
      offsetX: -50,
      offsetY: -50,
    });
  });

  it('pans by screen pixels and resets to the deterministic initial view', () => {
    expect(panBy({ scale: 2, offsetX: 10, offsetY: -5 }, 12, -8)).toEqual({
      scale: 2,
      offsetX: 22,
      offsetY: -13,
    });
    expect(resetView()).toEqual({ scale: 1, offsetX: 0, offsetY: 0 });
    expect(resetView()).not.toBe(INITIAL_VIEW);
  });

  it('builds a 24px screen-space grid and returns only the hovered cell plus eight neighbors', () => {
    const grid = buildSpatialGrid([
      { id: 'same-cell', x: 25, y: 25 },
      { id: 'neighbor-cell', x: 1, y: 1 },
      { id: 'far-cell', x: 73, y: 25 },
      { id: 'negative-cell', x: -10, y: 0 },
      { id: 'invalid-cell', x: Number.NaN, y: 0 },
    ]);

    expect(grid.cellSize).toBe(24);
    expect(getSpatialGridCandidateIndices(grid, { x: 25, y: 25 })).toEqual([0, 1]);
    expect(getSpatialGridCandidateIndices(grid, { x: 1, y: 1 })).toEqual([0, 1, 3]);
  });

  it('finds the nearest hittable point within radius', () => {
    const grid = buildSpatialGrid([
      { id: 'near', x: 11, y: 10, priority: HIT_PRIORITY.normal },
      { id: 'highlight-but-farther', x: 18, y: 10, priority: HIT_PRIORITY.highlight },
      { id: 'outside', x: 80, y: 80, priority: HIT_PRIORITY.query },
    ]);

    const hit = findNearestPoint(grid, { x: 10, y: 10 }, { radius: 20 });

    expect(hit?.point.id).toBe('near');
    expect(hit?.distance).toBe(1);
  });

  it('uses priority to resolve same-distance or overlapping hits', () => {
    const grid = buildSpatialGrid([
      { id: 'ordinary', x: 50, y: 50, priority: HIT_PRIORITY.normal },
      { id: 'top10', x: 50, y: 50, priority: HIT_PRIORITY.highlight },
      { id: 'query', x: 50, y: 50, priority: HIT_PRIORITY.query },
    ]);

    expect(findNearestPoint(grid, { x: 50, y: 50 }, { radius: 6 })?.point.id).toBe('query');
    expect(
      findNearestPoint(grid, { x: 50.5, y: 50 }, { radius: 6, priorityDistanceTolerance: 1 })
        ?.point.id,
    ).toBe('query');
  });

  it('handles empty grids, disabled points and non-finite hit inputs without throwing', () => {
    expect(findNearestPoint(buildSpatialGrid([]), { x: 0, y: 0 }, { radius: 10 })).toBeNull();

    const grid = buildSpatialGrid([
      { id: 'disabled', x: 0, y: 0, hittable: false },
      { id: 'enabled', x: 3, y: 4 },
      { id: 'invalid', x: Number.POSITIVE_INFINITY, y: 0 },
    ]);

    expect(findNearestPoint(grid, { x: 0, y: 0 }, { radius: 5 })?.point.id).toBe('enabled');
    expect(
      findNearestPoint(grid, { x: Number.NaN, y: Number.NEGATIVE_INFINITY }, { radius: 5 }),
    ).toBeNull();
    expect(findNearestPoint(grid, { x: 0, y: 0 }, { radius: Number.NaN })).toBeNull();
  });
});
