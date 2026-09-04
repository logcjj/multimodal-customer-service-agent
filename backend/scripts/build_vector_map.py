from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from app.contracts.models import ModelKind
from app.knowledge.service import KnowledgeService
from app.models.service import ModelService
from app.storage.database import Database


def build_vector_map(
    *,
    dataset_id: str,
    data_dir: Path,
    timeout_seconds: float,
    poll_interval_seconds: float = 0.5,
) -> dict[str, object]:
    database = Database(data_dir)
    model_service = ModelService(database)
    service = KnowledgeService(database)

    def embedding_model_name() -> str:
        runtime = model_service.get_default_runtime(ModelKind.EMBEDDING)
        return runtime[0].name if runtime else "embedding-unconfigured"

    service.embedding_model_provider = embedding_model_name
    started = time.monotonic()
    try:
        payload = service.rebuild_vector_map(dataset_id)
        while payload.get("status") in {"building", "stale", "missing"}:
            if time.monotonic() - started > timeout_seconds:
                raise TimeoutError(f"UMAP build timed out after {timeout_seconds:.0f}s")
            time.sleep(poll_interval_seconds)
            payload = service.vector_map(dataset_id)
        return payload
    finally:
        service.shutdown()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    backend_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="构建指定知识库的完整 UMAP 向量图")
    parser.add_argument("--dataset-id", required=True, help="知识库 ID")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=backend_root / "data",
        help="后端数据目录，默认使用 backend/data",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="构建超时秒数，默认 1800",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()
    started = time.monotonic()
    try:
        payload = build_vector_map(
            dataset_id=args.dataset_id,
            data_dir=data_dir,
            timeout_seconds=args.timeout,
        )
    except Exception as exc:
        print(f"status=failed error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    duration = time.monotonic() - started
    status = str(payload.get("status", "failed"))
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    point_count = int(meta.get("point_count", 0))
    published_version = str(meta.get("published_version", "unknown"))
    embedding_model = str(meta.get("embedding_model", "unknown"))
    output_root = data_dir / "vector_maps"
    print(
        f"status={status} point_count={point_count} duration_seconds={duration:.3f} "
        f"dataset_id={args.dataset_id} published_version={published_version} "
        f"embedding_model={embedding_model} output_root={output_root}"
    )
    if status != "ready":
        error = payload.get("error")
        if isinstance(error, dict):
            print(
                f"error_code={error.get('code', 'unknown')} "
                f"error_message={error.get('message', 'unknown')}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
