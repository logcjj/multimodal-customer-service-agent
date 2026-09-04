from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from app.contracts.models import ModelKind
from app.knowledge.index_bundle import IndexManifest
from app.knowledge.service import KnowledgeService
from app.models.service import ModelService
from app.storage.database import Database


def build_bundle(
    *,
    dataset_id: str,
    data_dir: Path,
    timeout_seconds: float,
    poll_interval_seconds: float = 0.05,
) -> IndexManifest:
    database = Database(data_dir)
    model_service = ModelService(database)
    service = KnowledgeService(database)

    def embedding_model_name() -> str:
        runtime = model_service.get_default_runtime(ModelKind.EMBEDDING)
        return runtime[0].name if runtime else "embedding-unconfigured"

    service.embedding_model_provider = embedding_model_name
    try:
        job = service.start_index_build(dataset_id)
        deadline = time.monotonic() + timeout_seconds
        while job.state in {"queued", "running"}:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"index bundle build timed out after {timeout_seconds:.1f}s")
            time.sleep(poll_interval_seconds)
            job = service.index_build_view(job.id)
        if job.state != "succeeded":
            raise RuntimeError(f"{job.error_code or 'IndexBuildFailed'}: {job.error_message or 'unknown'}")
        if not job.index_version:
            raise RuntimeError("IndexBuildFailed: succeeded build has no index version")
        return service.index_manifest(dataset_id, job.index_version)
    finally:
        service.shutdown()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    backend_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="为已发布知识库生成标准离线 Index Bundle")
    parser.add_argument("--dataset-id", required=True, help="已发布知识库 ID")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=backend_root / "data",
        help="后端数据目录，默认 backend/data",
    )
    parser.add_argument("--timeout", type=float, default=1800.0, help="构建超时秒数")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_bundle(
            dataset_id=args.dataset_id,
            data_dir=args.data_dir.expanduser().resolve(),
            timeout_seconds=args.timeout,
        )
    except Exception as exc:
        print(f"status=failed error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        "status=ready "
        f"dataset_id={manifest.dataset_id} index_version={manifest.index_version} "
        f"sources={manifest.counts.get('sources', 0)} "
        f"parent_chunks={manifest.counts.get('parent_chunks', 0)} "
        f"child_chunks={manifest.counts.get('child_chunks', 0)} "
        f"image_chunks={manifest.counts.get('image_chunks', 0)} "
        f"vector_dimension={manifest.vector_dimension}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
