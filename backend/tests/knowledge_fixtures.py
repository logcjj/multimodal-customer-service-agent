from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.main import create_app


TEST_MANUAL = """# 空气净化器滤网清洁
清洁前关闭电源并拔下插头，使用吸尘器清理预过滤网，完全晾干后再装回。

# 洗衣机 E03 排水故障
本节适用于 XQG100。洗衣机出现 E03 时，先关闭电源，检查排水管并清理排水过滤器。

# 健身追踪器表带安装
将表带对准接口插入，听到咔嗒声后轻拉确认已经卡紧。
"""


def create_client_with_manuals(data_dir: Path, **app_kwargs: Any) -> TestClient:
    client = TestClient(create_app(data_dir=data_dir, **app_kwargs))
    dataset = client.post(
        "/api/datasets",
        json={"name": "测试产品说明书", "parser_profile": "manual"},
    )
    assert dataset.status_code == 201
    dataset_id = dataset.json()["id"]
    uploaded = client.post(
        "/api/files",
        files={"file": ("test-manual.md", TEST_MANUAL.encode(), "text/markdown")},
    )
    assert uploaded.status_code == 201
    document = client.post(
        f"/api/datasets/{dataset_id}/documents",
        json={"file_id": uploaded.json()["id"], "parser_profile": "manual"},
    )
    assert document.status_code == 201
    parsed = client.post(f"/api/documents/{document.json()['id']}/parse")
    assert parsed.status_code == 200
    published = client.post(
        f"/api/datasets/{dataset_id}/publish",
        json={"index_version": parsed.json()["index_version"]},
    )
    assert published.status_code == 200
    return client
