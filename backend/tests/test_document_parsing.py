from __future__ import annotations

import pytest

from app.knowledge.chunking import ChunkingService
from app.knowledge.parsers import NormalizedBlock, NormalizedDocument, NormalizedPage, ParserRegistry


@pytest.fixture
def normalized_manual() -> NormalizedDocument:
    return NormalizedDocument(
        title="洗衣机说明书",
        pages=[
            NormalizedPage(
                page_number=1,
                blocks=[
                    NormalizedBlock(kind="heading", text="E03 排水故障", heading_level=1),
                    NormalizedBlock(kind="paragraph", text="显示 E03 时，请先关闭电源并拔下插头。"),
                    NormalizedBlock(kind="list", text="检查排水管是否弯折或堵塞。"),
                    NormalizedBlock(kind="list", text="清理排水过滤器后重新启动。"),
                ],
            )
        ],
    )


@pytest.mark.parametrize("profile", ["general", "manual", "qa", "table", "picture"])
def test_profile_produces_parent_and_child_chunks(profile, normalized_manual) -> None:
    result = ChunkingService().chunk(normalized_manual, profile)

    assert result.parents
    assert result.children
    assert all(child.parent_local_id for child in result.children)
    assert all(child.page_start >= 1 for child in result.children)


def test_text_parser_preserves_markdown_heading(tmp_path) -> None:
    path = tmp_path / "manual.md"
    path.write_text("# 安装\n\n先关闭电源。\n\n## 检查\n检查 E03。", encoding="utf-8")

    document = ParserRegistry().parse(path, "text/markdown")

    assert document.pages[0].blocks[0].kind == "heading"
    assert document.pages[0].blocks[0].text == "安装"
    assert any(block.text == "检查 E03。" for block in document.pages[0].blocks)

