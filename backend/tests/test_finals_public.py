from __future__ import annotations

import csv
import json

import httpx
import pytest

from app.evaluation.finals_public import (
    download_public_image,
    format_competition_ret,
    load_finals_questions,
)


def test_load_finals_questions_decodes_gb18030_and_extracts_image_urls(tmp_path) -> None:
    path = tmp_path / "evaluation_public.csv"
    with path.open("w", encoding="gb18030", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["id", "question"])
        writer.writeheader()
        writer.writerow(
            {
                "id": "12",
                "question": (
                    '"空调遥控器显示小松树是什么意思？'
                    'https://pic.imgdb.cn/i/example.jpg"'
                ),
            }
        )

    questions = load_finals_questions(path)

    assert len(questions) == 1
    assert questions[0].id == "12"
    assert questions[0].question == "空调遥控器显示小松树是什么意思？"
    assert questions[0].image_urls == ["https://pic.imgdb.cn/i/example.jpg"]


def test_load_finals_questions_repairs_exported_multi_part_quotes(tmp_path) -> None:
    path = tmp_path / "evaluation_public.csv"
    path.write_text(
        'id,question\n1,"""请问商品能提供试用吗？"",""试用后不满意，能退货吗？"""\n',
        encoding="utf-8",
    )

    questions = load_finals_questions(path)

    assert questions[0].question == "请问商品能提供试用吗？\n试用后不满意，能退货吗？"


def test_load_finals_questions_decodes_literal_unicode_escape_from_export(tmp_path) -> None:
    path = tmp_path / "evaluation_public.csv"
    path.write_text(
        'id,question\n41,"""What does Owner\\u2019s Manual include?"""\n',
        encoding="utf-8",
    )

    questions = load_finals_questions(path)

    assert questions[0].question == "What does Owner’s Manual include?"


def test_load_finals_questions_removes_quote_left_before_trailing_csv_comma(tmp_path) -> None:
    path = tmp_path / "evaluation_public.csv"
    path.write_text(
        'id,question\n6,"""商品与详情页描述不一致，请问可以退货吗？"","\n',
        encoding="utf-8",
    )

    questions = load_finals_questions(path)

    assert questions[0].question == "商品与详情页描述不一致，请问可以退货吗？"


def test_format_competition_ret_keeps_pic_and_image_counts_aligned() -> None:
    ret = format_competition_ret(
        "烤架用于承托烤盘。",
        ["Manual_1", "Manual_2", "Manual_1"],
        max_images=2,
    )
    decoder = json.JSONDecoder()
    answer, offset = decoder.raw_decode(ret)
    assert ret[offset] == ","
    images, end = decoder.raw_decode(ret, offset + 1)

    assert end == len(ret)
    assert answer.count("<PIC>") == 2
    assert images == ["Manual_1", "Manual_2"]


def test_download_public_image_accepts_verified_image_and_rejects_html() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("ok.png"):
            return httpx.Response(
                200,
                headers={"content-type": "image/png"},
                content=b"\x89PNG\r\n\x1a\nimage-bytes",
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html></html>",
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        downloaded = download_public_image(
            "https://pic.imgdb.cn/i/ok.png",
            client=client,
        )
        with pytest.raises(ValueError, match="not an image"):
            download_public_image(
                "https://pic.imgdb.cn/i/not-image",
                client=client,
            )

    assert downloaded.mime_type == "image/png"
    assert downloaded.data_url.startswith("data:image/png;base64,")
    assert len(downloaded.sha256) == 64
