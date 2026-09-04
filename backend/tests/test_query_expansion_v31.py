from __future__ import annotations

from app.knowledge.query_expansion import deterministic_expansion


def test_expands_common_chinese_manual_terms_without_dropping_codes() -> None:
    expanded = deterministic_expansion("洗衣机显示 E03，无法排水")

    assert "E03" in expanded
    assert "drain drainage" in expanded
    assert "error code troubleshooting" in expanded


def test_expands_compact_camera_error_code_with_equivalent_search_forms() -> None:
    expanded = deterministic_expansion("佳能相机显示 ERR02")

    assert "ERR02" in expanded
    assert "ERR_02" in expanded
    assert "ERR 02" in expanded


def test_expands_air_fryer_basket_cleaning_terms() -> None:
    expanded = deterministic_expansion("空气炸锅如何清洁炸篮？")

    assert "clean cleaning" in expanded
    assert "basket" in expanded


def test_expands_common_english_boarding_typo_for_manual_retrieval() -> None:
    expanded = deterministic_expansion("In deep water, how to borad the jetski alone?")

    assert "board boarding reboard reboarding" in expanded
    assert "solo" in expanded


def test_expands_visual_tree_symbol_into_air_purification_retrieval_terms() -> None:
    expanded = deterministic_expansion("空调遥控器显示小松树图标")

    assert "空气净化 等离子净化 离子发生器" in expanded
    assert "air purify ionizer plasma purification" in expanded
