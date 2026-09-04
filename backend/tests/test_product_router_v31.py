from __future__ import annotations

from app.knowledge.product_router import ProductRouter


def test_explicit_air_fryer_routes_to_english_manual() -> None:
    route = ProductRouter().route("空气炸锅显示 E1 应该怎么处理？")

    assert route.products == ("Air Fryer",)
    assert route.confidence >= 0.9


def test_explicit_washing_machine_routes_to_matching_manual() -> None:
    route = ProductRouter().route("洗衣机提示 E03，无法排水")

    assert route.products == ("Washing Machine",)
    assert "洗衣机" in route.matched_aliases


def test_weak_alias_does_not_hard_lock_retrieval() -> None:
    route = ProductRouter().route("滤网应该多久清洁一次？")

    assert route.products == ()
    assert route.reason == "只有弱别名，保持全库检索"


def test_ambiguous_camera_alias_keeps_both_candidates() -> None:
    route = ProductRouter().route("相机无法正常拍照")

    assert route.products == ("Camera", "相机手册")
    assert route.reason == "命中歧义别名，保留多个候选产品"


def test_jetski_common_spelling_and_typo_route_to_waverunner_manual() -> None:
    router = ProductRouter()

    assert router.route("How to board the jetski alone?").products == ("WaveRunner",)
    assert router.route("Does this jstski have storage?").products == ("WaveRunner",)
