from __future__ import annotations

from dataclasses import dataclass


# 产品路由只描述知识库目录，不包含题目答案。显式产品名用于缩小检索范围，
# 过短或跨品类的弱别名不会触发硬过滤，避免错误路由损害召回率。
PRODUCT_ALIASES: dict[str, tuple[str, ...]] = {
    "VR头显手册": ("vr头显", "vr眼镜", "虚拟现实头显", "虚拟现实"),
    "人体工学椅手册": ("人体工学椅", "办公椅"),
    "健身单车手册": ("健身单车", "健身车", "动感单车", "单车"),
    "健身追踪器手册": ("健身追踪器", "运动手环", "追踪器"),
    "儿童电动摩托车手册": ("儿童电动摩托车", "儿童摩托车", "电动摩托车"),
    "冰箱手册": ("冰箱", "冷藏室", "冷冻室"),
    "功能键盘手册": ("功能键盘", "机械键盘", "键盘"),
    "发电机手册": ("发电机", "generator"),
    "可编程温控器手册": ("可编程温控器", "温控器", "恒温器", "thermostat"),
    "吹风机手册": ("吹风机", "风嘴", "发梳"),
    "摩托艇手册": ("摩托艇",),
    "水泵手册": ("水泵", "抽水泵", "pump"),
    "洗碗机手册": ("洗碗机", "碗篮"),
    "烤箱手册": ("烤箱", "炉灯"),
    "电钻手册": ("电钻", "drill", "充电器"),
    "相机手册": ("相机", "照相机", "摄影", "拍照"),
    "空气净化器手册": ("空气净化器", "净化器", "滤网"),
    "空调手册": ("空调", "制冷", "制热", "遥控器"),
    "蒸汽清洁机手册": ("蒸汽清洁机", "蒸汽拖把", "清洁机"),
    "蓝牙激光鼠标手册": ("蓝牙激光鼠标", "激光鼠标", "laser mouse", "鼠标"),
    "Camera": ("camera", "相机", "拍照", "照片", "摄影"),
    "Espresso Machine": ("espresso machine", "意式咖啡机", "咖啡机"),
    "Air Fryer": ("air fryer", "airfryer", "空气炸锅"),
    "Boat": ("boat", "游艇", "划船", "钓鱼", "船"),
    "WaveRunner": (
        "waverunner",
        "jet ski",
        "jetski",
        "jstski",
        "personal watercraft",
        "watercraft",
    ),
    "Printer": ("printer", "打印机", "fax machine", "传真机", "传真", "多功能一体机"),
    "Earphones": ("earphones", "earbuds", "耳机", "耳塞"),
    "Media Player": (
        "media player",
        "micro sd",
        "ereader",
        "e-reader",
        "ebook reader",
        "e-book reader",
        "电子书阅读器",
        "电子书",
        "播放器",
    ),
    "Gas Grill": ("gas grill", "烧烤炉", "烤架"),
    "Snowmobile": ("snowmobile", "雪地摩托"),
    "TV": ("television", "电视机", "tv"),
    "Vacuum": ("robot vacuum", "robotic vacuum", "扫地机器人", "roomba", "吸尘器", "vacuum"),
    "Toothbrush": ("electric toothbrush", "电动牙刷", "toothbrush", "牙刷"),
    "Washing Machine": ("washing machine", "洗衣机"),
    "Pressure Cooker": ("pressure cooker", "高压锅", "压力锅"),
    "Microwave": ("microwave", "微波炉"),
    "Motherboard": ("motherboard", "主板", "bios"),
    "Phone": ("phone", "手机", "移动电话", "电话"),
    "Lawn Mower": ("lawn mower", "割草机"),
}


WEAK_ALIASES = frozenset({"滤网", "耳机", "耳塞", "烤架", "遥控器", "充电器", "单车", "船", "电话"})

COMPATIBILITY_PRODUCTS: dict[str, tuple[str, ...]] = {
    "空气净化器手册": ("air-purifier",),
    "Washing Machine": ("washing-machine",),
    "健身追踪器手册": ("fitness-tracker",),
}


@dataclass(frozen=True)
class ProductRoute:
    products: tuple[str, ...]
    retrieval_products: tuple[str, ...]
    matched_aliases: tuple[str, ...]
    confidence: float
    reason: str


class ProductRouter:
    def __init__(self, aliases: dict[str, tuple[str, ...]] | None = None) -> None:
        source = aliases or PRODUCT_ALIASES
        self._alias_to_products: dict[str, set[str]] = {}
        for product, product_aliases in source.items():
            for alias in product_aliases:
                self._alias_to_products.setdefault(alias.lower(), set()).add(product)

    def route(self, query: str) -> ProductRoute:
        normalized = query.lower().strip()
        matches = [alias for alias in self._alias_to_products if alias in normalized]
        strong = [alias for alias in matches if alias not in WEAK_ALIASES]
        if not strong:
            return ProductRoute(
                products=(),
                retrieval_products=(),
                matched_aliases=tuple(sorted(matches, key=lambda item: (-len(item), item))),
                confidence=0.0,
                reason="只有弱别名，保持全库检索" if matches else "未识别明确产品，保持全库检索",
            )

        longest_length = max(len(alias) for alias in strong)
        decisive_aliases = sorted(
            {alias for alias in strong if len(alias) == longest_length},
            key=lambda item: (-len(item), item),
        )
        products = sorted(
            {product for alias in decisive_aliases for product in self._alias_to_products[alias]},
            key=lambda value: (value.casefold(), value),
        )
        ambiguous = len(products) > 1
        return ProductRoute(
            products=tuple(products),
            retrieval_products=tuple(
                dict.fromkeys(
                    product
                    for canonical in products
                    for product in (canonical, *COMPATIBILITY_PRODUCTS.get(canonical, ()))
                )
            ),
            matched_aliases=tuple(decisive_aliases),
            confidence=0.86 if ambiguous else 0.97,
            reason="命中歧义别名，保留多个候选产品" if ambiguous else "命中显式产品别名",
        )
