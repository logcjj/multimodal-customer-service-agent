from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPlan:
    original: str
    normalized: str
    language: str
    target_manuals: frozenset[str]
    variants: tuple[str, ...]
    needs_images: bool


def normalize_query(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"^[\"'“”]+|[\"'“”]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_language(text: str) -> str:
    letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    return "en" if letters > cjk * 1.5 else "zh"


MANUAL_ALIAS_RULES = {
    "VR头显手册": ("vr头显", "vr", "headset"),
    "人体工学椅手册": ("人体工学椅", "ergonomic chair", "chair"),
    "健身单车手册": ("健身单车", "exercise bike", "bike"),
    "健身追踪器手册": ("健身追踪器", "fitness tracker", "tracker", "watch"),
    "儿童电动摩托车手册": ("儿童电动摩托车", "kids motorcycle", "electric motorcycle"),
    "冰箱手册": ("冰箱", "refrigerator", "fridge"),
    "功能键盘手册": ("功能键盘", "keyboard"),
    "发电机手册": ("发电机", "generator"),
    "可编程温控器手册": ("温控器", "thermostat"),
    "吹风机手册": ("吹风机", "blower", "hair dryer"),
    "摩托艇手册": ("摩托艇", "watercraft", "jet ski"),
    "水泵手册": ("水泵", "pump"),
    "洗碗机手册": ("洗碗机", "dishwasher"),
    "烤箱手册": ("烤箱", "oven"),
    "电钻手册": ("电钻", "drill", "dcb107", "dcb112"),
    "相机手册": ("相机", "camera"),
    "空气净化器手册": ("空气净化器", "air purifier"),
    "空调手册": ("空调", "air conditioner"),
    "蒸汽清洁机手册": ("蒸汽清洁机", "steam cleaner"),
    "蓝牙激光鼠标手册": ("蓝牙激光鼠标", "鼠标", "bluetooth mouse", "mouse"),
    "汇总英文手册": (
        "airfryer",
        "air fryer",
        "multi-use pressure cooker",
        "pressure cooker",
        "coffee machine",
        "earphones",
        "earbuds",
        "ereader",
        "e-reader",
        "fax",
        "grill",
        "landline",
        "lawn mower",
        "microwave",
        "motherboard",
        "vacuum",
        "snowmobile",
        "television",
        "toothbrush",
        "antenna",
        "dvd",
        "brake lever",
        "spark plug",
    ),
}


INTENT_EXPANSIONS = {
    "充电": ("充电线", "充电接口", "充满电", "电池", "续航"),
    "电量": ("电量低", "电池", "充电", "battery"),
    "表带": ("扣紧表带", "拆卸表带", "更换表带", "佩戴"),
    "扣紧": ("扣紧表带", "表扣", "环扣", "贴合"),
    "拆卸": ("拆卸", "取下", "松开卡扣", "拔下"),
    "安装": ("安装", "装入", "连接", "固定"),
    "设置": ("设置", "配置", "启用", "关闭"),
    "清洁": ("清洁", "擦拭", "维护", "clean"),
    "保修": ("三年有限保修", "免费服务", "维修", "warranty"),
    "维修": ("维修", "保养", "更换零件", "service"),
    "故障": ("故障排除", "无法启动", "问题", "troubleshooting"),
    "启动": ("启动", "start", "engine"),
    "停机": ("停机", "关闭", "stop"),
    "配对": ("配对", "蓝牙", "pairing"),
    "发票": ("发票", "抬头", "开具"),
    "退款": ("退款", "到账", "原路返回"),
    "退换货": ("退货", "换货", "7天无理由"),
    "charging": ("charge", "charger", "battery", "fully charged"),
    "charge": ("charging", "charger", "battery", "travel case"),
    "airfryer": ("air fryer", "multi-use pressure cooker", "before first use", "get started"),
    "air fryer": ("multi-use pressure cooker", "before first use", "quick release", "steam release"),
    "quick release": ("quick release button", "steam release valve", "pressure cooking"),
    "first use": ("before first use", "clean", "remove packaging", "water tank"),
    "coffee machine": ("espresso", "energy saving mode", "water volume", "factory settings"),
    "earphones": ("pairing", "charging case", "reset", "maintenance"),
    "ereader": ("ebook", "main menu", "browser history", "voice record", "video"),
    "fax": ("connect", "safety", "warning labels", "canada"),
    "grill": ("lp tank", "regulator", "leak testing", "indirect cooking"),
    "landline": ("base station", "handset", "battery level", "led indicator"),
    "lawn mower": ("roll bar", "height of cut", "engine oil", "filters", "mower belt"),
    "microwave": ("control", "light timer", "auto defrost", "grease filter", "charcoal filter"),
    "motherboard": ("bios", "cpu", "raid", "sata", "rear panel connectors"),
    "vacuum": ("robot anatomy", "home base", "bin", "filter", "sensors"),
    "cleaning": ("clean", "cleaning procedure", "maintenance"),
    "mounting": ("mount", "install", "steps"),
    "troubleshooting": ("problem", "fault", "steps"),
    "activate": ("activate", "deactivate", "features"),
    "deactivate": ("activate", "deactivate", "features"),
}


def detect_target_manuals(query: str, manual_names: set[str] | None = None) -> frozenset[str]:
    lower = query.lower()
    targets = set()
    for manual, aliases in MANUAL_ALIAS_RULES.items():
        if manual_names is not None and manual not in manual_names:
            continue
        if any(alias.lower() in lower for alias in aliases):
            targets.add(manual)
    return frozenset(targets)


def strip_product_aliases(query: str, target_manuals: set[str] | frozenset[str]) -> str:
    stripped = query
    for manual in target_manuals:
        for alias in MANUAL_ALIAS_RULES.get(manual, ()):
            stripped = re.sub(re.escape(alias), " ", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s+", " ", stripped).strip(" ，。?？")
    return stripped or query


def expansion_terms(query: str) -> list[str]:
    lower = query.lower()
    terms: list[str] = []
    for key, values in INTENT_EXPANSIONS.items():
        if key.lower() in lower:
            terms.extend(values)
    return terms


def unique_keep_order(items: list[str], limit: int = 10) -> tuple[str, ...]:
    seen = set()
    unique: list[str] = []
    for item in items:
        item = normalize_query(item)
        if not item or item.lower() in seen:
            continue
        seen.add(item.lower())
        unique.append(item)
        if len(unique) >= limit:
            break
    return tuple(unique)


def build_query_plan(query: str, manual_names: set[str] | None = None) -> QueryPlan:
    normalized = normalize_query(query)
    language = detect_language(normalized)
    target_manuals = detect_target_manuals(normalized, manual_names)
    core = strip_product_aliases(normalized, target_manuals)
    expansions = expansion_terms(normalized)

    lower = normalized.lower()
    product_prefixes = []
    for manual in target_manuals:
        for alias in MANUAL_ALIAS_RULES.get(manual, ()):
            if alias.lower() in lower:
                product_prefixes.append(alias)
                break

    variants = [normalized, core]
    for prefix in product_prefixes:
        variants.append(f"{prefix} {core}")
        if expansions:
            variants.append(f"{prefix} {' '.join(expansions[:6])}")
    if expansions:
        variants.append(" ".join(expansions[:8]))
        variants.append(f"{core} {' '.join(expansions[:5])}")

    needs_images = any(word in normalized.lower() for word in ("图", "图片", "pic", "figure", "diagram"))
    return QueryPlan(
        original=query,
        normalized=normalized,
        language=language,
        target_manuals=target_manuals,
        variants=unique_keep_order(variants, limit=10),
        needs_images=needs_images,
    )
