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
    "VR头显手册": ("vr头显", "vr", "headset", "遮光罩", "游玩区域", "处理器单元", "耳塞"),
    "人体工学椅手册": ("人体工学椅", "人体工学椅子", "椅子", "扶手", "ergonomic chair", "chair"),
    "健身单车手册": ("健身单车", "exercise bike", "bike"),
    "健身追踪器手册": ("健身追踪器", "fitness tracker", "tracker", "watch"),
    "儿童电动摩托车手册": ("儿童电动摩托车", "kids motorcycle", "electric motorcycle"),
    "冰箱手册": ("冰箱", "refrigerator", "fridge"),
    "功能键盘手册": ("功能键盘", "keyboard"),
    "发电机手册": ("发电机", "generator"),
    "可编程温控器手册": ("温控器", "thermostat"),
    "吹风机手册": ("吹风机", "blower", "hair dryer"),
    "摩托艇手册": ("摩托艇", "拖曳速度", "半滑航速度", "滑航速度", "watercraft", "jet ski"),
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
        "boat",
        "ship",
        "sailing",
        "on board",
        "bimini",
        "swim platform",
        "livewell",
        "jetski",
        "jet ski",
        "coffee maker",
        "bluetooth pairing",
        "af mode",
    ),
}


CONTEXT_MANUAL_RULES = (
    (
        "汇总英文手册",
        (
            "boat", "ship", "sailing", "on board", "bimini", "swim platform", "livewell",
            "anchor light", "battery compartment", "battery conversion", "bilge pump",
            "steering system", "engine oil level", "cooling system", "fire extinguishers",
        ),
    ),
    (
        "汇总英文手册",
        (
            "camera battery", "camera's", "camera when", "lens", "photography",
            "shutter button", "af mode", "af point", "camera image", "delete a single image",
            "erase all images", "print photos", "cp direct", "date/time battery",
        ),
    ),
    (
        "汇总英文手册",
        (
            "jetski", "jet ski", "qsts", "sponson", "impeller", "fuel filter",
            "fuel tank", "hood", "seat", "filler cap", "throttle lever",
        ),
    ),
    ("汇总英文手册", ("coffee maker", "empty the system", "factory settings", "water volume")),
    ("汇总英文手册", ("af mode", "af point", "metering mode", "drive mode")),
    ("汇总英文手册", ("bluetooth are pairing", "pairing and connecting")),
    ("人体工学椅手册", ("椅子", "扶手", "腰枕", "头枕", "按摩功能", "升降")),
    ("摩托艇手册", ("拖曳速度", "半滑航速度", "滑航速度", "深水", "尾流", "浪涌")),
    ("水泵手册", ("油箱滤网", "无法抽水", "排放燃油")),
    ("可编程温控器手册", ("警报界面", "警报代码", "热泵", "端子", "接线", "处理器单元")),
    ("VR头显手册", ("遮光罩", "游玩区域", "处理器单元", "耳塞", "镜罩", "ps vr")),
)


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
    "float valve": ("float valve", "silicone cap", "lid", "pressure cooking"),
    "anti-block shield": ("anti-block shield", "steam release pipe", "pressure regulation"),
    "steam release valve": ("steam release valve", "steam release pipe", "pressure cooking"),
    "sealing ring": ("sealing ring", "silicone sealing ring", "install", "remove"),
    "condensation collector": ("condensation collector", "install", "back of cooker base"),
    "natural release": ("natural release", "NR", "NPR", "depressurization"),
    "first use": ("before first use", "clean", "remove packaging", "water tank"),
    "coffee machine": ("espresso", "energy saving mode", "water volume", "factory settings"),
    "coffee maker": ("coffee machine", "clean", "empty the system", "descale"),
    "earphones": ("pairing", "charging case", "reset", "maintenance"),
    "pairing": ("pairing", "bluetooth", "connect", "reset"),
    "ereader": ("ebook", "main menu", "browser history", "voice record", "video", "music", "photo"),
    "music": ("music mode", "audio files", "mp3", "usb cable"),
    "video": ("video mode", "play video", "select video"),
    "photo viewer": ("photo viewer", "browse photo", "photo mode"),
    "fax": ("connect", "safety", "warning labels", "canada"),
    "grill": ("lp tank", "regulator", "leak testing", "indirect cooking"),
    "landline": ("base station", "handset", "battery level", "led indicator"),
    "lawn mower": ("roll bar", "height of cut", "engine oil", "filters", "mower belt"),
    "microwave": ("control", "light timer", "auto defrost", "grease filter", "charcoal filter"),
    "motherboard": ("bios", "cpu", "raid", "sata", "rear panel connectors"),
    "vacuum": ("robot anatomy", "home base", "bin", "filter", "sensors"),
    "boat": ("boat", "watercraft"),
    "approval label": ("approval label", "emission control certificate", "emission control information label"),
    "emission control": ("approval label", "emission control certificate", "emission control information label"),
    "turn a boat": ("boat characteristics", "jet thrust turns", "steering wheel", "jet thrust nozzles", "remote control lever"),
    "while sailing": ("boat characteristics", "steering", "jet thrust", "safe speeds"),
    "steering system": ("steering system checks", "steering wheel", "jet thrust nozzles", "free play"),
    "engine oil level": ("engine oil level", "dipstick", "minimum level mark", "maximum level mark"),
    "water supply": ("jet wash switch", "water supply", "water flow", "jet wash handle lever"),
    "start the boat": ("starting the engine", "battery switch", "blower switch", "engine shut-off cord", "main switch keys"),
    "boat's engine": ("starting the engine", "battery switch", "blower switch", "engine shut-off cord", "main switch keys"),
    "load the boat": ("deactivate the cruise assist", "remote control levers", "decrease the engine speed", "trailering"),
    "cruise is over": ("deactivate the cruise assist", "remote control levers", "decrease the engine speed"),
    "throttle-cable": ("throttle cable", "grease points", "grease the throttle-cable inner wires", "pulley wheel"),
    "throttle cable": ("throttle cable", "grease points", "grease the throttle-cable inner wires", "pulley wheel"),
    "jetski": ("watercraft",),
    "jet ski": ("watercraft",),
    "install the card": ("installing the card", "insert the CF card", "close the cover", "CF card"),
    "delete a single image": ("erasing a single image", "display the image", "select the image to be erased"),
    "cp direct": ("CP Direct", "start printing", "direct printing", "select OK"),
    "open and close the hood": ("engine hood", "engine hood latches", "open the engine hood", "lift the engine hood"),
    "filler caps": ("fuel tank filler cap", "oil tank filler cap", "fuel cock knob"),
    "engine switches": ("engine shut-off switch", "main switches", "START", "ON", "OFF"),
    "qsts": ("Quick Shift Trim System", "QSTS selector", "trim angle"),
    "sponson": ("adjustable sponson", "adjusting the adjustable sponson", "turning performance"),
    "af mode": ("selecting the AF mode", "AF point", "focusing", "metering"),
    "lens": ("mounting and detaching a lens", "attach the lens", "lens mount index"),
    "烤架": ("烤架", "烧烤", "支架", "层位"),
    "烤盘": ("烤盘", "烘焙", "饼干", "蛋糕", "披萨"),
    "接油盘": ("接油盘", "收集油脂", "食物碎屑", "少量水"),
    "油脂过滤器": ("油脂过滤器", "风扇", "热风循环", "洗碗机清洗"),
    "滑动搁架": ("滑动搁架", "半拉出", "配件", "洗碗机清洗"),
    "催化侧面板": ("催化侧面板", "搪瓷涂层", "油脂飞溅", "自动清洁"),
    "cleaning": ("clean", "cleaning procedure", "maintenance"),
    "mounting": ("mount", "install", "steps"),
    "troubleshooting": ("problem", "fault", "steps"),
    "activate": ("activate", "deactivate", "features"),
    "deactivate": ("activate", "deactivate", "features"),
}


def detect_target_manuals(query: str, manual_names: set[str] | None = None) -> frozenset[str]:
    lower = query.lower()
    targets = set()
    context_targets = {
        manual
        for manual, hints in CONTEXT_MANUAL_RULES
        if (manual_names is None or manual in manual_names)
        and any(hint.lower() in lower for hint in hints)
    }
    if context_targets:
        targets.update(context_targets)
    for manual, aliases in MANUAL_ALIAS_RULES.items():
        if manual_names is not None and manual not in manual_names:
            continue
        if any(alias.lower() in lower for alias in aliases):
            targets.add(manual)
    if "汇总英文手册" in context_targets:
        # In English bundled manuals, generic words like "pump" and "camera"
        # can be boat or camera parts. Keep contextual matches from being
        # diluted by unrelated Chinese product manuals.
        targets = {"汇总英文手册"}
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
