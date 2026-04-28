from __future__ import annotations

import re
from dataclasses import dataclass

from .products import (
    aliases_for_product,
    candidate_product_keys_for_query,
    english_manual_name,
    product_key_from_manual_name,
)


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
    "VR头显手册": ("vr头显", "vr", "headset", "处理器单元"),
    "人体工学椅手册": ("人体工学椅", "椅子", "扶手", "ergonomic chair", "chair"),
    "健身单车手册": ("健身单车", "exercise bike", "bike"),
    "健身追踪器手册": ("健身追踪器", "fitness tracker", "tracker", "watch"),
    "儿童电动摩托车手册": ("儿童电动摩托车", "kids motorcycle", "electric motorcycle"),
    "冰箱手册": ("冰箱", "refrigerator", "fridge"),
    "功能键盘手册": ("功能键盘", "keyboard"),
    "发电机手册": ("发电机", "generator"),
    "可编程温控器手册": ("温控器", "热泵", "thermostat"),
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
}


INTENT_EXPANSIONS = {
    "充电": ("充电线", "充电接口", "充满电", "电池", "续航"),
    "电量": ("电量低", "电池", "充电", "battery"),
    "电量状态": ("电量 LED 指示灯", "琥珀色", "控制面板", "鼠标", "蓝牙"),
    "表带": ("扣紧表带", "拆卸表带", "更换表带", "佩戴"),
    "扣紧": ("扣紧表带", "表扣", "环扣", "贴合"),
    "拆卸": ("拆卸", "取下", "松开卡扣", "拔下"),
    "卸载": ("卸载", "添加或删除程序", "删除", "重启"),
    "安装": ("安装", "装入", "连接", "固定"),
    "设置": ("设置", "配置", "启用", "关闭"),
    "锁屏": ("设置设备锁", "设备锁", "PIN码", "个人4位PIN码"),
    "设备锁": ("设置设备锁", "设备锁", "PIN码", "个人4位PIN码"),
    "部件": ("概览", "控制面板", "喷淋臂", "碗篮", "过滤器", "机门", "配件详情"),
    "处理器单元": ("处理器单元", "状态指示灯", "AUX端口", "HDMI输出端口", "HDMI电视端口", "HDMI PS4端口", "USB端口", "DC IN 12V接口", "通风口"),
    "松动": ("扶手变松", "常见问题", "重新拧紧螺丝"),
    "清洁": ("清洁", "擦拭", "维护", "clean"),
    "化油器": ("化油器调节", "高速油针", "低速油针", "怠速调节螺钉"),
    "冷机": ("冷机启动", "启动与停机", "阻风门", "泵油"),
    "热机": ("热机启动", "启动与停机", "启动油门", "泵油"),
    "防护装备": ("个人防护装备", "听力防护", "眼部防护", "防滑鞋"),
    "扶手": ("扶手", "安装扶手", "螺丝孔"),
    "触电": ("防止触电", "湿手", "接地", "接地导线"),
    "快速配对": ("Windows 系统快速配对", "预配对模式", "配对按钮", "USB 蓝牙接收器"),
    "油门": ("油门控制", "转向需要油门", "加油门", "松开油门"),
    "中低速": ("油门控制", "中低速转向", "半滑航速度", "转向需要油门"),
    "火花塞": ("火花塞检查", "检查火花塞型号及间隙", "清除积碳", "安装火花塞"),
    "发动机机油": ("发动机机油更换", "机油加注口盖", "放油接头", "添加发动机机油"),
    "标识": ("产品识别码", "识别码记录", "机器标识", "序列号", "记录"),
    "接线": ("连接电线", "热泵系统接线", "备用接线", "端子", "标签不匹配"),
    "保修": ("三年有限保修", "免费服务", "维修", "warranty"),
    "维修": ("维修", "保养", "更换零件", "service"),
    "故障": ("故障排除", "无法启动", "问题", "troubleshooting"),
    "问题": ("故障排除", "其他问题", "无法同步", "重启"),
    "启动": ("启动", "start", "engine"),
    "停机": ("停机", "关闭", "stop"),
    "配对": ("配对", "蓝牙", "pairing"),
    "发票": ("发票", "抬头", "开具"),
    "退款": ("退款", "到账", "原路返回"),
    "退换货": ("退货", "换货", "7天无理由"),
    "charging": ("charge", "charger", "battery", "fully charged"),
    "charge": ("charging", "charger", "battery", "travel case"),
    "quick release": ("quick release button", "steam release valve", "pressure cooking"),
    "first use": ("before first use", "clean", "remove packaging", "water tank"),
    "browser history": ("browser history", "clear browser history"),
    "ebook mode": ("ebook", "eBook Mode", "press M"),
    "pressing": ("press M", "Page Jump", "Save Mark", "Load Mark", "Browser Mode"),
    "video": ("Video", "Video mode", "time play", "full screen"),
    "photo viewer": ("Photo", "photo viewer", "automatic play"),
    "photo": ("Photo", "photo viewer", "automatic play"),
    "troubles": ("Trouble Shooting", "troubleshooting", "reset"),
    "trouble": ("Trouble Shooting", "troubleshooting", "reset"),
    "float valve": ("float valve", "install float valve", "remove float valve"),
    "silicone cap": ("silicone cap", "float valve and silicone cap", "install float valve"),
    "anti-block shield": ("anti-block shield", "install anti-block shield", "remove anti-block shield"),
    "steam release valve": ("steam release valve", "quick release button"),
    "silicone sealing ring": ("silicone sealing ring", "sealing ring", "remove sealing ring"),
    "sealing ring": ("silicone sealing ring", "sealing ring", "remove sealing ring"),
    "natural release": ("natural release", "NR", "NPR"),
    "full bin sensors": ("cleaning the full bin sensors", "wipe the sensors"),
    "extractors": ("cleaning the extractors", "brushes", "rollers"),
    "leak testing": ("leak testing valves hose regulator", "turn all grill control knobs"),
    "indirect cooking": ("indirect cooking", "preheat", "burners"),
    "anchor light": ("anchor light", "install the anchor light", "navigation and anchor lights"),
    "bimini top": ("bimini top", "installing the bimini top", "remove the bimini top"),
    "engine oil level": ("engine oil level", "check the engine oil level", "oil level"),
    "throttle-cable": ("throttle-cable", "throttle cable", "grease the throttle-cable inner wires"),
    "throttle cable": ("throttle cable", "throttle-cable", "adjuster lock nut"),
    "mount the lens": ("mounting a lens", "mounting and detaching a lens", "lens"),
    "lens": ("mounting a lens", "mounting and detaching a lens", "lens"),
    "install the card": ("installing the card", "installing and removing the CF card", "CF card"),
    "cf card": ("installing and removing the CF card", "installing the card", "CF card"),
    "shutter button": ("shutter button", "pressing halfway", "pressing completely"),
    "eyepiece cover": ("using the eyepiece cover", "attaching the eyepiece cover", "eyepiece cover"),
    "single image": ("erasing a single image", "erase", "display the image"),
    "delete a single image": ("erasing a single image", "erase", "display the image"),
    "cp direct": ("CP Direct", "Direct Printing from the Camera", "Start printing"),
    "observing views": ("front view", "navigation button view", "bottom view", "buttons", "interfaces"),
    "front view": ("front view", "home/esc button", "prev/next page"),
    "navigation button view": ("navigation button view", "button operation", "navigation button"),
    "bottom view": ("bottom view", "interfaces"),
    "serial port connector": ("serial port connector", "10-1 pin COM", "COM port"),
    "tpm connector": ("TPM connector", "14-1 pin TPM", "trusted platform module"),
    "pressure cooking lid": ("pressure cooking lid", "removing the lid", "closing the lid"),
    "components": ("what is in the box", "components", "earbuds", "charging case"),
    "what should i have": ("what is in the box", "components", "earbuds", "charging case"),
    "two primary modes": ("dual mode virtual wall", "virtual wall mode", "halo mode", "spot cleaning"),
    "primary modes": ("dual mode virtual wall", "virtual wall mode", "halo mode"),
    "engine switches": ("engine shut-off switch", "engine stop switch", "shut-off cord", "black button", "red button"),
    "engine switch": ("engine shut-off switch", "engine stop switch", "shut-off cord"),
    "cleaning": ("clean", "cleaning procedure", "maintenance"),
    "mounting": ("mount", "install", "steps"),
    "troubleshooting": ("problem", "fault", "steps"),
    "activate": ("activate", "deactivate", "features"),
    "deactivate": ("activate", "deactivate", "features"),
}


ENGLISH_FOCUS_PHRASES = (
    "browser history",
    "main menu",
    "ebook mode",
    "ebook",
    "video",
    "photo viewer",
    "trouble shooting",
    "troubleshooting",
    "silicone cap",
    "float valve",
    "anti-block shield",
    "steam release valve",
    "silicone sealing ring",
    "sealing ring",
    "natural release",
    "full bin sensors",
    "extractors",
    "throttle-cable",
    "throttle cable",
    "leak testing",
    "indirect cooking",
    "led indicator",
    "base station",
    "anchor light",
    "bimini top",
    "engine oil level",
    "mounting a lens",
    "mount the lens",
    "lens",
    "cf card",
    "installing the card",
    "shutter button",
    "eyepiece cover",
    "single image",
    "delete a single image",
    "cp direct",
    "observing views",
    "front view",
    "navigation button view",
    "bottom view",
    "serial port connector",
    "tpm connector",
    "pressure cooking lid",
    "components",
    "what should i have",
)

CHINESE_FOCUS_TERMS = (
    "个人防护装备",
    "防护装备",
    "化油器调节",
    "化油器",
    "冷机启动",
    "热机启动",
    "冷机",
    "热机",
    "启动与停机",
    "扶手",
    "扶手变松",
    "松动",
    "处理器单元",
    "部件",
    "概览",
    "不适合",
    "燃油高度易燃且有毒",
    "防止触电",
    "触电",
    "识别码记录",
    "标识信息",
    "卸载WIDCOMM蓝牙驱动程序",
    "卸载",
    "快速配对",
    "WIDCOMM蓝牙驱动程序配对",
    "电量状态",
    "其他问题",
    "故障排除",
    "设置设备锁",
    "设备锁",
    "锁屏",
)

COMPONENT_ACTION_KEYS = {
    "float valve",
    "anti-block shield",
    "steam release valve",
    "silicone sealing ring",
    "sealing ring",
}


def aliases_for_manual(manual: str) -> tuple[str, ...]:
    product_key = product_key_from_manual_name(manual)
    if product_key:
        return aliases_for_product(product_key)
    return MANUAL_ALIAS_RULES.get(manual, ())


def detect_target_manuals(query: str, manual_names: set[str] | None = None) -> frozenset[str]:
    lower = query.lower()
    targets = set()
    for manual, aliases in MANUAL_ALIAS_RULES.items():
        if manual_names is not None and manual not in manual_names:
            continue
        if any(alias.lower() in lower for alias in aliases):
            targets.add(manual)

    product_keys = candidate_product_keys_for_query(query)
    for product_key in product_keys:
        product_manual = english_manual_name(product_key)
        if manual_names is None:
            targets.add(product_manual)
        elif product_manual in manual_names:
            targets.add(product_manual)
        elif "汇总英文手册" in manual_names:
            targets.add("汇总英文手册")
    return frozenset(targets)


def strip_product_aliases(query: str, target_manuals: set[str] | frozenset[str]) -> str:
    stripped = query
    for manual in target_manuals:
        for alias in aliases_for_manual(manual):
            stripped = re.sub(re.escape(alias), " ", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s+的(?=[\u4e00-\u9fffA-Za-z0-9])", " ", stripped)
    stripped = re.sub(r"(?<=[\u4e00-\u9fffA-Za-z0-9])的\s+", " ", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip(" ，。?？")
    return stripped or query


def is_component_overview_query(query: str) -> bool:
    lower = query.lower()
    if re.search(r"\b(?:set|install|remove|clean|adjust|connect|delete|print|mount)\b", lower):
        return False
    return bool(
        re.search(r"\b(?:what do you know about|know about|overview of|tell me about)\b", lower)
        or re.search(r"是什么|有哪些|哪些|介绍|说明", query)
    )


def expansion_terms(query: str, target_manuals: set[str] | frozenset[str] | None = None) -> list[str]:
    lower = query.lower()
    target_manuals = target_manuals or frozenset()
    overview_query = is_component_overview_query(query)
    terms: list[str] = []
    for key, values in INTENT_EXPANSIONS.items():
        if key.lower() in lower:
            if key == "部件" and "处理器单元" in query:
                continue
            if key == "photo" and "汇总英文手册::ereader" not in target_manuals:
                continue
            if key in COMPONENT_ACTION_KEYS and overview_query:
                terms.append(key)
                continue
            terms.extend(values)
    return terms


def contextual_expansion_terms(query: str, target_manuals: set[str] | frozenset[str]) -> list[str]:
    terms: list[str] = []
    if "发电机手册" in target_manuals:
        if "开关" in query:
            terms.extend(("发动机开关", "经济控制开关", "燃油开关旋钮", "控制面板"))
        if "停机" in query:
            terms.extend(("AE01025 停机", "断开所有电气设备", "发动机开关", "燃油开关旋钮", "油箱盖通气旋钮"))
    return terms


def focused_phrases_from_query(query: str) -> list[str]:
    text = query.strip().strip("\"'“”")
    lower = text.lower()
    phrases: list[str] = []
    for match in re.finditer(r"的([\u4e00-\u9fffA-Za-z0-9（）()\-_/]{2,16})", text):
        phrase = match.group(1).strip(" ？?。；;，,")
        phrase = re.sub(r"(?:该如何|如何|怎么|是什么|有哪些|哪些|吗)$", "", phrase)
        phrase = re.sub(r"(?:类别时|类别|时)$", "", phrase)
        if phrase == "发动机停机":
            phrase = "停机"
        base_phrase = re.sub(r"[（(][^（）()]{1,12}[）)]$", "", phrase)
        if 2 <= len(phrase) <= 16 and phrase not in {
            "正确步骤",
            "详细步骤",
            "注意事项",
            "使用方法",
            "操作方法",
            "开关",
            "稳定性和操控性",
            "建议",
            "发生",
        }:
            phrases.append(phrase)
            if base_phrase != phrase and 2 <= len(base_phrase) <= 16:
                phrases.append(base_phrase)
    if "最高难度" in text and "挑战" in text:
        phrases.append("挑战")
    for phrase in CHINESE_FOCUS_TERMS:
        if phrase in text:
            phrases.append(phrase)
    for phrase in ENGLISH_FOCUS_PHRASES:
        if phrase in lower:
            phrases.append(phrase)
    for quoted in re.findall(r"[“\"']([^“”\"']{2,32})[”\"']", text):
        if quoted.lower() not in {"m", "auto", "manual"}:
            phrases.append(quoted)
    seen = set()
    unique: list[str] = []
    for phrase in phrases:
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            unique.append(phrase)
    return unique[:3]


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
    expansions = [*expansion_terms(normalized, target_manuals), *contextual_expansion_terms(normalized, target_manuals)]
    focus_phrases = focused_phrases_from_query(normalized)

    lower = normalized.lower()
    product_prefixes = []
    for manual in target_manuals:
        for alias in aliases_for_manual(manual):
            if alias.lower() in lower:
                product_prefixes.append(alias)
                break

    variants = [normalized, core]
    variants.extend(focus_phrases)
    for prefix in product_prefixes:
        variants.append(f"{prefix} {core}")
        for phrase in focus_phrases:
            variants.append(f"{prefix} {phrase}")
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
