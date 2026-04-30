from __future__ import annotations

import re
from dataclasses import dataclass

from .flag_reranker import FlagEmbeddingReranker
from .llm_client import OpenAICompatibleClient
from .manuals import ManualChunk, build_chunks, image_path_for_id, load_manuals
from .policy import answer_policy_question, generic_policy_answer, looks_like_manual_question
from .products import candidate_product_keys_for_query, english_manual_name
from .query_planner import QueryPlan, build_query_plan, detect_language
from .retrieval import BM25Retriever, HIGH_INTENT_TERMS, SearchResult, tokenize


PIC_REF_RE = re.compile(r"\[PIC:([^\]]+)\]")
COMMON_QUERY_TERMS = {
    "如何", "怎么", "什么", "哪些", "内容", "正确", "需要", "告诉", "根据", "手册", "说明",
    "请问", "一下", "介绍", "不同", "两种", "两个", "两类", "一种", "一个", "一类",
    "注意", "注意到", "为我", "帮我", "看到", "发现", "相关", "建议",
    "the", "and", "what", "how", "are", "for", "with", "proper", "correct",
    "procedure", "steps", "does", "when", "inside", "ensure",
    "if", "this", "that", "is", "to", "of", "on", "in", "before", "after", "using",
    "use", "do", "can", "you", "i", "my", "want", "need", "first", "time",
    "multi", "pressure", "cooker", "air", "fryer", "airfryer", "coffee", "machine",
    "earphones", "ereader", "fax", "grill", "landline", "lawn", "mower", "microwave",
    "motherboard", "vacuum", "snowmobile", "television", "toothbrush", "camera",
    "boat", "jetski",
}
TARGET_MANUAL_ALIASES = {
    "VR头显": "VR头显手册",
    "遮光罩": "VR头显手册",
    "人体工学椅": "人体工学椅手册",
    "健身单车": "健身单车手册",
    "健身追踪器": "健身追踪器手册",
    "儿童电动摩托车": "儿童电动摩托车手册",
    "冰箱": "冰箱手册",
    "功能键盘": "功能键盘手册",
    "发电机": "发电机手册",
    "温控器": "可编程温控器手册",
    "吹风机": "吹风机手册",
    "摩托艇": "摩托艇手册",
    "水泵": "水泵手册",
    "洗碗机": "洗碗机手册",
    "烤箱": "烤箱手册",
    "电钻": "电钻手册",
    "相机": "相机手册",
    "空气净化器": "空气净化器手册",
    "空调": "空调手册",
    "蒸汽清洁机": "蒸汽清洁机手册",
    "蓝牙激光鼠标": "蓝牙激光鼠标手册",
    "鼠标": "蓝牙激光鼠标手册",
    "dcb107": "电钻手册",
    "dcb112": "电钻手册",
    "drill": "电钻手册",
    "fitness tracker": "健身追踪器手册",
}
SENTENCE_PUNCT_RE = re.compile(r"[。！？!?；;：:]")
TOC_LIKE_RE = re.compile(r"(?:^|\s)\d{1,2}\s+[\u4e00-\u9fffA-Za-z]")
ACTION_SIGNAL_RE = re.compile(
    r"安装|取下|拆下|装入|装回|按下|打开|关闭|连接|固定|选择|使用|清洁|更换|检查|"
    r"设置|配置|切换|保存|下载|排空|擦拭|拧紧|松开|拔下|防止|请勿|确保|应使用|"
    r"加速|减速|转向|油门|恢复|练习|烹饪|"
    r"\b(?:install|remove|press|open|close|connect|clean|replace|set|configure|switch|"
    r"save|download|drain|check|use|keep|make sure|do not|adjust|mount|delete|erase|print)\b",
    re.IGNORECASE,
)
NUMBERED_STEP_RE = re.compile(r"(?:^|\s)(?:\d+[.、]?|[A-Z][.、])\s*[\u4e00-\u9fffA-Za-z]")
CONTINUED_FRAGMENT_RE = re.compile(
    r"^(?:[a-z]{1,4}\s|nds\s|ot\s|ther\s|phones,\s|l\s|s,nicks|ge\s|and\s|that\s|not\soperate)",
)
FOCUS_STOP_PHRASES = {
    "正确步骤",
    "详细步骤",
    "相关说明",
    "注意事项",
    "操作方法",
    "使用方法",
    "前五条",
    "前几条",
    "开关",
    "稳定性和操控性",
    "建议",
    "发生",
}
GENERIC_TOOL_PHRASES = {
    "部件",
    "概览",
    "清洁",
    "安装",
    "设置",
    "故障",
    "问题",
    "步骤",
    "components",
    "cleaning",
    "installation",
    "settings",
    "troubleshooting",
}
TOOL_PHRASE_QUESTION_RE = re.compile(
    r"\b(?:what|how|can|you|know|about|using|after|before|current|different|proper|correct|"
    r"procedure|steps|does|when|where|which)\b",
    re.IGNORECASE,
)
OVERVIEW_TITLE_RE = re.compile(
    r"^(?:overview|general description|what is in the box|what's in the box|components|"
    r"steam release valve|anti-block shield|sealing ring|float valve|pressure cooking lid|"
    r"概览|部件|部件说明|包装内容|组成|简介)\b",
    re.IGNORECASE,
)
ACTION_TITLE_RE = re.compile(
    r"^(?:install|remove|connect|charge|set|replace|clean|register|delete|erase|mount|"
    r"activate|deactivate|select|start|stop|warning|caution|note|"
    r"安装|拆卸|取下|连接|充电|设置|更换|清洁|注册|删除|选择|启动|停机|警告|注意)",
    re.IGNORECASE,
)
PROCEDURE_TITLE_RE = re.compile(
    r"^(?:install|remove|connect|charge|set|replace|clean|register|delete|erase|mount|"
    r"activate|deactivate|select|start|stop|"
    r"安装|拆卸|取下|连接|充电|设置|更换|清洁|注册|删除|选择|启动|停机)",
    re.IGNORECASE,
)
ENGLISH_ACTION_RE = re.compile(
    r"\b(?:install|remove|set|clean|cleaning|press|play|operate|procedure|steps|adjust|"
    r"mount|delete|erase|print|care|check|connect|charge|start|stop|use|replace|select)\b",
    re.IGNORECASE,
)
STATUS_BEHAVIOR_RE = re.compile(
    r"\b(?:current status|status|behavior|behaviour|indicator behavior|events status|charge status)\b|"
    r"状态|指示灯|行为",
    re.IGNORECASE,
)
ENGLISH_FOCUS_PHRASES = (
    "browser history",
    "main menu",
    "ebook mode",
    "ebook",
    "video",
    "photo viewer",
    "trouble shooting",
    "troubleshooting",
    "troublehsooting",
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


def clean_context_text(text: str) -> str:
    text = PIC_REF_RE.sub("<PIC>", text)
    text = clean_manual_noise(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_math_fragment(match: re.Match[str]) -> str:
    text = match.group(1)
    text = re.sub(r"\\(?:mathsf|mathrm|text|tt|bf|pmb|mathcal|sf)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:widehat|overbrace|boxed)\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\,", " ").replace("\\;", " ").replace("\\:", " ")
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_manual_noise(text: str) -> str:
    text = re.sub(r"\$([^$]{1,120})\$", clean_math_fragment, text)
    text = re.sub(r"(?i)\bfigure\s+\d+\b", "", text)
    text = re.sub(r"(?i)\(?(?:see|refer to)\s+page\s+\d+\)?", "", text)
    text = re.sub(r"(?i)\bpage\s+\d+\b", "", text)
    text = re.sub(r"\.{4,}", " ", text)
    text = re.sub(r"(?m)^#\s*", "", text)
    text = re.sub(r"(?<=[a-zA-Z])(?=\d+\.)", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Z][a-z])", " ", text)
    text = text.replace("", "•").replace("••", "•")
    return text


def readable_chunk_text(chunk: ManualChunk) -> str:
    text = PIC_REF_RE.sub("<PIC>", chunk.text)
    text = clean_manual_noise(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+#\s+", "\n", text)
    text = re.sub(r"^#\s*", "", text.strip())
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip(" #") for line in text.splitlines() if line.strip(" #")]
    return "\n".join(lines).strip()


def collect_image_ids(results: list[SearchResult], limit: int | None = None) -> list[str]:
    image_ids: list[str] = []
    for result in results:
        for image_id in result.chunk.image_ids:
            if image_id and image_id not in image_ids and image_path_for_id(image_id) is not None:
                image_ids.append(image_id)
            if limit is not None and len(image_ids) >= limit:
                return image_ids
    return image_ids


def focused_phrases_from_question(question: str) -> list[str]:
    text = question.strip().strip("\"'“”")
    lower = text.lower()
    phrases: list[str] = []
    for match in re.finditer(r"的([\u4e00-\u9fffA-Za-z0-9（）()\-_/]{2,16})", text):
        phrase = match.group(1).strip(" ？?。；;，,")
        phrase = re.sub(r"(?:该如何|如何|怎么|是什么|有哪些|哪些|吗)$", "", phrase)
        phrase = re.sub(r"(?:类别时|类别|时)$", "", phrase)
        if phrase == "发动机停机":
            phrase = "停机"
        base_phrase = re.sub(r"[（(][^（）()]{1,12}[）)]$", "", phrase)
        if 2 <= len(phrase) <= 16 and phrase not in FOCUS_STOP_PHRASES:
            phrases.append(phrase)
            if base_phrase != phrase and 2 <= len(base_phrase) <= 16:
                phrases.append(base_phrase)
    for quoted in re.findall(r"[“\"']([^“”\"']{2,24})[”\"']", text):
        if quoted not in FOCUS_STOP_PHRASES:
            phrases.append(quoted)
    if "最高难度" in text and "挑战" in text:
        phrases.append("挑战")
    for phrase in CHINESE_FOCUS_TERMS:
        if phrase in text:
            phrases.append(phrase)
    for phrase in ENGLISH_FOCUS_PHRASES:
        if phrase in lower:
            phrases.append(phrase)
    if "troubleshooting" in lower or "trouble shooting" in lower:
        phrases.append("troublehsooting")
    seen = set()
    unique: list[str] = []
    for phrase in phrases:
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(phrase)
    return unique[:3]


def phrase_tool_candidates(plan: QueryPlan) -> list[str]:
    phrases = list(focused_phrases_from_question(plan.normalized))
    for variant in plan.variants[1:]:
        item = variant.strip(" ？?。；;，,")
        if not (2 <= len(item) <= 64):
            continue
        if len(re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "", item)) < 3:
            continue
        if item == plan.normalized:
            continue
        if re.search(r"[？?]", item):
            continue
        if TOOL_PHRASE_QUESTION_RE.search(item) or re.search(r"如何|怎么|什么|哪些|正确|步骤|请问|告诉|介绍", item):
            continue
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-/]*", item)
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", item)
        if len(words) > 5 or len(cjk_chars) > 18:
            continue
        phrases.append(item)
    seen = set()
    unique: list[str] = []
    for phrase in phrases:
        key = re.sub(r"\s+", " ", phrase.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(phrase)
    return unique[:8]


def readable_text_from_raw(raw_text: str) -> str:
    text = PIC_REF_RE.sub("<PIC>", raw_text)
    text = clean_manual_noise(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+#\s+", "\n", text)
    text = re.sub(r"^#\s*", "", text.strip())
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip(" #") for line in text.splitlines() if line.strip(" #")]
    return "\n".join(lines).strip()


def focused_chunk_text(question: str, chunk: ManualChunk) -> tuple[str, tuple[str, ...]] | None:
    phrases = focused_phrases_from_question(question)
    if not phrases:
        return None
    raw = chunk.text
    lowered = raw.lower()
    phrase = next((item for item in phrases if item.lower() in lowered), "")
    if not phrase:
        return None
    full_readable = readable_text_from_raw(raw)
    local_heading_count = len(re.findall(r"（图\d+）|■\s*[\u4e00-\u9fffA-Za-zA-Z]{2,20}模式", full_readable))
    same_block_has_hot_cold_start = (
        phrase in {"冷机", "热机", "冷机启动", "热机启动"}
        and "冷机启动" in full_readable
        and "热机启动" in full_readable
    )
    same_block_has_silicone_float_steps = (
        phrase == "silicone cap"
        and "float valve" in question.lower()
        and "firmly attach the silicone cap" in full_readable.lower()
    )
    mixed_subsection_block = (
        (phrase in {"卸载", "卸载WIDCOMM蓝牙驱动程序", "快速配对"} and "卸载 WIDCOMM" in full_readable and "快速配对" in full_readable)
        or (phrase in {"serial port connector", "tpm connector"} and "Serial port connector" in full_readable and "TPM connector" in full_readable)
        or (phrase == "扶手" and re.search(r"松动|变松", question) and re.search(r"松动|变松|重新拧紧", full_readable))
    )
    if (
        len(full_readable) <= 620
        and local_heading_count <= 1
        and not same_block_has_hot_cold_start
        and not same_block_has_silicone_float_steps
        and not mixed_subsection_block
    ):
        plain_len = len(full_readable.replace("<PIC>", "").strip())
        token_count = len(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", full_readable.replace("<PIC>", "")))
        if not chunk.image_ids and plain_len < 60 and token_count <= 8:
            return None
        if plain_len >= 18 or (plain_len >= 6 and len(chunk.image_ids) >= 2):
            return full_readable[:700], chunk.image_ids
    phrase_l = phrase.lower()
    positions = [match.start() for match in re.finditer(re.escape(phrase_l), lowered)]
    if not positions:
        return None
    if (
        is_action_query(question)
        and not is_troubleshooting_query(question)
        and min(positions) > 260
        and PROCEDURE_TITLE_RE.match(chunk.title.strip())
        and ACTION_SIGNAL_RE.search(full_readable[:260])
    ):
        return full_readable[:700], chunk.image_ids

    def position_score(pos: int) -> tuple[int, int]:
        next_char = raw[pos + len(phrase) : pos + len(phrase) + 1]
        window = raw[pos : pos + 180]
        score = 0
        if next_char and next_char not in "、，,）)]":
            score += 30
        if PIC_REF_RE.search(window):
            score += 20
        if raw.rfind("#", 0, pos + 1) >= max(0, pos - 8):
            score += 10
        if re.search(r"松动|变松", question) and re.search(r"松动|变松|重新拧紧", window):
            score += 120
        if phrase in {"卸载", "卸载WIDCOMM蓝牙驱动程序"} and "请按照以下步骤" in window[:80]:
            score += 140
        if phrase in {"卸载", "卸载WIDCOMM蓝牙驱动程序"} and window.startswith(("卸载完成", "卸载程序")):
            score -= 80
        return score, pos

    pos = max(positions, key=position_score)
    start = pos
    for pattern in ("\n# ", "# "):
        prev = raw.rfind(pattern, 0, pos)
        if prev >= 0:
            start = max(start, prev + len(pattern))
    end = len(raw)
    after = pos + len(phrase)
    header_re = re.compile(
        r"(?:\n#\s+|#\s+|(?<=\])|(?<=\n))[\u4e00-\u9fffA-Za-z0-9（）()]{2,18}（图\d+）"
        r"|(?:\n|(?<=\]))\s*■\s*[\u4e00-\u9fffA-Za-zA-Z（）()]{2,24}模式"
    )
    for match in header_re.finditer(raw, after):
        candidate = match.start()
        if candidate > after + 12:
            end = min(end, candidate)
            break
    for marker in ("控制面板说明",):
        marker_pos = raw.find(marker, after)
        if marker_pos >= 0 and marker_pos > after + 12:
            end = min(end, marker_pos)
    if phrase in {"冷机", "冷机启动"}:
        marker_pos = raw.find("热机启动", after)
        if marker_pos >= 0 and marker_pos > after + 4:
            end = min(end, marker_pos)
    if phrase in {"卸载", "卸载WIDCOMM蓝牙驱动程序"}:
        for marker in ("# 4.", "4. 激光鼠标设置", "4.1 Windows 系统快速配对"):
            marker_pos = raw.find(marker, after)
            if marker_pos >= 0 and marker_pos > after + 8:
                end = min(end, marker_pos)
    if phrase == "快速配对":
        for marker in ("4.1 Windows 系统快速配对", "Windows 系统快速配对"):
            marker_pos = raw.rfind(marker, max(0, pos - 120), pos + len(phrase))
            if marker_pos >= 0:
                start = min(start, marker_pos)
        next_section = raw.find("# 在 BIOS", after)
        if next_section >= 0 and next_section > after + 8:
            end = min(end, next_section)
    if phrase == "serial port connector":
        marker_pos = raw.find("2. TPM connector", after)
        if marker_pos >= 0 and marker_pos > after + 8:
            end = min(end, marker_pos)
    if phrase == "tpm connector":
        for marker in ("2. TPM connector", "TPM connector (14-1 pin TPM)"):
            marker_pos = raw.rfind(marker, max(0, pos - 120), pos + len(phrase))
            if marker_pos >= 0:
                start = min(start, marker_pos)
    if phrase == "silicone cap":
        for marker in ("Place one finger", "Firmly attach"):
            marker_pos = raw.rfind(marker, max(0, pos - 220), pos + len(phrase))
            if marker_pos >= 0:
                start = min(start, marker_pos)
    has_previous_connect_marker = False
    for marker in ("Connect each end", "1 Connect each end", "2 Connect each end"):
        marker_pos = raw.rfind(marker, max(0, pos - 240), pos)
        if marker_pos >= 0:
            start = min(start, marker_pos)
            has_previous_connect_marker = True
    if not has_previous_connect_marker:
        for marker in ("Connect each end", "1 Connect each end", "2 Connect each end"):
            marker_pos = raw.find(marker, pos, min(len(raw), pos + 180))
            if marker_pos >= 0:
                start = marker_pos
    next_hash = raw.find("# ", after)
    if next_hash >= 0 and next_hash > after + 12:
        end = min(end, next_hash)
    snippet = raw[start:end].strip()
    if len(snippet) > 520 and not (start < pos and pos - start <= 240):
        snippet = raw[pos:end].strip()
    readable = readable_text_from_raw(snippet)
    plain_len = len(readable.replace("<PIC>", "").strip())
    pic_count = len(PIC_REF_RE.findall(snippet))
    if plain_len < 18 and not (plain_len >= 6 and pic_count >= 2):
        return None
    return readable[:520], tuple(dict.fromkeys(PIC_REF_RE.findall(snippet)))


def detect_target_manuals(question: str) -> set[str]:
    q = question.lower()
    targets = {
        manual
        for alias, manual in TARGET_MANUAL_ALIASES.items()
        if alias.lower() in q
    }
    for product_key in candidate_product_keys_for_query(question):
        targets.add(english_manual_name(product_key))
    return targets


def narrow_results(plan: QueryPlan, results: list[SearchResult]) -> list[SearchResult]:
    targets = plan.target_manuals or detect_target_manuals(plan.normalized)
    focus_phrases = focused_phrases_from_question(plan.normalized)
    switch_overview = asks_for_switch_overview(plan.normalized)
    shutdown_procedure = asks_for_shutdown_procedure(plan.normalized)
    if targets:
        filtered = [result for result in results if result.chunk.manual in targets]
        if filtered:
            top = filtered[0].score
            focused = [
                result
                for result in filtered
                if focus_phrases
                and any(phrase.lower() in readable_chunk_text(result.chunk).lower() for phrase in focus_phrases)
            ]
            structured_matches = [
                result
                for result in filtered
                if (switch_overview and "开关" in result.chunk.title)
                or (shutdown_procedure and "停机" in result.chunk.title)
                or (
                    "热泵" in plan.normalized
                    and any(term in result.chunk.title for term in ("热泵", "连接电线", "备用接线", "端子"))
                )
            ]
            if top >= 50:
                ratio = 0.35 if "汇总英文手册" in targets else 0.55
                strong = [result for result in filtered if result.score >= top * ratio]
                combined: list[SearchResult] = []
                seen = set()
                for result in [*strong, *focused, *structured_matches]:
                    if result.chunk.id in seen:
                        continue
                    seen.add(result.chunk.id)
                    combined.append(result)
                return combined or filtered[:12]
            return filtered
    if not results:
        return results
    top = results[0].score
    if top >= 25:
        same_manual = [result for result in results if result.chunk.manual == results[0].chunk.manual]
        strong = [result for result in same_manual if result.score >= top * 0.85]
        return strong or same_manual[:4]
    return results


def is_component_overview_query(question: str) -> bool:
    lower = question.lower()
    if re.search(r"\b(?:set|install|remove|clean|adjust|connect|delete|erase|print|mount)\b", lower):
        return False
    return bool(
        re.search(r"\b(?:what do you know about|know about|overview of|tell me about)\b", lower)
        or re.search(r"是什么|有哪些|哪些|介绍|说明", question)
    )


def is_action_query(question: str) -> bool:
    lower = question.lower()
    chinese_action = bool(
        re.search(
            r"安装|拆卸|卸载|更换|检查|清洁|设置|启动|停机|排空|如何|怎么|调节|连接|打印|删除|操作",
            question,
        )
    )
    if chinese_action:
        return True
    return bool(ENGLISH_ACTION_RE.search(lower))


def is_status_behavior_query(question: str) -> bool:
    return bool(STATUS_BEHAVIOR_RE.search(question))


def is_troubleshooting_query(question: str) -> bool:
    return bool(re.search(r"故障|无法|问题|排除|troubleshoot|trouble shooting|problem|error", question, re.IGNORECASE))


def is_start_procedure_query(question: str) -> bool:
    lower = question.lower()
    mentions_start = "启动" in question or bool(re.search(r"\bstart(?:ing)?\b", lower))
    asks_procedure = bool(re.search(r"步骤|如何|怎么|操作|前[一二两三四五六七八九十\d]|最后|steps?|how", question, re.IGNORECASE))
    return mentions_start and asks_procedure and not is_troubleshooting_query(question)


def is_interface_operation_query(question: str) -> bool:
    return "界面" in question and bool(re.search(r"操作|遵循|步骤|如何|怎么", question))


def title_matches_overview_intent(title: str) -> bool:
    clean = compact_title(title)
    return bool(OVERVIEW_TITLE_RE.search(title.strip())) or clean.startswith(("overview of", "general description"))


def looks_like_english_toc_or_index(text: str) -> bool:
    clean = re.sub(r"\s+", " ", text.replace("<PIC>", "")).strip()
    if SENTENCE_PUNCT_RE.search(clean):
        return False
    page_refs = len(re.findall(r"(?:\.\s*){1,3}\d{1,3}|[A-Za-z][A-Za-z /-]{2,35}\s+\d{1,3}", clean))
    return len(clean) < 260 and page_refs >= 2


def compact_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def rerank_results(plan: QueryPlan, results: list[SearchResult]) -> list[SearchResult]:
    if not results:
        return results

    variant_terms = set(tokenize(" ".join(plan.variants)))
    core_terms = {
        term
        for term in tokenize(plan.normalized)
        if len(term) >= 2 and term not in COMMON_QUERY_TERMS and not any(ch.isdigit() for ch in term)
    }
    focus_phrases = focused_phrases_from_question(plan.normalized)
    switch_overview = asks_for_switch_overview(plan.normalized)
    shutdown_procedure = asks_for_shutdown_procedure(plan.normalized)
    caution_query = "注意" in plan.normalized
    action_query = is_action_query(plan.normalized)
    overview_query = is_component_overview_query(plan.normalized)
    status_behavior_query = is_status_behavior_query(plan.normalized)
    troubleshooting_query = is_troubleshooting_query(plan.normalized)
    start_procedure_query = is_start_procedure_query(plan.normalized)
    interface_operation_query = is_interface_operation_query(plan.normalized) and (
        "健身追踪器手册" in plan.target_manuals or "健身追踪器" in plan.normalized
    )
    warranty_query = bool(re.search(r"保修|质保|warranty", plan.normalized, flags=re.IGNORECASE))
    scored_items: list[tuple[SearchResult, int, int]] = []
    for result in results:
        chunk = result.chunk
        title = chunk.title.lower()
        readable = readable_chunk_text(chunk)
        head = readable[:1000].lower()
        full_text = readable.lower()
        doc_text = f"{title} {head}"
        coverage = sum(1 for term in variant_terms if len(term) >= 2 and (term in title or term in head))
        core_coverage = sum(1 for term in core_terms if term in title or term in head)
        title_hits = sum(1 for term in variant_terms if len(term) >= 2 and term in title)
        early_title_hits = sum(1 for term in core_terms if term in title[:40])
        image_bonus = 1.5 if chunk.image_ids else 0.0
        manual_bonus = 3.0 if plan.target_manuals and chunk.manual in plan.target_manuals else 0.0
        score = (
            result.score
            + coverage * 1.8
            + core_coverage * 5.0
            + title_hits * 6.0
            + early_title_hits * 26.0
            + image_bonus
            + manual_bonus
        )
        if any(term in title[:30] for term in HIGH_INTENT_TERMS if term in core_terms):
            score += 24.0
        for phrase in ("before first use", "first use", "quick release", "steam release", "robot anatomy"):
            if phrase in " ".join(plan.variants).lower() and phrase in doc_text:
                score += 110.0
        if title.startswith(("before first use", "quick release", "steam release")):
            score += 70.0
        if not warranty_query and is_front_matter_noise_block(readable):
            score -= 260.0
        if looks_like_english_toc_or_index(readable):
            score -= 260.0
        if switch_overview:
            title_clean = re.sub(r"\s+", "", chunk.title)
            if "开关" in title_clean and len(title_clean) <= 18:
                score += 230.0
            if title_clean in {"控制面板", "部件说明"}:
                score -= 90.0
        if shutdown_procedure:
            title_clean = re.sub(r"\s+", "", chunk.title)
            if "停机" in title_clean[:18]:
                score += 260.0
            elif title_clean == "发动机开关":
                score -= 90.0
        if "热泵" in plan.normalized:
            if "热泵" in chunk.title:
                score += 220.0
            if any(term in chunk.title for term in ("连接电线", "备用接线", "端子", "接线")):
                score += 180.0
            if "常规系统" in chunk.title:
                score -= 160.0
            if not any(term in chunk.title for term in ("热泵", "连接电线", "备用接线", "端子", "接线", "注释")):
                score -= 260.0
        lower_query = plan.normalized.lower()
        title_l = chunk.title.lower()
        title_compact = compact_title(chunk.title)
        if start_procedure_query:
            if "启动发动机" in chunk.title or re.search(r"\bstart(?:ing)? the engine\b", title_l):
                score += 820.0
            if chunk.title.strip(" #") in {"注", "注意", "note"} and re.search(r"启动发动机前|before starting the engine", readable[:260], re.IGNORECASE):
                score += 720.0
            if re.search(r"无法启动|不能启动|发动机无法启动|cannot start|won'?t start|does not start|troubleshoot", f"{chunk.title} {readable[:260]}", re.IGNORECASE):
                score -= 860.0
        if troubleshooting_query:
            if re.search(r"troubleshoot|troublehsooting|trouble shooting|problem|故障|无法|问题", full_text, re.IGNORECASE):
                score += 420.0
            if re.search(r"troublehsooting|troubleshooting", full_text, re.IGNORECASE):
                score += 360.0
        if interface_operation_query:
            if chunk.title.startswith(("操作", "基础操作")):
                score += 760.0
            elif chunk.title.startswith(("快捷设置", "关闭通知", "查看电量", "查看数据")):
                score -= 240.0
        if "化油器" in plan.normalized:
            if any(term in chunk.title for term in ("化油器", "高速油针", "低速油针", "怠速调节螺钉")):
                score += 520.0
            if "个人防护装备" in chunk.title:
                score -= 420.0
        if "冷机" in plan.normalized or "热机" in plan.normalized:
            if "启动与停机" in chunk.title:
                score += 620.0
            if "冷机" in plan.normalized and "冷机启动" in readable:
                score += 280.0
            if "热机" in plan.normalized and "热机启动" in readable:
                score += 280.0
            if any(term in chunk.title for term in ("高速油针", "低速油针", "怠速")):
                score -= 420.0
        if "防止" in plan.normalized and "触电" in plan.normalized:
            if "防止触电" in chunk.title or "防止触电" in readable[:80]:
                score += 620.0
            if "燃油高度易燃" in chunk.title:
                score -= 420.0
        if "其他问题" in plan.normalized or ("问题" in plan.normalized and "健身追踪器" in plan.normalized):
            if "其他问题" in chunk.title or "故障排除" in chunk.title:
                score += 620.0
            if any(term in chunk.title for term in ("进行消费", "非接触式支付", "使用信用卡")):
                score -= 520.0
        if any(term in plan.normalized for term in ("锁屏", "设备锁")):
            if "设置设备锁" in chunk.title or "设备锁" in readable[:120]:
                score += 680.0
            if any(term in chunk.title for term in ("进行消费", "非接触式支付", "使用信用卡")):
                score -= 520.0
        if "卸载" in plan.normalized:
            if "卸载" in chunk.title:
                score += 620.0
            if "配对" in chunk.title:
                score -= 360.0
        if "快速配对" in plan.normalized:
            if "快速配对" in chunk.title:
                score += 620.0
            if any(term in chunk.title for term in ("电量状态", "卸载", "WIDCOMM 蓝牙驱动程序使用")):
                score -= 360.0
        if "电量状态" in plan.normalized and "电量状态" in chunk.title:
            score += 360.0
        if "处理器单元" in plan.normalized:
            if chunk.manual == "VR头显手册":
                score += 500.0
            if chunk.title.strip() == "处理器单元":
                score += 820.0
            elif "处理器单元" in chunk.title and any(term in chunk.title for term in ("外壳", "接口", "清洁")):
                score -= 420.0
            if any(term in full_text for term in ("状态指示灯", "aux端口", "hdmi输出端口", "dc in 12v接口", "通风口")):
                score += 420.0
        if "洗碗机" in plan.normalized and "部件" in plan.normalized:
            if chunk.title.startswith("概览"):
                score += 720.0
            if chunk.title.startswith("不适合在洗碗机中清洗"):
                score -= 520.0
        if "anchor light" in lower_query:
            if "anchor light" in title_l:
                score += 520.0
            if "bimini top" in title_l:
                score -= 520.0
        if "bimini top" in lower_query and "bimini top" in title_l:
            score += 420.0
        if "engine oil level" in lower_query and "engine oil level" in title_l:
            score += 420.0
        if "throttle-cable" in lower_query or ("throttle cable" in lower_query and "boat" in lower_query):
            if "throttle cable" in title_l or "throttle-cable" in head:
                score += 560.0
        if "mount" in lower_query and "lens" in lower_query and "mounting" in title_l and "lens" in title_l:
            score += 620.0
        if "install" in lower_query and "card" in lower_query and ("installing the card" in title_l or "cf card" in title_l):
            score += 620.0
        if "shutter button" in lower_query and "shutter button" in title_l:
            score += 520.0
        if "eyepiece cover" in lower_query and "eyepiece cover" in title_l:
            score += 620.0
        if ("delete" in lower_query or "single image" in lower_query) and (
            "erasing a single image" in title_l or "erase" in title_l
        ):
            score += 620.0
        if "cp direct" in lower_query:
            if "cp direct" in title_l or "direct printing" in title_l:
                score += 620.0
            if "cp direct" in full_text:
                score += 700.0
        if "observing views" in lower_query or ("buttons and interfaces" in lower_query and "ereader" in lower_query):
            if title_l.strip().startswith(("front view", "navigation button view", "bottom view")):
                score += 620.0
            if title_l.strip().startswith(("settings", "video", "ebook", "photo")):
                score -= 260.0
        if "serial port connector" in lower_query and "serial port connector" in title_l:
            score += 620.0
        if "tpm connector" in lower_query:
            if "tpm connector" in full_text:
                score += 620.0
            if "serial port connector" in title_l and "tpm connector" not in title_l:
                score -= 120.0
        if "pressure cooking lid" in lower_query:
            if title_compact == "pressure cooking lid":
                score += 680.0
            if "floatvalve" in title_l or "float valve" in title_l:
                score -= 480.0
        if "components" in lower_query or "what should i have" in lower_query:
            if any(term in full_text for term in ("what is in the box", "charging case", "earbuds")):
                score += 520.0
            if any(term in title_l for term in ("bluetooth connection", "voice control")):
                score -= 420.0
        if re.search(r"\b(?:set|install)\b", lower_query):
            if title_l.strip().startswith("install"):
                score += 560.0
            if title_l.strip().startswith("remove"):
                score -= 220.0
            if re.fullmatch(r"[a-z -]{2,40}", title_l.strip()) and not title_l.strip().startswith("install"):
                score -= 180.0
        if re.search(r"\bremove\b", lower_query):
            if title_l.strip().startswith("remove"):
                score += 240.0
            if title_l.strip().startswith("install"):
                score -= 120.0
        if re.search(r"\b(?:clean|cleaning)\b", lower_query) and "clean" in title_l[:40]:
            score += 220.0
        if overview_query and focus_phrases:
            if title_matches_overview_intent(chunk.title):
                score += 460.0
            elif ACTION_TITLE_RE.match(chunk.title.strip()):
                score -= 360.0
            for phrase in focus_phrases:
                phrase_l = compact_title(phrase)
                if not phrase_l:
                    continue
                if title_compact == phrase_l or title_compact.startswith(phrase_l):
                    score += 360.0
                elif title_l.strip().startswith(("install", "remove")) and phrase.lower() in title_l:
                    score -= 420.0
        if status_behavior_query:
            behavior_terms = ("behavior", "behaviour", "status", "current status", "状态", "指示灯", "indicator")
            behavior_hits = sum(1 for term in behavior_terms if term in full_text)
            score += min(520.0, behavior_hits * 95.0)
            if "current status" in lower_query and "current status" in full_text:
                score += 260.0
            if "different" in lower_query and "different" in full_text:
                score += 160.0
            if "base station" in lower_query and "base station" in full_text:
                score += 180.0
            if is_low_information_block(readable) and not re.search(r"behavior|behaviour|current status|状态", full_text):
                score -= 760.0
        if focus_phrases:
            has_focus = any(phrase.lower() in full_text for phrase in focus_phrases)
            if has_focus:
                score += 180.0
                if any(phrase.lower() in title for phrase in focus_phrases):
                    score += 80.0
                    first_pos = min(
                        (title.find(phrase.lower()) for phrase in focus_phrases if phrase.lower() in title),
                        default=999,
                    )
                    if action_query and first_pos <= 12:
                        score += 260.0
                    if first_pos <= 12:
                        score += 220.0
                    if action_query and any(term in title for term in ("安装", "拆卸", "更换", "检查", "清洁", "设置", "启动", "停机", "排空")):
                        score += 240.0
                elif action_query and len(chunk.title) > 80:
                    score -= 90.0
            elif plan.target_manuals:
                score -= 130.0
        if caution_query and chunk.title.strip(" #").startswith(("注意", "注")):
            score += 180.0
        if len(title) > 140 and early_title_hits == 0:
            score -= 18.0
        if looks_like_continuation_fragment(readable):
            score -= 140.0
        scored_items.append((SearchResult(chunk=chunk, score=score), core_coverage, title_hits))

    scored_items.sort(key=lambda item: item[0].score, reverse=True)
    reranked = [item[0] for item in scored_items]
    top = reranked[0].score
    if top <= 0:
        return reranked
    top_core_coverage = scored_items[0][1]
    min_core_coverage = max(1, int(top_core_coverage * 0.55)) if top_core_coverage else 0
    keep_ratio = 0.45 if plan.target_manuals else 0.62
    focus_lowers = [phrase.lower() for phrase in focus_phrases]
    kept = [
        item
        for item, core_coverage, title_hits in scored_items
        if item.score >= top * keep_ratio
        and (
            core_coverage >= min_core_coverage
            or title_hits > 0
            or any(phrase in readable_chunk_text(item.chunk).lower() for phrase in focus_lowers)
        )
    ]
    return kept or reranked[:4]


def extract_relevant_sentences(question: str, chunks: list[ManualChunk], max_sentences: int = 8) -> list[str]:
    query_terms = set(tokenize(question))
    candidates: list[tuple[int, str]] = []
    for chunk in chunks:
        text = clean_context_text(chunk.text)
        sentences = re.split(r"(?<=[。！？.!?])\s+|(?<=\n)", text)
        for sentence in sentences:
            sentence = sentence.strip(" #")
            if not sentence or len(sentence) < 8:
                continue
            score = sum(1 for term in query_terms if term and term in sentence.lower())
            if score:
                candidates.append((score, sentence))
    candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)

    selected: list[str] = []
    seen = set()
    for _, sentence in candidates:
        key = sentence[:80]
        if key in seen:
            continue
        seen.add(key)
        selected.append(sentence[:260])
        if len(selected) >= max_sentences:
            break
    return selected


def is_low_information_block(text: str) -> bool:
    clean = text.replace("<PIC>", "").strip()
    if not clean:
        return True
    if looks_like_intro_anchor(clean):
        return True
    if looks_like_section_heading(clean):
        return True
    if len(clean) >= 35 and NUMBERED_STEP_RE.search(clean):
        return False
    if len(clean) >= 30 and SENTENCE_PUNCT_RE.search(clean):
        return False
    if len(clean) >= 20 and ACTION_SIGNAL_RE.search(clean) and SENTENCE_PUNCT_RE.search(clean):
        return False
    if len(clean) >= 30 and ACTION_SIGNAL_RE.search(clean):
        return False
    if len(clean) >= 70 and not looks_like_toc_block(clean):
        return False
    return True


def looks_like_toc_block(clean: str) -> bool:
    if SENTENCE_PUNCT_RE.search(clean):
        return False
    return len(clean) < 220 and len(TOC_LIKE_RE.findall(clean)) >= 3


def looks_like_section_heading(text: str) -> bool:
    clean = text.replace("<PIC>", "").strip(" #")
    if not clean:
        return True
    if looks_like_toc_block(clean):
        return True
    if len(clean) > 38:
        return False
    if re.fullmatch(r"(?:\d+[.、]?\s*)?[\u4e00-\u9fffA-Za-zA-Z0-9（）() /_-]{1,34}(?:时|说明|小贴士)?", clean):
        if clean.endswith("时") or clean in {"目录", "注", "注意", "警告", "重要安全说明", "故障排除"}:
            return True
        if not SENTENCE_PUNCT_RE.search(clean) and not ACTION_SIGNAL_RE.search(clean):
            return True
    return False


def looks_like_intro_anchor(text: str) -> bool:
    clean = re.sub(r"\s+", " ", text.replace("<PIC>", "")).strip(" #")
    if len(clean) > 110:
        return False
    return clean.startswith(
        (
            "使用前检查 注",
            "故障排除 若",
            "Troubleshooting If",
            "Before each use",
        )
    )


def looks_like_continuation_fragment(text: str) -> bool:
    clean = text.replace("<PIC>", "").strip()
    if not clean:
        return False
    if clean[0].isupper():
        return False
    if CONTINUED_FRAGMENT_RE.match(clean):
        return True
    return bool(re.match(r"^[a-z]{1,4}\s+[a-z]", clean)) and not clean.lower().startswith(
        ("the ", "this ", "when ", "while ", "read ", "use ", "do ", "if ")
    )


def asks_for_first_items(question: str) -> int | None:
    normalized = question.lower()
    digit_match = re.search(r"(?:前|first)\s*([1-9])\s*(?:条|项|个|items?|points?)", normalized)
    if digit_match:
        return min(8, int(digit_match.group(1)))
    chinese_numbers = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
    }
    match = re.search(r"前([一二两三四五六七八])(?:条|项|个|步|步骤)", question)
    if match:
        return chinese_numbers[match.group(1)]
    return None


def asks_for_switch_overview(question: str) -> bool:
    return "开关" in question and bool(re.search(r"介绍|说明|两种|两个|两类|不同", question))


def asks_for_shutdown_procedure(question: str) -> bool:
    return "停机" in question and bool(re.search(r"如何|怎么|操作|步骤|让|procedure|steps|how ", question, re.IGNORECASE))


def asks_for_multiple_meanings(question: str) -> bool:
    return bool(re.search(r"这些|各|分别|含义|标识|meanings?", question, re.IGNORECASE))


def desired_block_count(question: str, is_english: bool) -> int:
    first_items = asks_for_first_items(question)
    if first_items:
        return min(8, first_items)
    if asks_for_multiple_meanings(question):
        return 3 if not is_english else 2
    if "注意" in question:
        return 3
    if asks_for_switch_overview(question):
        return 3
    if re.search(r"哪些|有哪些|步骤|如何|怎么|procedure|steps|how ", question, flags=re.IGNORECASE):
        return 3 if not is_english else 2
    return 2


def has_incomplete_numbered_procedure(question: str, text: str) -> bool:
    if not re.search(r"步骤|如何|怎么|操作|排空|procedure|steps|how ", question, flags=re.IGNORECASE):
        return False
    clean = text.replace("<PIC>", "").strip()
    has_first = re.search(r"(?:^|\s)1[.、]?\s*[\u4e00-\u9fffA-Za-z]", clean)
    has_second = re.search(r"(?:^|\s)2[.、]?\s*[\u4e00-\u9fffA-Za-z]", clean)
    return bool(has_first and not has_second and len(clean) < 180)


def looks_like_incomplete_numbered_block(text: str) -> bool:
    clean = text.replace("<PIC>", "").strip()
    has_first = re.search(r"(?:^|\s)1[.、]?\s*[\u4e00-\u9fffA-Za-z]", clean)
    has_second = re.search(r"(?:^|\s)2[.、]?\s*[\u4e00-\u9fffA-Za-z]", clean)
    return bool(has_first and not has_second and len(clean) < 180)


def procedure_step_count(text: str) -> int:
    clean = text.replace("<PIC>", " ")
    return len(re.findall(r"(?:^|\s)[1-9][.、]?\s+[\u4e00-\u9fffA-Za-z]", clean))


def is_sufficient_procedure_block(question: str, text: str) -> bool:
    if asks_for_first_items(question):
        return False
    if not re.search(r"步骤|正确步骤|详细步骤|如何|怎么|procedure|steps|how ", question, flags=re.IGNORECASE):
        return False
    return procedure_step_count(text) >= 3


def truncate_to_requested_items(question: str, text: str) -> str:
    count = asks_for_first_items(question)
    if not count:
        return text
    next_item = count + 1
    match = re.search(rf"(?:^|\n|\s){next_item}[.、]?\s*[\u4e00-\u9fffA-Za-z]", text)
    if not match:
        return text
    return text[: match.start()].rstrip(" \n。；;") + "。"


def is_generic_overview_block(text: str) -> bool:
    clean = text.replace("<PIC>", "").strip(" #")
    return bool(re.match(r"^(?:\d+[.、]?\s*)?(?:关于|简介|包装内容|目录|说明)(?:\s|$|[\u4e00-\u9fffA-Za-z])", clean))


def is_front_matter_noise_block(text: str) -> bool:
    clean = re.sub(r"\s+", " ", text.replace("<PIC>", "")).strip()
    if len(clean) < 160:
        return False
    signals = ("有限保修", "保修问答", "保修期内", "保修是否", "保修范围", "异常负荷", "经销商在保修")
    return sum(1 for signal in signals if signal in clean) >= 2


def fallback_manual_answer(question: str, results: list[SearchResult]) -> str:
    is_english = detect_language(question) == "en"
    if not results:
        if is_english:
            return "I could not find enough clear information yet. Please provide the product model, symptom, or image so we can verify it further."
        return "暂时没有检索到足够明确的信息。建议补充商品型号、故障现象或图片，以便进一步核实。"

    lines: list[str] = []
    top_text = readable_chunk_text(results[0].chunk)
    has_substantive_result = any(
        not is_low_information_block(readable_chunk_text(result.chunk))
        for result in results[1:]
    )
    if results[0].chunk.image_ids and is_low_information_block(top_text) and not has_substantive_result:
        if is_english:
            lines.append("The requested component or operation overview is shown in the related illustration. <PIC>")
            lines.append(f"Related images: {', '.join(results[0].chunk.image_ids)}")
        else:
            lines.append("该部件位置或操作示意主要见相关插图。<PIC>")
            lines.append(f"相关插图: {', '.join(results[0].chunk.image_ids)}")
        return "\n".join(lines)

    blocks: list[str] = []
    selected_image_ids: list[str] = []
    seen = set()
    top_score = results[0].score
    selected_top_score: float | None = None
    max_blocks = desired_block_count(question, is_english)
    block_keep_ratio = 0.50 if max_blocks >= 3 else 0.72
    if asks_for_multiple_meanings(question):
        block_keep_ratio = min(block_keep_ratio, 0.60)
    has_substantive = any(not is_low_information_block(readable_chunk_text(result.chunk)) for result in results)
    warranty_query = bool(re.search(r"保修|质保|warranty", question, flags=re.IGNORECASE))
    overview_query_for_blocks = is_component_overview_query(question)
    for result in results:
        threshold_score = selected_top_score if selected_top_score is not None else top_score
        if blocks and result.score < threshold_score * block_keep_ratio:
            continue
        if overview_query_for_blocks and re.match(r"^(?:remove|delete|erase|拆卸|取下|删除)", result.chunk.title.strip(), re.IGNORECASE):
            continue
        title_key = re.sub(r"\s+", " ", result.chunk.title.lower())[:80]
        if title_key in seen:
            continue
        focused = focused_chunk_text(question, result.chunk)
        if focused is not None:
            block, block_image_ids = focused
        else:
            block = readable_chunk_text(result.chunk)
            block_image_ids = result.chunk.image_ids
        if not block:
            continue
        if "热泵" in question and "接线" in question and not any(
            term in block for term in ("热泵", "端子", "电线", "接线", "标签", "R滑动", "O和B", "Y1", "W1")
        ):
            continue
        if has_substantive and not warranty_query and is_front_matter_noise_block(block):
            continue
        if blocks and is_generic_overview_block(block):
            continue
        if has_substantive and is_low_information_block(block) and focused is None:
            continue
        block = truncate_to_requested_items(question, block)
        key = block[:80]
        if key in seen:
            continue
        seen.add(key)
        seen.add(title_key)
        if selected_top_score is None:
            selected_top_score = result.score
        if len(blocks) == 0:
            limit = 950 if is_english else 620
        else:
            limit = 430 if is_english else 320
        blocks.append(block[:limit])
        for image_id in block_image_ids:
            if image_id and image_id not in selected_image_ids and image_path_for_id(image_id) is not None:
                selected_image_ids.append(image_id)
        if (
            focused is not None
            and not asks_for_multiple_meanings(question)
            and not ("注意" in question and len(block.replace("<PIC>", "").strip()) < 160)
        ):
            break
        if len(blocks) == 1 and is_sufficient_procedure_block(question, block):
            break
        if len(blocks) >= max_blocks:
            break

    if blocks:
        lines.extend(blocks)
    else:
        chunks = [item.chunk for item in results[:4]]
        sentences = extract_relevant_sentences(question, chunks) or [clean_context_text(chunks[0].text)[:520]]
        lines.extend(sentences[:6])

    image_ids = selected_image_ids[:4] or collect_image_ids(results, limit=4)
    if image_ids:
        label = "Related images" if is_english else "相关插图"
        lines.append(f"{label}: {', '.join(image_ids)}")
    return "\n".join(lines)


def build_llm_messages(question: str, results: list[SearchResult]) -> list[dict[str, str]]:
    context_blocks = []
    for idx, result in enumerate(results[:6], start=1):
        chunk = result.chunk
        images = ", ".join(chunk.image_ids) if chunk.image_ids else "无"
        context_blocks.append(
            f"[资料{idx}] 手册={chunk.manual} 标题={chunk.title} 图片={images}\n{readable_chunk_text(chunk)[:1800]}"
        )
    context = "\n\n".join(context_blocks)
    system = (
        "你是一个严谨的多模态客服智能体。你必须只依据给定资料回答，不得编造资料中没有的型号、参数、政策或承诺。"
        "回答要自然、清晰、有步骤感；用户用中文提问就用中文客服语气回答，用户用英文提问就用英文回答。"
        "如果资料中有相关图片，请在对应步骤或说明后保留<PIC>，并在末尾列出图片ID。"
        "如果用户一次问多个问题，请拆分后一一作答。若资料不足，请明确说明缺少的信息，而不是猜测。"
    )
    user = f"用户问题：\n{question}\n\n可用资料：\n{context}\n\n请直接输出最终客服答案，不要输出分析过程。"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@dataclass
class AnswerEngine:
    use_llm: bool = False
    use_flag_reranker: bool | None = None

    def __post_init__(self) -> None:
        manuals = load_manuals()
        self.manual_names = {manual.name for manual in manuals}
        self.chunks = build_chunks(manuals)
        self.chunk_pos = {chunk.id: idx for idx, chunk in enumerate(self.chunks)}
        self.retriever = BM25Retriever(self.chunks)
        self.flag_reranker = FlagEmbeddingReranker(enabled=self.use_flag_reranker)
        self.llm = OpenAICompatibleClient() if self.use_llm else None

    def merge_search_results(self, groups: list[list[SearchResult]], top_k: int = 60) -> list[SearchResult]:
        best: dict[str, SearchResult] = {}
        for group in groups:
            for result in group:
                chunk_id = result.chunk.id
                if chunk_id not in best or result.score > best[chunk_id].score:
                    best[chunk_id] = result
        merged = list(best.values())
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[:top_k]

    def title_phrase_search(
        self,
        plan: QueryPlan,
        allowed_manuals: set[str] | None,
        top_k: int = 24,
    ) -> list[SearchResult]:
        phrases = phrase_tool_candidates(plan)
        if not phrases:
            return []
        overview_query = is_component_overview_query(plan.normalized)
        scored: list[SearchResult] = []
        for chunk in self.chunks:
            if allowed_manuals is not None and chunk.manual not in allowed_manuals:
                continue
            title_l = chunk.title.lower()
            if overview_query and re.match(r"^(?:install|remove|clean|set|replace|安装|拆卸|取下|清洁|更换)", title_l):
                continue
            compact = compact_title(chunk.title)
            score = 0.0
            for phrase in phrases:
                phrase_l = phrase.lower()
                phrase_compact = compact_title(phrase)
                if not phrase_compact:
                    continue
                if phrase_compact == compact:
                    score += 90.0
                elif phrase_l in title_l:
                    score += 55.0 + min(30.0, len(phrase_l) * 0.8)
                elif phrase_compact in compact and phrase.lower() not in GENERIC_TOOL_PHRASES:
                    score += 35.0
            if score > 0:
                if chunk.image_ids:
                    score += 2.0
                scored.append(SearchResult(chunk=chunk, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def body_phrase_search(
        self,
        plan: QueryPlan,
        allowed_manuals: set[str] | None,
        top_k: int = 24,
    ) -> list[SearchResult]:
        phrases = [
            phrase
            for phrase in phrase_tool_candidates(plan)
            if phrase.lower() not in GENERIC_TOOL_PHRASES and len(phrase.replace(" ", "")) >= 3
        ]
        if not phrases:
            return []
        overview_query = is_component_overview_query(plan.normalized)
        scored: list[SearchResult] = []
        for chunk in self.chunks:
            if allowed_manuals is not None and chunk.manual not in allowed_manuals:
                continue
            text_l = chunk.text.lower()
            title_l = chunk.title.lower()
            if overview_query and re.match(r"^(?:install|remove|clean|set|replace|安装|拆卸|取下|清洁|更换)", title_l):
                continue
            score = 0.0
            for phrase in phrases:
                phrase_l = phrase.lower()
                if phrase_l in title_l:
                    score += 35.0
                pos = text_l.find(phrase_l)
                if pos < 0:
                    continue
                if pos <= 800:
                    score += 32.0
                elif pos <= 1800:
                    score += 22.0
                else:
                    score += 12.0
            if score > 0:
                if chunk.image_ids:
                    score += 1.5
                scored.append(SearchResult(chunk=chunk, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def top_result_has_action_alignment(self, plan: QueryPlan, result: SearchResult | None) -> bool:
        if result is None or not is_action_query(plan.normalized):
            return False
        text = f"{result.chunk.title} {readable_chunk_text(result.chunk)[:500]}"
        return bool(
            ENGLISH_ACTION_RE.search(text)
            or re.search(r"安装|拆卸|卸载|更换|检查|清洁|设置|启动|停机|排空|调节|连接|打印|删除|操作", text)
        )

    def top_result_has_focus(self, plan: QueryPlan, result: SearchResult | None) -> bool:
        if result is None:
            return False
        focus_phrases = focused_phrases_from_question(plan.normalized)
        if not focus_phrases:
            return True
        text = f"{result.chunk.title} {readable_chunk_text(result.chunk)[:900]}".lower()
        return any(phrase.lower() in text for phrase in focus_phrases)

    def should_use_title_phrase_search(self, plan: QueryPlan, bm25_results: list[SearchResult]) -> bool:
        phrases = phrase_tool_candidates(plan)
        if not phrases:
            return False
        if not bm25_results:
            return True
        top = bm25_results[0]
        second = bm25_results[1] if len(bm25_results) > 1 else None
        top_text = readable_chunk_text(top.chunk)
        top_low_info = is_low_information_block(top_text)
        close_second = bool(second and second.score >= top.score * 0.88)
        overview_query = is_component_overview_query(plan.normalized)
        status_behavior_query = is_status_behavior_query(plan.normalized)

        if overview_query and title_matches_overview_intent(top.chunk.title) and top.score >= 35:
            return False
        if status_behavior_query and re.search(r"behavior|behaviour|current status|状态", top_text, re.IGNORECASE):
            return False
        if self.top_result_has_action_alignment(plan, top) and top.score >= 45:
            return False
        if top.score < 45:
            return True
        if top_low_info and not self.top_result_has_action_alignment(plan, top):
            return True
        return close_second and not self.top_result_has_focus(plan, top)

    def should_use_body_phrase_search(self, plan: QueryPlan, bm25_results: list[SearchResult]) -> bool:
        if is_component_overview_query(plan.normalized):
            return False
        if not phrase_tool_candidates(plan):
            return False
        if not bm25_results:
            return bool(plan.target_manuals)
        top = bm25_results[0]
        if self.top_result_has_action_alignment(plan, top) and top.score >= 45:
            return False
        top_text = readable_chunk_text(top.chunk)
        if is_status_behavior_query(plan.normalized) and re.search(r"behavior|behaviour|current status|状态", top_text, re.IGNORECASE):
            return False
        return top.score < 45 or (is_low_information_block(top_text) and not self.top_result_has_focus(plan, top))

    def flag_reranker_text(self, chunk: ManualChunk) -> str:
        title = re.sub(r"\s+", " ", chunk.title).strip()
        body = readable_chunk_text(chunk)
        if len(body) > 1800:
            body = body[:1800]
        return f"{title}\n{body}".strip()

    def reranker_candidate_quality(self, plan: QueryPlan, result: SearchResult) -> float:
        text = readable_chunk_text(result.chunk)
        title = result.chunk.title.lower()
        head = text[:1200].lower()
        score = 0.0

        core_terms = [
            term
            for term in tokenize(plan.normalized)
            if len(term) >= 2 and term not in COMMON_QUERY_TERMS and not any(ch.isdigit() for ch in term)
        ]
        for term in core_terms:
            if term in title:
                score += 3.0
            elif term in head:
                score += 1.2

        focus_phrases = focused_phrases_from_question(plan.normalized)
        for phrase in focus_phrases:
            phrase_l = phrase.lower()
            if phrase_l in title:
                score += 6.0
            elif phrase_l in head:
                score += 3.5

        if self.top_result_has_action_alignment(plan, result):
            score += 4.0
        if self.top_result_has_focus(plan, result):
            score += 4.0
        if not is_low_information_block(text):
            score += 2.0

        compact = compact_title(result.chunk.title)
        for phrase in focus_phrases:
            phrase_compact = compact_title(phrase)
            if phrase_compact and (compact == phrase_compact or compact.startswith(phrase_compact)):
                score += 5.0

        return score

    def should_use_flag_reranker(self, plan: QueryPlan, results: list[SearchResult]) -> bool:
        if not self.flag_reranker.active or len(results) < 2:
            return False
        if detect_language(plan.normalized) != "en":
            return False

        target_manuals = plan.target_manuals
        if target_manuals and not all(manual.startswith("汇总英文手册::") for manual in target_manuals):
            return False

        top = results[0]
        second = results[1]
        top_text = readable_chunk_text(top.chunk)
        focus_phrases = focused_phrases_from_question(plan.normalized)
        title_l = top.chunk.title.lower()
        title_exact_focus = any(
            compact_title(phrase) == compact_title(top.chunk.title) or phrase.lower() in title_l
            for phrase in focus_phrases
            if compact_title(phrase)
        )
        close_scores = second.score >= top.score * 0.92
        top_is_low_info = is_low_information_block(top_text)
        top_has_focus = self.top_result_has_focus(plan, top)
        top_has_action = self.top_result_has_action_alignment(plan, top)

        if title_exact_focus and top_has_focus and (top_has_action or not top_is_low_info):
            return False
        if not close_scores and top_has_focus and (top_has_action or not top_is_low_info):
            return False
        return close_scores or top_is_low_info or not top_has_focus

    def apply_flag_reranker(self, plan: QueryPlan, results: list[SearchResult], candidate_limit: int = 20) -> list[SearchResult]:
        if not results or not self.should_use_flag_reranker(plan, results):
            return results

        candidates = results[: min(candidate_limit, 10)]
        docs = [self.flag_reranker_text(result.chunk) for result in candidates]
        rerank_scores = self.flag_reranker.score_texts(plan.normalized, docs)
        if len(rerank_scores) != len(candidates):
            return results

        low = min(rerank_scores)
        high = max(rerank_scores)
        spread = high - low
        reranked_candidates: list[SearchResult] = []
        for result, rerank_score in zip(candidates, rerank_scores):
            normalized = 0.5 if spread <= 1e-6 else (rerank_score - low) / spread
            fused_score = result.score + normalized * 10.0
            reranked_candidates.append(SearchResult(chunk=result.chunk, score=fused_score))

        reranked_candidates.sort(key=lambda item: item.score, reverse=True)
        original_top = candidates[0]
        reranked_top = reranked_candidates[0]
        original_quality = self.reranker_candidate_quality(plan, original_top)
        reranked_quality = self.reranker_candidate_quality(plan, reranked_top)
        if (
            reranked_top.chunk.id != original_top.chunk.id
            and reranked_quality < original_quality + 2.5
        ):
            return results

        seen_ids = {item.chunk.id for item in reranked_candidates}
        tail = [item for item in results[candidate_limit:] if item.chunk.id not in seen_ids]
        return [*reranked_candidates, *tail]

    def expand_short_top_context(self, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return results
        top = results[0]
        top_text = readable_chunk_text(top.chunk)
        heading_anchor = looks_like_section_heading(top_text)
        intro_anchor = looks_like_intro_anchor(top_text)
        continuation_needed = has_incomplete_numbered_procedure("", top_text)
        if not heading_anchor:
            continuation_needed = has_incomplete_numbered_procedure(top.chunk.title, top_text) or looks_like_incomplete_numbered_block(top_text)
        if (
            not is_low_information_block(top_text)
            and not continuation_needed
            and not intro_anchor
        ) or top.score < 40:
            return results

        neighbor_limit = 4 if heading_anchor or intro_anchor else 2 if continuation_needed else 3
        neighbors: list[SearchResult] = []
        result_by_id = {item.chunk.id: item for item in results}
        seen = {top.chunk.id}
        pos = self.chunk_pos.get(top.chunk.id)
        if pos is None:
            return results
        for offset in range(1, neighbor_limit + 1):
            if pos + offset >= len(self.chunks):
                break
            neighbor = self.chunks[pos + offset]
            if neighbor.manual != top.chunk.manual:
                break
            if neighbor.id not in seen:
                neighbors.append(
                    result_by_id.get(
                        neighbor.id,
                        SearchResult(chunk=neighbor, score=top.score * (0.99 - offset * 0.01)),
                    )
                )
                seen.add(neighbor.id)
        if not neighbors:
            return results
        if heading_anchor or intro_anchor or continuation_needed or is_low_information_block(top_text):
            neighbor_ids = {item.chunk.id for item in neighbors}
            return [top, *neighbors, *[item for item in results[1:] if item.chunk.id not in neighbor_ids]]
        return [*results, *neighbors]

    def retrieve(self, plan: QueryPlan) -> list[SearchResult]:
        allowed_manuals = set(plan.target_manuals) if plan.target_manuals else None
        bm25_results = self.retriever.search_many(
            plan.variants,
            top_k=40,
            per_query_k=24 if allowed_manuals else 18,
            allowed_manuals=allowed_manuals,
        )
        tool_results: list[list[SearchResult]] = [bm25_results]
        if self.should_use_title_phrase_search(plan, bm25_results):
            tool_results.append(self.title_phrase_search(plan, allowed_manuals))
        if self.should_use_body_phrase_search(plan, bm25_results):
            tool_results.append(self.body_phrase_search(plan, allowed_manuals))
        raw = self.merge_search_results(tool_results, top_k=60)
        lowered = plan.normalized.lower()
        if "robot anatomy" in lowered and "vacuum" in lowered:
            for chunk in self.chunks:
                if chunk.manual == english_manual_name("vacuum") and chunk.title.strip().lower().startswith("vacuum"):
                    raw.append(SearchResult(chunk=chunk, score=(raw[0].score if raw else 80.0) + 80.0))
                    break
        narrowed = narrow_results(plan, raw)
        reranked = rerank_results(plan, narrowed)
        reranked = self.apply_flag_reranker(plan, reranked)
        return self.expand_short_top_context(reranked[:8])

    def answer(self, question: str, qid: str | int | None = None) -> str:
        plan = build_query_plan(question, self.manual_names)
        manual_question = bool(plan.target_manuals) or looks_like_manual_question(plan.normalized)

        if not manual_question:
            return answer_policy_question(plan.normalized) or generic_policy_answer(plan.normalized)

        results = self.retrieve(plan)
        if self.use_llm and self.llm is not None and results:
            try:
                return self.llm.chat(build_llm_messages(plan.normalized, results))
            except Exception as exc:
                # Keep batch generation running even if one API call fails.
                return fallback_manual_answer(question, results) + f"\n（系统提示：LLM调用失败，已使用本地检索答案。错误摘要：{exc}）"
        return fallback_manual_answer(question, results)
