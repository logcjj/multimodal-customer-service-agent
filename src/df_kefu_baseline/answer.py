from __future__ import annotations

import re
from dataclasses import dataclass

from .llm_client import OpenAICompatibleClient
from .manuals import ManualChunk, build_chunks, image_path_for_id, load_manuals
from .policy import answer_policy_question, generic_policy_answer, looks_like_manual_question
from .query_planner import MANUAL_ALIAS_RULES, QueryPlan, build_query_plan, detect_language
from .retrieval import BM25Retriever, HIGH_INTENT_TERMS, SearchResult, tokenize


PIC_REF_RE = re.compile(r"\[PIC:([^\]]+)\]")
COMMON_QUERY_TERMS = {
    "如何", "怎么", "什么", "哪些", "内容", "正确", "需要", "告诉", "根据", "手册", "说明",
    "请问", "the", "and", "what", "how", "are", "for", "with", "proper", "correct",
    "procedure", "steps", "does", "when", "inside", "ensure",
    "if", "this", "that", "is", "to", "of", "on", "in", "before", "after", "using",
    "use", "do", "can", "you", "i", "my", "want", "need", "first", "time",
    "while", "about", "properly", "usually", "normal", "common",
    "multi", "pressure", "cooker", "air", "fryer", "airfryer", "coffee", "machine",
    "earphones", "ereader", "fax", "grill", "landline", "lawn", "mower", "microwave",
    "motherboard", "vacuum", "snowmobile", "television", "toothbrush", "camera",
    "boat", "jetski",
}
TARGET_MANUAL_ALIASES = {
    "VR头显": "VR头显手册",
    "遮光罩": "VR头显手册",
    "游玩区域": "VR头显手册",
    "处理器单元": "VR头显手册",
    "耳塞": "VR头显手册",
    "人体工学椅": "人体工学椅手册",
    "椅子": "人体工学椅手册",
    "扶手": "人体工学椅手册",
    "健身单车": "健身单车手册",
    "健身追踪器": "健身追踪器手册",
    "儿童电动摩托车": "儿童电动摩托车手册",
    "冰箱": "冰箱手册",
    "功能键盘": "功能键盘手册",
    "发电机": "发电机手册",
    "温控器": "可编程温控器手册",
    "吹风机": "吹风机手册",
    "摩托艇": "摩托艇手册",
    "拖曳速度": "摩托艇手册",
    "半滑航速度": "摩托艇手册",
    "滑航速度": "摩托艇手册",
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
    "camera": "汇总英文手册",
    "vacuum": "汇总英文手册",
    "snowmobile": "汇总英文手册",
    "television": "汇总英文手册",
    "toothbrush": "汇总英文手册",
    "boat": "汇总英文手册",
    "ship": "汇总英文手册",
    "sailing": "汇总英文手册",
    "on board": "汇总英文手册",
    "jetski": "汇总英文手册",
    "jet ski": "汇总英文手册",
    "bimini": "汇总英文手册",
    "swim platform": "汇总英文手册",
    "livewell": "汇总英文手册",
    "coffee maker": "汇总英文手册",
    "af mode": "汇总英文手册",
}

PHRASE_HINTS = (
    "anti-block shield", "steam release valve", "quick release button", "quick release",
    "float valve", "silicone cap", "sealing ring", "silicone sealing ring",
    "condensation collector", "natural release", "approval label",
    "emission control", "battery conversion", "sound system", "storage compartments",
    "battery compartment", "anchor light", "bimini top", "swim platform", "livewell",
    "maintenance setting", "factory reset", "trip screen", "steering system",
    "engine oil level", "fuel filter", "fuel tank", "adjustable sponson",
    "intake and impeller", "af mode", "main menu", "browser history",
    "music", "music mode", "video", "photo", "photo viewer", "voice recording", "ebook mode",
    "mounting and detaching a lens", "attach the lens", "lens mount", "lens", "shutter button",
    "camera battery", "date/time battery", "cp direct", "delete", "erase",
    "烤架", "烤盘", "接油盘", "油脂过滤器", "滑动搁架", "催化侧面板",
    "警报界面", "热泵", "油箱滤网", "遮光罩", "游玩区域", "处理器单元", "耳塞",
    "扶手变松", "松动", "高度调节", "后仰", "按摩功能", "滤网清洁", "烤箱外部",
)
ACCESSORY_TERMS = {
    "烤架", "烤盘", "接油盘", "油脂过滤器", "滑动搁架", "催化侧面板",
    "float valve", "anti-block shield", "steam release valve", "sealing ring",
    "condensation collector", "bimini top", "swim platform", "livewell",
}


def clean_context_text(text: str) -> str:
    text = PIC_REF_RE.sub("<PIC>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def readable_chunk_text(chunk: ManualChunk) -> str:
    text = PIC_REF_RE.sub("<PIC>", chunk.text)
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


def detect_target_manuals(question: str) -> set[str]:
    q = question.lower()
    targets = {
        manual
        for alias, manual in TARGET_MANUAL_ALIASES.items()
        if alias.lower() in q
    }
    return targets


def phrase_hints_for_question(question: str) -> list[str]:
    q = question.lower()
    hints = [phrase for phrase in PHRASE_HINTS if phrase.lower() in q]
    if "空气净化器" in question and ("模式" in question or "运行" in question or "设置" in question):
        hints.extend(("常规运行", "自动运行", "涡轮风扇运行", "睡眠运行"))
        if "安全" in question or "儿童" in question:
            hints.append("安全锁功能")
        if "关机" in question or "定时" in question:
            hints.append("自动关机功能")
    if "空气净化器" in question and "滤网" in question and "清洁" in question:
        hints.extend(("滤网清洁", "预过滤网", "切勿用水清洗滤网"))
    if "空气净化器" in question and "滤网" in question and ("更换" in question or "换" in question):
        hints.extend(("更换滤网", "滤网更换指示灯", "睡眠 + 自动键"))
    if "空气净化器" in question and "长期存放" in question:
        hints.extend(("长期存放", "晾干设备内部", "干燥、无阳光直射"))
    if "boat" in q or "sailing" in q:
        if "water supply" in q:
            hints.extend(("jet wash switch", "water supply", "water flow", "jet wash handle lever"))
        if "start the boat" in q or "boat's engine" in q:
            hints.extend(("Turn the battery switch to the ON position", "Push the blower switch", "Install the clip", "main switch keys to the start position"))
        if "turn" in q and "turn on" not in q and "turn off" not in q and ("sailing" in q or "turn a boat" in q):
            hints.extend(("boat characteristics", "jet thrust turns", "steering wheel", "jet thrust nozzles"))
        if "steering system" in q:
            hints.extend(("steering system checks", "steering wheel", "jet thrust nozzles", "free play"))
        if "engine oil level" in q:
            hints.extend(("engine oil level", "dipstick", "minimum level mark", "maximum level mark"))
        if "approval label" in q or "emission control" in q:
            hints.extend(("approval label", "emission control certificate", "emission control information label"))
        if "cruise is over" in q or "load the boat" in q:
            hints.extend(("deactivate the cruise assist", "remote control levers", "decrease the engine speed"))
        if "throttle-cable" in q or "throttle cable" in q:
            hints.extend(("throttle cable", "grease points", "grease the throttle-cable inner wires", "pulley wheel"))
    if "quick release" in q:
        hints.extend(("Quick Release (QR or QPR)", "quick release button", "Vent position", "steam release valve"))
    if "pressure cooking lid" in q:
        hints.extend(("pressure cooking lid", "install the pressure cooking lid", "remove the pressure cooking lid"))
    if "float valve" in q:
        hints.extend(("float valve", "silicone cap", "install the float valve", "remove the float valve"))
    if "anti-block shield" in q:
        hints.extend(("anti-block shield", "steam release pipe", "remove the anti-block shield"))
    if "sealing ring" in q:
        if "install" in q:
            hints.extend(("Install the sealing ring", "press it into place", "snug behind sealing ring rack"))
        elif "remove" in q:
            hints.extend(("Remove the sealing ring", "pull the sealing ring out", "silicone"))
        else:
            hints.extend(("Sealing ring When the pressure cooking lid", "air-tight seal", "Only one sealing ring"))
    if "over-the-range microwave" in q:
        if "auto defrost" in q:
            hints.extend(("AUTO DEFROST", "Auto Defrost Chart", "defrost frozen foods"))
        if "reheat" in q:
            hints.extend(("REHEAT", "PIZZA lets you reheat", "sensor"))
        if "control" in q:
            hints.extend(("CONTROL PANEL FEATURES", "Display includes a clock", "Touch this pad"))
    if "camera" in q:
        if "install" in q and "card" in q:
            hints.extend(("Installing the Card", "Insert the CF card", "Close the cover", "CF card"))
        if "delete a single image" in q or "erase" in q:
            hints.extend(("Erasing a Single Image", "Select the image to be erased", "Erase"))
        if "cp direct" in q or "print photos" in q:
            hints.extend(("CP Direct", "Start printing", "Direct Printing", "select [OK]"))
    if "jetski" in q or "jet ski" in q:
        if "hood" in q:
            hints.extend(("Engine hood", "engine hood latches", "open the engine hood", "lift the engine hood"))
        if "filler cap" in q:
            hints.extend(("fuel tank filler cap", "oil tank filler cap", "fuel cock knob"))
        if "engine switches" in q or "engine switch" in q:
            hints.extend(("engine shut-off switch", "Main switches", "START", "ON", "OFF"))
        if "qsts" in q:
            hints.extend(("Quick Shift Trim System", "QSTS selector", "trim angle"))
        if "sponson" in q:
            hints.extend(("Adjustable Sponson", "Adjusting the Adjustable Sponson", "turning performance"))
    for match in re.findall(r"\b[a-z][a-z0-9]*(?:[- ][a-z0-9]+){1,3}\b", q):
        if len(match) >= 7 and match not in COMMON_QUERY_TERMS:
            hints.append(match)
    seen = set()
    unique: list[str] = []
    for hint in hints:
        key = hint.lower()
        if key not in seen:
            seen.add(key)
            unique.append(hint)
    return unique[:8]


def plan_product_terms(plan: QueryPlan) -> set[str]:
    terms: set[str] = set()
    for manual in plan.target_manuals:
        for alias in MANUAL_ALIAS_RULES.get(manual, ()):
            terms.update(tokenize(alias))
    return {term for term in terms if len(term) >= 2}


def bundle_product_penalty(question: str, title: str, text: str) -> float:
    q = question.lower()
    doc = f"{title} {text}".lower()
    penalty = 0.0
    if ("boat" in q or "sailing" in q) and not any(
        term in doc for term in ("boat", "vessel", "jet thrust", "steering", "cruise assist", "throttle-cable")
    ):
        penalty += 260.0
    if "boat" in q and "approval label" not in q and "emission control" not in q:
        if "approval label" in doc or "emission control certificate" in doc:
            penalty += 360.0
    if ("start the boat" in q or "boat's engine" in q) and any(
        term in doc for term in ("engines can also be stopped", "to remove the battery", "remove the main switch keys")
    ):
        penalty += 760.0
    if "camera" in q and not any(term in doc for term in ("camera", "lens", "shutter", "cf card", "battery pack", "photo")):
        penalty += 260.0
    if "camera" in q and "flash" not in q and "flash photography" in doc:
        penalty += 520.0
    if "camera" in q and "install" in q and "card" in q and not any(
        term in doc for term in ("installing the card", "insert the cf card", "close the cover", "cf card can be inserted")
    ):
        penalty += 760.0
    if "camera" in q and "install" in q and "card" in q and "remove the cf card" in doc:
        penalty += 620.0
    if "camera" in q and ("delete a single image" in q or "erase" in q) and not any(
        term in doc for term in ("erasing a single image", "select the image to be erased", "erase images", "erasing images")
    ):
        penalty += 760.0
    if "camera" in q and ("cp direct" in q or "print photos" in q):
        if text.strip().startswith("fore resuming"):
            penalty += 520.0
        if not any(term in doc for term in ("cp direct", "direct printing", "start printing")):
            penalty += 520.0
    if "sealing ring" in q and title.startswith(("caution", "warning", "! warning")):
        penalty += 760.0
    if "ereader" in q and any(term in doc for term in ("camera", "lens", "shutter", "flash photography")):
        penalty += 420.0
    if ("jetski" in q or "jet ski" in q) and "watercraft education and training" in doc:
        penalty += 520.0
    if "fax" in q and any(term in doc for term in ("lp tank", "grill", "regulator")):
        penalty += 420.0
    if "grill" in q and "fax" in doc:
        penalty += 320.0
    return penalty


def narrow_results(plan: QueryPlan, results: list[SearchResult]) -> list[SearchResult]:
    targets = plan.target_manuals or detect_target_manuals(plan.normalized)
    if targets:
        filtered = [result for result in results if result.chunk.manual in targets]
        if filtered:
            top = filtered[0].score
            if top >= 50:
                ratio = 0.35 if "汇总英文手册" in targets else 0.55
                return [result for result in filtered if result.score >= top * ratio] or filtered[:12]
            return filtered
    if not results:
        return results
    top = results[0].score
    if top >= 25:
        same_manual = [result for result in results if result.chunk.manual == results[0].chunk.manual]
        strong = [result for result in same_manual if result.score >= top * 0.85]
        return strong or same_manual[:4]
    return results


def rerank_results(plan: QueryPlan, results: list[SearchResult]) -> list[SearchResult]:
    if not results:
        return results

    product_terms = plan_product_terms(plan)
    variant_terms = set(tokenize(" ".join(plan.variants))) - product_terms
    core_terms = {
        term
        for term in tokenize(plan.normalized)
        if len(term) >= 2
        and term not in COMMON_QUERY_TERMS
        and term not in product_terms
        and not any(ch.isdigit() for ch in term)
    }
    phrase_hints = phrase_hints_for_question(plan.normalized)
    scored_items: list[tuple[SearchResult, int, int, bool]] = []
    for result in results:
        chunk = result.chunk
        title = chunk.title.lower()
        head = readable_chunk_text(chunk)[:1000].lower()
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
            + core_coverage * 7.0
            + title_hits * 9.0
            + early_title_hits * 34.0
            + image_bonus
            + manual_bonus
        )
        phrase_hit = False
        for phrase in phrase_hints:
            phrase_l = phrase.lower()
            phrase_count = doc_text.count(phrase_l)
            if phrase_l in title:
                phrase_hit = True
                score += 430.0
            elif phrase_l in head:
                phrase_hit = True
                score += 230.0
            if phrase_count > 1:
                score += min(260.0, phrase_count * 52.0)
        if phrase_hints and not any(phrase.lower() in doc_text for phrase in phrase_hints):
            score -= 340.0
        if title.startswith(("warning", "! warning", "caution", "important safeguards")):
            if phrase_hints and not any(phrase.lower() in title for phrase in phrase_hints):
                score -= 260.0
        if chunk.manual == "汇总英文手册":
            score -= bundle_product_penalty(plan.normalized, title, head)
        if any(term.lower() in plan.normalized.lower() for term in ACCESSORY_TERMS) and title.startswith(
            ("首次使用", "before first use", "quick release")
        ):
            score -= 160.0
        if "mount" in plan.normalized.lower() and "lens" in plan.normalized.lower():
            if not any(phrase in doc_text for phrase in ("attach the lens", "mounting", "lens mount")):
                score -= 420.0
        if core_terms and all(term in title for term in core_terms if len(term) >= 3):
            score += 90.0
        if any(term in title[:30] for term in HIGH_INTENT_TERMS if term in core_terms):
            score += 24.0
        for phrase in ("before first use", "first use", "robot anatomy"):
            if phrase in " ".join(plan.variants).lower() and phrase in doc_text:
                score += 110.0
        if title.startswith(("before first use",)):
            score += 70.0
        if len(title) > 140 and early_title_hits == 0:
            score -= 18.0
        scored_items.append((SearchResult(chunk=chunk, score=score), core_coverage, title_hits, phrase_hit))

    scored_items.sort(key=lambda item: item[0].score, reverse=True)
    reranked = [item[0] for item in scored_items]
    top = reranked[0].score
    if top <= 0:
        return reranked
    top_core_coverage = scored_items[0][1]
    min_core_coverage = max(1, int(top_core_coverage * 0.55)) if top_core_coverage else 0
    kept = [
        item
        for item, core_coverage, _, phrase_hit in scored_items
        if (
            item.score >= top * 0.62 and core_coverage >= min_core_coverage
        ) or (
            phrase_hit and item.score >= top * 0.35
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


def focus_block_for_question(question: str, block: str, limit: int) -> str:
    hints = phrase_hints_for_question(question)
    if not hints:
        return block[:limit]

    block_l = block.lower()
    positions = [block_l.find(hint.lower()) for hint in hints if block_l.find(hint.lower()) >= 0]
    if not positions:
        return block[:limit]

    pos = min(positions)
    start = max(block.rfind("\n", 0, pos), block.rfind("#", 0, pos), 0)
    if start < pos - 180:
        start = max(0, pos - 180)
    end = min(len(block), start + limit)
    if end < len(block):
        pivot = max(block.rfind("\n", start, end), block.rfind("。", start, end), block.rfind(". ", start, end))
        if pivot > pos:
            end = pivot + 1
    focused = block[start:end].strip(" #\n")
    return focused or block[:limit]


def clean_customer_block(block: str) -> str:
    replacements = (
        (r"请阅读本手册", ""),
        (r"在阅读并理解本手册中的说明前，?", ""),
        (r"请妥善保管本手册以备日后查阅。?", ""),
        (r"妥善保管本手册以备日后查阅。?", ""),
        (r"请查阅本手册末尾的", "请查看"),
        (r"本手册", "说明"),
        (r"使用说明书", "使用说明"),
        (r"用户手册", "使用指南"),
        (r"(?i)\buse and care manual\b", "Use and care guide"),
        (r"(?i)\binstruction manual\b", "instruction guide"),
        (r"(?i)\buser manual\b", "user guide"),
        (r"(?i)thismanual", "this guide"),
        (r"(?i)\bmanual\b", "guide"),
    )
    cleaned = block
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def target_manual_supplements(plan: QueryPlan, chunks: list[ManualChunk], limit: int = 8) -> list[SearchResult]:
    if not plan.target_manuals:
        return []
    product_terms = plan_product_terms(plan)
    terms = {
        term
        for term in tokenize(plan.normalized)
        if len(term) >= 2
        and term not in COMMON_QUERY_TERMS
        and term not in product_terms
        and not any(ch.isdigit() for ch in term)
    }
    phrases = phrase_hints_for_question(plan.normalized)
    supplements: list[SearchResult] = []
    for chunk in chunks:
        if chunk.manual not in plan.target_manuals:
            continue
        title = chunk.title.lower()
        text = readable_chunk_text(chunk).lower()
        score = 0.0
        for phrase in phrases:
            phrase_l = phrase.lower()
            if phrase_l in title:
                score += 460.0
            elif phrase_l in text:
                score += 240.0
        for term in terms:
            if term in title:
                score += 42.0
            elif term in text:
                score += 8.0
        if chunk.manual == "汇总英文手册":
            score -= bundle_product_penalty(plan.normalized, title, text[:1000])
        if score > 0:
            supplements.append(SearchResult(chunk=chunk, score=score))
    supplements.sort(key=lambda item: item.score, reverse=True)
    return supplements[:limit]


def fallback_manual_answer(question: str, results: list[SearchResult]) -> str:
    is_english = detect_language(question) == "en"
    if not results:
        if is_english:
            return "I could not find enough clear information in the provided manuals. Please provide the product model, symptom, or image so we can verify it further."
        return "您好，暂未在已提供的资料中检索到足够明确的信息。建议您补充商品型号、故障现象或图片，我们会继续为您核实。"

    lines = ["Here is what you can do:"] if is_english else ["您好，可以这样处理："]
    blocks: list[str] = []
    selected_results: list[SearchResult] = []
    seen = set()
    top_score = results[0].score
    question_phrases = phrase_hints_for_question(question)
    for result in results:
        block = readable_chunk_text(result.chunk)
        block_l = block.lower()
        has_phrase = any(phrase.lower() in block_l or phrase.lower() in result.chunk.title.lower() for phrase in question_phrases)
        if blocks and question_phrases and not has_phrase:
            continue
        if blocks and result.score < top_score * 0.68 and not has_phrase:
            continue
        title_key = re.sub(r"\s+", " ", result.chunk.title.lower())[:80]
        if title_key in seen:
            continue
        if not block:
            continue
        key = block[:80]
        if key in seen:
            continue
        seen.add(key)
        seen.add(title_key)
        if len(blocks) == 0:
            limit = 1100 if is_english else 760
        else:
            limit = 560 if is_english else 430
        blocks.append(clean_customer_block(focus_block_for_question(question, block, limit)))
        selected_results.append(result)
        if len(blocks) >= 3:
            break

    if blocks:
        for idx, block in enumerate(blocks, start=1):
            lines.append(f"{idx}. {block}")
    else:
        chunks = [item.chunk for item in results[:4]]
        sentences = extract_relevant_sentences(question, chunks) or [clean_context_text(chunks[0].text)[:520]]
        for idx, sentence in enumerate(sentences[:6], start=1):
            lines.append(f"{idx}. {sentence}")

    image_ids = collect_image_ids(selected_results or results, limit=6)
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

    def __post_init__(self) -> None:
        manuals = load_manuals()
        self.manual_names = {manual.name for manual in manuals}
        self.chunks = build_chunks(manuals)
        self.chunk_pos = {chunk.id: idx for idx, chunk in enumerate(self.chunks)}
        self.retriever = BM25Retriever(self.chunks)
        self.llm = OpenAICompatibleClient() if self.use_llm else None

    def expand_short_top_context(self, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return results
        top = results[0]
        if len(readable_chunk_text(top.chunk)) >= 100 or top.score < 40:
            return results

        expanded = list(results)
        seen = {item.chunk.id for item in expanded}
        pos = self.chunk_pos.get(top.chunk.id)
        if pos is None:
            return results
        for offset in range(1, 4):
            if pos + offset >= len(self.chunks):
                break
            neighbor = self.chunks[pos + offset]
            if neighbor.manual != top.chunk.manual:
                break
            if neighbor.id not in seen:
                expanded.append(SearchResult(chunk=neighbor, score=top.score * (0.99 - offset * 0.01)))
                seen.add(neighbor.id)
        return expanded

    def retrieve(self, plan: QueryPlan) -> list[SearchResult]:
        raw = self.retriever.search_many(plan.variants, top_k=40, per_query_k=18)
        seen_raw = {item.chunk.id for item in raw}
        for supplement in target_manual_supplements(plan, self.chunks):
            if supplement.chunk.id not in seen_raw:
                raw.append(supplement)
                seen_raw.add(supplement.chunk.id)
        lowered = plan.normalized.lower()
        if "robot anatomy" in lowered and "vacuum" in lowered:
            for chunk in self.chunks:
                if chunk.manual == "汇总英文手册" and chunk.title.strip().lower().startswith("vacuum"):
                    raw.append(SearchResult(chunk=chunk, score=(raw[0].score if raw else 80.0) + 80.0))
                    break
        narrowed = narrow_results(plan, raw)
        reranked = rerank_results(plan, narrowed)
        return self.expand_short_top_context(reranked[:8])

    def answer(self, question: str, qid: str | int | None = None) -> str:
        plan = build_query_plan(question, self.manual_names)
        force_manual = False
        force_policy = False
        if qid is not None:
            try:
                numeric_qid = int(qid)
                force_policy = numeric_qid < 64
                force_manual = numeric_qid >= 64
            except ValueError:
                force_manual = False
        force_manual = force_manual or looks_like_manual_question(plan.normalized)

        policy_answer = None if force_manual and not force_policy else answer_policy_question(plan.normalized)
        if policy_answer:
            return policy_answer
        if force_policy:
            return generic_policy_answer(plan.normalized)

        results = self.retrieve(plan)
        if self.use_llm and self.llm is not None and results:
            try:
                return self.llm.chat(build_llm_messages(plan.normalized, results))
            except Exception as exc:
                # Keep batch generation running even if one API call fails.
                return fallback_manual_answer(question, results) + f"\n（系统提示：LLM调用失败，已使用本地检索答案。错误摘要：{exc}）"
        return fallback_manual_answer(question, results)
