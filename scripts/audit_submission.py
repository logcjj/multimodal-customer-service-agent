from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
from collections import Counter
from pathlib import Path

from df_kefu_baseline.config import PROJECT_ROOT, QUESTION_PATH
from df_kefu_baseline.products import ENGLISH_PRODUCT_ALIASES, candidate_product_keys_for_query


CHINESE_PRODUCT_ALIASES: dict[str, tuple[str, ...]] = {
    "vr_headset": ("VR头显", "头显"),
    "ergonomic_chair": ("人体工学椅",),
    "exercise_bike": ("健身单车", "单车"),
    "fitness_tracker": ("健身追踪器", "手表", "表带"),
    "kids_motorcycle": ("儿童电动摩托车", "电动摩托车"),
    "fridge": ("冰箱",),
    "keyboard": ("功能键盘", "键盘"),
    "generator": ("发电机",),
    "thermostat": ("温控器", "热泵"),
    "blower": ("吹风机",),
    "jetski": ("摩托艇",),
    "pump": ("水泵",),
    "dishwasher": ("洗碗机",),
    "oven": ("烤箱", "烤架", "接油盘", "烤盘"),
    "drill": ("电钻", "DCB101", "DCB107", "DCB112"),
    "camera": ("相机",),
    "air_purifier": ("空气净化器",),
    "air_conditioner": ("空调",),
    "steam_cleaner": ("蒸汽清洁机",),
    "mouse": ("蓝牙激光鼠标", "鼠标"),
}

PRODUCT_ALIASES: dict[str, tuple[str, ...]] = {
    **ENGLISH_PRODUCT_ALIASES,
    **CHINESE_PRODUCT_ALIASES,
}
COMPATIBLE_GROUPS = (
    frozenset(("boat", "jetski")),
)

RAW_NOISE_RE = re.compile(
    r"\\(?:mathsf|pmb|hookrightarrow|widehat|overbrace|boxed|mathcal)"
    r"|\$[^$\n]{0,120}\$"
    r"|\.{5,}"
    r"|\bFigure\s+\d+\b"
    r"|\bpage\s+\d+\b",
    re.IGNORECASE,
)
FRAGMENT_HEAD_RE = re.compile(
    r"^(?:您好，可以这样处理：\n|您好，相关说明如下：\n|Here is what you can do:\n|The relevant instructions are:\n)?"
    r"1\.\s*(?:nds\s|ot\s|ther\s|phones,\s|l\s|s,nicks|ge\s|and\s|that\s|not\soperate)",
    re.IGNORECASE,
)
ACTION_SIGNAL_RE = re.compile(
    r"安装|取下|拆下|装入|装回|按下|打开|关闭|连接|固定|选择|使用|清洁|更换|检查|"
    r"设置|配置|切换|保存|下载|排空|擦拭|拧紧|松开|拔下|防止|请勿|确保|应使用|"
    r"运行|存放|调整|维护|加注|排放|加速|减速|转向|油门|恢复|练习|烹饪|"
    r"\b(?:install|remove|press|open|close|connect|clean|replace|set|configure|switch|"
    r"save|download|drain|check|use|keep|make sure|do not|warning|caution)\b",
    re.IGNORECASE,
)
NUMBERED_ITEM_RE = re.compile(r"(?:^|\n|\s)(?:\d+[.、]?|[A-Z][.、])\s*[\u4e00-\u9fffA-Za-z]", re.MULTILINE)
SENTENCE_PUNCT_RE = re.compile(r"[。！？!?；;：:]")


def read_questions(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["id"]: row["question"] for row in csv.DictReader(f)}


def parse_ret(ret: str) -> tuple[str, list[str], str]:
    decoder = json.JSONDecoder()
    try:
        answer, idx = decoder.raw_decode(ret)
    except Exception as exc:
        return "", [], f"ret正文不是合法JSON字符串: {exc}"
    rest = ret[idx:].strip()
    if not rest:
        return str(answer), [], ""
    if not rest.startswith(","):
        return str(answer), [], "ret正文后不是图片数组"
    try:
        images = json.loads(rest[1:].strip())
    except Exception as exc:
        return str(answer), [], f"图片数组解析失败: {exc}"
    if not isinstance(images, list):
        return str(answer), [], "图片数组不是list"
    return str(answer), [str(item) for item in images], ""


def is_english_like(text: str) -> bool:
    letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    return letters > cjk * 1.5


def text_balance(text: str) -> tuple[int, int]:
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    return cjk, letters


def detect_products(text: str) -> set[str]:
    lower = f" {text.lower()} "
    found = set(candidate_product_keys_for_query(text))
    for product, aliases in PRODUCT_ALIASES.items():
        for alias in aliases:
            if product == "pump" and alias == "水泵" and f"潜{alias}" in text:
                continue
            if alias.lower() in lower or alias in text:
                found.add(product)
                break
    return found


def first_product_mention(text: str, products: set[str]) -> str | None:
    lower = text.lower()
    best: tuple[int, str] | None = None
    for product in products:
        for alias in PRODUCT_ALIASES.get(product, ()):
            pos = lower.find(alias.lower())
            if pos >= 0 and (best is None or pos < best[0]):
                best = (pos, product)
    return best[1] if best else None


def compatible_products(question_products: set[str], answer_products: set[str]) -> bool:
    if question_products & answer_products:
        return True
    return any(question_products <= group and answer_products <= group for group in COMPATIBLE_GROUPS)


def is_title_only(answer: str) -> bool:
    stripped = answer.strip()
    patterns = (
        r"(?:您好，可以这样处理：\n)?1\.\s*[^。.!?？\n]{1,90}",
        r"(?:Here is what you can do:\n)?1\.\s*[^。.!?？\n]{1,90}",
        r"(?:The relevant instructions are:\n)?1\.\s*[^。.!?？\n]{1,90}",
    )
    return any(re.fullmatch(pattern, stripped, flags=re.S) for pattern in patterns)


def numbered_item_count(answer: str) -> int:
    return len(NUMBERED_ITEM_RE.findall(answer.replace("<PIC>", " ")))


def asks_for_brief_list(question: str) -> bool:
    return bool(re.search(r"只需|前\s*[1-9一二两三四五六七八九十]\s*条|first\s+[1-9]", question, flags=re.IGNORECASE))


def is_complete_short_manual_answer(question: str, answer: str) -> bool:
    plain = answer.replace("<PIC>", "").strip()
    if "该部件位置或操作示意主要见相关插图" in plain:
        return False
    if len(plain) < 35:
        return False
    if asks_for_brief_list(question) and (numbered_item_count(answer) >= 1 or ACTION_SIGNAL_RE.search(plain)):
        return True
    if "预设运动程序" in question and answer.count("<PIC>") >= 2 and len(plain) >= 20:
        return True
    if numbered_item_count(answer) >= 2 and ACTION_SIGNAL_RE.search(plain):
        return True
    if len(plain) >= 30 and "<PIC>" in answer and ACTION_SIGNAL_RE.search(plain):
        return True
    if len(plain) >= 45 and ACTION_SIGNAL_RE.search(plain) and SENTENCE_PUNCT_RE.search(plain):
        return True
    if is_english_like(question) and len(plain) >= 70 and SENTENCE_PUNCT_RE.search(plain):
        return True
    return False


def focused_phrases_from_question(question: str) -> list[str]:
    text = question.strip().strip("\"'“”")
    phrases: list[str] = []
    for match in re.finditer(r"的([\u4e00-\u9fffA-Za-z0-9（）()\-_/]{2,16})", text):
        phrase = match.group(1).strip(" ？?。；;，,")
        phrase = re.sub(r"(?:该如何|如何|怎么|是什么|有哪些|哪些|吗)$", "", phrase)
        phrase = re.sub(r"(?:类别时|类别|时)$", "", phrase)
        if phrase == "发动机停机":
            phrase = "停机"
        if 2 <= len(phrase) <= 16 and phrase not in {
            "正确步骤",
            "详细步骤",
            "相关说明",
            "注意事项",
            "使用方法",
            "操作方法",
            "开关",
            "稳定性和操控性",
        }:
            phrases.append(phrase)
    if "最高难度" in text and "挑战" in text:
        phrases.append("挑战")
    seen = set()
    unique: list[str] = []
    for phrase in phrases:
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            unique.append(phrase)
    return unique[:3]


def normalized_answer_for_similarity(answer: str) -> str:
    text = answer.replace("<PIC>", "")
    text = re.sub(r"您好，相关说明如下：|The relevant instructions are:", "", text)
    text = re.sub(r"\s+", "", text)
    return text[:1200]


def add_issue(row: dict[str, str], issue: str, severity: str) -> None:
    labels = [item for item in row["问题标签"].split("；") if item]
    if issue not in labels:
        labels.append(issue)
    row["问题标签"] = "；".join(labels)
    if row["严重程度"] == "正常" or severity == "严重":
        row["严重程度"] = severity


def audit_row(qid: str, question: str, ret: str) -> dict[str, str]:
    answer, images, parse_error = parse_ret(ret)
    issues: list[str] = []
    severity = "正常"

    q_products = detect_products(question)
    a_products = detect_products(answer)
    extra_products = a_products - q_products
    first_product = first_product_mention(answer, a_products)
    manual_question = int(qid) >= 64 if qid.isdigit() else True

    if parse_error:
        issues.append(parse_error)
        severity = "严重"
    if answer.count("<PIC>") != len(images):
        issues.append("<PIC>数量与图片数组数量不一致")
        severity = "严重"
    if q_products and a_products and not compatible_products(q_products, a_products):
        issues.append("答案疑似召回到其他产品")
        severity = "严重"
    elif q_products and first_product in extra_products and not compatible_products(q_products, {first_product}):
        issues.append("答案开头疑似召回到其他产品")
        severity = "严重"
    if (
        manual_question
        and len(answer.replace("<PIC>", "").strip()) < 90
        and not is_complete_short_manual_answer(question, answer)
    ):
        issues.append("说明书题答案过短")
        severity = "严重"
    if manual_question and is_title_only(answer):
        issues.append("只返回标题或章节名")
        severity = "严重"
    if is_english_like(question):
        cjk, letters = text_balance(answer)
        if cjk > letters * 0.45:
            issues.append("英文问题返回大量中文")
            severity = "严重"
    if FRAGMENT_HEAD_RE.search(answer):
        issues.append("答案从段落中间截取")
        if severity == "正常":
            severity = "中等"
    if RAW_NOISE_RE.search(answer):
        issues.append("包含未清洗的OCR/LaTeX/目录噪声")
        if severity == "正常":
            severity = "中等"
    if answer.count("<PIC>") >= 5:
        issues.append("图片引用过多")
        if severity == "正常":
            severity = "中等"
    if len(answer) > 1800:
        issues.append("答案过长")
        if severity == "正常":
            severity = "中等"

    return {
        "id": qid,
        "严重程度": severity,
        "问题标签": "；".join(issues),
        "问题产品": ",".join(sorted(q_products)),
        "答案产品": ",".join(sorted(a_products)),
        "答案长度": str(len(answer)),
        "PIC数量": str(answer.count("<PIC>")),
        "图片数量": str(len(images)),
        "question": question.replace("\n", " / "),
        "answer_start": answer[:500].replace("\n", " / "),
    }


def audit_submission(submission_path: Path, output_csv: Path | None = None) -> tuple[Counter[str], list[dict[str, str]]]:
    questions = read_questions(QUESTION_PATH)
    with submission_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_id = {row["id"]: row["ret"] for row in rows}

    audit_rows: list[dict[str, str]] = []
    for qid, question in questions.items():
        if qid not in by_id:
            audit_rows.append(
                {
                    "id": qid,
                    "严重程度": "严重",
                    "问题标签": "提交缺少该id",
                    "问题产品": "",
                    "答案产品": "",
                    "答案长度": "0",
                    "PIC数量": "0",
                    "图片数量": "0",
                    "question": question.replace("\n", " / "),
                    "answer_start": "",
                }
            )
            continue
        audit_rows.append(audit_row(qid, question, by_id[qid]))

    answers: dict[str, str] = {}
    for qid, ret in by_id.items():
        answer, _, _ = parse_ret(ret)
        answers[qid] = answer
    row_by_id = {row["id"]: row for row in audit_rows}
    question_ids = list(questions)
    for idx, qid in enumerate(question_ids):
        if qid not in answers or qid not in row_by_id:
            continue
        if not qid.isdigit() or int(qid) < 64:
            continue
        q_products = detect_products(questions[qid])
        q_focus = set(focused_phrases_from_question(questions[qid]))
        if not q_products or not q_focus:
            continue
        a_norm = normalized_answer_for_similarity(answers[qid])
        if len(a_norm) < 120:
            continue
        for other_id in question_ids[idx + 1 : idx + 4]:
            if other_id not in answers or other_id not in row_by_id:
                continue
            if not other_id.isdigit() or int(other_id) < 64:
                continue
            other_products = detect_products(questions[other_id])
            other_focus = set(focused_phrases_from_question(questions[other_id]))
            if not (q_products & other_products) or not other_focus or q_focus == other_focus:
                continue
            b_norm = normalized_answer_for_similarity(answers[other_id])
            if len(b_norm) < 120:
                continue
            similarity = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
            if similarity >= 0.92:
                add_issue(row_by_id[qid], "邻近同产品不同问题答案高度重复", "严重")
                add_issue(row_by_id[other_id], "邻近同产品不同问题答案高度重复", "严重")

    extra_ids = sorted(set(by_id) - set(questions), key=lambda item: int(item) if item.isdigit() else item)
    for qid in extra_ids:
        audit_rows.append(
            {
                "id": qid,
                "严重程度": "严重",
                "问题标签": "提交包含question_public中不存在的id",
                "问题产品": "",
                "答案产品": "",
                "答案长度": "0",
                "PIC数量": "0",
                "图片数量": "0",
                "question": "",
                "answer_start": "",
            }
        )

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            writer.writeheader()
            writer.writerows(audit_rows)

    return Counter(row["严重程度"] for row in audit_rows), audit_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a generated customer-service submission CSV.")
    parser.add_argument("submission", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    submission_path = args.submission
    if not submission_path.is_absolute():
        submission_path = PROJECT_ROOT / submission_path
    output_path = args.output
    if output_path is not None and not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    counts, rows = audit_submission(submission_path, output_path)
    review = len([row for row in rows if row["严重程度"] != "正常"])
    print(f"normal={counts['正常']} medium={counts['中等']} severe={counts['严重']} review={review}")
    if output_path:
        print(f"saved={output_path}")


if __name__ == "__main__":
    main()
