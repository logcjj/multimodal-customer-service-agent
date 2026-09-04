from __future__ import annotations

from app.runtime.error_codes import error_code_search_forms


# 小型跨语言检索词典只补充说明书中的通用动作和部件名称，不生成答案。
# 型号、错误码和用户原问题始终原样保留，LLM 改写失败时仍可稳定检索。
TERM_EXPANSIONS: tuple[tuple[str, str], ...] = (
    ("清洁", "clean cleaning"),
    ("清洗", "clean wash cleaning"),
    ("炸篮", "basket"),
    ("滤网", "filter"),
    ("排水", "drain drainage"),
    ("报错", "error code troubleshooting"),
    ("错误码", "error code troubleshooting"),
    ("显示", "display error code troubleshooting"),
    ("故障", "fault troubleshooting"),
    ("更换", "replace replacement"),
    ("安装", "install installation"),
    ("连接", "connect connection"),
    ("无法", "cannot failed troubleshooting"),
    ("启动", "start power on"),
    ("充电", "charge charging"),
    ("重置", "reset"),
    ("配对", "pair pairing"),
    ("温度", "temperature"),
    ("时间", "time timer"),
    (
        "小松树",
        "空气净化 等离子净化 离子发生器 air purify ionizer plasma purification",
    ),
    (
        "树形图标",
        "空气净化 等离子净化 离子发生器 air purify ionizer plasma purification",
    ),
    ("borad", "board boarding reboard reboarding"),
    ("board", "board boarding reboard reboarding"),
    ("alone", "solo"),
    ("rinse", "rinsing overflow rinse"),
    ("tool kit", "owner operator manual tool kit"),
    ("transport", "transporting trailer towing"),
)


def deterministic_expansion(query: str) -> str:
    terms = [query.strip()]
    terms.extend(error_code_search_forms(query))
    lowered = query.lower()
    for source, target in TERM_EXPANSIONS:
        if source in lowered:
            terms.append(target)
    return " ".join(dict.fromkeys(term for term in terms if term))
