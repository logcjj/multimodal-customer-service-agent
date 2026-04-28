from __future__ import annotations

import re


ENGLISH_PRODUCT_ALIASES: dict[str, tuple[str, ...]] = {
    "camera": (
        "dslr camera",
        "cf card",
        "af mode",
        "date/time battery",
        "camera battery",
        "household power outlet",
        "cp direct",
        "picture style",
        "shutter button",
        "lens",
    ),
    "coffee_machine": (
        "coffee machine",
        "espresso",
        "lungo",
        "energy saving mode",
        "water volume",
        "factory settings",
        "19 bar",
    ),
    "air_fryer": (
        "airfryer",
        "air fryer",
        "nutriu",
        "airfrying",
        "keep warm mode",
        "food table",
    ),
    "boat": (
        "boat",
        "bilge pump",
        "bimini",
        "livewell",
        "cruise assist",
        "jet wash",
        "swim platform",
        "fire extinguisher",
        "throttle-cable",
        "throttle cable",
        "on board",
        "sailing",
    ),
    "jetski": (
        "jetski",
        "jet ski",
        "waverunner",
        "watercraft",
        "qsts",
        "sponson",
    ),
    "fax": (
        "fax",
        "product safety guide",
        "warning labels",
        "fingers",
    ),
    "earphones": (
        "earphones",
        "earbuds",
        "earbud",
        "active noise canceling",
        "charging case",
        "multifunction button",
    ),
    "ereader": (
        "ereader",
        "e-reader",
        "ebook",
        "browser history",
        "music mode",
        "voice record",
        "photo viewer",
    ),
    "grill": (
        "grill",
        "lp tank",
        "regulator",
        "indirect cooking",
        "burner",
    ),
    "snowmobile": (
        "snowmobile",
        "v-belt",
        "brake lever",
        "ski runner",
        "spark plug",
        "riding uphill",
        "side hill",
    ),
    "television": (
        "television",
        " tv ",
        "caption",
        "channels",
        "dvd player",
        "outdoor antenna",
        "poor reception",
        "key lock",
    ),
    "vacuum": (
        "vacuum",
        "roomba",
        "home base",
        "dual mode virtual wall",
        "full bin",
        "extractors",
        "caster wheel",
        "spot cleaning",
    ),
    "security_camera": (
        "camera installation guide",
        "mv camera",
        "t-rail",
        "drop ceiling",
        "dashboard",
    ),
    "toothbrush": (
        "toothbrush",
        "sonic",
        "brush head",
        "plaque",
    ),
    "washing_machine": (
        "washing machine",
        "washing procedure",
        "spin dry",
        "rinsing",
        "drain hose",
    ),
    "pressure_cooker_air_fryer": (
        "multi-use pressure cooker",
        "pressure cooker",
        "quick release",
        "natural release",
        "sealing ring",
        "float valve",
        "anti-block shield",
        "steam release",
        "condensation collector",
    ),
    "microwave": (
        "microwave",
        "over-the-range",
        "auto defrost",
        "grease filter",
        "charcoal filter",
        "light timer",
        "favorite recipe",
    ),
    "motherboard": (
        "motherboard",
        "pci express",
        "bios",
        "sata",
        "raid",
        "cpu socket",
        "rear panel connectors",
        "system memory",
        "t_sensor",
    ),
    "landline": (
        "landline",
        "base station",
        "handset",
        "phonebook",
        "led indicator",
        "xl490",
        "xl495",
    ),
    "lawn_mower": (
        "lawn mower",
        "mower belt",
        "roll bar",
        "height of cut",
        "rear-shock",
        "mower deck",
        "engine oil",
    ),
}


PRODUCT_NAME_HINTS: dict[str, tuple[str, ...]] = {
    "air fryer": ("air_fryer", "pressure_cooker_air_fryer"),
    "airfryer": ("air_fryer",),
    "multi-use pressure cooker": ("pressure_cooker_air_fryer",),
    "pressure cooker": ("pressure_cooker_air_fryer",),
    "coffee machine": ("coffee_machine",),
    "earphones": ("earphones",),
    "earbuds": ("earphones",),
    "ereader": ("ereader",),
    "e-reader": ("ereader",),
    "fax": ("fax",),
    "grill": ("grill",),
    "landline": ("landline",),
    "lawn mower": ("lawn_mower",),
    "microwave": ("microwave",),
    "motherboard": ("motherboard",),
    "vacuum": ("vacuum",),
    "snowmobile": ("snowmobile",),
    "television": ("television",),
    "toothbrush": ("toothbrush",),
    "boat": ("boat",),
    "jetski": ("jetski",),
    "jet ski": ("jetski",),
    "watercraft": ("jetski",),
    "camera": ("camera", "security_camera"),
    "t-rail": ("security_camera",),
}


def english_manual_name(product_key: str) -> str:
    return f"汇总英文手册::{product_key}"


def product_key_from_manual_name(manual_name: str) -> str | None:
    prefix = "汇总英文手册::"
    if manual_name.startswith(prefix):
        return manual_name[len(prefix) :]
    return None


def aliases_for_product(product_key: str) -> tuple[str, ...]:
    return ENGLISH_PRODUCT_ALIASES.get(product_key, ())


def infer_english_product_key(text: str, fallback_index: int | None = None) -> str:
    window = f" {text[:12000].lower()} "
    best_key = ""
    best_score = 0
    for key, aliases in ENGLISH_PRODUCT_ALIASES.items():
        score = 0
        for alias in aliases:
            alias_l = f" {alias.lower()} " if alias.strip() == alias and " " not in alias.strip() else alias.lower()
            if alias_l in window:
                score += max(8, len(alias.strip()) // 2)
                if alias.lower() in text[:800].lower():
                    score += 20
        if score > best_score:
            best_key = key
            best_score = score

    if best_key:
        return best_key
    if fallback_index is not None:
        return f"part_{fallback_index:02d}"
    return "part"


def candidate_product_keys_for_query(query: str) -> set[str]:
    lower = f" {query.lower()} "
    candidates: set[str] = set()

    if "t-rail" in lower or "drop ceiling" in lower or "camera installation" in lower:
        candidates.add("security_camera")
    if any(term in lower for term in (" af ", "af mode", "cf card", "date/time", "photo", "picture", "lens", "shutter")):
        candidates.add("camera")

    for alias, keys in PRODUCT_NAME_HINTS.items():
        if alias in lower:
            candidates.update(keys)

    for key, aliases in ENGLISH_PRODUCT_ALIASES.items():
        if any(alias.lower() in lower for alias in aliases):
            candidates.add(key)

    if "camera" in candidates and "security_camera" in candidates:
        if "t-rail" in lower or "installation guide" in lower or "drop ceiling" in lower:
            candidates.discard("camera")
        else:
            candidates.discard("security_camera")
    if "ereader" in candidates and "camera" in candidates:
        if "ereader" in lower or "e-reader" in lower or "ebook" in lower:
            candidates.discard("camera")

    return candidates
