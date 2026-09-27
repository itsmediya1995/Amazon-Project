"""Country-open string parsing for names and addresses. No closed US/India one-hots."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

LEGAL_SUFFIXES = frozenset(
    {
        "inc",
        "incorporated",
        "llc",
        "ltd",
        "limited",
        "llp",
        "lp",
        "pc",
        "pllc",
        "corp",
        "corporation",
        "company",
        "co",
        "pvt",
        "pvtltd",
        "private",
        "plc",
        "sarl",
        "sasu",
        "sas",
        "sa",
        "sci",
        "eurl",
        "snc",
        "gmbh",
        "pte",
        "llc.",
        "partners",
        "group",
        "holdings",
        "services",
        "service",
        "enterprises",
        "enterprise",
        "associates",
        "association",
        "foundation",
        "trust",
        "clinic",
        "hospital",
        "bank",
    }
)

NAME_STOP = frozenset(
    {
        "the",
        "of",
        "and",
        "a",
        "an",
        "dba",
        "aka",
        "www",
        "http",
        "https",
        "com",
        "net",
        "org",
        "pvt",
        "private",
        "limited",
        "ltd",
        "llc",
        "inc",
        "corp",
        "sarl",
        "sasu",
        "sas",
    }
)

STREET_MAP = {
    "st": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "ct": "court",
    "hwy": "highway",
    "pkwy": "parkway",
    "rte": "route",
    "rue": "rue",
    "pl": "place",
    "cir": "circle",
    "ter": "terrace",
    "trl": "trail",
    "apt": "unit",
    "apartment": "unit",
    "ste": "unit",
    "suite": "unit",
    "fl": "floor",
    "floor": "floor",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
}

ADDR_STOP = frozenset(
    {
        "unit",
        "floor",
        "near",
        "opp",
        "opposite",
        "behind",
        "next",
        "po",
        "box",
        "plot",
        "door",
        "house",
        "hno",
        "no",
        "nr",
        "c",
        "o",
    }
)

NON_ALNUM_RE = re.compile(r"[^a-z0-9\u0900-\u097f\u0c80-\u0cff]+")
DIGIT_RE = re.compile(r"\d+")
PIN6_RE = re.compile(r"\b(\d{6})\b")
ZIP5_RE = re.compile(r"\b(\d{5})\b")
HOUSE_RE = re.compile(
    r"\b(\d{1,6}[a-z]?)\b|#\s*(\d{1,6}[a-z]?)|h\.?\s*no\.?\s*[-:]?\s*(\d{1,6}[a-z]?)",
    re.I,
)


def _fold(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("&", " and ")
    text = text.replace("pvtltd", "pvt ltd").replace("pvt.ltd", "pvt ltd")
    text = re.sub(r"https?://", " ", text)
    text = text.replace("www.", " ")
    return NON_ALNUM_RE.sub(" ", text)


def name_tokens(text: str) -> list[str]:
    toks = [t for t in _fold(text).split() if t and t not in NAME_STOP and t not in LEGAL_SUFFIXES]
    # drop 1-char latin noise but keep digits
    out = []
    for t in toks:
        if t.isdigit() or len(t) >= 2:
            out.append(t)
    return out


def addr_tokens(text: str) -> list[str]:
    toks = []
    for t in _fold(text).split():
        t = STREET_MAP.get(t, t)
        if t in ADDR_STOP or t in NAME_STOP:
            continue
        if t.isdigit() or len(t) >= 2:
            toks.append(t)
    return toks


def postal_code(address: str, country: str) -> str:
    raw = str(address or "")
    c = (country or "").strip().lower()
    if c == "india":
        m = PIN6_RE.search(raw)
        return m.group(1) if m else ""
    m = ZIP5_RE.search(raw)
    return m.group(1) if m else ""


def house_number(address: str) -> str:
    raw = str(address or "")
    m = HOUSE_RE.search(raw)
    if not m:
        return ""
    for g in m.groups():
        if g:
            return g.lower()
    return ""


def locality_tokens(address: str) -> list[str]:
    """Use trailing comma segments (city / state / region) which survive format noise."""
    parts = [p.strip() for p in str(address or "").split(",") if p.strip()]
    tail = " ".join(parts[-3:]) if parts else str(address or "")
    toks = addr_tokens(tail)
    return toks[:6]


def digit_sig(name: str, address: str) -> tuple[str, ...]:
    found = DIGIT_RE.findall(f"{name} {address}")
    # keep distinctive numbers (skip years-ish optional)
    return tuple(sorted({d for d in found if 2 <= len(d) <= 8})[:8])


def name_prefix(tokens: list[str], n: int = 2) -> str:
    return " ".join(tokens[:n])


@lru_cache(maxsize=1)
def _noop():
    return None
