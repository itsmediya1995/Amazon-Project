from __future__ import annotations

from collections import defaultdict

MAX_BLOCK = {
    "ph": 120,
    "pn": 80,
    "ln": 64,
    "np": 64,
    "ex": 200,
    "t": 36,
    "d": 40,
}
MIN_TOKEN_LEN = 4


def _add(index: dict[str, list[int]], key: str, i: int) -> None:
    if key:
        index[key].append(i)


def build_index(recs) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Rare-key inverted index. Oversized keys are dropped after a count pass."""
    df: dict[str, int] = defaultdict(int)
    keys_per_row: list[list[str]] = []

    for rec in recs:
        keys = []
        c = rec.country
        if rec.postal and rec.house:
            keys.append(f"{c}|ph|{rec.postal}|{rec.house}")
        if rec.postal and rec.prefix1:
            keys.append(f"{c}|pn|{rec.postal}|{rec.prefix1}")
        if rec.prefix2 and rec.loc_set:
            loc = next(iter(sorted(rec.loc_set, key=len, reverse=True)), "")
            if loc and len(loc) >= 4:
                keys.append(f"{c}|ln|{loc}|{rec.prefix2}")
        if rec.prefix2:
            keys.append(f"{c}|np|{rec.prefix2}")
        if rec.core:
            keys.append(f"{c}|ex|{rec.core}")
        for tok in rec.name_set:
            if len(tok) >= MIN_TOKEN_LEN:
                keys.append(f"{c}|t|{tok}")
        for d in rec.digits:
            if len(d) >= 4:
                keys.append(f"{c}|d|{d}")
        # unique keys per row
        keys = list(dict.fromkeys(keys))
        keys_per_row.append(keys)
        for k in keys:
            df[k] += 1

    index: dict[str, list[int]] = defaultdict(list)
    kept_df: dict[str, int] = {}
    for i, keys in enumerate(keys_per_row):
        for k in keys:
            cnt = df[k]
            kind = k.split("|")[1]
            if cnt > MAX_BLOCK.get(kind, 48):
                continue
            if cnt < 1:
                continue
            index[k].append(i)
            kept_df[k] = cnt
    return index, kept_df


def candidates_for(rec, index: dict[str, list[int]], cap: int = 80) -> list[int]:
    seen: dict[int, int] = {}
    c = rec.country
    keys = []
    if rec.postal and rec.house:
        keys.append(f"{c}|ph|{rec.postal}|{rec.house}")
    if rec.postal and rec.prefix1:
        keys.append(f"{c}|pn|{rec.postal}|{rec.prefix1}")
    if rec.prefix2 and rec.loc_set:
        loc = next(iter(sorted(rec.loc_set, key=len, reverse=True)), "")
        if loc and len(loc) >= 4:
            keys.append(f"{c}|ln|{loc}|{rec.prefix2}")
    if rec.prefix2:
        keys.append(f"{c}|np|{rec.prefix2}")
    if rec.core:
        keys.append(f"{c}|ex|{rec.core}")
    for tok in rec.name_set:
        if len(tok) >= MIN_TOKEN_LEN:
            keys.append(f"{c}|t|{tok}")
    for d in rec.digits:
        if len(d) >= 4:
            keys.append(f"{c}|d|{d}")

    for k in keys:
        bucket = index.get(k)
        if not bucket:
            continue
        for j in bucket:
            seen[j] = seen.get(j, 0) + 1

    if not seen:
        return []
    # prefer IDs that hit multiple keys
    ranked = sorted(seen.items(), key=lambda x: -x[1])
    return [j for j, _ in ranked[:cap]]
