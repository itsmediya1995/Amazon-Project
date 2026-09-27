from __future__ import annotations

import numpy as np

from .records import Record

FEATURE_NAMES = [
    "name_jaccard",
    "name_overlap",
    "name_contain",
    "addr_jaccard",
    "addr_overlap",
    "loc_jaccard",
    "postal_match",
    "house_match",
    "digit_jaccard",
    "prefix_equal",
    "core_equal",
    "same_country",
    "name_len_ratio",
    "missing_addr",
]


def _jac(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def _overlap(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _contain(a: frozenset, b: frozenset) -> float:
    if not a:
        return 0.0
    return len(a & b) / len(a)


def pair_features(a: Record, b: Record) -> np.ndarray:
    if a.country != b.country:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    nj = _jac(a.name_set, b.name_set)
    aj = _jac(a.addr_set, b.addr_set)
    nlen = min(len(a.name_set), len(b.name_set)) / max(len(a.name_set), len(b.name_set), 1)
    feats = np.array(
        [
            nj,
            _overlap(a.name_set, b.name_set),
            _contain(a.name_set, b.name_set),
            aj,
            _overlap(a.addr_set, b.addr_set),
            _jac(a.loc_set, b.loc_set),
            1.0 if a.postal and a.postal == b.postal else 0.0,
            1.0 if a.house and a.house == b.house else 0.0,
            _jac(a.digits, b.digits),
            1.0 if a.prefix2 and a.prefix2 == b.prefix2 else 0.0,
            1.0 if a.core and a.core == b.core else 0.0,
            1.0,
            nlen,
            1.0 if (not a.addr_set or not b.addr_set) else 0.0,
        ],
        dtype=np.float32,
    )
    return feats


def cheap_score(feats: np.ndarray) -> float:
    """Precision-heavy blend used before / without the GBDT."""
    return float(
        0.38 * feats[0]
        + 0.16 * feats[1]
        + 0.10 * feats[2]
        + 0.14 * feats[3]
        + 0.08 * feats[5]
        + 0.08 * feats[6]
        + 0.06 * feats[7]
    )
