from __future__ import annotations

import numpy as np


def f05_entity(pred: set[str], true: set[str]) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    inter = len(pred & true)
    if inter == 0:
        return 0.0
    prec = inter / len(pred)
    rec = inter / len(true)
    return (1.25 * prec * rec) / (0.25 * prec + rec)


def macro_f05(preds: dict[str, set[str]], truths: dict[str, set[str]]) -> float:
    scores = [f05_entity(preds.get(k, set()), truths.get(k, set())) for k in truths]
    return float(np.mean(scores)) if scores else 0.0
