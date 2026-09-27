"""AetherLink ER pipeline: parse → retrieve → rank → assign.

Original team pipeline (not a Kaggle notebook clone).
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# allow `python src/run.py` from this folder
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR.parent))

from src.blocking import candidates_for, build_index  # noqa: E402
from src.features import cheap_score, pair_features  # noqa: E402
from src.metrics import macro_f05  # noqa: E402
from src.records import Record, build_record  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "dataset"
DEFAULT_OUT = ROOT / "output"
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "rankers.pkl"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_source(path: Path) -> list[Record]:
    log(f"Loading {path.name} ...")
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    recs: list[Record] = []
    for i, row in enumerate(df.itertuples(index=False)):
        recs.append(build_record(row.entity_id, row.business_name, row.business_address, row.country))
        if (i + 1) % 500000 == 0:
            log(f"  parsed {i + 1:,}")
    log(f"  {len(recs):,} records in {time.time() - t0:.1f}s")
    return recs


def load_gt(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            eid = parts[0]
            ids = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            out[eid] = {x for x in ids if x}
    return out


def predict_proba(model: HistGradientBoostingClassifier | None, X: np.ndarray) -> np.ndarray:
    if model is None or len(X) == 0:
        return np.zeros(len(X), dtype=np.float32)
    return model.predict_proba(X)[:, 1].astype(np.float32)


def collect_pairs(s1_recs, other_recs, index, gt_sets=None, cap=80):
    Xs = []
    ys = []
    for rec in s1_recs:
        true = gt_sets.get(rec.eid, set()) if gt_sets is not None else None
        for j in candidates_for(rec, index, cap=cap):
            other = other_recs[j]
            feats = pair_features(rec, other)
            Xs.append(feats)
            if true is not None:
                ys.append(1 if other.eid in true else 0)
    if not Xs:
        return np.zeros((0, 14), dtype=np.float32), np.zeros(0, dtype=np.int32)
    return np.vstack(Xs), np.array(ys, dtype=np.int32)


def train_ranker(X, y, name: str):
    if len(y) == 0 or y.sum() == 0:
        log(f"  skip {name}: no positives")
        return None
    log(f"  train {name}: n={len(y):,} pos={int(y.sum()):,} ({y.mean():.3f})")
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        max_iter=90,
        learning_rate=0.08,
        l2_regularization=0.15,
        min_samples_leaf=40,
        random_state=7,
    )
    clf.fit(X, y)
    return clf


def score_candidates(s1: Record, others: list[Record], index, model, cap=80):
    idxs = candidates_for(s1, index, cap=cap)
    if not idxs:
        return [], np.zeros(0, dtype=np.float32)
    feats = np.vstack([pair_features(s1, others[j]) for j in idxs])
    if model is not None:
        proba = predict_proba(model, feats)
        # blend a little cheap score so empty-address pairs with strong names still rank
        cheap = np.array([cheap_score(f) for f in feats], dtype=np.float32)
        scores = 0.85 * proba + 0.15 * cheap
    else:
        scores = np.array([cheap_score(f) for f in feats], dtype=np.float32)
    return idxs, scores


def assign_exclusive(best: dict[str, tuple[float, int]], thresh: float) -> dict[int, list[str]]:
    pred: dict[int, list[str]] = defaultdict(list)
    for eid, (sc, s1_i) in best.items():
        if sc >= thresh:
            pred[s1_i].append(eid)
    return pred


def run_split(
    s1_recs: list[Record],
    s2_recs: list[Record],
    s3_recs: list[Record],
    s2_index,
    s3_index,
    model_s2,
    model_s3,
    thresh: float,
    cap: int = 80,
):
    best_s2: dict[str, tuple[float, int]] = {}
    best_s3: dict[str, tuple[float, int]] = {}
    cand_map: dict[str, list[str]] = {}

    t0 = time.time()
    for i, rec in enumerate(s1_recs):
        i2, sc2 = score_candidates(rec, s2_recs, s2_index, model_s2, cap=cap)
        i3, sc3 = score_candidates(rec, s3_recs, s3_index, model_s3, cap=cap)
        cids = []
        for j, sc in zip(i2, sc2):
            eid = s2_recs[j].eid
            cids.append(eid)
            prev = best_s2.get(eid)
            if prev is None or sc > prev[0]:
                best_s2[eid] = (float(sc), i)
        for j, sc in zip(i3, sc3):
            eid = s3_recs[j].eid
            cids.append(eid)
            prev = best_s3.get(eid)
            if prev is None or sc > prev[0]:
                best_s3[eid] = (float(sc), i)
        cand_map[rec.eid] = list(dict.fromkeys(cids))
        if (i + 1) % 20000 == 0:
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            log(f"  scored {i + 1:,}/{len(s1_recs):,} ({rate:.0f}/s)")

    pred2 = assign_exclusive(best_s2, thresh)
    pred3 = assign_exclusive(best_s3, thresh)
    matches: dict[str, list[str]] = {}
    for i, rec in enumerate(s1_recs):
        ids = pred2.get(i, []) + pred3.get(i, [])
        matches[rec.eid] = list(dict.fromkeys(ids))
    return matches, cand_map


def write_tsv(path: Path, s1_ids: list[str], mapping: dict[str, list[str]], col: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for eid in s1_ids:
            ids = mapping.get(eid, [])
            f.write(f"{eid}\t{','.join(ids)}\n")
    log(f"Wrote {path} ({len(s1_ids):,} rows)")


def tune_threshold(s1_recs, s2_recs, s3_recs, s2_index, s3_index, model_s2, model_s3, gt, cap):
    # collect best exclusive scores then sweep
    best_s2: dict[str, tuple[float, int]] = {}
    best_s3: dict[str, tuple[float, int]] = {}
    for i, rec in enumerate(s1_recs):
        i2, sc2 = score_candidates(rec, s2_recs, s2_index, model_s2, cap=cap)
        i3, sc3 = score_candidates(rec, s3_recs, s3_index, model_s3, cap=cap)
        for j, sc in zip(i2, sc2):
            eid = s2_recs[j].eid
            prev = best_s2.get(eid)
            if prev is None or sc > prev[0]:
                best_s2[eid] = (float(sc), i)
        for j, sc in zip(i3, sc3):
            eid = s3_recs[j].eid
            prev = best_s3.get(eid)
            if prev is None or sc > prev[0]:
                best_s3[eid] = (float(sc), i)

    true = {r.eid: gt.get(r.eid, set()) for r in s1_recs}
    best_t, best_s = 0.55, -1.0
    for t in np.linspace(0.35, 0.85, 21):
        p2 = assign_exclusive(best_s2, float(t))
        p3 = assign_exclusive(best_s3, float(t))
        preds = {}
        for i, rec in enumerate(s1_recs):
            preds[rec.eid] = set(p2.get(i, []) + p3.get(i, []))
        s = macro_f05(preds, true)
        log(f"  thresh={t:.2f}  F0.5={s:.4f}")
        if s > best_s:
            best_t, best_s = float(t), s
    log(f"  selected thresh={best_t:.2f}  F0.5={best_s:.4f}")
    return best_t, best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--train-s1", type=int, default=18000)
    ap.add_argument("--valid-s1", type=int, default=4000)
    ap.add_argument("--cap", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--infer-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="Train and tune on holdout; skip test inference")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train_dir = args.data_root / "train"
    test_dir = args.data_root / "test"
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    model_s2 = model_s3 = None
    thresh = 0.58

    if not args.infer_only:
        s1 = load_source(train_dir / "train_source1.tsv")
        s2 = load_source(train_dir / "train_source2.tsv")
        s3 = load_source(train_dir / "train_source3.tsv")
        gt = load_gt(train_dir / "train_ground_truth.tsv")

        log("Building train blocking indexes ...")
        s2_index, _ = build_index(s2)
        s3_index, _ = build_index(s3)
        log(f"  S2 keys={len(s2_index):,}  S3 keys={len(s3_index):,}")

        ids = list(range(len(s1)))
        random.shuffle(ids)
        train_ids = ids[: args.train_s1]
        valid_ids = ids[args.train_s1 : args.train_s1 + args.valid_s1]
        train_recs = [s1[i] for i in train_ids]
        valid_recs = [s1[i] for i in valid_ids]

        if not args.skip_train:
            log("Collecting training pairs ...")
            X2, y2 = collect_pairs(train_recs, s2, s2_index, gt, cap=args.cap)
            X3, y3 = collect_pairs(train_recs, s3, s3_index, gt, cap=args.cap)
            model_s2 = train_ranker(X2, y2, "S1-S2")
            model_s3 = train_ranker(X3, y3, "S1-S3")
            with args.model_path.open("wb") as f:
                pickle.dump({"s2": model_s2, "s3": model_s3}, f)
            log(f"Saved {args.model_path}")
        else:
            with args.model_path.open("rb") as f:
                blob = pickle.load(f)
            model_s2, model_s3 = blob["s2"], blob["s3"]

        log("Tuning threshold on holdout S1 ...")
        thresh, val_score = tune_threshold(
            valid_recs, s2, s3, s2_index, s3_index, model_s2, model_s3, gt, args.cap
        )
        with args.model_path.open("wb") as f:
            pickle.dump({"s2": model_s2, "s3": model_s3, "thresh": thresh, "val_f05": val_score}, f)

        if args.eval_only:
            log("eval-only: stopping after threshold tune")
            return

        # free train memory before test
        del s1, s2, s3, s2_index, s3_index, gt
    else:
        with args.model_path.open("rb") as f:
            blob = pickle.load(f)
        model_s2, model_s3 = blob["s2"], blob["s3"]
        thresh = float(blob.get("thresh", 0.58))
        log(f"Loaded model thresh={thresh:.3f} val_f05={blob.get('val_f05')}")

    log("Loading test sources ...")
    t1 = load_source(test_dir / "test_source1.tsv")
    t2 = load_source(test_dir / "test_source2.tsv")
    t3 = load_source(test_dir / "test_source3.tsv")
    log("Building test blocking indexes ...")
    t2_index, _ = build_index(t2)
    t3_index, _ = build_index(t3)
    log(f"  S2 keys={len(t2_index):,}  S3 keys={len(t3_index):,}")

    log("Inference on full test S1 ...")
    matches, cands = run_split(
        t1, t2, t3, t2_index, t3_index, model_s2, model_s3, thresh, cap=args.cap
    )
    s1_ids = [r.eid for r in t1]
    write_tsv(args.out_dir / "matching_results.tsv", s1_ids, matches, "matched_entity_ids")
    write_tsv(args.out_dir / "candidate_pairs.tsv", s1_ids, cands, "candidate_entity_ids")

    n_match = sum(1 for v in matches.values() if v)
    n_links = sum(len(v) for v in matches.values())
    n_cand = sum(len(v) for v in cands.values())
    log(
        f"Done. S1={len(s1_ids):,} with_matches={n_match:,} links={n_links:,} "
        f"avg_cand={n_cand / max(len(s1_ids), 1):.2f} thresh={thresh:.3f}"
    )


if __name__ == "__main__":
    main()
