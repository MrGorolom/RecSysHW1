"""Download, clean and chronologically split Amazon Video Games 2014."""

import ast
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parent
URLS = {
    "reviews_Video_Games_5.json.gz": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/reviews_Video_Games_5.json.gz",
    "meta_Video_Games.json.gz": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/meta_Video_Games.json.gz",
}


def download_data():
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        path = raw / name
        if not path.exists() or path.stat().st_size == 0:
            temp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, temp)
            temp.replace(path)
    return raw


def records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield ast.literal_eval(line)


def make_matrix(frame, shape, values=None):
    if values is None:
        values = np.ones(len(frame), dtype=np.float64)
    result = csr_matrix((values, (frame.user_idx, frame.item_idx)), shape=shape)
    result.sort_indices()
    return result


def prepare_data():
    raw = download_data()
    rows = [{k: row.get(k) for k in ("reviewerID", "asin", "overall", "unixReviewTime")}
            for row in records(raw / "reviews_Video_Games_5.json.gz")]
    reviews = pd.DataFrame(rows).rename(columns={"reviewerID": "user_id", "asin": "item_id",
                                               "overall": "rating", "unixReviewTime": "timestamp"})
    audit = {"raw_interactions": len(reviews), "missing": reviews.isna().sum().to_dict()}
    reviews["rating"] = pd.to_numeric(reviews.rating, errors="coerce")
    reviews["timestamp"] = pd.to_numeric(reviews.timestamp, errors="coerce")
    clean = reviews.dropna().copy()
    clean = clean[clean.rating.between(1, 5) & (clean.timestamp > 0)]
    audit["invalid_or_missing_removed"] = len(reviews) - len(clean)
    audit["duplicate_pairs_removed"] = int(clean.duplicated(["user_id", "item_id"]).sum())
    # Keep the first observation: later edits must not move an old pair into the future.
    clean = clean.sort_values(["timestamp", "user_id", "item_id"], kind="stable")
    clean = clean.drop_duplicates(["user_id", "item_id"], keep="first").reset_index(drop=True)
    audit.update(clean_interactions=len(clean), users=int(clean.user_id.nunique()),
                 items=int(clean.item_id.nunique()))
    assert 100_000 <= len(clean) <= 2_000_000
    assert audit["users"] >= 1000 and audit["items"] >= 1000
    # Date-level timestamps: all events on a boundary date go to the later period.
    t1 = int(clean.timestamp.quantile(0.80, interpolation="nearest"))
    t2 = int(clean.timestamp.quantile(0.90, interpolation="nearest"))
    train_raw = clean[clean.timestamp < t1].copy()
    valid_raw = clean[(clean.timestamp >= t1) & (clean.timestamp < t2)].copy()
    test_raw = clean[clean.timestamp >= t2].copy()
    # Fixed train catalog permits exactly identical candidates throughout the experiment.
    users = sorted(train_raw.user_id.unique())
    items = sorted(train_raw.item_id.unique())
    umap, imap = dict(zip(users, range(len(users)))), dict(zip(items, range(len(items))))

    def encode(frame):
        frame = frame[frame.user_id.isin(umap) & frame.item_id.isin(imap)].copy()
        frame["user_idx"] = frame.user_id.map(umap).astype(int)
        frame["item_idx"] = frame.item_id.map(imap).astype(int)
        return frame.reset_index(drop=True)

    train, valid, test = map(encode, (train_raw, valid_raw, test_raw))
    counts = train.groupby("user_idx").size()
    warm_users = counts[counts >= 3].index.to_numpy()
    # Eligibility depends on pre-validation history, never on future activity.
    eval_users = set(warm_users.tolist())
    audit["periods"] = {}
    for name, original, encoded in (("train", train_raw, train), ("validation", valid_raw, valid),
                                    ("test", test_raw, test)):
        audit["periods"][name] = {
            "raw_events": len(original), "warm_events": len(encoded),
            "cold_events_excluded": len(original) - len(encoded),
            "relevant_warm_events": int((encoded.rating >= 4).sum()),
            "start": pd.to_datetime(original.timestamp.min(), unit="s", utc=True).isoformat(),
            "end": pd.to_datetime(original.timestamp.max(), unit="s", utc=True).isoformat(),
        }
    for name, encoded in (("validation", valid), ("test", test)):
        audit["periods"][name]["evaluated_users"] = int(encoded.loc[
            (encoded.rating >= 4) & encoded.user_idx.isin(eval_users), "user_idx"].nunique())
    audit["train_users"] = len(users)
    audit["train_items"] = len(items)
    audit["eligible_history_users"] = len(warm_users)
    audit["split_boundaries"] = [pd.to_datetime(t, unit="s", utc=True).isoformat() for t in (t1, t2)]
    audit["sparsity"] = 1 - len(clean) / (audit["users"] * audit["items"])
    meta_rows = {}
    for row in records(raw / "meta_Video_Games.json.gz"):
        asin = row.get("asin")
        if asin in imap and asin not in meta_rows:
            paths = row.get("categories") or []
            cats = sorted({str(c) for p in paths for c in (p if isinstance(p, list) else [p])})
            # The root category is constant and adds no descriptive distinction.
            cats = [c for c in cats if c != "Video Games"]
            meta_rows[asin] = {"title": row.get("title", ""),
                               "brand": str(row.get("brand") or ""), "categories": cats}
    metadata = [meta_rows.get(i, {"title": "", "brand": "", "categories": []}) for i in items]
    audit["metadata"] = {"catalog_items": len(items), "matched": len(meta_rows),
                         "with_title": sum(bool(m["title"]) for m in metadata),
                         "with_brand": sum(bool(m["brand"]) for m in metadata),
                         "with_categories": sum(bool(m["categories"]) for m in metadata),
                         "distinct_categories": len({c for m in metadata for c in m["categories"]})}
    audit["sha256"] = {n: hashlib.sha256((raw / n).read_bytes()).hexdigest() for n in URLS}
    return {"clean": clean, "train": train, "validation": valid, "test": test,
            "metadata": metadata, "items": items, "users": users, "eligible_users": warm_users,
            "shape": (len(users), len(items)), "audit": audit}
