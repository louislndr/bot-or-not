"""
Utility functions: I/O helpers, scoring, logging.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


def competition_score(tp: int, fp: int, fn: int) -> float:
    """Exact competition scoring: +2 per TP, -2 per FN, -6 per FP."""
    return 2 * tp - 6 * fp - 2 * fn


def expected_score_at_threshold(y_true, y_prob, threshold: float) -> float:
    """Compute competition score at a given probability threshold."""
    tp = fp = fn = 0
    for label, prob in zip(y_true, y_prob):
        predicted = 1 if prob >= threshold else 0
        if predicted == 1 and label == 1:
            tp += 1
        elif predicted == 1 and label == 0:
            fp += 1
        elif predicted == 0 and label == 1:
            fn += 1
    return competition_score(tp, fp, fn)


def find_optimal_threshold(y_true, y_prob, min_t: float = 0.05, max_t: float = 0.95, step: float = 0.01):
    """Scan thresholds and return (best_threshold, best_score, all_results)."""
    results = []
    t = min_t
    while t <= max_t + 1e-9:
        score = expected_score_at_threshold(y_true, y_prob, t)
        results.append((round(t, 3), score))
        t += step
    best_t, best_score = max(results, key=lambda x: x[1])
    return best_t, best_score, results


def _convert_vendor_format(raw: list) -> dict:
    """
    Convert a raw Twitter API tweet list (Indiana University / vendor format)
    to the competition dataset format expected by score_dataset.

    Each entry in the list is {"created_at": ..., "user": {...twitter user object...}}.
    Tweet text is not present in this format, so content/temporal features will be 0.
    Profile features (username, description, follower counts, etc.) work normally.
    """
    from datetime import datetime, timezone
    seen_users = {}
    for entry in raw:
        u = entry.get("user", {})
        uid = str(u.get("id", ""))
        if not uid or uid in seen_users:
            continue
        # Parse account creation date to generate a synthetic timestamp for the "post"
        created_at_str = entry.get("created_at", "")
        try:
            ts = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S +0000 %Y")
            ts = ts.replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            ts = "2019-01-01T00:00:00+00:00"

        seen_users[uid] = {
            "user": {
                "id": uid,
                "username": u.get("screen_name", ""),
                "name": u.get("name", ""),
                "description": u.get("description", "") or "",
                "location": u.get("location", "") or "",
                "tweet_count": u.get("statuses_count", 0),
                "z_score": 0.0,
            },
            "ts": ts,
            "lang": u.get("lang", "en") or "en",
        }

    users = [v["user"] for v in seen_users.values()]

    # Determine dataset language from majority of user langs
    lang_counts: dict = {}
    for v in seen_users.values():
        l = v["lang"]
        lang_counts[l] = lang_counts.get(l, 0) + 1
    dataset_lang = max(lang_counts, key=lang_counts.get) if lang_counts else "en"
    dataset_lang = "fr" if dataset_lang in ("fr",) else "en"

    return {
        "lang": dataset_lang,
        "metadata": {},
        "users": users,
        "posts": [],  # no tweet text available in this format
    }


def load_dataset(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    # Detect vendor/Indiana University format: top-level list of tweet objects with a "user" key
    if isinstance(raw, list) and raw and "user" in raw[0]:
        return _convert_vendor_format(raw)
    return raw


def load_bot_ids(path: str) -> set:
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_detections(user_ids: list, path: str) -> None:
    """Write one user ID per line."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for uid in user_ids:
            f.write(str(uid) + "\n")


def precision_recall_f1(tp: int, fp: int, fn: int):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1
