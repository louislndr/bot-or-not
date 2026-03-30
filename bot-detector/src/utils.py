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


def load_dataset(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
