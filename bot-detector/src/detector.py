"""
Inference logic: load model artifacts, score users, apply threshold.
Used by run_detector.py.
"""
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.features import build_feature_matrix
from src.model import load_artifacts, features_to_matrix
from src.utils import setup_logger

logger = setup_logger(__name__)


def score_dataset(
    dataset: dict,
    artifacts_dir: str,
) -> Tuple[List[str], List[float], List[dict]]:
    """
    Score all users in a dataset.
    Returns (flagged_user_ids, all_scores_list, score_records).
    score_records is a list of dicts: {user_id, bot_score, flagged, username, z_score}.
    """
    raw_lang = dataset.get("lang", "en").lower().strip()
    lang = "fr" if raw_lang in ("fr", "french", "français", "francais") else "en"
    logger.info(f"Scoring dataset lang={lang}")

    # Load artifacts
    model, scaler, threshold, feature_names = load_artifacts(lang, artifacts_dir)
    logger.info(f"Loaded model for lang={lang}, threshold={threshold}")

    # Extract features
    feature_dicts, _, user_ids = build_feature_matrix(dataset, bot_ids=None)

    # Build matrix
    X = features_to_matrix(feature_dicts, feature_names)
    X_scaled = scaler.transform(X)

    # Score
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[:, 1]
    else:
        raw = model.decision_function(X_scaled)
        mn, mx = raw.min(), raw.max()
        probs = (raw - mn) / (mx - mn) if mx > mn else np.full_like(raw, 0.5)

    # Build records
    score_records = []
    flagged_ids = []
    for uid, prob, fd in zip(user_ids, probs, feature_dicts):
        flagged = bool(prob >= threshold)
        if flagged:
            flagged_ids.append(uid)
        score_records.append({
            "user_id": uid,
            "bot_score": float(prob),
            "flagged": flagged,
            "username": fd.get("_username", ""),
            "z_score": fd.get("_z_score_raw", 0.0),
            "hashtag_rate": fd.get("hashtag_rate", 0.0),
            "link_rate": fd.get("link_rate", 0.0),
        })

    # Sort by score descending
    score_records.sort(key=lambda r: -r["bot_score"])

    logger.info(
        f"Total users: {len(user_ids)} | Flagged as bots: {len(flagged_ids)} "
        f"({100*len(flagged_ids)/max(len(user_ids),1):.1f}%)"
    )

    return flagged_ids, probs.tolist(), score_records
