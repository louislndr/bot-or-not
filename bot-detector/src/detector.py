"""
Inference logic: load model artifacts, score users, apply threshold.
Used by run_detector.py.
"""
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.features import build_feature_matrix
from src.model import load_artifacts, features_to_matrix, clip_outliers
from src.utils import setup_logger

logger = setup_logger(__name__)

# ML scores above this are flagged unconditionally — no veto needed.
# Set equal to the model threshold so the veto only applies to sub-threshold scores.
HIGH_CONFIDENCE_THRESHOLD = 0.65


def _is_obvious_human(feat: dict) -> bool:
    """
    Returns True if the account shows unambiguous human signals.
    ALL four conditions must hold — intentionally strict so few accounts qualify.
    """
    return (
        feat.get("posting_hour_entropy", 0) > 4.0
        and feat.get("lexical_diversity", 0) > 0.75
        and feat.get("hashtag_sequence_repeat", 1.0) < 0.05
        and feat.get("cross_account_dup_score", 1.0) < 0.02
    )


def _hard_bot_rules(feat: dict) -> List[str]:
    """
    Returns rule names that fire for obviously bot accounts.
    Flagged unconditionally regardless of ML score.
    """
    fired = []
    if feat.get("compression_ratio", 1.0) < 0.35 and feat.get("hashtag_sequence_repeat", 0) > 0.6:
        fired.append("high_compression_hashtag_repeat")
    if feat.get("cross_account_dup_score", 0) > 0.25:
        fired.append("coordinated_campaign")
    if feat.get("fixed_gap_score", 0) > 0.75 and feat.get("inter_post_gap_cv", 1.0) < 0.15:
        fired.append("machine_clock_timing")
    if feat.get("near_duplicate_ratio", 0) > 0.55:
        fired.append("extreme_near_duplicates")
    return fired


def _corroborating_rules(feat: dict) -> List[str]:
    """
    Returns the names of rules that fire for this user.
    Used as corroborating evidence for borderline ML flags (threshold ≤ score < 0.72).
    Any single rule firing is enough to confirm the flag.
    """
    fired = []

    if feat.get("hashtag_sequence_repeat", 0) > 0.4:
        fired.append("hashtag_sequence_repeat")

    if feat.get("cross_account_dup_score", 0) > 0.1:
        fired.append("cross_account_coordination")

    if feat.get("near_duplicate_ratio", 0) > 0.35:
        fired.append("near_duplicate_posts")

    if feat.get("fixed_gap_score", 0) > 0.6 and feat.get("inter_post_gap_cv", 1.0) < 0.3:
        fired.append("machine_timing")

    if feat.get("compression_ratio", 1.0) < 0.45:
        fired.append("repetitive_content")

    if feat.get("exact_duplicate_ratio", 0) > 0.2:
        fired.append("exact_duplicates")

    if feat.get("template_score", 0) > 0.2:
        fired.append("template_posting")

    if feat.get("cross_account_near_dup_score", 0) > 0.1:
        fired.append("cross_account_near_dup")

    return fired


def score_dataset(
    dataset: dict,
    artifacts_dir: str,
) -> Tuple[List[str], List[float], List[dict]]:
    """
    Score all users in a dataset.
    Returns (flagged_user_ids, all_scores_list, score_records).
    score_records is a list of dicts: {user_id, bot_score, flagged, username, z_score,
                                        vetoed, corroborating_rules}.
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
    X_scaled = clip_outliers(scaler.transform(X))

    # Score
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[:, 1]
    else:
        raw = model.decision_function(X_scaled)
        mn, mx = raw.min(), raw.max()
        probs = (raw - mn) / (mx - mn) if mx > mn else np.full_like(raw, 0.5)

    # Build records with veto logic
    score_records = []
    flagged_ids = []
    n_vetoed = 0
    n_whitelisted = 0
    n_hard_flagged = 0

    for uid, prob, fd in zip(user_ids, probs, feature_dicts):
        prob = float(prob)

        # ── Hard bot rules: flag unconditionally, skip everything else ────────
        # Only fire if ML score is at least somewhat suspicious (>= 0.25).
        # This prevents hard rules from overriding a confident human verdict.
        hard_rules = _hard_bot_rules(fd) if prob >= 0.35 else []
        if hard_rules:
            flagged_ids.append(uid)
            n_hard_flagged += 1
            score_records.append({
                "user_id": uid,
                "bot_score": prob,
                "flagged": True,
                "vetoed": False,
                "whitelisted": False,
                "hard_rule_fired": True,
                "corroborating_rules": hard_rules,
                "username": fd.get("_username", ""),
                "z_score": fd.get("_z_score_raw", 0.0),
                "hashtag_rate": fd.get("hashtag_rate", 0.0),
                "link_rate": fd.get("link_rate", 0.0),
            })
            continue

        # ── Whitelist: obvious humans — never flag ────────────────────────────
        if _is_obvious_human(fd):
            n_whitelisted += 1
            score_records.append({
                "user_id": uid,
                "bot_score": prob,
                "flagged": False,
                "vetoed": False,
                "whitelisted": True,
                "hard_rule_fired": False,
                "corroborating_rules": [],
                "username": fd.get("_username", ""),
                "z_score": fd.get("_z_score_raw", 0.0),
                "hashtag_rate": fd.get("hashtag_rate", 0.0),
                "link_rate": fd.get("link_rate", 0.0),
            })
            continue

        # ── ML scoring with veto layer ────────────────────────────────────────
        rules = _corroborating_rules(fd)

        if prob >= HIGH_CONFIDENCE_THRESHOLD:
            flagged = True
            vetoed = False
        elif prob >= threshold:
            if rules:
                flagged = True
                vetoed = False
            else:
                flagged = False
                vetoed = True
                n_vetoed += 1
        else:
            flagged = False
            vetoed = False

        if flagged:
            flagged_ids.append(uid)

        score_records.append({
            "user_id": uid,
            "bot_score": prob,
            "flagged": flagged,
            "vetoed": vetoed,
            "whitelisted": False,
            "hard_rule_fired": False,
            "corroborating_rules": rules,
            "username": fd.get("_username", ""),
            "z_score": fd.get("_z_score_raw", 0.0),
            "hashtag_rate": fd.get("hashtag_rate", 0.0),
            "link_rate": fd.get("link_rate", 0.0),
        })

    # Sort by score descending
    score_records.sort(key=lambda r: -r["bot_score"])

    logger.info(
        f"Total users: {len(user_ids)} | Flagged: {len(flagged_ids)} "
        f"({100*len(flagged_ids)/max(len(user_ids),1):.1f}%) | "
        f"Hard-flagged: {n_hard_flagged} | Whitelisted: {n_whitelisted} | Vetoed: {n_vetoed}"
    )

    return flagged_ids, probs.tolist(), score_records
