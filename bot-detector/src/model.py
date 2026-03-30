"""
Model training, cross-validation, threshold calibration, and artifact saving.
Optimizes the exact competition score: 2*TP - 2*FN - 6*FP.
"""
import os
import pickle
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.utils import (
    competition_score,
    find_optimal_threshold,
    precision_recall_f1,
    save_json,
    load_json,
    setup_logger,
)

warnings.filterwarnings("ignore")
logger = setup_logger(__name__)

# ─── feature groups for ablation ─────────────────────────────────────────────
FEATURE_GROUPS = {
    "profile": [
        "username_has_digits", "username_digit_ratio", "username_length",
        "username_entropy", "name_has_control_chars", "name_length",
        "description_is_empty", "description_length", "description_generic_score",
        "location_is_null", "name_username_similarity",
    ],
    "volume": ["tweet_count", "z_score", "abs_z_score", "near_tweet_cap"],
    "hashtag": [
        "hashtag_rate", "posts_with_hashtag_frac", "all_posts_have_hashtag",
        "hashtag_diversity", "max_hashtag_freq", "hashtag_count_total",
        "hashtag_sequence_repeat",
    ],
    "text": [
        "avg_post_length", "stdev_post_length", "sentence_length_cv",
        "avg_word_count", "lexical_diversity", "moving_avg_lex_diversity",
        "vocab_growth_rate", "exclamation_rate", "question_rate",
        "emoji_rate", "caps_ratio", "exact_duplicate_ratio",
        "compression_ratio", "lang_mismatch_rate", "lang_flip_rate",
        "mention_pattern_repeat",
    ],
    "links": ["link_rate", "mention_rate", "url_always_at_end"],
    "temporal": [
        "unique_hours", "posting_hour_entropy", "sleep_hour_frac",
        "burst_score", "inter_post_gap_mean", "inter_post_gap_stdev",
        "inter_post_gap_min", "inter_post_gap_cv", "fixed_gap_score",
        "posts_in_first_hour", "time_span_minutes",
    ],
    "structural": [
        "near_duplicate_ratio", "template_score",
        "topic_hashtag_rate", "topic_focus_score",
        "cross_account_dup_score", "cross_account_near_dup_score",
    ],
    "semantic": ["quote_tweet_ratio", "generic_phrase_count"],
}

ALL_FEATURE_NAMES = [f for group in FEATURE_GROUPS.values() for f in group]

# Conservative threshold bump: if CV-optimal threshold < this, nudge up
CONSERVATIVE_THRESHOLD_FLOOR = 0.65


def _build_models() -> Dict[str, Any]:
    """Return model candidates."""
    models = {
        "logistic_regression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42
        ),
    }
    # Wrap gradient boosting in calibrated wrapper for probability output
    models["gradient_boosting"] = CalibratedClassifierCV(
        models["gradient_boosting"], method="isotonic", cv=3
    )
    return models


def _rule_based_predict(df: pd.DataFrame, hashtag_threshold: float = 0.6) -> np.ndarray:
    """
    Rule-based baseline: flag if hashtag_rate > threshold AND at least one
    secondary signal fires (name_has_control_chars, all_posts_have_hashtag,
    or high generic_phrase_count).
    """
    primary = df["hashtag_rate"] > hashtag_threshold
    secondary = (
        (df["name_has_control_chars"] == 1)
        | (df["all_posts_have_hashtag"] == 1)
        | (df["generic_phrase_count"] >= 2)
        | (df["link_rate"] < 0.15)
    )
    return (primary & secondary).astype(int).values


def features_to_matrix(feature_dicts: List[dict], feature_names: List[str]) -> np.ndarray:
    rows = []
    for fd in feature_dicts:
        rows.append([fd.get(f, 0.0) for f in feature_names])
    return np.array(rows, dtype=np.float64)


def run_cross_validation(
    feature_dicts: List[dict],
    labels: List[int],
    lang: str,
    n_splits: int = 5,
) -> Tuple[Dict, str, float]:
    """
    Run StratifiedKFold CV for all models. Returns (results dict, best_model_name, best_threshold).
    """
    X_full = features_to_matrix(feature_dicts, ALL_FEATURE_NAMES)
    y = np.array(labels)

    model_candidates = _build_models()
    results: Dict[str, Dict] = {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for model_name, model_proto in model_candidates.items():
        fold_scores = []
        fold_thresholds = []
        all_val_true = []
        all_val_prob = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y)):
            X_train, X_val = X_full[train_idx], X_full[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            import copy
            model = copy.deepcopy(model_proto)
            model.fit(X_train_s, y_train)

            if hasattr(model, "predict_proba"):
                val_probs = model.predict_proba(X_val_s)[:, 1]
            else:
                val_probs = model.decision_function(X_val_s)
                # min-max normalize to [0,1]
                mn, mx = val_probs.min(), val_probs.max()
                if mx > mn:
                    val_probs = (val_probs - mn) / (mx - mn)
                else:
                    val_probs = np.full_like(val_probs, 0.5)

            t_opt, score_opt, _ = find_optimal_threshold(y_val.tolist(), val_probs.tolist())
            fold_scores.append(score_opt)
            fold_thresholds.append(t_opt)
            all_val_true.extend(y_val.tolist())
            all_val_prob.extend(val_probs.tolist())

        # Compute competition score over all OOF predictions at best threshold
        oof_t, oof_score, threshold_curve = find_optimal_threshold(all_val_true, all_val_prob)

        # Apply conservative floor
        final_threshold = max(oof_t, CONSERVATIVE_THRESHOLD_FLOOR)

        tp = fp = fn = 0
        for label, prob in zip(all_val_true, all_val_prob):
            pred = 1 if prob >= final_threshold else 0
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 1:
                fn += 1

        prec, rec, f1 = precision_recall_f1(tp, fp, fn)

        results[model_name] = {
            "fold_scores": fold_scores,
            "mean_fold_score": float(np.mean(fold_scores)),
            "oof_score_at_optimal_t": oof_score,
            "oof_score_at_conservative_t": competition_score(tp, fp, fn),
            "optimal_threshold": oof_t,
            "conservative_threshold": final_threshold,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "threshold_curve": threshold_curve[:20],  # first 20 for reporting
        }

        logger.info(
            f"[{lang}] {model_name}: fold_scores={[round(s,1) for s in fold_scores]}"
            f" mean={np.mean(fold_scores):.1f}"
            f" oof_score={oof_score:.1f} (t={oof_t})"
            f" conservative_score={competition_score(tp, fp, fn):.1f} (t={final_threshold})"
            f" P={prec:.3f} R={rec:.3f} F1={f1:.3f}"
        )

    # Rule-based baseline
    df_full = pd.DataFrame(feature_dicts)
    for col in ALL_FEATURE_NAMES:
        if col not in df_full.columns:
            df_full[col] = 0.0

    rule_preds = _rule_based_predict(df_full)
    rb_tp = int(((rule_preds == 1) & (y == 1)).sum())
    rb_fp = int(((rule_preds == 1) & (y == 0)).sum())
    rb_fn = int(((rule_preds == 0) & (y == 1)).sum())
    rb_prec, rb_rec, rb_f1 = precision_recall_f1(rb_tp, rb_fp, rb_fn)
    rb_score = competition_score(rb_tp, rb_fp, rb_fn)

    results["rule_based"] = {
        "fold_scores": [rb_score],
        "mean_fold_score": rb_score,
        "oof_score_at_optimal_t": rb_score,
        "oof_score_at_conservative_t": rb_score,
        "optimal_threshold": None,
        "conservative_threshold": None,
        "precision": rb_prec,
        "recall": rb_rec,
        "f1": rb_f1,
        "tp": rb_tp, "fp": rb_fp, "fn": rb_fn,
    }
    logger.info(
        f"[{lang}] rule_based: score={rb_score:.1f}"
        f" P={rb_prec:.3f} R={rb_rec:.3f}"
    )

    # Select best: highest conservative score; tie-break by simplicity.
    # If all models score negatively at the conservative floor, fall back to the
    # optimal threshold for the best-scoring model (negative score < 0 expected score).
    simplicity_order = [
        "logistic_regression", "random_forest", "gradient_boosting", "rule_based"
    ]
    model_conservative_scores = {
        k: results[k]["oof_score_at_conservative_t"] for k in simplicity_order if k in results
    }
    model_optimal_scores = {
        k: results[k]["oof_score_at_optimal_t"] for k in simplicity_order if k in results
    }

    best_conservative = max(model_conservative_scores.values())
    best_optimal = max(model_optimal_scores.values())

    # Prefer conservative threshold if it yields a non-negative expected score.
    # If forcing t=0.65 gives negative score for ALL models, the floor is hurting
    # us — fall back to the CV-optimal threshold instead.
    use_conservative = best_conservative >= 0

    if use_conservative:
        score_map = model_conservative_scores
        threshold_key = "conservative_threshold"
    else:
        # Fall back to optimal threshold — it's better than guaranteed negative score
        score_map = model_optimal_scores
        threshold_key = "optimal_threshold"
        logger.warning(
            f"[{lang}] All conservative scores negative ({best_conservative}). "
            f"Using optimal threshold instead."
        )

    best_score_val = max(score_map.values())
    best_model = None
    for name in simplicity_order:
        if name in score_map:
            # Tie-break: within 5 competition points of best (absolute, not relative)
            if score_map[name] >= best_score_val - 5:
                best_model = name
                break

    # Fallback: pick the model with the absolute highest score
    if best_model is None:
        best_model = max(score_map, key=score_map.get)

    best_threshold = results[best_model][threshold_key]
    # Ensure threshold is valid
    if best_threshold is None:
        best_threshold = CONSERVATIVE_THRESHOLD_FLOOR

    # Annotate the winning model's result with the actual score at the chosen threshold
    actual_score_key = "oof_score_at_optimal_t" if not use_conservative else "oof_score_at_conservative_t"
    results[best_model]["actual_cv_score"] = results[best_model][actual_score_key]

    logger.info(f"[{lang}] Selected model: {best_model} (threshold={best_threshold})")

    return results, best_model, best_threshold


def run_ablation(
    feature_dicts: List[dict],
    labels: List[int],
    lang: str,
    model_name: str = "logistic_regression",
    n_splits: int = 5,
) -> Dict[str, float]:
    """
    For each feature group, re-run CV with that group removed and report score drop.
    """
    y = np.array(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    ablation: Dict[str, float] = {}

    # Baseline score with all features
    X_all = features_to_matrix(feature_dicts, ALL_FEATURE_NAMES)
    base_oof_true, base_oof_prob = _oof_probs(X_all, y, model_name, skf)
    _, base_score, _ = find_optimal_threshold(base_oof_true, base_oof_prob)
    ablation["_baseline"] = base_score

    for group_name, group_feats in FEATURE_GROUPS.items():
        remaining = [f for f in ALL_FEATURE_NAMES if f not in group_feats]
        X_ablated = features_to_matrix(feature_dicts, remaining)
        oof_true, oof_prob = _oof_probs(X_ablated, y, model_name, skf)
        _, score, _ = find_optimal_threshold(oof_true, oof_prob)
        ablation[group_name] = score
        drop = base_score - score
        logger.info(f"[{lang}] ablation remove {group_name}: score={score:.1f} (drop={drop:+.1f})")

    return ablation


def _oof_probs(X, y, model_name, skf):
    """Helper: get OOF probability predictions."""
    import copy
    models = _build_models()
    model_proto = models.get(model_name, models["logistic_regression"])
    oof_true, oof_prob = [], []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        model = copy.deepcopy(model_proto)
        model.fit(X_train_s, y_train)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val_s)[:, 1]
        else:
            probs = model.decision_function(X_val_s)
            mn, mx = probs.min(), probs.max()
            probs = (probs - mn) / (mx - mn) if mx > mn else np.full_like(probs, 0.5)
        oof_true.extend(y_val.tolist())
        oof_prob.extend(probs.tolist())
    return oof_true, oof_prob


def train_final_model(
    feature_dicts: List[dict],
    labels: List[int],
    model_name: str,
    threshold: float,
) -> Tuple[Any, StandardScaler]:
    """Train the selected model on ALL data. Returns (fitted_model, fitted_scaler)."""
    X = features_to_matrix(feature_dicts, ALL_FEATURE_NAMES)
    y = np.array(labels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = _build_models()
    import copy
    model = copy.deepcopy(models.get(model_name, models["logistic_regression"]))
    model.fit(X_scaled, y)
    return model, scaler


def save_artifacts(
    model, scaler, threshold: float, model_name: str, cv_score: float,
    lang: str, output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"model_{lang}.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(output_dir, f"scaler_{lang}.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    save_json(
        {"threshold": threshold, "model": model_name, "cv_score": cv_score},
        os.path.join(output_dir, f"threshold_{lang}.json"),
    )
    save_json(ALL_FEATURE_NAMES, os.path.join(output_dir, "feature_names.json"))
    logger.info(f"Saved artifacts for lang={lang} to {output_dir}")


def load_artifacts(lang: str, artifacts_dir: str):
    """Load model, scaler, and threshold for inference."""
    with open(os.path.join(artifacts_dir, f"model_{lang}.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(artifacts_dir, f"scaler_{lang}.pkl"), "rb") as f:
        scaler = pickle.load(f)
    threshold_info = load_json(os.path.join(artifacts_dir, f"threshold_{lang}.json"))
    feature_names = load_json(os.path.join(artifacts_dir, "feature_names.json"))
    return model, scaler, threshold_info["threshold"], feature_names


def get_feature_importances(model, feature_names: List[str]) -> List[Tuple[str, float]]:
    """Extract feature importances or coefficients from a fitted model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    elif hasattr(model, "calibrated_classifiers_"):
        # CalibratedClassifierCV: average importances from base estimators
        base = model.calibrated_classifiers_[0].estimator
        if hasattr(base, "feature_importances_"):
            importances = base.feature_importances_
        elif hasattr(base, "coef_"):
            importances = np.abs(base.coef_[0])
        else:
            return []
    else:
        return []

    pairs = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    return pairs
