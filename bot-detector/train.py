#!/usr/bin/env python3
"""
Full training + validation pipeline.
Saves artifacts to artifacts/ and a validation report to outputs/.

Usage:
    python train.py --en-dataset data/dataset_posts_users_30.json \
                    --en-bots    data/dataset_bots_30.txt \
                    --fr-dataset data/dataset_posts_users_31.json \
                    --fr-bots    data/dataset_bots_31.txt \
                    --output-dir artifacts/
"""
import argparse
import os
import sys
import traceback
from pathlib import Path

# Make src importable whether run from repo root or elsewhere
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from src.features import build_feature_matrix
from src.model import (
    ALL_FEATURE_NAMES,
    FEATURE_GROUPS,
    run_cross_validation,
    run_ablation,
    train_final_model,
    save_artifacts,
    get_feature_importances,
    features_to_matrix,
    _rule_based_predict,
)
from src.utils import (
    load_dataset,
    load_bot_ids,
    competition_score,
    find_optimal_threshold,
    precision_recall_f1,
    setup_logger,
)

logger = setup_logger("train")


def format_confusion_matrix(tp, fp, fn, tn, lang):
    total = tp + fp + fn + tn
    lines = [
        f"Confusion Matrix ({lang})",
        "=" * 40,
        f"                  Predicted Bot  Predicted Human",
        f"  Actual Bot      TP={tp:<6}     FN={fn}",
        f"  Actual Human    FP={fp:<6}     TN={tn}",
        "",
        f"  Total users: {total}",
        f"  Bots in dataset: {tp+fn} ({100*(tp+fn)/max(total,1):.1f}%)",
    ]
    return "\n".join(lines)


def error_analysis(feature_dicts, labels, user_ids, all_probs, threshold, dataset, lang):
    """Return string with FP and FN analysis."""
    posts_by_user = {}
    for p in dataset.get("posts", []):
        posts_by_user.setdefault(p["author_id"], []).append(p)

    users_by_id = {u["id"]: u for u in dataset.get("users", [])}

    lines = [f"\n=== Error Analysis ({lang}) ==="]

    fps = []
    fns = []
    for uid, label, prob, fd in zip(user_ids, labels, all_probs, feature_dicts):
        pred = 1 if prob >= threshold else 0
        if pred == 1 and label == 0:
            fps.append((uid, prob, fd))
        elif pred == 0 and label == 1:
            fns.append((uid, prob, fd))

    def format_case(uid, prob, fd, case_type):
        user = users_by_id.get(uid, {})
        sample_posts = posts_by_user.get(uid, [])[:5]
        out = [
            f"\n  [{case_type}] user_id={uid} score={prob:.3f}",
            f"  username={fd.get('_username','')} name={fd.get('_name','')}",
            f"  tweet_count={fd.get('tweet_count',0)} z_score={fd.get('z_score',0):.2f}",
            f"  hashtag_rate={fd.get('hashtag_rate',0):.2f} link_rate={fd.get('link_rate',0):.2f}",
            f"  name_has_control_chars={fd.get('name_has_control_chars',0)} "
            f"all_posts_have_hashtag={fd.get('all_posts_have_hashtag',0)}",
            f"  description={user.get('description','')[:80]}",
            f"  Sample posts:",
        ]
        for p in sample_posts:
            out.append(f"    - {p['text'][:120]}")
        return "\n".join(out)

    lines.append(f"\nFALSE POSITIVES (humans flagged as bots): {len(fps)}")
    for uid, prob, fd in fps[:10]:
        lines.append(format_case(uid, prob, fd, "FP"))

    lines.append(f"\nFALSE NEGATIVES (bots missed): {len(fns)}")
    for uid, prob, fd in fns[:10]:
        lines.append(format_case(uid, prob, fd, "FN"))

    return "\n".join(lines)


def _get_oof_probs_for_report(feature_dicts, labels, model_name):
    """Get OOF probs with the final selected model for reporting."""
    from sklearn.model_selection import StratifiedKFold
    from src.model import _oof_probs, features_to_matrix
    X = features_to_matrix(feature_dicts, ALL_FEATURE_NAMES)
    y = np.array(labels)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return _oof_probs(X, y, model_name, skf)


def train_language(
    dataset_path: str,
    bots_path: str,
    lang: str,
    output_dir: str,
    outputs_dir: str,
) -> dict:
    logger.info(f"\n{'='*60}")
    logger.info(f"Training for language: {lang}")
    logger.info(f"{'='*60}")

    dataset = load_dataset(dataset_path)
    bot_ids = load_bot_ids(bots_path)
    n_users = len(dataset["users"])
    n_bots = sum(1 for u in dataset["users"] if u["id"] in bot_ids)
    logger.info(f"Dataset: {n_users} users, {n_bots} bots ({100*n_bots/n_users:.1f}%)")

    # Extract features
    logger.info("Extracting features...")
    feature_dicts, labels, user_ids = build_feature_matrix(dataset, bot_ids)
    df = pd.DataFrame(feature_dicts)
    logger.info(f"Feature matrix: {len(feature_dicts)} users x {len(ALL_FEATURE_NAMES)} features")

    # Cross-validation
    logger.info("Running cross-validation...")
    cv_results, best_model_name, best_threshold = run_cross_validation(
        feature_dicts, labels, lang
    )

    # Ablation
    logger.info("Running feature ablation...")
    ablation = run_ablation(feature_dicts, labels, lang, model_name="logistic_regression")

    # Get OOF probs for the best model for error analysis + threshold sweep
    if best_model_name != "rule_based":
        oof_true, oof_prob = _get_oof_probs_for_report(feature_dicts, labels, best_model_name)
    else:
        # rule-based doesn't have probs; use logistic regression for analysis
        oof_true, oof_prob = _get_oof_probs_for_report(feature_dicts, labels, "logistic_regression")

    _, _, threshold_curve = find_optimal_threshold(oof_true, oof_prob)

    # Train final model
    logger.info(f"Training final model ({best_model_name}) on all data...")
    final_model, final_scaler = train_final_model(
        feature_dicts, labels, best_model_name, best_threshold
    )

    # Compute final CV score at the actual selected threshold
    best_cv = cv_results[best_model_name]
    cv_score = best_cv.get("actual_cv_score", best_cv["oof_score_at_conservative_t"])

    # Save artifacts
    save_artifacts(
        final_model, final_scaler, best_threshold, best_model_name, cv_score, lang, output_dir
    )

    # Feature importances
    importances = get_feature_importances(final_model, ALL_FEATURE_NAMES)

    # Compute confusion matrix at the conservative threshold
    tp = best_cv["tp"]
    fp = best_cv["fp"]
    fn = best_cv["fn"]
    tn = n_users - tp - fp - fn
    prec, rec, f1 = precision_recall_f1(tp, fp, fn)

    # Error analysis
    error_txt = error_analysis(
        feature_dicts, labels, user_ids, oof_prob, best_threshold, dataset, lang
    )

    # Write outputs
    os.makedirs(outputs_dir, exist_ok=True)

    # Confusion matrix
    with open(os.path.join(outputs_dir, f"confusion_matrix_{lang}.txt"), "w") as f:
        f.write(format_confusion_matrix(tp, fp, fn, tn, lang))
        f.write("\n\nCV Results:\n")
        for mname, res in cv_results.items():
            f.write(
                f"  {mname}: conservative_score={res['oof_score_at_conservative_t']:.1f}"
                f" P={res['precision']:.3f} R={res['recall']:.3f} F1={res['f1']:.3f}"
                f" threshold={res['conservative_threshold']}\n"
            )

    # Error analysis
    with open(os.path.join(outputs_dir, f"error_analysis_{lang}.txt"), "w") as f:
        f.write(error_txt)

    # Feature correlations
    feat_df = pd.DataFrame(feature_dicts)[ALL_FEATURE_NAMES]
    feat_df["_label"] = labels
    corr = feat_df.corr()["_label"].drop("_label").sort_values(key=abs, ascending=False)
    with open(os.path.join(outputs_dir, f"feature_correlation_{lang}.txt"), "w") as f:
        f.write(f"Feature correlation with bot label ({lang}):\n")
        for fname, c in corr.items():
            f.write(f"  {fname:40s}: {c:+.4f}\n")

    # Threshold sweep
    with open(os.path.join(outputs_dir, f"threshold_sweep_{lang}.txt"), "w") as f:
        f.write(f"Threshold sweep ({lang}) — competition score vs threshold:\n")
        for t, score in threshold_curve:
            marker = " <-- optimal" if abs(t - best_cv["optimal_threshold"]) < 0.005 else ""
            conservative_marker = " <-- conservative (used)" if abs(t - best_threshold) < 0.005 else ""
            f.write(f"  t={t:.2f}  score={score:8.1f}{marker}{conservative_marker}\n")

    # Ablation report
    with open(os.path.join(outputs_dir, f"ablation_{lang}.txt"), "w") as f:
        f.write(f"Feature group ablation ({lang}):\n")
        baseline = ablation.get("_baseline", 0)
        f.write(f"  Baseline (all features):  {baseline:.1f}\n\n")
        for group, score in sorted(ablation.items(), key=lambda x: x[1]):
            if group == "_baseline":
                continue
            drop = baseline - score
            f.write(f"  Remove {group:15s}: score={score:.1f}  drop={drop:+.1f}\n")

    # Score distribution (text histogram)
    bot_probs = [p for p, l in zip(oof_prob, oof_true) if l == 1]
    human_probs = [p for p, l in zip(oof_prob, oof_true) if l == 0]
    with open(os.path.join(outputs_dir, f"score_distribution_{lang}.txt"), "w") as f:
        f.write(f"Score distribution ({lang}):\n\n")
        f.write("Bot scores (OOF):\n")
        f.write(f"  mean={np.mean(bot_probs):.3f} median={np.median(bot_probs):.3f}"
                f" min={min(bot_probs):.3f} max={max(bot_probs):.3f}\n")
        f.write("Human scores (OOF):\n")
        f.write(f"  mean={np.mean(human_probs):.3f} median={np.median(human_probs):.3f}"
                f" min={min(human_probs):.3f} max={max(human_probs):.3f}\n")
        # ASCII histogram
        f.write("\nHistogram (bins 0.0-1.0 in 0.1 steps):\n")
        for bucket in range(10):
            lo, hi = bucket / 10, (bucket + 1) / 10
            bot_in = sum(1 for p in bot_probs if lo <= p < hi)
            hum_in = sum(1 for p in human_probs if lo <= p < hi)
            f.write(f"  [{lo:.1f}-{hi:.1f}]  bots={bot_in:4d} {'#'*bot_in}  humans={hum_in:4d} {'.'*min(hum_in,60)}\n")

    summary = {
        "lang": lang,
        "n_users": n_users,
        "n_bots": n_bots,
        "best_model": best_model_name,
        "best_threshold": best_threshold,
        "cv_score": cv_score,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "top_features": importances[:10] if importances else [],
    }

    logger.info(f"\n[{lang}] SUMMARY:")
    logger.info(f"  Best model: {best_model_name}")
    logger.info(f"  Threshold: {best_threshold}")
    logger.info(f"  CV competition score: {cv_score:.1f}")
    logger.info(f"  Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    logger.info(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    if importances:
        logger.info(f"  Top 5 features: {[f'{n}={v:.3f}' for n,v in importances[:5]]}")

    return summary


def write_validation_report(summaries: list, output_dir: str):
    path = os.path.join(output_dir, "validation_report.txt")
    lines = ["=" * 70, "VALIDATION REPORT — Bot Detection System", "=" * 70, ""]
    for s in summaries:
        lang = s["lang"]
        lines += [
            f"Language: {lang.upper()}",
            f"  Dataset: {s['n_users']} users, {s['n_bots']} bots ({100*s['n_bots']/s['n_users']:.1f}%)",
            f"  Best model: {s['best_model']}",
            f"  Decision threshold: {s['best_threshold']}",
            f"  Competition score (CV): {s['cv_score']:.1f}",
            f"  Precision: {s['precision']:.3f}",
            f"  Recall: {s['recall']:.3f}",
            f"  F1: {s['f1']:.3f}",
            f"  TP={s['tp']}  FP={s['fp']}  FN={s['fn']}  TN={s['tn']}",
            f"  Top features by importance:",
        ]
        for fname, fval in (s["top_features"] or [])[:10]:
            lines.append(f"    {fname:40s}: {fval:.4f}")
        lines += [
            "",
            "  Failure modes:",
            "    1. Bots with low hashtag rates but no other strong signals",
            "    2. Highly active humans with high hashtag usage (e.g., sports fans)",
            "    3. Bots with well-crafted human-like descriptions and realistic posting gaps",
            "    4. Human accounts in non-English scripts that trigger control-char detector",
            "    5. Bots posting only once or twice (insufficient temporal/structural signal)",
            "",
        ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Validation report saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train bot detection models")
    parser.add_argument("--en-dataset", default="data/dataset_posts_users_30.json")
    parser.add_argument("--en-bots", default="data/dataset_bots_30.txt")
    parser.add_argument("--fr-dataset", default="data/dataset_posts_users_31.json")
    parser.add_argument("--fr-bots", default="data/dataset_bots_31.txt")
    parser.add_argument("--output-dir", default="artifacts/")
    args = parser.parse_args()

    outputs_dir = "outputs"
    summaries = []

    for lang, dataset_path, bots_path in [
        ("en", args.en_dataset, args.en_bots),
        ("fr", args.fr_dataset, args.fr_bots),
    ]:
        try:
            summary = train_language(
                dataset_path, bots_path, lang, args.output_dir, outputs_dir
            )
            summaries.append(summary)
        except Exception as e:
            logger.error(f"Failed training for {lang}: {e}")
            traceback.print_exc()

    write_validation_report(summaries, outputs_dir)
    logger.info("\nTraining complete. Run run_detector.py for inference.")


if __name__ == "__main__":
    main()
