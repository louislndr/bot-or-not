#!/usr/bin/env python3
"""
Final inference script. One command, outputs submission file.

Usage:
    python run_detector.py --input dataset_posts_users_XX.json \
                           --output flagr.detections.en.txt \
                           --artifacts artifacts/

Completes in under 60 seconds for ~300-user datasets.
Requires no internet access, no API keys.
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.detector import score_dataset
from src.utils import load_dataset, write_detections, setup_logger

logger = setup_logger("run_detector")


def main():
    parser = argparse.ArgumentParser(description="Bot detection inference")
    parser.add_argument("--input", required=True, help="Path to input JSON dataset")
    parser.add_argument("--output", required=True, help="Output detections file path")
    parser.add_argument("--artifacts", default="artifacts/", help="Path to saved artifacts")
    args = parser.parse_args()

    t0 = time.time()

    # Load dataset
    logger.info(f"Loading dataset: {args.input}")
    dataset = load_dataset(args.input)
    raw_lang = dataset.get("lang", "en").lower().strip()
    lang = "fr" if raw_lang in ("fr", "french", "français", "francais") else "en"
    n_users = len(dataset.get("users", []))
    n_posts = len(dataset.get("posts", []))
    logger.info(f"Language: {lang} | Users: {n_users} | Posts: {n_posts}")

    # Score
    flagged_ids, all_probs, score_records = score_dataset(dataset, args.artifacts)

    # Write detections file
    write_detections(flagged_ids, args.output)
    logger.info(f"Detections written to: {args.output}")

    # Write scores CSV for inspection
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs", exist_ok=True)
    scores_path = os.path.join("outputs", f"scores_{ts}.csv")
    with open(scores_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "user_id", "bot_score", "flagged", "vetoed", "whitelisted",
            "hard_rule_fired", "corroborating_rules", "username",
            "z_score", "hashtag_rate", "link_rate",
        ])
        writer.writeheader()
        for rec in score_records:
            row = dict(rec)
            row["corroborating_rules"] = "|".join(rec.get("corroborating_rules", []))
            writer.writerow(row)
    logger.info(f"Score details written to: {scores_path}")

    # Print summary to stdout
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"BOT DETECTION RESULTS — {lang.upper()}")
    print(f"{'='*60}")
    print(f"Total users:    {n_users}")
    print(f"Flagged bots:   {len(flagged_ids)} ({100*len(flagged_ids)/max(n_users,1):.1f}%)")
    print(f"Elapsed:        {elapsed:.1f}s")
    n_vetoed = sum(1 for r in score_records if r.get("vetoed"))
    n_whitelisted = sum(1 for r in score_records if r.get("whitelisted"))
    n_hard = sum(1 for r in score_records if r.get("hard_rule_fired"))
    print(f"Hard-flagged (rules): {n_hard}")
    print(f"Whitelisted (human):  {n_whitelisted}")
    print(f"Vetoed (no evidence): {n_vetoed}")
    print(f"\nTop 10 highest-scored users:")
    print(f"{'rank':>4}  {'score':>6}  {'status':>10}  {'username':25}  {'rules'}")
    print("-" * 95)
    for i, rec in enumerate(score_records[:10], 1):
        if rec.get("hard_rule_fired"):
            status = "HARD-BOT"
        elif rec["flagged"]:
            status = "BOT"
        elif rec.get("whitelisted"):
            status = "HUMAN"
        elif rec.get("vetoed"):
            status = "VETOED"
        else:
            status = ""
        rules_str = "|".join(rec.get("corroborating_rules", [])) or "-"
        print(
            f"{i:>4}  {rec['bot_score']:.4f}  {status:>10}  "
            f"{str(rec['username'])[:25]:25}  {rules_str}"
        )
    print(f"\nOutput file: {args.output}")
    print(f"Scores file: {scores_path}")


if __name__ == "__main__":
    main()
