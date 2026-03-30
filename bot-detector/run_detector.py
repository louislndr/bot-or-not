#!/usr/bin/env python3
"""
Final inference script. One command, outputs submission file.

Usage:
    python run_detector.py --input dataset_posts_users_XX.json \
                           --output myteam.detections.en.txt \
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
        writer = csv.DictWriter(
            f, fieldnames=["user_id", "bot_score", "flagged", "username", "z_score", "hashtag_rate", "link_rate"]
        )
        writer.writeheader()
        writer.writerows(score_records)
    logger.info(f"Score details written to: {scores_path}")

    # Print summary to stdout
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"BOT DETECTION RESULTS — {lang.upper()}")
    print(f"{'='*60}")
    print(f"Total users:    {n_users}")
    print(f"Flagged bots:   {len(flagged_ids)} ({100*len(flagged_ids)/max(n_users,1):.1f}%)")
    print(f"Elapsed:        {elapsed:.1f}s")
    print(f"\nTop 10 highest-scored users:")
    print(f"{'rank':>4}  {'score':>6}  {'flagged':>7}  {'username':30}  {'z_score':>8}  {'hashtag_rate':>12}")
    print("-" * 80)
    for i, rec in enumerate(score_records[:10], 1):
        flag_str = "BOT" if rec["flagged"] else "   "
        print(
            f"{i:>4}  {rec['bot_score']:.4f}  {flag_str:>7}  "
            f"{str(rec['username'])[:30]:30}  {rec['z_score']:>8.2f}  "
            f"{rec['hashtag_rate']:>12.2f}"
        )
    print(f"\nOutput file: {args.output}")
    print(f"Scores file: {scores_path}")


if __name__ == "__main__":
    main()
