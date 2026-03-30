# Bot-or-Not: Bot Detection System

A machine-learning bot detection system for the [Bot or Not Competition](https://forms.gle/fpe9cFkgn1e6BYEG8). Consumes a time-ordered corpus of tweets (`dataset.posts&users.json`) and outputs the user IDs it believes are bots.

## How It Works

### Pipeline Overview

```
dataset.posts&users.json
        │
        ▼
  Feature Extraction  ──►  62 per-user features (profile, hashtag, text,
  (src/features.py)         temporal, structural, semantic, links, volume)
        │
        ▼
  ML Model Inference  ──►  GradientBoosting (EN) / LogisticRegression (FR)
  (src/detector.py)         trained on practice datasets
        │
        ▼
  Threshold Filter    ──►  Threshold calibrated to maximize competition score
                            (+2 TP, −2 FN, −6 FP)
        │
        ▼
  [team_name].detections.[lang].txt   (one user_id per line)
```

### Feature Groups (62 features total)

| Group | # Features | Key Signals |
|-------|-----------|-------------|
| Profile | 11 | Digit ratio in username, control chars in name, empty description, null location |
| Hashtag | 7 | Hashtag rate per post, diversity, sequence repetition — **strongest English signal** |
| Text/Lexical | 16 | Lexical diversity, exclamation rate, compression ratio, duplicate ratio |
| Links | 3 | Link rate, mention rate, URL always at end of post |
| Temporal | 11 | Posting hour entropy, inter-post gap CV, burst score, fixed-gap score |
| Structural | 6 | Near-duplicate ratio, template score, cross-account coordination score |
| Semantic | 2 | Generic phrase count, quote tweet ratio |
| Volume | 4 | Tweet count, z-score, near-100-tweet-cap flag |

### Scoring Objective

The threshold is not tuned for F1 or accuracy. It is tuned to directly maximise:

```
score = 2×TP − 2×FN − 6×FP
```

Since flagging a human as a bot costs 3× more than missing a bot, the decision threshold is set conservatively (≥ 0.65 for English).

---

## Setup

```bash
cd bot-detector
pip install -r requirements.txt
```

**Python 3.9+ required.** Key dependencies: `scikit-learn`, `numpy`, `pandas`, `scipy`.

---

## Usage

### Step 1 — Train (run once on practice datasets)

```bash
cd bot-detector
python train.py \
  --en-dataset data/dataset_posts_users_30.json \
  --en-bots    data/dataset_bots_30.txt \
  --fr-dataset data/dataset_posts_users_31.json \
  --fr-bots    data/dataset_bots_31.txt \
  --output-dir artifacts/
```

Pre-trained artifacts are already committed in `artifacts/` — you can skip this step and go straight to inference.

### Step 2 — Run Detector (on final evaluation dataset)

```bash
cd bot-detector
python run_detector.py \
  --input  /path/to/dataset.posts_users.json \
  --output teamname.detections.en.txt \
  --artifacts artifacts/
```

The script auto-detects language from the dataset's `lang` field and loads the correct model. It completes in under 60 seconds for ~300-user datasets.

**Output:** One user ID (UUID) per line in the detections file — same format as `dataset.bots.txt`.

---

## Differences Between French and English Detection

The French and English models are **trained and calibrated separately**. They differ in the following ways:

### English Model
- **Algorithm:** GradientBoostingClassifier
- **Decision threshold:** 0.65 (conservative floor applied)
- **Dominant features:** `hashtag_count_total`, `hashtag_diversity`, `posting_hour_entropy`
- **Why:** English bots in the training data heavily rely on hashtag campaigns. High hashtag volume combined with low diversity (repeating the same tags) is a near-definitive English bot signal.

### French Model
- **Algorithm:** LogisticRegression
- **Decision threshold:** 0.87 (higher — fewer training samples so false-positive risk is elevated)
- **Dominant features:** `exclamation_rate`, `quote_tweet_ratio`, `abs_z_score`
- **Why:** French bots in the training data show different behavioural patterns — high exclamation usage, heavy quote-tweeting, and unusually high/low activity relative to dataset average. Hashtag signals are weaker in French.

### Common Patterns (both languages)
- Temporal irregularity: very regular posting gaps or extreme burst posting
- Template posting: near-duplicate content across multiple posts
- Cross-account coordination: multiple accounts posting identical or near-identical text
- Profile anomalies: auto-generated usernames, empty descriptions

---

## Project Structure

```
bot-detector/
├── src/
│   ├── features.py     # Feature extraction (62 features per user)
│   ├── model.py        # Training, CV, threshold calibration
│   ├── detector.py     # Inference pipeline
│   └── utils.py        # Scoring, I/O helpers
├── data/               # Practice datasets (JSON + bot labels)
├── artifacts/          # Saved models, scalers, thresholds
├── outputs/            # Validation reports, confusion matrices
├── train.py            # Training entrypoint
└── run_detector.py     # Inference entrypoint
```

---

## Human Detection Note

The system runs fully automated — no human review is performed on the output. The final submission is the direct output of `run_detector.py`.
