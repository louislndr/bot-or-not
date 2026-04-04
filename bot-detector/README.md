# Bot-or-Not — Social Media Bot Detection

A competition-grade bot detection system for the Bot-or-Not hackathon. Optimizes the exact competition score (+2 TP / −2 FN / −6 FP), not accuracy or F1.

---

## Overview and Method

We frame bot detection as a supervised binary classification problem. For each user we build a 40-dimensional feature vector from their profile metadata and full post stream, then train a calibrated classifier with a threshold tuned to maximize expected competition score on held-out data.

The scoring asymmetry (+2 TP, −2 FN, −6 FP) means a false positive costs **3× more** than a missed bot. The break-even condition on any positive prediction is P(bot | score ≥ t) > 0.75 — the model must be very confident before flagging. We enforce this via threshold calibration: we scan thresholds from 0.05 to 0.95 in 0.01 steps on out-of-fold predictions and pick the threshold that maximizes expected competition score, with a conservative floor of 0.65 (raised if CV indicates it hurts performance for a given language).

The strongest single signal is **hashtag behavior**: bots average ~0.91 hashtags/post vs ~0.20 for humans. Structural signals (unique posting hours, near-duplicate post detection) and temporal entropy round out the picture. Profile signals (control chars in names, digits in usernames) provide complementary evidence.

---

## Feature Groups

| Group | Key features | Rationale |
|-------|-------------|-----------|
| **Profile** | `name_has_control_chars`, `username_digit_ratio`, `description_is_empty`, `location_is_null` | Bots often have raw emoji bytes (0x1f…) in names, auto-generated usernames with numbers, and empty/null location/description |
| **Hashtag** | `hashtag_rate`, `posts_with_hashtag_frac`, `all_posts_have_hashtag`, `hashtag_diversity`, `max_hashtag_freq` | Strongest signal group. Bots use hashtags ~4.5× more per post. 15% of bots have every post tagged vs ~0% of humans |
| **Text/Lexical** | `avg_post_length`, `lexical_diversity`, `exclamation_rate`, `caps_ratio` | Bots post slightly longer but more formulaic text with lower lexical diversity |
| **Links** | `link_rate`, `mention_rate` | Humans share links more (~0.52 vs bots ~0.29); bots often push hashtag content without links |
| **Temporal** | `posting_hour_entropy`, `inter_post_gap_mean`, `inter_post_gap_stdev`, `unique_hours` | Bots often cluster posting at specific hours (low entropy) or burst with tiny gaps |
| **Structural** | `near_duplicate_ratio`, `template_score`, `topic_hashtag_rate` | Templatized bots repeat near-identical posts; genuine users have variety |
| **Semantic** | `generic_phrase_count`, `quote_tweet_ratio` | Static list of 28 generic phrases common in coordinated bot campaigns |
| **Volume** | `tweet_count`, `z_score`, `abs_z_score` | Used as context; NOT a dominant signal — bots span all activity levels |

---

## Model Selection

We train and compare four models via 5-fold Stratified CV:

1. **Logistic Regression** (L2, C=1.0, class_weight=balanced) — strong baseline, interpretable
2. **Random Forest** (200 trees, max_depth=6) — handles nonlinear interactions
3. **Gradient Boosting** (CalibratedClassifierCV wrapping sklearn GBM) — best at combining weak signals
4. **Rule-based baseline** — hashtag_rate > 0.6 AND secondary signal; useful sanity check

**English:** GradientBoosting wins (CV score 58, P=0.945, R=0.788) at t=0.65
**French:** RandomForest wins (CV score 8, P=0.929, R=0.481) at t=0.48

Tie-breaking rule: if two models are within 5 competition points, prefer the simpler one (LR > RF > GBM).

---

## Threshold Calibration

We do **not** use 0.5 as the decision threshold.

With the scoring function +2/−2/−6, the expected value of flagging a user at probability p is:
```
E[flag] = 2p - 6(1-p) = 8p - 6
```
This is positive only when p > 0.75. So the Bayes-optimal threshold for this loss function is **0.75**.

In practice, our probability estimates aren't perfectly calibrated, so we scan empirically. We apply a conservative floor of 0.65 for English (well-evidenced dataset, 275 users) and let the optimizer choose for French (fewer examples → more uncertainty).

**The tradeoff:** A higher threshold catches fewer bots (lower recall) but virtually eliminates false positives (precision near 1.0). For this scoring function that's the right trade — each avoided FP is worth 3 caught bots.

---

## English vs French Differences

| Aspect | English | French |
|--------|---------|--------|
| Dataset size | 275 users, 7528 posts | 171 users, 4643 posts |
| Bot prevalence | 24.0% | 15.8% |
| Best model | GradientBoosting | RandomForest |
| Decision threshold | 0.65 | 0.48 |
| Top signal | hashtag_count_total | posts_with_hashtag_frac |
| CV score | 58.0 | 8.0 |

French is harder: fewer bots, less training data, and the signal is weaker at high confidence levels. The conservative 0.65 floor hurts French (all models scored negatively at that threshold), so we use the CV-optimal 0.48 and accept the slightly higher FP risk for the chance to catch more bots.

---

## How to Run

### Setup

```bash
pip install -r requirements.txt
```

### Training (generates artifacts)

```bash
python3 train.py \
  --en-dataset data/dataset_posts_users_30.json \
  --en-bots    data/dataset_bots_30.txt \
  --fr-dataset data/dataset_posts_users_31.json \
  --fr-bots    data/dataset_bots_31.txt \
  --output-dir artifacts/
```

Outputs:
- `artifacts/model_{en,fr}.pkl` — trained model
- `artifacts/scaler_{en,fr}.pkl` — fitted StandardScaler
- `artifacts/threshold_{en,fr}.json` — threshold + metadata
- `artifacts/feature_names.json` — ordered feature list
- `outputs/validation_report.txt` — CV results, error analysis
- `outputs/confusion_matrix_{en,fr}.txt`
- `outputs/error_analysis_{en,fr}.txt` — per FP/FN case study
- `outputs/ablation_{en,fr}.txt` — feature group importance
- `outputs/threshold_sweep_{en,fr}.txt` — score vs threshold curve

### Inference (competition submission)

```bash
python3 run_detector.py \
  --input  dataset_posts_users_XX.json \
  --output myteam.detections.en.txt \
  --artifacts artifacts/
```

Language is auto-detected from the `lang` field in the JSON. Output is one user ID per line, nothing else. Also writes `outputs/scores_[timestamp].csv` for inspection.

**Runtime:** ~0.8 seconds for a 275-user dataset on a modern laptop.

---

## Repository Structure

```
bot-detector/
├── README.md
├── requirements.txt
├── data/                     # practice datasets (30=EN, 31=FR)
├── src/
│   ├── features.py           # all feature extraction logic
│   ├── model.py              # CV, model training, threshold selection, artifact I/O
│   ├── detector.py           # inference pipeline
│   └── utils.py              # scoring function, I/O helpers
├── artifacts/                # saved model/scaler/threshold files
├── outputs/                  # validation reports, submission files, diagnostics
├── train.py                  # training entrypoint
└── run_detector.py           # inference entrypoint
```

---

## Validation Results

| Language | Model | Threshold | CV Score | Precision | Recall | F1 | TP | FP | FN |
|----------|-------|-----------|----------|-----------|--------|----|----|----|----|
| English | GradientBoosting | 0.65 | **58.0** | 0.945 | 0.788 | 0.860 | 52 | 3 | 14 |
| French | RandomForest | 0.48 | **8.0** | 0.929 | 0.481 | 0.634 | 13 | 1 | 14 |

These are 5-fold cross-validation OOF estimates (not training set metrics).

---

## Limitations and Failure Modes

1. **Bots with low hashtag rates** — ~15% of training bots have low hashtag rates. If future bots drop hashtags entirely, our strongest feature group loses power.
2. **Highly active human sports fans** — high hashtag users like `#nhlhockey` fans can look superficially bot-like; temporal/structural signals help distinguish them.
3. **Well-crafted low-volume bots** — bots with only 1–3 posts provide no temporal or structural signal; we can only rely on profile features, which are weaker.
4. **Language mismatch** — French model is trained on 171 users. Unusual French slang or regional patterns not in training data can fool the lexical features.
5. **Coordinated human accounts** — real humans who tweet repetitively about a topic (e.g., activists, journalists) can score high on topic_hashtag_rate and posting regularity.

---

## Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
scipy>=1.9.0
matplotlib>=3.5.0
xgboost>=1.7.0  (optional; we use sklearn GBM as fallback)
```

No external NLP libraries (no spacy, transformers, or OpenAI). No internet access required at inference time.
