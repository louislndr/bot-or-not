"""
Feature extraction: per-user aggregation from posts + profile.
All features operate at the user level (one vector per user).
"""
import bisect
import math
import random
import re
import statistics
import zlib
from datetime import datetime, timezone
from difflib import SequenceMatcher  # used only for name_username_similarity
from typing import Dict, List, Any, Optional, Set, Tuple

# ─── constants ────────────────────────────────────────────────────────────────

GENERIC_PHRASES = [
    "can't believe", "cant believe", "time flies", "still a masterpiece",
    "proud of our hustle", "make sure to", "don't miss", "dont miss",
    "check out", "mark your calendar", "spread the word", "join us",
    "exciting news", "big news", "stay tuned", "coming soon",
    "limited time", "don't forget", "dont forget", "follow for more",
    "like and share", "tag a friend", "drop a comment", "hit the link",
    "link in bio", "swipe up", "click the link", "subscribe now",
]

# Common filler words that appear in generic bot descriptions
GENERIC_DESC_WORDS = {
    "love", "life", "world", "music", "art", "sports", "fan", "news",
    "official", "account", "follow", "tweet", "sharing", "info",
    "updates", "latest", "daily", "passion", "dream", "happy",
    "positive", "vibes", "blessed", "grateful", "inspire", "motivation",
    "success", "hustle", "grind", "entrepreneur", "influencer",
}

HASHTAG_RE = re.compile(r"#\w+")
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F9FF"
    "\U00002700-\U000027BF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)


def _parse_dt(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f+00:00").replace(tzinfo=timezone.utc)


def _char_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _entropy(counts: List[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def _string_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: Dict[str, int] = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    return _entropy(list(counts.values()))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


def _temporal_coordination_score(
    timestamps: List[datetime],
    user_id: str,
    global_post_index: List[Tuple[float, str]],
    global_post_ts_keys: List[float],
    window_sec: int = 60,
) -> float:
    """
    Fraction of this user's posts that have ≥1 post from a different user within window_sec.
    High score = posting in lock-step with other accounts (coordination signal).
    """
    if not timestamps or not global_post_index:
        return 0.0
    coordinated = 0
    for ts in timestamps:
        t = ts.timestamp()
        lo = bisect.bisect_left(global_post_ts_keys, t - window_sec)
        hi = bisect.bisect_right(global_post_ts_keys, t + window_sec)
        for other_t, other_uid in global_post_index[lo:hi]:
            if other_uid != user_id:
                coordinated += 1
                break
    return coordinated / len(timestamps)


def extract_features(
    user: dict,
    posts: List[dict],
    dataset_metadata: dict,
    cross_account_text_counts: Optional[Dict[str, int]] = None,
    cross_account_post_sample: Optional[List[tuple]] = None,
    dataset_lang: str = "en",
    global_post_index: Optional[List[Tuple[float, str]]] = None,
    global_post_ts_keys: Optional[List[float]] = None,
    global_mention_counts: Optional[Dict[str, int]] = None,
) -> dict:
    """
    Extract all features for a single user.

    cross_account_text_counts: normalised_text → count of distinct users posting it.
    cross_account_post_sample: list of (user_id, word_set) for posts from OTHER users.
    """
    feat: Dict[str, Any] = {}

    # ── A: Profile features ───────────────────────────────────────────────────
    username: str = user.get("username") or ""
    name: str = user.get("name") or ""
    description: str = user.get("description") or ""
    location = user.get("location")

    feat["username_has_digits"] = int(any(c.isdigit() for c in username))
    feat["username_digit_ratio"] = (
        sum(1 for c in username if c.isdigit()) / len(username) if username else 0.0
    )
    feat["username_length"] = len(username)
    # High entropy = random-looking / auto-generated username
    feat["username_entropy"] = _string_entropy(username.lower())
    # Only flag actual invisible/control chars — NOT regular emojis (common in human names)
    feat["name_has_control_chars"] = int(
        any(
            ord(c) < 32                          # ASCII control chars
            or ord(c) in (0x200B, 0x200C, 0x200D, 0x200E, 0x200F)  # zero-width chars
            or (0x2060 <= ord(c) <= 0x206F)      # invisible formatting chars
            or (0xFFF0 <= ord(c) <= 0xFFFF)      # specials block
            for c in name
        )
    )
    feat["name_length"] = len(name)
    feat["description_is_empty"] = int(len(description.strip()) == 0)
    feat["description_length"] = len(description)

    # Generic/template description: fraction of words from bot-typical vocabulary
    desc_words = set(description.lower().split())
    feat["description_generic_score"] = (
        len(desc_words & GENERIC_DESC_WORDS) / len(desc_words) if desc_words else 0.0
    )

    feat["location_is_null"] = int(
        location is None or location == ":null:" or str(location).strip() == ""
    )
    feat["name_username_similarity"] = _char_similarity(name, username)

    # ── A: Volume / activity features ────────────────────────────────────────
    tweet_count = user.get("tweet_count", 0)
    feat["tweet_count"] = tweet_count
    z = user.get("z_score", 0.0) or 0.0
    feat["z_score"] = z
    feat["abs_z_score"] = abs(z)
    # Posting near 100-tweet cap (each user has 10–100 tweets in dataset)
    feat["near_tweet_cap"] = int(tweet_count >= 90)

    # ── Text preparation ──────────────────────────────────────────────────────
    texts = [p["text"] for p in posts] if posts else []
    n_posts = len(texts)
    post_langs = [p.get("lang", dataset_lang) for p in posts] if posts else []

    hashtags_per_post: List[List[str]] = [HASHTAG_RE.findall(t.lower()) for t in texts]
    all_hashtags: List[str] = [h for hs in hashtags_per_post for h in hs]

    # ── B: Hashtag features ───────────────────────────────────────────────────
    if n_posts > 0:
        counts_per_post = [len(hs) for hs in hashtags_per_post]
        feat["hashtag_rate"] = sum(counts_per_post) / n_posts
        feat["posts_with_hashtag_frac"] = sum(1 for c in counts_per_post if c > 0) / n_posts
        feat["all_posts_have_hashtag"] = int(all(c > 0 for c in counts_per_post))
    else:
        feat["hashtag_rate"] = 0.0
        feat["posts_with_hashtag_frac"] = 0.0
        feat["all_posts_have_hashtag"] = 0

    if all_hashtags:
        unique_ht = set(all_hashtags)
        feat["hashtag_diversity"] = len(unique_ht) / len(all_hashtags)
        counter: Dict[str, int] = {}
        for h in all_hashtags:
            counter[h] = counter.get(h, 0) + 1
        feat["max_hashtag_freq"] = max(counter.values()) / len(all_hashtags)
    else:
        feat["hashtag_diversity"] = 1.0
        feat["max_hashtag_freq"] = 0.0
    feat["hashtag_count_total"] = len(all_hashtags)

    # LLM-bot signals: always trail posts with hashtags + consistent count per post
    if n_posts > 0:
        # Fraction of posts where all hashtags appear in the last 30% of the text
        trailing_count = 0
        for t, hs in zip(texts, hashtags_per_post):
            if not hs:
                continue
            cutoff = int(len(t) * 0.70)
            # find position of first hashtag
            first_ht_pos = min((t.lower().find(h) for h in hs if t.lower().find(h) >= 0), default=len(t))
            if first_ht_pos >= cutoff:
                trailing_count += 1
        posts_with_ht = sum(1 for hs in hashtags_per_post if hs)
        feat["hashtag_trailing_rate"] = trailing_count / posts_with_ht if posts_with_ht > 0 else 0.0

        # Consistency of hashtag count per post (low stdev = bot uses same # of tags every time)
        ht_counts = [len(hs) for hs in hashtags_per_post if hs]
        if len(ht_counts) >= 3:
            ht_mean = statistics.mean(ht_counts)
            ht_stdev = statistics.stdev(ht_counts)
            feat["hashtag_count_consistency"] = 1.0 - min(ht_stdev / ht_mean, 1.0) if ht_mean > 0 else 0.0
        else:
            feat["hashtag_count_consistency"] = 0.0
    else:
        feat["hashtag_trailing_rate"] = 0.0
        feat["hashtag_count_consistency"] = 0.0

    # C: Repeated hashtag pattern in the same order (tuple repeat)
    if n_posts >= 2:
        ht_tuples = [tuple(hs) for hs in hashtags_per_post if hs]
        if ht_tuples:
            ht_seen: Dict[tuple, int] = {}
            for t in ht_tuples:
                ht_seen[t] = ht_seen.get(t, 0) + 1
            feat["hashtag_sequence_repeat"] = sum(
                1 for t in ht_tuples if ht_seen[t] > 1
            ) / len(ht_tuples)
        else:
            feat["hashtag_sequence_repeat"] = 0.0
    else:
        feat["hashtag_sequence_repeat"] = 0.0

    # ── C+D: Text / lexical features ─────────────────────────────────────────
    if n_posts > 0:
        lengths = [len(t) for t in texts]
        feat["avg_post_length"] = statistics.mean(lengths)
        feat["stdev_post_length"] = statistics.stdev(lengths) if n_posts > 1 else 0.0

        # D: Low syntactic variety — CV of sentence length
        feat["sentence_length_cv"] = (
            (statistics.stdev(lengths) / statistics.mean(lengths))
            if n_posts > 1 and statistics.mean(lengths) > 0
            else 0.0
        )

        words_per_post = [t.split() for t in texts]
        word_counts = [len(w) for w in words_per_post]
        feat["avg_word_count"] = statistics.mean(word_counts)

        all_words = [w.lower() for ws in words_per_post for w in ws]
        feat["lexical_diversity"] = (
            len(set(all_words)) / len(all_words) if all_words else 1.0
        )

        # D: Moving-average lexical diversity — computed on last 10 posts only
        # More robust than full-corpus TTR for high-volume users
        window_texts = texts[-10:]
        window_words = [w.lower() for t in window_texts for w in t.split()]
        feat["moving_avg_lex_diversity"] = (
            len(set(window_words)) / len(window_words) if window_words else 1.0
        )

        # D: Vocab growth rate — new words in 2nd half vs 1st half
        if n_posts >= 4:
            mid = n_posts // 2
            first_half_words = set(w.lower() for ws in words_per_post[:mid] for w in ws)
            second_half_words = set(w.lower() for ws in words_per_post[mid:] for w in ws)
            new_in_second = len(second_half_words - first_half_words)
            total_unique = len(first_half_words | second_half_words)
            feat["vocab_growth_rate"] = new_in_second / total_unique if total_unique > 0 else 0.0
        else:
            feat["vocab_growth_rate"] = 1.0

        feat["exclamation_rate"] = sum(t.count("!") for t in texts) / n_posts
        feat["question_rate"] = sum(t.count("?") for t in texts) / n_posts
        feat["emoji_rate"] = sum(1 for t in texts if EMOJI_RE.search(t)) / n_posts

        all_chars = "".join(texts)
        alpha_chars = [c for c in all_chars if c.isalpha()]
        feat["caps_ratio"] = (
            sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if alpha_chars else 0.0
        )

        # C: Exact self-duplicate ratio
        text_norm = [t.strip().lower() for t in texts]
        seen: Dict[str, int] = {}
        for t in text_norm:
            seen[t] = seen.get(t, 0) + 1
        feat["exact_duplicate_ratio"] = (
            sum(1 for t in text_norm if seen[t] > 1) / n_posts
        )

        # F: Language mismatch rate
        feat["lang_mismatch_rate"] = (
            sum(1 for pl in post_langs if pl != dataset_lang) / n_posts
        )

        # F: Language flip rate — consecutive post language changes
        if len(post_langs) > 1:
            flips = sum(
                1 for i in range(len(post_langs) - 1)
                if post_langs[i] != post_langs[i + 1]
            )
            feat["lang_flip_rate"] = flips / (len(post_langs) - 1)
        else:
            feat["lang_flip_rate"] = 0.0

        # C: Low compression complexity — zlib compression ratio
        # Low ratio = repetitive content (compresses well)
        combined = " ".join(texts).encode("utf-8")
        if len(combined) > 0:
            compressed = zlib.compress(combined, level=6)
            feat["compression_ratio"] = len(compressed) / len(combined)
        else:
            feat["compression_ratio"] = 1.0

        # C: Mention placeholder pattern repeat — "@mention" appearing in same position
        mention_positions = []
        for t in texts:
            words = t.split()
            positions = [i for i, w in enumerate(words) if w == "@mention"]
            mention_positions.append(tuple(positions))
        if len(mention_positions) >= 2:
            pos_seen: Dict[tuple, int] = {}
            for p in mention_positions:
                if p:  # skip posts with no mentions
                    pos_seen[p] = pos_seen.get(p, 0) + 1
            posts_with_mention = [p for p in mention_positions if p]
            if posts_with_mention:
                feat["mention_pattern_repeat"] = sum(
                    1 for p in posts_with_mention if pos_seen[p] > 1
                ) / len(posts_with_mention)
            else:
                feat["mention_pattern_repeat"] = 0.0
        else:
            feat["mention_pattern_repeat"] = 0.0

    else:
        feat["avg_post_length"] = 0.0
        feat["stdev_post_length"] = 0.0
        feat["sentence_length_cv"] = 0.0
        feat["avg_word_count"] = 0.0
        feat["lexical_diversity"] = 1.0
        feat["moving_avg_lex_diversity"] = 1.0
        feat["vocab_growth_rate"] = 1.0
        feat["exclamation_rate"] = 0.0
        feat["question_rate"] = 0.0
        feat["emoji_rate"] = 0.0
        feat["caps_ratio"] = 0.0
        feat["exact_duplicate_ratio"] = 0.0
        feat["lang_mismatch_rate"] = 0.0
        feat["lang_flip_rate"] = 0.0
        feat["compression_ratio"] = 1.0
        feat["mention_pattern_repeat"] = 0.0

    # ── Link / mention features ───────────────────────────────────────────────
    if n_posts > 0:
        feat["link_rate"] = sum(1 for t in texts if "https://t.co/twitter_link" in t) / n_posts
        feat["mention_rate"] = sum(1 for t in texts if MENTION_RE.search(t)) / n_posts
        # C: URL placeholder always in same structural position
        url_bearing = [t for t in texts if "https://t.co/twitter_link" in t]
        if len(url_bearing) >= 2:
            feat["url_always_at_end"] = sum(
                1 for t in url_bearing if t.strip().endswith("https://t.co/twitter_link")
            ) / len(url_bearing)
        else:
            feat["url_always_at_end"] = 0.0

        # ── Network / mention graph features ─────────────────────────────────
        all_mentions = [m.lower() for t in texts for m in MENTION_RE.findall(t)]
        if all_mentions:
            mention_counter: Dict[str, int] = {}
            for m in all_mentions:
                mention_counter[m] = mention_counter.get(m, 0) + 1
            unique_mentions = len(mention_counter)
            # How varied are the targets? Low = always same account
            feat["mention_unique_count"] = unique_mentions
            feat["mention_diversity"] = unique_mentions / len(all_mentions)
            # Fraction of all mentions going to the single top target
            feat["mention_top_account_rate"] = max(mention_counter.values()) / len(all_mentions)
            # Shared campaign target: fraction of mentions going to accounts
            # that are frequently targeted across the full dataset (≥3 users)
            if global_mention_counts:
                coordinated = sum(
                    cnt for target, cnt in mention_counter.items()
                    if global_mention_counts.get(target, 0) >= 3
                )
                feat["shared_mention_targets_score"] = coordinated / len(all_mentions)
            else:
                feat["shared_mention_targets_score"] = 0.0
        else:
            feat["mention_unique_count"] = 0
            feat["mention_diversity"] = 1.0
            feat["mention_top_account_rate"] = 0.0
            feat["shared_mention_targets_score"] = 0.0
    else:
        feat["link_rate"] = 0.0
        feat["mention_rate"] = 0.0
        feat["url_always_at_end"] = 0.0
        feat["mention_unique_count"] = 0
        feat["mention_diversity"] = 1.0
        feat["mention_top_account_rate"] = 0.0
        feat["shared_mention_targets_score"] = 0.0

    # ── A: Temporal features ──────────────────────────────────────────────────
    if posts:
        try:
            timestamps = sorted([_parse_dt(p["created_at"]) for p in posts])
            hours = [ts.hour for ts in timestamps]
            feat["unique_hours"] = len(set(hours))

            hour_counts = [0] * 24
            for h in hours:
                hour_counts[h] += 1
            feat["posting_hour_entropy"] = _entropy(hour_counts)

            # A: Sleep-hour fraction (1–6 AM UTC) — no circadian rhythm
            feat["sleep_hour_frac"] = sum(1 for h in hours if 1 <= h <= 6) / len(hours)

            # A: Burst score — fraction of posts in the first 10% of the time span
            span_sec = (timestamps[-1] - timestamps[0]).total_seconds()
            if span_sec > 0:
                burst_window = span_sec * 0.10
                burst_count = sum(
                    1 for ts in timestamps
                    if (ts - timestamps[0]).total_seconds() <= burst_window
                )
                feat["burst_score"] = burst_count / n_posts
            else:
                feat["burst_score"] = 1.0  # all posts at same second = extreme burst

            if len(timestamps) > 1:
                gaps = [
                    (timestamps[i + 1] - timestamps[i]).total_seconds()
                    for i in range(len(timestamps) - 1)
                ]
                gap_mean = statistics.mean(gaps)
                gap_stdev = statistics.stdev(gaps) if len(gaps) > 1 else 0.0
                feat["inter_post_gap_mean"] = gap_mean
                feat["inter_post_gap_stdev"] = gap_stdev
                feat["inter_post_gap_min"] = min(gaps)
                # A: Low CV = unnaturally regular timing
                feat["inter_post_gap_cv"] = (
                    gap_stdev / gap_mean if gap_mean > 0 else 0.0
                )
                # A: Fixed-gap score — fraction of gaps within 20% of median
                if len(gaps) >= 3:
                    median_gap = statistics.median(gaps)
                    if median_gap > 0:
                        near_median = sum(
                            1 for g in gaps if abs(g - median_gap) / median_gap <= 0.20
                        )
                        feat["fixed_gap_score"] = near_median / len(gaps)
                    else:
                        feat["fixed_gap_score"] = 0.0
                else:
                    feat["fixed_gap_score"] = 0.0
            else:
                feat["inter_post_gap_mean"] = 0.0
                feat["inter_post_gap_stdev"] = 0.0
                feat["inter_post_gap_min"] = 0.0
                feat["inter_post_gap_cv"] = 0.0
                feat["fixed_gap_score"] = 0.0

            first_ts = timestamps[0]
            one_hour_later = first_ts.timestamp() + 3600
            feat["posts_in_first_hour"] = sum(
                1 for ts in timestamps if ts.timestamp() <= one_hour_later
            )
            feat["time_span_minutes"] = span_sec / 60.0

            uid = user.get("id", "")
            if global_post_index is not None and global_post_ts_keys is not None:
                feat["temporal_coordination_score"] = _temporal_coordination_score(
                    timestamps, uid, global_post_index, global_post_ts_keys
                )
            else:
                feat["temporal_coordination_score"] = 0.0
        except Exception:
            feat["unique_hours"] = 0
            feat["posting_hour_entropy"] = 0.0
            feat["sleep_hour_frac"] = 0.0
            feat["burst_score"] = 0.0
            feat["inter_post_gap_mean"] = 0.0
            feat["inter_post_gap_stdev"] = 0.0
            feat["inter_post_gap_min"] = 0.0
            feat["inter_post_gap_cv"] = 0.0
            feat["fixed_gap_score"] = 0.0
            feat["posts_in_first_hour"] = len(posts)
            feat["time_span_minutes"] = 0.0
            feat["temporal_coordination_score"] = 0.0
    else:
        feat["unique_hours"] = 0
        feat["posting_hour_entropy"] = 0.0
        feat["sleep_hour_frac"] = 0.0
        feat["burst_score"] = 0.0
        feat["inter_post_gap_mean"] = 0.0
        feat["inter_post_gap_stdev"] = 0.0
        feat["inter_post_gap_min"] = 0.0
        feat["inter_post_gap_cv"] = 0.0
        feat["fixed_gap_score"] = 0.0
        feat["posts_in_first_hour"] = 0
        feat["time_span_minutes"] = 0.0
        feat["temporal_coordination_score"] = 0.0

    # ── C: Structural / template detection ───────────────────────────────────
    if n_posts >= 2:
        capped_texts = texts[:30]
        n_capped = len(capped_texts)
        word_sets = [set(t.lower().split()) for t in capped_texts]
        dup_flags = [False] * n_capped
        for i in range(n_capped):
            if dup_flags[i]:
                continue
            for j in range(i + 1, n_capped):
                if _jaccard(word_sets[i], word_sets[j]) >= 0.70:
                    dup_flags[i] = True
                    dup_flags[j] = True
        feat["near_duplicate_ratio"] = sum(dup_flags) / n_capped

        prefix_len = 20
        pair_count = 0
        template_count = 0
        for i in range(min(n_posts, 30)):
            for j in range(i + 1, min(n_posts, 30)):
                pair_count += 1
                if texts[i][:prefix_len] == texts[j][:prefix_len] and len(texts[i]) >= prefix_len:
                    template_count += 1
        feat["template_score"] = template_count / pair_count if pair_count > 0 else 0.0
    else:
        feat["near_duplicate_ratio"] = 0.0
        feat["template_score"] = 0.0

    # F: Topic hashtag rate — fraction of posts containing dataset seed keywords
    topic_keywords: List[str] = []
    for topic_entry in dataset_metadata.get("topics", []):
        for kw in topic_entry.get("keywords", []):
            topic_keywords.append(kw.lower())

    if n_posts > 0 and topic_keywords:
        on_topic = sum(
            1 for t in texts if any(kw in t.lower() for kw in topic_keywords)
        )
        feat["topic_hashtag_rate"] = on_topic / n_posts

        # F: Topic focus score — how narrowly focused on ONE topic vs all topics
        # Bots that relentlessly push a single topic while humans spread across many
        topic_counts = []
        for topic_entry in dataset_metadata.get("topics", []):
            kwds = [kw.lower() for kw in topic_entry.get("keywords", [])]
            topic_counts.append(
                sum(1 for t in texts if any(kw in t.lower() for kw in kwds))
            )
        if sum(topic_counts) > 0:
            # Gini-like measure: 0 = balanced, 1 = all posts on one topic
            total = sum(topic_counts)
            fracs = [c / total for c in topic_counts]
            feat["topic_focus_score"] = max(fracs)  # fraction on dominant topic
        else:
            feat["topic_focus_score"] = 0.0
    else:
        feat["topic_hashtag_rate"] = 0.0
        feat["topic_focus_score"] = 0.0

    # ── B+C: Cross-account coordination signals ───────────────────────────────
    if cross_account_text_counts and n_posts > 0:
        text_norm = [t.strip().lower() for t in texts]
        feat["cross_account_dup_score"] = sum(
            1 for t in text_norm if cross_account_text_counts.get(t, 0) > 1
        ) / n_posts
    else:
        feat["cross_account_dup_score"] = 0.0

    # B+C: Cross-account near-duplicate score
    # Uses a sample of posts from other users passed in from build_feature_matrix
    if cross_account_post_sample and n_posts > 0:
        own_word_sets = [set(t.lower().split()) for t in texts[:30]]
        matched = 0
        for own_ws in own_word_sets:
            if not own_ws:
                continue
            for _, other_ws in cross_account_post_sample:
                if _jaccard(own_ws, other_ws) >= 0.55:
                    matched += 1
                    break  # one match is enough for this post
        feat["cross_account_near_dup_score"] = matched / min(n_posts, 30)
    else:
        feat["cross_account_near_dup_score"] = 0.0

    # ── Semantic / generic phrase features ───────────────────────────────────
    if n_posts > 0:
        joined_lower = " ".join(texts).lower()
        feat["quote_tweet_ratio"] = sum(1 for t in texts if t.startswith('"')) / n_posts
        total_words = sum(len(t.split()) for t in texts)
        phrase_hits = sum(1 for phrase in GENERIC_PHRASES if phrase in joined_lower)
        feat["generic_phrase_rate"] = phrase_hits / total_words if total_words > 0 else 0.0
    else:
        feat["quote_tweet_ratio"] = 0.0
        feat["generic_phrase_rate"] = 0.0

    return feat


def build_feature_matrix(dataset: dict, bot_ids: set = None):
    """
    Extract features for all users in a dataset.
    Returns (feature_dicts, labels, user_ids).
    """
    posts_by_user: Dict[str, List[dict]] = {}
    for p in dataset.get("posts", []):
        posts_by_user.setdefault(p["author_id"], []).append(p)

    metadata = dataset.get("metadata", {})
    dataset_lang = dataset.get("lang", "en")

    # ── Build cross-account text corpus ──────────────────────────────────────
    # Exact duplicate coordination
    text_to_users: Dict[str, set] = {}
    for uid, user_posts in posts_by_user.items():
        for p in user_posts:
            norm = p["text"].strip().lower()
            text_to_users.setdefault(norm, set()).add(uid)
    cross_account_text_counts: Dict[str, int] = {
        t: len(uids) for t, uids in text_to_users.items()
    }

    # ── Build per-user word-set index for near-dup cross-account detection ───
    # (user_id, word_set) for a capped sample of posts per user
    all_user_post_wordsets: Dict[str, List[Set[str]]] = {}
    for uid, user_posts in posts_by_user.items():
        all_user_post_wordsets[uid] = [
            set(p["text"].lower().split()) for p in user_posts[:30]
        ]

    # ── Build global sorted (timestamp, user_id) index for temporal coordination ─
    _raw_events: List[Tuple[float, str]] = []
    for uid, user_posts in posts_by_user.items():
        for p in user_posts:
            try:
                ts = _parse_dt(p["created_at"]).timestamp()
                _raw_events.append((ts, uid))
            except Exception:
                pass
    _raw_events.sort(key=lambda x: x[0])
    global_post_index: List[Tuple[float, str]] = _raw_events
    global_post_ts_keys: List[float] = [e[0] for e in _raw_events]

    # ── Build global mention target → user count index ────────────────────────
    # mention_target → number of distinct users who mention it
    mention_target_user_counts: Dict[str, int] = {}
    _target_to_users: Dict[str, Set[str]] = {}
    for uid, user_posts in posts_by_user.items():
        for p in user_posts:
            for m in MENTION_RE.findall(p["text"].lower()):
                _target_to_users.setdefault(m, set()).add(uid)
    mention_target_user_counts = {t: len(uids) for t, uids in _target_to_users.items()}

    # ── Build hashtag → users + top hashtag per user (single pass) ────────────
    hashtag_to_users: Dict[str, Set[str]] = {}
    user_top_hashtag: Dict[str, Optional[str]] = {}
    for uid, user_posts in posts_by_user.items():
        ht_counter: Dict[str, int] = {}
        for p in user_posts:
            for ht in HASHTAG_RE.findall(p["text"].lower()):
                ht_counter[ht] = ht_counter.get(ht, 0) + 1
                hashtag_to_users.setdefault(ht, set()).add(uid)
        user_top_hashtag[uid] = max(ht_counter, key=ht_counter.get) if ht_counter else None

    feature_dicts = []
    labels = [] if bot_ids is not None else None
    user_ids = []

    for user in dataset.get("users", []):
        uid = user["id"]
        user_posts = posts_by_user.get(uid, [])

        # Build a sample of posts from OTHER users (~200 posts sampled)
        other_users = [k for k in all_user_post_wordsets if k != uid]
        sample_posts: List[tuple] = []
        # Sample up to 2 posts per other user to keep diversity
        for other_uid in other_users:
            wsets = all_user_post_wordsets[other_uid]
            for ws in wsets[:2]:
                sample_posts.append((other_uid, ws))
        # If too many, random sample down to 300
        if len(sample_posts) > 300:
            random.seed(42)
            sample_posts = random.sample(sample_posts, 300)

        feat = extract_features(
            user, user_posts, metadata,
            cross_account_text_counts=cross_account_text_counts,
            cross_account_post_sample=sample_posts,
            dataset_lang=dataset_lang,
            global_post_index=global_post_index,
            global_post_ts_keys=global_post_ts_keys,
            global_mention_counts=mention_target_user_counts,
        )
        feat["_user_id"] = uid
        feat["_username"] = user.get("username", "")
        feat["_name"] = user.get("name", "")
        feat["_z_score_raw"] = user.get("z_score", 0.0)
        feature_dicts.append(feat)
        user_ids.append(uid)
        if bot_ids is not None:
            labels.append(1 if uid in bot_ids else 0)

    # ── Second pass: cluster features ────────────────────────────────────────
    # Requires all individual features to be computed first.
    uid_to_feat: Dict[str, dict] = {fd["_user_id"]: fd for fd in feature_dicts}

    def _is_suspicious(fd: dict) -> bool:
        return (
            fd.get("near_duplicate_ratio", 0) > 0.4
            or fd.get("hashtag_sequence_repeat", 0) > 0.5
            or fd.get("cross_account_dup_score", 0) > 0.15
        )

    for fd in feature_dicts:
        uid = fd["_user_id"]
        top_ht = user_top_hashtag.get(uid)
        if top_ht is not None:
            cluster_members = hashtag_to_users[top_ht]
            fd["cluster_size"] = len(cluster_members)
            fd["cluster_bot_density"] = (
                sum(1 for m_uid in cluster_members if _is_suspicious(uid_to_feat[m_uid]))
                / len(cluster_members)
            )
        else:
            fd["cluster_size"] = 0
            fd["cluster_bot_density"] = 0.0

    return feature_dicts, labels, user_ids
