import pandas as pd

# Compute karma score from post metrics and metadata
def compute_karma(row):
    likes = row.get("Likes", 0)
    comments = row.get("Comments", 0)
    shares = row.get("Shares", 0)
    impressions = row.get("Impressions", 0)
    reach = row.get("Reach", 0)
    engagement_rate = row.get("Engagement Rate", 0)
    time_match_score = row.get("time_match_score", 0)
    is_buddy = row.get("is_buddy_post", False)

    # Normalize core metrics to [0, 1]
    norm_likes = min(likes / 500, 1)
    norm_comments = min(comments / 100, 1)
    norm_shares = min(shares / 100, 1)
    norm_impressions = min(impressions / 10000, 1)
    norm_reach = min(reach / 10000, 1)
    norm_engagement = min(engagement_rate / 100, 1)

    # Compute weighted base score (0–1)
    score = 0
    score += norm_likes * 0.15
    score += norm_comments * 0.15
    score += norm_shares * 0.15
    score += norm_engagement * 0.20
    score += norm_impressions * 0.15
    score += norm_reach * 0.10
    score += time_match_score * 0.05
    if is_buddy:
        score += 0.03

    sentiment = str(row.get("Sentiment", "")).lower()
    if sentiment == "positive":
        score += 0.02

    post_type = str(row.get("Post Type", "")).lower()
    if post_type == "video":
        score += 0.02
    elif post_type == "image":
        score += 0.01

    # Cap score at 1
    base_score = min(score, 1.0)

    # Convert to scaled karma band
    if base_score <= 0.33:
        return round(base_score * 33) or 1  # ensure ≥1
    elif base_score <= 0.66:
        return round(33 + (base_score - 0.33) * (33 / 0.33))
    else:
        return round(66 + (base_score - 0.66) * (34 / 0.34))

# Determine karma bucket from score
def assign_karma_bucket(score):
    if score <= 33:
        return "low"
    elif score <= 66:
        return "medium"
    else:
        return "high"

if __name__ == "__main__":
    print("📂 Loading data from scored_posts_with_users.csv...")
    df = pd.read_csv("data/processed/scored_posts_with_users.csv")

    print("💯 Calculating karma scores in low-medium-high bands...")
    df["karma"] = df.apply(compute_karma, axis=1)

    print("🪣 Assigning karma buckets...")
    df["karma_bucket"] = df["karma"].apply(assign_karma_bucket)

    print("💾 Saving updated data with karma and karma_bucket columns...")
    df.to_csv("data/processes/scored_posts_with_users.csv", index=False)

    print("✅ Karma and karma_bucket columns appended successfully.")
