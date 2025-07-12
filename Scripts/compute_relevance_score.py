import pandas as pd

# Load the Excel file
file_path = "data/raw/social media engagement data.xlsx"
df = pd.read_excel(file_path)

# Mapping sentiment to numerical score
sentiment_map = {
    "Positive": 1.0,
    "Neutral": 0.5,
    "Negative": 0.0,
    "Mixed": 0.7
}

# Map sentiments to numeric scores
df["sentiment_score"] = df["Sentiment"].map(sentiment_map)

# Compute raw engagement score
df["engagement_score"] = df[["Likes", "Comments", "Shares"]].sum(axis=1)

# Normalize engagement score
df["engagement_score_normalized"] = (df["engagement_score"] - df["engagement_score"].min()) / (
    df["engagement_score"].max() - df["engagement_score"].min()
)

# Compute final relevance score
df["relevance_score"] = 0.6 * df["engagement_score_normalized"] + 0.4 * df["sentiment_score"]

# ✅ Save as CSV
df.to_csv("data/processed/scored_posts_with_users.csv", index=False)
print("✅ Relevance scores saved to scored_posts_with_users.csv")
