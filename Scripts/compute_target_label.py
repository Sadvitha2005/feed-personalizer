import pandas as pd

# Load the data
input_file = "data/processed/scored_posts_with_users.csv"
df = pd.read_csv(input_file)

# Fill missing values with safe defaults
df["relevance_score"] = df["relevance_score"].fillna(0.5)
df["karma"] = df["karma"].fillna(0)
df["time_match_score"] = df["time_match_score"].fillna(0)
df["user_follows_tag"] = df["user_follows_tag"].fillna(False)
df["is_buddy_post"] = df["is_buddy_post"].fillna(False)
df["Post Type"] = df["Post Type"].fillna("text")

# Compute the hybrid target label
def compute_target_label(row):
    # Base score from real engagement (optional realism)
    base_score = row["relevance_score"]

    # Initialize contextual boost
    contextual_score = 0.0

    # Heuristic 1: Buddy + Tag following (boosted)
    if row["is_buddy_post"] and row["user_follows_tag"]:
        contextual_score += 0.6
    elif row["is_buddy_post"] or row["user_follows_tag"]:
        contextual_score += 0.3
    else:
        contextual_score -= 0.1  # Penalize unrelated posts

    # Heuristic 2: Karma boost
    if row["karma"] >= 67:
        contextual_score += 0.3
    elif row["karma"] >= 34:
        contextual_score += 0.2
    else:
        contextual_score += 0.05
    
    if row["karma"] < 20:
        contextual_score -=0.02
    elif row["karma"] < 50:
        contextual_score -= 0.01

    # Heuristic 3: Time match (0–1 scale)
    contextual_score += 0.2 * row["time_match_score"]


    # Final score: stronger weight on heuristic behavior
    final_score = 0.5 * base_score + 0.5 * min(max(contextual_score, 0.0), 1.0)

    return min(final_score, 1.0)

# Apply function
df["target_label"] = df.apply(compute_target_label, axis=1)

# Save to file
df.to_csv(input_file, index=False)
print("✅ target_label computed and saved to scored_posts_with_users.csv")