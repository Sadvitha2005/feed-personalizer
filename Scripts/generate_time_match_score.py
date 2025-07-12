import pandas as pd
import ast

# Function to convert HH:MM:SS to minutes since midnight
def time_to_minutes(time_str):
    h, m, _ = map(int, time_str.split(":"))
    return h * 60 + m

# Enhanced function to compute time match score
def compute_time_match_score(post_time_str, active_hours_list):
    post_minutes = time_to_minutes(post_time_str)
    min_distance = float("inf")

    for time_range in active_hours_list:
        start_str, end_str = time_range.split("-")
        start_minutes = time_to_minutes(start_str + ":00")
        end_minutes = time_to_minutes(end_str + ":00")

        # Handle overnight time windows (e.g., 22:00-02:00)
        if start_minutes <= end_minutes:
            if start_minutes <= post_minutes <= end_minutes:
                return 1.0
        else:
            # Overnight case (e.g., 22:00 to 02:00 spans midnight)
            if post_minutes >= start_minutes or post_minutes <= end_minutes:
                return 1.0

        # Compute circular distance to range edges
        distance_to_start = min(abs(post_minutes - start_minutes), 1440 - abs(post_minutes - start_minutes))
        distance_to_end = min(abs(post_minutes - end_minutes), 1440 - abs(post_minutes - end_minutes))
        min_distance = min(min_distance, distance_to_start, distance_to_end)

    return round(max(0.0, 1 - (min_distance / 600)), 2)  # max distance = 10 hours (600 min)

# Load the posts data
print("📂 Loading posts data...")
posts_df = pd.read_csv("data/processed/scored_posts_with_users.csv")

# Load the simulated users data
print("📂 Loading user active hours...")
users_df = pd.read_csv("data/intermediate/simulated_users.csv")

# Merge posts and user active hours on user_id
merged_df = posts_df.merge(users_df[["user_id", "active_hours"]], on="user_id", how="left")

# Convert string representation of list to actual list
merged_df["active_hours"] = merged_df["active_hours"].apply(ast.literal_eval)

# Compute time match score
print("⏱️ Computing time match scores...")
merged_df["time_match_score"] = merged_df.apply(
    lambda row: compute_time_match_score(row["Time"], row["active_hours"]), axis=1
)

# Drop active_hours if not needed in final output
merged_df.drop(columns=["active_hours"], inplace=True)

# Save the updated DataFrame
merged_df.to_csv("data/processed/scored_posts_with_users.csv", index=False)
print("✅ Done. Output saved to scored_posts_with_users.csv")
