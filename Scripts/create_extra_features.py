import pandas as pd
from datetime import datetime, timezone

# Load your CSV
df = pd.read_csv("data/processed/scored_posts_with_users.csv")

# Ensure Post Timestamp is parsed as datetime in UTC
df['Post Timestamp'] = pd.to_datetime(df['Post Timestamp'], utc=True)

# Current time for computing post recency (UTC)
current_time = datetime.now(timezone.utc)

# 1. Post Hour
df['post_hour'] = df['Post Timestamp'].dt.hour

# 2. Post Recency in hours
df['post_recency_hours'] = (current_time - df['Post Timestamp']).dt.total_seconds() / 3600.0

# Avoid division by zero
df['post_recency_hours'] = df['post_recency_hours'].replace(0, 0.01)

# 3. buddy_followed_tag = is_buddy_post AND user_follows_tag
df['buddy_followed_tag'] = (df['is_buddy_post'] == True) & (df['user_follows_tag'] == True)

# 4. time_weighted_karma = karma / post_recency_hours
df['time_weighted_karma'] = df['karma'] / df['post_recency_hours']

# Save the updated CSV
df.to_csv("data/processed/scored_posts_with_users.csv", index=False)
