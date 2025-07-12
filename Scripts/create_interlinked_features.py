import pandas as pd
import numpy as np

# Load your CSV
df = pd.read_csv("data/processed/scored_posts_with_users.csv")

# 1. nonbuddy_tag_followed: not a buddy, but follows the tag
df["nonbuddy_tag_followed"] = (~df["is_buddy_post"]) & (df["user_follows_tag"])

# 2. buddy_tag_unfollowed: buddy, but does not follow the tag
df["buddy_tag_unfollowed"] = (df["is_buddy_post"]) & (~df["user_follows_tag"])

# 3. nonbuddy_tag_unfollowed: not a buddy and does not follow the tag
df["nonbuddy_tag_unfollowed"] = (~df["is_buddy_post"]) & (~df["user_follows_tag"])

# 4. either_buddy_or_followed: either buddy OR follows the tag
df["either_buddy_or_followed"] = (df["is_buddy_post"]) | (df["user_follows_tag"])

# 5. karma_per_time_match: ratio of karma to time match score
df["karma_per_time_match"] = np.where(
    df["time_match_score"] == 0,
    0,
    df["karma"] / df["time_match_score"]
)

# 6. Normalize karma_per_time_match to 0–1
min_kptm = df["karma_per_time_match"].min()
max_kptm = df["karma_per_time_match"].max()
df["karma_per_time_match_normalized"] = (
    (df["karma_per_time_match"] - min_kptm) / (max_kptm - min_kptm)
    if max_kptm - min_kptm != 0 else 0
)

# 7. karma_per_post_hour: ratio of karma to post_hour
df["karma_per_post_hour"] = np.where(
    df["post_hour"] == 0,
    0,
    df["karma"] / df["post_hour"]
)

# 8. Normalize karma_per_post_hour to 0–1
min_kpph = df["karma_per_post_hour"].min()
max_kpph = df["karma_per_post_hour"].max()
df["karma_per_post_hour_normalized"] = (
    (df["karma_per_post_hour"] - min_kpph) / (max_kpph - min_kpph)
    if max_kpph - min_kpph != 0 else 0
)

# 9. karma_x_time_match: product of karma and time_match_score
df["karma_x_time_match"] = df["karma"] * df["time_match_score"]

# 10. buddy_and_high_karma: buddy and karma_bucket is high
df["buddy_and_high_karma"] = (df["is_buddy_post"]) & (df["karma_bucket"] == 'high')
#11
df["buddy_and_medium_karma"] = (df["is_buddy_post"]) & (df["karma_bucket"] == 'medium')
#12
df["buddy_and_low_karma"] = (df["is_buddy_post"]) & (df["karma_bucket"] == 'low')
#13
df["Image_and_high_karma"] = (df["Post Type"] == 'Image') & (df["karma_bucket"] == 'high')
#14
df["Video_and_high_karma"] = (df["Post Type"] == 'Video') & (df["karma_bucket"] == 'high')
#15
df["either_buddy_or_followed_high_karma"] = (df["either_buddy_or_followed"]) & (df["karma_bucket"] == 'high')
#16
df["either_buddy_or_followed_medium_karma"] = (df["either_buddy_or_followed"]) & (df["karma_bucket"] == 'medium')
#17
df["user_follows_tag_high_karma"] = df["user_follows_tag"] & (df["karma_bucket"] == 'high')
#18
df["user_follows_tag_medium_karma"] = df["user_follows_tag"] & (df["karma_bucket"] == 'medium')
#19
df["either_buddy_or_followed_low_karma"] = (df["either_buddy_or_followed"]) & (df["karma_bucket"] == 'low')
#20
df["user_follows_tag_low_karma"] = df["user_follows_tag"] & (df["karma_bucket"] == 'low')
#21
df["either_buddy_or_followed_high_karma_Video"] = (df["either_buddy_or_followed_high_karma"]) & (df["Post Type"] == 'Video')
#22
df["either_buddy_or_followed_high_karma_Image"] = (df["either_buddy_or_followed_high_karma"]) & (df["Post Type"] == 'Image')
# Save the updated CSV
df.to_csv("data/processed/scored_posts_with_users.csv", index=False)
