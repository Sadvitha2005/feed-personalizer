import pandas as pd
import random
import json
import numpy as np

# Load users and posts
users_df = pd.read_csv("data/intermediate/simulated_users.csv")
users_df["buddies"] = users_df["buddies"].apply(json.loads)

posts_df = pd.read_csv("data/processed/scored_posts_with_users.csv", low_memory=False)
assert len(posts_df) == 100000, "scored_posts_with_users.csv must have exactly 100,000 rows"

user_ids = users_df["user_id"].tolist()
buddies_lookup = users_df.set_index("user_id")["buddies"].to_dict()

min_posts, max_posts = 20, 25
num_users = len(user_ids)
total_posts_needed = 100000

# Step 1: Assign post counts per user between 20 and 25, summing to exactly 100,000
post_counts = np.array([min_posts] * num_users)
remaining = total_posts_needed - post_counts.sum()

# Distribute remaining posts
while remaining > 0:
    updated = False
    for i in range(num_users):
        if post_counts[i] < max_posts:
            post_counts[i] += 1
            remaining -= 1
            updated = True
            if remaining == 0:
                break
    if not updated:
        raise ValueError("⚠️ No users left to assign extra posts, but remaining > 0")

# Step 2: Shuffle posts
posts_df = posts_df.sample(frac=1, random_state=42).reset_index(drop=True)

assigned_user_ids = []
author_ids = []
is_buddy_flags = []

post_index = 0

# Step 3: Assign posts to users with buddy logic
for user_id, count in zip(user_ids, post_counts):
    buddies = buddies_lookup.get(user_id, [])
    for _ in range(count):
        if post_index >= len(posts_df):
            break

        # 40% chance to assign a buddy author
        if random.random() < 0.4 and buddies:
            author_id = random.choice(buddies)
            is_buddy = True
        else:
            non_buddies = list(set(user_ids) - set(buddies) - {user_id})
            author_id = random.choice(non_buddies)
            is_buddy = False

        assigned_user_ids.append(user_id)
        author_ids.append(author_id)
        is_buddy_flags.append(is_buddy)
        post_index += 1

# Final assignment to DataFrame
posts_df = posts_df.iloc[:len(assigned_user_ids)].copy()
posts_df["user_id"] = assigned_user_ids
posts_df["author_id"] = author_ids
posts_df["is_buddy_post"] = is_buddy_flags

# Save to file
posts_df.to_csv("data/processed/scored_posts_with_users.csv", index=False)
print(f"✅ Assigned 100,000 posts across {num_users} users with 20–25 posts each.")
