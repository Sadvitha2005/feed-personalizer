import pandas as pd
import random
import json

# Define pools
branches = ["CSE", "ECE", "IT", "Mech", "AI", "DS"]
tags_followed_pool = ["coding", "startups", "python", "clubs", "events", "ML", "design", "internships", "project", "career"]
time_slots = ["06:00-09:00", "08:00-11:00", "12:00-14:00", "17:00-19:00", "20:00-23:00"]

# Generate user_ids
num_users = 5000
user_ids = [f"stu_{i:04d}" for i in range(1, num_users + 1)]

# Simulate base user profiles and assign buddies
user_profiles = []
user_id_set = set(user_ids)

for user_id in user_ids:
    possible_buddies = list(user_id_set - {user_id})  # exclude self
    profile = {
        "user_id": user_id,
        "branches_of_interest": random.sample(branches, k=random.choice([1, 2])),
        "tags_followed": random.sample(tags_followed_pool, k=random.randint(3, 5)),
        "active_hours": random.sample(time_slots, k=2),
        "buddies": random.sample(possible_buddies, k=random.randint(2, 4))
    }
    user_profiles.append(profile)

# Convert to DataFrame
user_df = pd.DataFrame(user_profiles)

# Convert list columns to JSON strings for Excel compatibility
list_columns = ["branches_of_interest", "tags_followed", "active_hours", "buddies"]
for col in list_columns:
    user_df[col] = user_df[col].apply(json.dumps)

# Save to Excel
user_df.to_csv("data/intermediate/simulated_users.csv", index=False)
print("✅ Saved 5000 users to simulated_users.csv with buddies from same ID range")
