import pandas as pd

# Load the dataset
df = pd.read_csv("data/processed/scored_posts_with_users.csv")

# Fill missing data to ensure safe evaluations
df["user_follows_tag"] = df["user_follows_tag"].fillna(False)
df["is_buddy_post"] = df["is_buddy_post"].fillna(False)
df["karma"] = df["karma"].fillna(0)
df["target_label"] = df["target_label"].fillna(0.5)

print("\n📊 Dataset Overview:")
print(df.head())

# ─────────────────────────────────────────────
# Define karma level
def karma_level(k):
    if k >= 67:
        return "High"
    elif k >= 34:
        return "Medium"
    else:
        return "Low"

df["karma_level"] = df["karma"].apply(karma_level)

# ─────────────────────────────────────────────
# Rule 1: Buddy + Followed Tag → label > 0.8
rule1 = df[(df["user_follows_tag"] == True) & (df["is_buddy_post"] == True)]
rule1_high = rule1[rule1["target_label"] > 0.8]

# Rule 2: (Buddy XOR Followed Tag) → label 0.5 to 0.79
rule2 = df[
    ((df["user_follows_tag"] == True) ^ (df["is_buddy_post"] == True))  # XOR
]
rule2_mid = rule2[(rule2["target_label"] >= 0.5) & (rule2["target_label"] < 0.8)]

# Rule 3: Neither Buddy Nor Followed Tag → label < 0.5
rule3 = df[(df["user_follows_tag"] == False) & (df["is_buddy_post"] == False)]
rule3_low = rule3[rule3["target_label"] < 0.5]

# ─────────────────────────────────────────────
# Karma-level based scoring
karma_stats = df.groupby("karma_level")["target_label"].mean()

# ─────────────────────────────────────────────
# Display Results
print("\n✅ Heuristic Validation Results:\n")

print("🔹 Rule 1: Buddy + Followed Tag")
print(f"  → Count: {len(rule1)}")
print(f"  → Average score: {rule1['target_label'].mean():.3f}")
print(f"  → % with score > 0.8: {(len(rule1_high) / len(rule1) * 100):.2f}%")

print("\n🔹 Rule 2: Either Buddy OR Followed Tag (not both)")
print(f"  → Count: {len(rule2)}")
print(f"  → % in range [0.5 – 0.79]: {(len(rule2_mid) / len(rule2) * 100):.2f}%")

print("\n🔹 Rule 3: No Buddy, No Followed Tag")
print(f"  → Count: {len(rule3)}")
print(f"  → Average score: {rule3['target_label'].mean():.3f}")
print(f"  → % with score < 0.5: {(len(rule3_low) / len(rule3) * 100):.2f}%")

print("\n🔹 Rule 4: Karma vs Target Label")
for level in ["Low", "Medium", "High"]:
    avg_score = karma_stats.get(level, float("nan"))
    print(f"  → {level} karma: avg score = {avg_score:.3f}")
