import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import numpy as np

# ================================
# ✅ Load model and data
# ================================
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv("data/processed/scored_posts_with_users.csv")

# ================================
# ✅ Drop rows with missing values
# ================================
df = df.dropna(subset=["user_tags_text", "Audience Interests", "Post Content"]).reset_index(drop=True)

# ================================
# ✅ Create text to embed
# ================================
df["post_text"] = df["Audience Interests"].astype(str).str.strip() + " " + df["Post Content"].astype(str).str.strip()

# ================================
# ✅ Prepare containers
# ================================
similarities = []
overlap_buckets = []

BATCH_SIZE = 128

print("⚡ Computing semantic features...")

for start in tqdm(range(0, len(df), BATCH_SIZE), desc="🔍 Batching"):
    end = start + BATCH_SIZE
    posts = df["post_text"].iloc[start:end].tolist()
    user_tags = df["user_tags_text"].iloc[start:end].tolist()

    post_embs = model.encode(posts, convert_to_tensor=True, show_progress_bar=False)
    user_embs = model.encode(user_tags, convert_to_tensor=True, show_progress_bar=False)

    sims = util.cos_sim(post_embs, user_embs).diagonal()

    similarities.extend(sims.cpu().numpy())

# ================================
# ✅ Assign columns
# ================================
df["semantic_tag_similarity"] = similarities
df["semantic_overlap_bucket"] = pd.qcut(df["semantic_tag_similarity"], q=5, labels=False)


# ================================
# ✅ Save
# ================================
df.to_csv("data/processed/scored_posts_with_users.csv", index=False)
print("✅ Done! Semantic feature columns added.")
