import pandas as pd
import ast
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch
import numpy as np

# ================================
# ✅ Load model & data
# ================================
model = SentenceTransformer('all-MiniLM-L6-v2')

df_users = pd.read_csv("data/intermediate/simulated_users.csv")
df_posts = pd.read_csv("data/processed/scored_posts_with_users.csv")

# ================================
# ✅ Preprocess tags_followed
# ================================
df_users["tags_followed"] = df_users["tags_followed"].apply(lambda x: ast.literal_eval(str(x)) if pd.notna(x) else [])
user_tag_map = dict(zip(df_users["user_id"], [" ".join(tags) for tags in df_users["tags_followed"]]))

df_posts["user_tags_text"] = df_posts["user_id"].map(user_tag_map)

# ================================
# ✅ Drop incomplete rows
# ================================
df_posts = df_posts.dropna(subset=["user_tags_text", "Post Content", "Audience Interests"]).reset_index(drop=True)

# ================================
# ✅ Combine post fields
# ================================
df_posts["post_text"] = df_posts["Audience Interests"].astype(str).str.strip() + " " + df_posts["Post Content"].astype(str).str.strip()

# ================================
# ✅ Batch encoding & similarity
# ================================
BATCH_SIZE = 128  # Optimized for 8GB RAM
all_similarities = []

print("⚡ Step 1: Calculating cosine similarities...")

for start in tqdm(range(0, len(df_posts), BATCH_SIZE), desc="🔍 Embedding Batches"):
    end = start + BATCH_SIZE
    batch_post = df_posts["post_text"].iloc[start:end].tolist()
    batch_tags = df_posts["user_tags_text"].iloc[start:end].tolist()

    # Get embeddings
    post_embeddings = model.encode(batch_post, convert_to_tensor=True, show_progress_bar=False)
    tag_embeddings = model.encode(batch_tags, convert_to_tensor=True, show_progress_bar=False)

    # Cosine similarities (diagonal of each pair)
    batch_similarities = util.cos_sim(post_embeddings, tag_embeddings).diagonal().cpu().numpy()
    all_similarities.extend(batch_similarities)

# ================================
# ✅ Auto threshold for balance
# ================================
SIMILARITY_THRESHOLD = float(np.median(all_similarities))
print(f"✅ Using median similarity threshold: {SIMILARITY_THRESHOLD:.4f}")

# ================================
# ✅ Assign user_follows_tag
# ================================
df_posts["user_follows_tag"] = [sim >= SIMILARITY_THRESHOLD for sim in all_similarities]

# ================================
# ✅ Save results
# ================================
df_posts.to_csv("data/processed/scored_posts_with_users.csv", index=False)
print("✅ Done! Saved with balanced 'user_follows_tag' column.")
