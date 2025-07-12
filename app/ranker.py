import json
import pandas as pd
import joblib
from .extract_features import extract_features

# Add root directory to the system path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# 🔧 Load config.json
with open("config/config.json") as f:
    config = json.load(f)
    
model_type = config.get("model_type", "lightgbm")
model_path = config["model_path"].get(model_type)
expected_columns = config["feature_columns"].get(model_type)

if not model_path or not expected_columns:
    raise ValueError(f"Missing model_path or feature_columns for model_type: {model_type}")


try:
    model, _ = joblib.load(model_path)
except Exception as e:
    print(f"❌ Failed to load model from {model_path}")
    raise e


# Content type encoding (used during training)
content_type_map = {"video": 0.08, "image": 0.04, "text": 0.02}
weekday_type_map = {"Weekday": 0, "Weekend": 1}
time_period_map = {"Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 2}
karma_bucket_map = {"low": 0, "medium": 1, "high": 2}

def rank_posts(user_id, posts, user_profile):
    threshold = config.get("model_threshold", 0.5) 
    feature_list = []
    post_ids = []

    for post in posts:
        # Feature extraction
        features = extract_features(post, user_profile)

        # Convert content_type to numeric
        features["Post Type"] = content_type_map.get(post["content_type"], 0)
        features['Weekday Type'] = weekday_type_map.get(features.get('Weekday Type'), 0)
        features["Time Periods"] = time_period_map.get(features.get("Time Periods"), 0)
        features['karma_bucket'] = karma_bucket_map.get(features.get("karma_bucket"), 0)
        # Make sure new features are present (assumed to be added in extract_features)
        if "Weekday Type" not in features:
            features["Weekday Type"] = 0  # default if missing
        if "Time Periods" not in features:
            features["Time Periods"] = 0  # default if missing

        feature_list.append(features)
        post_ids.append(post["post_id"])

    # Convert to DataFrame
    X = pd.DataFrame(feature_list)
    # Defensive check for empty DataFrame
    if X.empty:
        return {
            "user_id": user_id,
            "ranked_posts": [],
            "status": "empty"
        }
    # Reorder columns to match expected model input
    X = X[expected_columns]
    scores = model.predict(X)

    # Check if any score meets the threshold
    if any(score >= threshold for score in scores):
        filtered = [
            {"post_id": pid, "score": float(score)}
            for pid, score in zip(post_ids, scores)
            if score >= threshold
        ]
        ranked_posts = sorted(filtered, key=lambda x: x["score"], reverse=True)
        return {
            "user_id": user_id,
            "ranked_posts": ranked_posts,
            "status": "ranked"
        }
    else:
        # fallback: scale down original scores
        fallback_posts = [
            {"post_id": pid, "score": round(float(score * 0.8), 4)}  # or any penalty strategy
            for pid, score in zip(post_ids, scores)
        ]
        ranked_posts = sorted(fallback_posts, key=lambda x: x["score"], reverse=True)
        return {
            "user_id": user_id,
            "ranked_posts": ranked_posts,
            "status": "fallback_used"
        }
def main():
    test_posts = [
        # Buddy + followed tag + high karma + within active hours
        {"post_id": "p1", "author_id": "stu_1010", "tags": ["ai"], "content_type": "text", "karma": 90, "created_at": "2025-05-27T07:30:00Z"},
        
        # Buddy + followed tag + high karma + outside active hours
        {"post_id": "p2", "author_id": "stu_2020", "tags": ["ml"], "content_type": "image", "karma": 80, "created_at": "2025-05-27T14:00:00Z"},
        
        # Buddy + followed tag + low karma + within active hours
        {"post_id": "p3", "author_id": "stu_1010", "tags": ["coding"], "content_type": "text", "karma": 5, "created_at": "2025-05-27T21:00:00Z"},
        
        # Buddy + unfollowed tag + high karma + within active hours
        {"post_id": "p4", "author_id": "stu_2020", "tags": ["events"], "content_type": "video", "karma": 95, "created_at": "2025-05-27T21:30:00Z"},
        
        # Buddy + unfollowed tag + low karma + outside active hours
        {"post_id": "p5", "author_id": "stu_3030", "tags": ["fest"], "content_type": "text", "karma": 2, "created_at": "2025-05-27T15:00:00Z"},
        
        # Non-buddy + followed tag + high karma + within active hours
        {"post_id": "p6", "author_id": "stu_4040", "tags": ["python"], "content_type": "image", "karma": 85, "created_at": "2025-05-27T08:30:00Z"},
        
        # Non-buddy + followed tag + low karma + outside active hours
        {"post_id": "p7", "author_id": "stu_4041", "tags": ["ml"], "content_type": "text", "karma": 1, "created_at": "2025-05-27T12:00:00Z"},
        
        # Non-buddy + unfollowed tag + high karma + within active hours
        {"post_id": "p8", "author_id": "stu_5050", "tags": ["travel"], "content_type": "text", "karma": 90, "created_at": "2025-05-27T20:30:00Z"},
        
        # Non-buddy + unfollowed tag + low karma + outside active hours
        {"post_id": "p9", "author_id": "stu_5051", "tags": ["food"], "content_type": "image", "karma": 3, "created_at": "2025-05-27T13:30:00Z"},
        
        # Random post (unrelated)
        {"post_id": "p10", "author_id": "stu_6060", "tags": ["sports"], "content_type": "video", "karma": 15, "created_at": "2025-05-27T18:00:00Z"},
        
        # Add 20 more variations
        {"post_id": "p11", "author_id": "stu_1010", "tags": ["coding"], "content_type": "image", "karma": 50, "created_at": "2025-05-27T20:00:00Z"},
        {"post_id": "p12", "author_id": "stu_2020", "tags": ["python"], "content_type": "text", "karma": 75, "created_at": "2025-05-27T07:15:00Z"},
        {"post_id": "p13", "author_id": "stu_4040", "tags": ["python"], "content_type": "text", "karma": 25, "created_at": "2025-05-27T22:15:00Z"},
        {"post_id": "p14", "author_id": "stu_2020", "tags": ["ds"], "content_type": "image", "karma": 70, "created_at": "2025-05-27T08:00:00Z"},
        {"post_id": "p15", "author_id": "stu_5050", "tags": ["coding"], "content_type": "video", "karma": 65, "created_at": "2025-05-27T10:00:00Z"},
        {"post_id": "p16", "author_id": "stu_2020", "tags": ["random"], "content_type": "text", "karma": 60, "created_at": "2025-05-27T06:00:00Z"},
        {"post_id": "p17", "author_id": "stu_1010", "tags": ["ai"], "content_type": "text", "karma": 95, "created_at": "2025-05-27T21:15:00Z"},
        {"post_id": "p18", "author_id": "stu_1010", "tags": ["ai"], "content_type": "image", "karma": 10, "created_at": "2025-05-27T10:30:00Z"},
        {"post_id": "p19", "author_id": "stu_3030", "tags": ["python"], "content_type": "video", "karma": 80, "created_at": "2025-05-27T23:00:00Z"},
        {"post_id": "p20", "author_id": "stu_4041", "tags": ["ml"], "content_type": "text", "karma": 0, "created_at": "2025-05-27T05:00:00Z"},
        {"post_id": "p21", "author_id": "stu_6060", "tags": ["ml"], "content_type": "image", "karma": 45, "created_at": "2025-05-27T20:30:00Z"},
        {"post_id": "p22", "author_id": "stu_2020", "tags": ["fest"], "content_type": "text", "karma": 15, "created_at": "2025-05-27T07:00:00Z"},
        {"post_id": "p23", "author_id": "stu_7070", "tags": ["coding"], "content_type": "text", "karma": 70, "created_at": "2025-05-27T21:45:00Z"},
        {"post_id": "p24", "author_id": "stu_8080", "tags": ["python"], "content_type": "image", "karma": 30, "created_at": "2025-05-27T10:00:00Z"},
        {"post_id": "p25", "author_id": "stu_1010", "tags": ["ai"], "content_type": "video", "karma": 100, "created_at": "2025-05-27T08:00:00Z"},
        {"post_id": "p26", "author_id": "stu_2020", "tags": ["travel"], "content_type": "image", "karma": 50, "created_at": "2025-05-27T08:00:00Z"},
        {"post_id": "p27", "author_id": "stu_3030", "tags": ["startups"], "content_type": "text", "karma": 80, "created_at": "2025-05-27T23:30:00Z"},
        {"post_id": "p28", "author_id": "stu_4040", "tags": ["python"], "content_type": "video", "karma": 85, "created_at": "2025-05-27T09:30:00Z"},
        {"post_id": "p29", "author_id": "stu_5050", "tags": ["ai"], "content_type": "text", "karma": 90, "created_at": "2025-05-27T07:45:00Z"},
        {"post_id": "p30", "author_id": "stu_6060", "tags": ["food"], "content_type": "text", "karma": 20, "created_at": "2025-05-27T19:00:00Z"},
        {"post_id": "p31", "author_id": "stu_1025", "tags": ["ai", "projects"], "content_type": "text", "karma": 65, "created_at": "2025-06-01T10:30:00Z"}
    ]

    input_json = {
         "user_id": "stu_9999",
        "posts": test_posts,
        "user_profile": {
            "branches_of_interest": ["AI", "DS"],
            "tags_followed": ["coding", "python", "ai", "ml"],
            "buddies": ["stu_1010", "stu_2020", "stu_3030"],
            "active_hours": ["07:00-09:00", "20:00-23:00"]
        }
    }

    output = rank_posts(
        user_id=input_json["user_id"],
        posts=input_json["posts"],
        user_profile=input_json["user_profile"]
    )

    print(json.dumps(output, indent=2))

# Sample usage for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run feed ranker")
    parser.add_argument("--print", action="store_true", help="Print input features")
    args = parser.parse_args()
    main()
