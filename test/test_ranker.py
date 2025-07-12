# test/test_ranker.py
import pytest
import sys
import os

# Add the root project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ranker import rank_posts


# Sample input
user_profile = {
    "branches_of_interest": ["AI"],
    "tags_followed": ["python", "ml"],
    "buddies": ["u1"],
    "active_hours": ["07:00-09:00", "20:00-23:00"]
}

def sample_post(pid, karma, author, tags):
    return {
        "post_id": pid,
        "author_id": author,
        "tags": tags,
        "content_type": "text",
        "karma": karma,
        "created_at": "2025-05-27T07:30:00Z"
    }

def test_rank_logic_high_signal():
    posts = [sample_post("p1", 90, "u1", ["ml"])]
    result = rank_posts("user1", posts, user_profile)
    assert result["ranked_posts"][0]["score"] > 0.8

def test_rank_logic_irrelevant():
    posts = [sample_post("p2", 10, "u9", ["random"])]
    result = rank_posts("user1", posts, user_profile)
    assert result["ranked_posts"][0]["score"] < 0.5

def test_fallback_trigger():
    posts = [sample_post("p3", 2, "u9", ["unknown"])]
    result = rank_posts("user1", posts, user_profile)
    assert len(result["ranked_posts"]) == 1  # should not crash
