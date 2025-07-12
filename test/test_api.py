from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import app
client = TestClient(app)

def test_rank_feed_endpoint():
    response = client.post("/rank-feed", json={
        "user_id": "testuser",
        "user_profile": {
            "branches_of_interest": ["AI"],
            "tags_followed": ["ml"],
            "buddies": ["u1"],
            "active_hours": ["07:00-09:00"]
        },
        "posts": [{
            "post_id": "p1",
            "author_id": "u1",
            "tags": ["ml"],
            "content_type": "text",
            "karma": 90,
            "created_at": "2025-05-27T07:30:00Z"
        }]
    })
    assert response.status_code == 200
    assert response.json()["ranked_posts"][0]["score"] > 0.8
def test_rank_feed_missing_post_id():
    response = client.post("/rank-feed", json={
        "user_id": "testuser",
        "user_profile": {
            "branches_of_interest": ["AI"],
            "tags_followed": ["ml"],
            "buddies": ["u1"],
            "active_hours": ["07:00-09:00"]
        },
        "posts": [{
            # "post_id" is missing
            "author_id": "u1",
            "tags": ["ml"],
            "content_type": "text",
            "karma": 90,
            "created_at": "2025-05-27T07:30:00Z"
        }]
    })
    assert response.status_code == 422  # Unprocessable Entity (FastAPI validation)
def test_rank_feed_empty_post_list():
    response = client.post("/rank-feed", json={
        "user_id": "testuser",
        "user_profile": {
            "branches_of_interest": ["AI"],
            "tags_followed": ["ml"],
            "buddies": ["u1"],
            "active_hours": ["07:00-09:00"]
        },
        "posts": []
    })
    assert response.status_code == 200
    assert response.json()["ranked_posts"] == []
    assert response.json()["status"] == "empty"
def test_rank_feed_fallback_trigger():
    response = client.post("/rank-feed", json={
        "user_id": "testuser",
        "user_profile": {
            "branches_of_interest": [],
            "tags_followed": [],
            "buddies": [],
            "active_hours": ["00:00-01:00"]
        },
        "posts": [{
            "post_id": "fallback_1",
            "author_id": "nonbuddy",
            "tags": ["random"],
            "content_type": "text",
            "karma": 0,
            "created_at": "2025-05-27T12:00:00Z"
        }]
    })
    assert response.status_code == 200
    assert response.json()["ranked_posts"][0]["score"] >= 0  # fallback assigns valid score
def test_missing_user_id():
    response = client.post("/rank-feed", json={
        # "user_id": "testuser",  # intentionally omitted
        "user_profile": {
            "branches_of_interest": ["AI"],
            "tags_followed": ["ml"],
            "buddies": ["u1"],
            "active_hours": ["07:00-09:00"]
        },
        "posts": [{
            "post_id": "p1",
            "author_id": "u1",
            "tags": ["ml"],
            "content_type": "text",
            "karma": 90,
            "created_at": "2025-05-27T07:30:00Z"
        }]
    })
    assert response.status_code == 422
def test_invalid_karma_type():
    response = client.post("/rank-feed", json={
        "user_id": "testuser",
        "user_profile": {
            "branches_of_interest": ["AI"],
            "tags_followed": ["ml"],
            "buddies": ["u1"],
            "active_hours": ["07:00-09:00"]
        },
        "posts": [{
            "post_id": "p1",
            "author_id": "u1",
            "tags": ["ml"],
            "content_type": "text",
            "karma": "high",  # should be an int
            "created_at": "2025-05-27T07:30:00Z"
        }]
    })
    assert response.status_code == 422
def test_empty_user_profile():
    response = client.post("/rank-feed", json={
        "user_id": "testuser",
        "user_profile": {},  # empty dict
        "posts": [{
            "post_id": "p1",
            "author_id": "u1",
            "tags": ["ml"],
            "content_type": "text",
            "karma": 90,
            "created_at": "2025-05-27T07:30:00Z"
        }]
    })
    assert response.status_code == 422
def test_missing_post_id():
    response = client.post("/rank-feed", json={
        "user_id": "testuser",
        "user_profile": {
            "branches_of_interest": ["AI"],
            "tags_followed": ["ml"],
            "buddies": ["u1"],
            "active_hours": ["07:00-09:00"]
        },
        "posts": [{
            # "post_id": "p1",  # intentionally omitted
            "author_id": "u1",
            "tags": ["ml"],
            "content_type": "text",
            "karma": 90,
            "created_at": "2025-05-27T07:30:00Z"
        }]
    })
    assert response.status_code == 422
def test_malformed_json():
    response = client.post("/rank-feed", data="{not: valid json}")
    assert response.status_code == 422 or response.status_code == 400
def test_unexpected_extra_field():
    response = client.post("/rank-feed", json={
        "user_id": "testuser",
        "user_profile": {
            "branches_of_interest": ["AI"],
            "tags_followed": ["ml"],
            "buddies": ["u1"],
            "active_hours": ["07:00-09:00"]
        },
        "posts": [{
            "post_id": "p1",
            "author_id": "u1",
            "tags": ["ml"],
            "content_type": "text",
            "karma": 90,
            "created_at": "2025-05-27T07:30:00Z",
            "extra_field": "oops"
        }]
    })
    # FastAPI ignores extra fields unless you disallow them explicitly
    assert response.status_code == 200
