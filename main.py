from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json
from app import ranker


app = FastAPI(title="Feed Ranking API", version="1.0")

# -------------------- Pydantic Models -------------------- #

class UserProfile(BaseModel):
    branches_of_interest: List[str]
    tags_followed: List[str]
    buddies: List[str]
    active_hours: List[str]

class PostInput(BaseModel):
    post_id: str
    author_id: str
    tags: List[str]
    content_type: str
    karma: int
    created_at: str

class RankRequest(BaseModel):
    user_id: str
    user_profile: UserProfile
    posts: List[PostInput]

class RankedPost(BaseModel):
    post_id: str
    score: float

class RankResponse(BaseModel):
    user_id: str
    ranked_posts: List[RankedPost]
    status: str

# -------------------- Endpoints -------------------- #
@app.get("/")
def read_root():
    return {"message": "Welcome to the Feed Personalization API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/version")
def get_version():
    return {"version": app.version}

@app.post("/rank-feed", response_model=RankResponse)
def rank_feed(request: RankRequest):
    result = ranker.rank_posts(
        user_id=request.user_id,
        posts=[post.model_dump() for post in request.posts],
        user_profile=request.user_profile.model_dump()
    )
    return result
