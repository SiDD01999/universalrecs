import pytest
import pandas as pd
import numpy as np
from src.recommender import RecommenderEngine

@pytest.fixture
def engine():
    # Force retraining for tests if needed, but the constructor loads data
    return RecommenderEngine()

def test_engine_initialization(engine):
    assert engine.movies is not None
    assert engine.ratings is not None
    assert not engine.movies.empty
    assert not engine.ratings.empty

def test_get_popular_items(engine):
    popular = engine.get_popular_items(n=5)
    assert len(popular) == 5
    assert "title" in popular[0]
    assert "score" in popular[0]
    assert "reason" in popular[0]
    assert popular[0]["reason"] == "Popular Outcome"

def test_recommend_new_user(engine):
    # User 9999 is likely a new user (not in dummy data)
    recs, method = engine.recommend(user_id=9999, n=5)
    assert method == "Popularity (New User)"
    assert len(recs) == 5

def test_recommend_existing_user(engine):
    # User 1 exists in dummy data
    recs, method = engine.recommend(user_id=1, n=5)
    assert method == "Hybrid"
    assert len(recs) == 5
    for r in recs:
        reason = r["reason"]
        assert any(reason.startswith(p) for p in ["Because you liked", "Users like you also enjoyed", "Popular Outcome"])

def test_add_feedback(engine):
    initial_count = len(engine.ratings)
    # Use a likely unique item for this user to test addition
    engine.add_feedback(user_id=1, movie_id=999, rating=5.0)
    assert len(engine.ratings) == initial_count + 1
    # Check if it retrains (or at least doesn't crash)
    assert engine.user_item_matrix is not None
