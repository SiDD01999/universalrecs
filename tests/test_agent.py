import pytest
from src.agent import search_movies, recommend_movies, router_node, AgentState
from langchain_core.messages import HumanMessage, AIMessage

def test_search_movies_tool():
    result = search_movies.invoke({"query": "Action"})
    assert isinstance(result, str)
    assert "Here are some movies" in result or "No movies found" in result

def test_recommend_movies_tool():
    # Testing for User 1
    result = recommend_movies.invoke({"user_id": 1})
    assert isinstance(result, str)
    assert "personalized recommendations" in result

def test_router_node_fallback():
    # Mock state
    state = {
        "messages": [HumanMessage(content="recommend a movie")],
        "user_id": 1
    }
    # Since we have a key in .env, it should attempt to call Gemini.
    # If the key is invalid or network fails, it should fallback.
    result = router_node(state)
    assert "messages" in result
    assert "final_response" in result
