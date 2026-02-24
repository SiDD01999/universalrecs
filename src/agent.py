import os
from typing import TypedDict, Annotated, List, Union
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from .recommender import RecommenderEngine

# Load environment variables
load_dotenv()

# --- 1. Define State ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    user_id: int
    google_api_key: str
    final_response: str

# --- 2. Define Tools ---
_engine = RecommenderEngine()

@tool
def search_movies(query: str):
    """
    Search for movies by title, genre, or description.
    Use this when the user asks for specific types of movies (e.g., 'action movies', 'movies about space').
    """
    results = _engine.search_items(query, n=5)
    if not results:
        return "No movies found matching that query."
    
    response = "Here are some movies I found:\n"
    for m in results:
        response += f"- {m['title']} ({m['genres']})\n"
    return response

@tool
def recommend_movies(user_id: int):
    """
    Get personalized movie recommendations for a user.
    Use this when the user asks for general suggestions (e.g., 'what should I watch?', 'recommend something').
    The user_id is required.
    """
    recs, method = _engine.recommend(user_id, n=5)
    response = f"Here are your personalized recommendations ({method}):\n"
    for m in recs:
        response += f"- {m['title']} ({m['score']:.2f}): {m['reason']}\n"
    return response

# --- 3. Define Nodes ---

def router_node(state: AgentState):
    """
    Decides which tool to call based on the user's last message.
    """
    messages = state['messages']
    last_message = messages[-1].content
    user_id = state.get('user_id', 1) 

    # Gemini Router
    api_key = state.get('google_api_key') or os.getenv("GOOGLE_API_KEY")
    
    if api_key:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        tools = [search_movies, recommend_movies]
        llm_with_tools = llm.bind_tools(tools)
        
        try:
            response = llm_with_tools.invoke(messages)
            
            if response.tool_calls:
                return {"messages": [response]}
            else:
                return {"messages": [response], "final_response": response.content}
        except Exception as e:
            print(f"Gemini API Error: {e}")
            # Fallback will handle it
    
    # Fallback: Keyword Router
    print("Warning: Gemini integration issue or missing key. Using keyword router.")
    lower_msg = last_message.lower()
    if "recommend" in lower_msg or "watch" in lower_msg or "suggest" in lower_msg:
         result = recommend_movies.invoke({"user_id": user_id})
         return {"messages": [AIMessage(content=result)], "final_response": result}
    else:
         query = last_message.replace("search", "").strip() or "action"
         result = search_movies.invoke({"query": query})
         return {"messages": [AIMessage(content=result)], "final_response": result}

def tool_execution_node(state: AgentState):
    """
    Executes the tool selected by the LLM.
    """
    messages = state['messages']
    last_message = messages[-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {}

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call['name']
    tool_args = tool_call['args']
    
    result = ""
    if tool_name == "search_movies":
        result = search_movies.invoke(tool_args)
    elif tool_name == "recommend_movies":
        if 'user_id' not in tool_args:
             tool_args['user_id'] = state.get('user_id', 1)
        result = recommend_movies.invoke(tool_args)
    
    return {"messages": [AIMessage(content=result)], "final_response": result}

# --- 4. Define Graph ---

def route_condition(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("tools", tool_execution_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    route_condition,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge("tools", END)

app_graph = workflow.compile()
