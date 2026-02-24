import streamlit as st
import pandas as pd
import time
import os
from dotenv import load_dotenv
from src.recommender import RecommenderEngine
from src.evaluator import Evaluator

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="UniversalRecs", layout="wide")

# Custom CSS for "Premium" feel
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
    .card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .card-title {
        color: #FFFFFF;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .card-genre {
        color: #AAAAAA;
        font-size: 0.9rem;
        margin-bottom: 10px;
    }
    .card-reason {
        color: #4CAF50;
        font-size: 0.85rem;
        font-style: italic;
    }
    .metric-box {
        background-color: #2D2D2D;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("UniversalRecs 🎬")
st.caption("Hybrid Recommendation Engine powered by Gemini & Collaborative Filtering")

# Initialize Engine (Cached)
@st.cache_resource
def get_engine():
    return RecommenderEngine()

engine = get_engine()

# --- Sidebar ---
st.sidebar.header("User Profile")
user_ids = sorted(engine.ratings['userId'].unique())
selected_user = st.sidebar.selectbox("Select User", [0] + list(user_ids), format_func=lambda x: "New User" if x == 0 else f"User {x}")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ API Settings")
env_key = os.getenv("GOOGLE_API_KEY")
api_key = st.sidebar.text_input("Gemini API Key", value=env_key if env_key else "", type="password", help="Enter your Gemini API key here. It will override the one in .env if provided.")

if api_key:
    if not env_key or api_key != env_key:
        st.sidebar.success("Using Manual API Key")
    else:
        st.sidebar.success("Using .env API Key")
else:
    st.sidebar.warning("No Gemini Key! Assistant will use keyword fallback.")

# Main Content Logic
if selected_user == 0:
    st.info("Welcome! As a new user, we'll show you what's popular.")
    current_uid = max(user_ids) + 1 if user_ids else 1
    is_new = True
else:
    current_uid = selected_user
    is_new = False

# Layout: Recommendations & Feedback
col_recs, col_stats = st.columns([3, 1])

with col_recs:
    st.subheader(f"Top Picks for {('New User' if is_new else f'User {current_uid}')}")
    
    with st.spinner("Crunching the numbers..."):
        recs, method = engine.recommend(current_uid, n=10)
    
    st.markdown(f"**Engine Mode:** `{method}`")
    
    for item in recs:
        with st.container():
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">{item['title']}</div>
                    <div class="card-genre">{item['genres']}</div>
                    <div class="card-reason">💡 {item['reason']}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                if st.button("👍 Like", key=f"like_{item['movieId']}"):
                    engine.add_feedback(current_uid, item['movieId'], 5.0)
                    st.toast(f"Liked {item['title']}! Retraining...", icon="🎉")
                    time.sleep(1)
                    st.rerun()
                    
                if st.button("👎 Dislike", key=f"dislike_{item['movieId']}"):
                    engine.add_feedback(current_uid, item['movieId'], 1.0)
                    st.toast(f"Disliked {item['title']}. Tuning...", icon="🔧")
                    time.sleep(1)
                    st.rerun()

with col_stats:
    st.subheader("Engine Stats")
    
    if st.button("Calculate Metrics"):
        evaluator = Evaluator(engine)
        with st.spinner("Evaluating..."):
            rmse = evaluator.calculate_rmse()
            coverage = evaluator.calculate_coverage()
        
        st.metric("RMSE Error", f"{rmse:.3f}" if not pd.isna(rmse) else "N/A", delta_color="inverse")
        st.metric("Catalog Coverage", f"{coverage:.1%}")
        
    st.markdown("---")
    st.write("### Data Overview")
    st.write(f"**Users:** {len(engine.ratings['userId'].unique())}")
    st.write(f"**Items:** {len(engine.movies)}")
    st.write(f"**Interactions:** {len(engine.ratings)}")

# --- Agentic Chat Interface ---
st.markdown("---")
st.header("🤖 AI Assistant (Gemini Powered)")
st.caption("Ask me anything! e.g., 'I want a funny movie' or 'What should I watch?'")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What are you looking for?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    from src.agent import workflow, AgentState
    from langchain_core.messages import HumanMessage
    
    app_graph = workflow.compile()
    
    with st.spinner("Gemini is thinking..."):
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "user_id": current_uid,
            "google_api_key": api_key
        }
        
        try:
            result = app_graph.invoke(initial_state)
            response_text = result.get("final_response", "I'm not sure how to help with that.")
        except Exception as e:
            response_text = f"Error calling Gemini: {e}. Please check your API key."
        
        with st.chat_message("assistant"):
            st.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})
