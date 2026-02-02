import streamlit as st
import pandas as pd
import time
from src.recommender import RecommenderEngine
from src.evaluator import Evaluator

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
st.caption("Hybrid Recommendation Engine with content-based & collaborative filtering")

# Initialize Engine (Cached)
@st.cache_resource
def get_engine():
    return RecommenderEngine()

engine = get_engine()

# Sidebar - User Selection
st.sidebar.header("User Profile")
user_ids = sorted(engine.ratings['userId'].unique())
selected_user = st.sidebar.selectbox("Select User", [0] + list(user_ids), format_func=lambda x: "New User" if x == 0 else f"User {x}")

# Main Content
if selected_user == 0:
    st.info("Welcome! As a new user, we'll show you what's popular.")
    # Use a dummy fresh ID for prediction (e.g., max + 1)
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
        # Card-like layout
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
                # Feedback Buttons
                # Unique keys required for Streamlit buttons
                if st.button("👍 Like", key=f"like_{item['movieId']}"):
                    engine.add_feedback(current_uid, item['movieId'], 5.0)
                    st.toast(f"Liked {item['title']}! Retraining...", icon="🎉")
                    time.sleep(1) # slight pause
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

