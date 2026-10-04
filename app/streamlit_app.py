"""
streamlit_app.py

Web page for the Customer Review Aspect Analyzer.
Run from the project folder with:

    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

# so that "import config" and "import src..." work when streamlit runs this file
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st

import config
from src.aggregate import analyze_review
from src.predict import AspectSentimentPredictor

st.set_page_config(page_title="Review Aspect Analyzer", layout="centered")

SENTIMENT_COLORS = {"negative": "#d9534f", "neutral": "#9e9e9e", "positive": "#4caf50"}

# ready-made reviews for people who don't want to type
EXAMPLES = [
    "The food was amazing but the service was really slow and the prices are too high.",
    "Great pizza and friendly staff. The ambience was lovely too.",
    "The waiter was rude and the soup was cold. Never coming back.",
    "The dessert was heavenly, but the music was too loud and the tables were cramped.",
    "The biryani was flavorful and the portion was generous.",
    "Terrible service and the bill was far too expensive. We waited an hour for our food.",
    "We ordered pasta and a salad. The restaurant is near the station.",
]


@st.cache_resource(show_spinner="Loading model...")
def load_predictor():
    return AspectSentimentPredictor(config.MODEL_DIR)  # loaded only once


def color_sentiment(value):
    return f"color: {SENTIMENT_COLORS[value]}; font-weight: 600"


def use_example():
    # runs when the dropdown changes: copy the chosen example into the text box
    if st.session_state.example_choice in EXAMPLES:
        st.session_state.review_text = st.session_state.example_choice


st.title("Customer Review Aspect Analyzer")
st.caption("What are customers talking about, and how do they feel about each thing?")

if not (config.MODEL_DIR / "config.json").exists():
    st.error("No trained model found. Train one first (see README) and copy it into models/.")
    st.stop()

predictor = load_predictor()

if "review_text" not in st.session_state:
    st.session_state.review_text = EXAMPLES[0]

st.selectbox("Don't want to type? Pick an example review",
             ["-- choose an example --"] + EXAMPLES,
             key="example_choice", on_change=use_example)

review = st.text_area("Or write / paste your own review", key="review_text", height=120)

if st.button("Analyze review", type="primary"):
    if not review.strip():
        st.warning("Please write a review or pick an example first.")
    else:
        result = analyze_review(review, predictor)
        if result.empty:
            st.warning("No aspects were found in this review. Try a review that "
                       "mentions things like food, service or price.")
        else:
            st.dataframe(result.style.map(color_sentiment, subset=["sentiment"]),
                         use_container_width=True)
