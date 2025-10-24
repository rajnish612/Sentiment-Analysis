import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st


st.set_page_config(page_title="Sentiment Analysis App", layout="centered")


@st.cache_resource
def load_models():
    model = load_model("sentiment_analysis_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


model, tokenizer = load_models()


def predict(text):
    tokenize = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(tokenize, maxlen=600, padding="post")
    prediction = model.predict(sequence, verbose=0)
    return prediction[0][0]


st.title(" Sentiment Analysis App")

st.markdown(
    """
<div style='background-color: #1e3a5f; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
    <p style='margin: 0; font-size: 16px; color: white;'><strong>Project:</strong> NLP-based Sentiment Analysis</p>
    <p style='margin: 5px 0; font-size: 16px; color: white;'><strong>Developer:</strong> Rajnish Nath</p>
    <p style='margin: 5px 0 0 0; font-size: 16px; color: white;'><strong>Education:</strong> BCA Undergraduate</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("### Analyze the sentiment of your text using LSTM model")
st.markdown("---")

# Text input
user_input = st.text_area(
    "Enter your text here:",
    placeholder="Type or paste your review, comment, or any text...",
    height=150,
)

# Predict button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button("Analyze Sentiment", use_container_width=True)

# Prediction and results
if predict_button:
    if user_input.strip():
        with st.spinner("Analyzing..."):
            sentiment_score = predict(user_input)

        st.markdown("---")
        st.subheader("Results:")

        # Display sentiment
        col1, col2 = st.columns(2)

        with col1:
            if sentiment_score > 0.5:
                st.success("Positive")
            else:
                st.error("Negative")

    else:
        st.warning("Please enter some text to analyze.")

st.sidebar.header("Example Texts")

examples = [
    "This movie was fantastic! I really loved it.",
    "Terrible experience. Would not recommend to anyone.",
    "It was okay, nothing special but not bad either.",
    "Absolutely amazing! Best movie I've seen in years!",
    "Waste of time and money. Very disappointed.",
]

for example in examples:
    if st.sidebar.button(example[:50] + "...", key=example):
        st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("About the Project")
st.sidebar.markdown(
    """
**Project:** NLP-based Sentiment Analysis

**Developer:** Rajnish Nath

**Education:** BCA Undergraduate

**Description:** This app uses a Bidirectional LSTM model trained on IMDB reviews to predict sentiment from text.

**Technology Stack:**
- TensorFlow/Keras
- Natural Language Processing (NLP)
- Streamlit
- Python
"""
)
