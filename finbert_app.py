import streamlit as st
import torch
import scipy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config('Sentiment-Analysis', layout="centered")

# Set up the app title and description
st.title("FinBERT Financial Sentiment Analysis")
st.write("""This app uses the FinBERT model to analyze sentiment in financial text.""")

# Add some examples
st.write("#### Example Inputs")
st.write("""
    - The company's financial performance was great this quarter.
    - ABC Company goods drop off by 5% and company to go on bankrupt.
    - AP Oil CEO says 'Sales at no positive or negative changes' after 6 months of production
    """)

# Load the model and tokenizer
@st.cache_resource(show_spinner=False)  # Added show_spinner=False
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Function to predict sentiment
def predict_sentiment(text):
    tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}
    
    with torch.no_grad():
        input_sequence = tokenizer(text, return_tensors="pt", **tokenizer_kwargs)
        logits = model(**input_sequence).logits
        scores = {
            k: v
            for k, v in zip(
                model.config.id2label.values(),
                scipy.special.softmax(logits.numpy().squeeze()),
            )
        }
    
    predicted_sentiment = max(scores, key=scores.get)
    confidence_score = max(scores.values())
    
    return predicted_sentiment, confidence_score, scores

# Create input text area
user_input = st.text_area(label="Enter financial text to analyze:", height=100)

if st.button("Predict Sentiment"):
    if user_input:
        # Get prediction
        sentiment, confidence, all_scores = predict_sentiment(user_input)
        
        # Display results
        st.subheader("Results")
        
        # Sentiment with color coding
        if sentiment == "positive":
            st.success(f"Sentiment: {sentiment.title()} (Confidence: {confidence:.2%})")
        elif sentiment == "negative":
            st.error(f"Sentiment: {sentiment.title()} (Confidence: {confidence:.2%})")
        else:
            st.info(f"Sentiment: {sentiment.title()} (Confidence: {confidence:.2%})")

    else:
        st.warning("Please enter some text to analyze.")


# Add footer
st.markdown("---")
st.markdown("Built with [FinBERT](https://huggingface.co/ProsusAI/finbert) and deployed on [Streamlit](https://streamlit.io/) for 3mtt Deep Drive Project by Godsgift Olomu")